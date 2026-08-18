"""Acoustic signal processing for ReTrace-ASR.

This module supplies the *acoustic* side of the pipeline, which was previously
missing: instead of feeding audio to an LLM and asking it to decide, we compute
classical acoustic features and use a second, independent ASR (paraformer) to
estimate where the audio itself is ambiguous.

Two capabilities:

1. ``detect_asr_disagreement`` — run paraformer on an audio window and diff its
   transcript against the first-pass (Qwen-Omni) transcript. Spans where two
   independent acoustic models disagree are strong "acoustic uncertainty"
   signals, and are surfaced to the judge / relisten machinery.

2. ``mfcc_distance`` — extract MFCC features for two audio windows and measure
   their acoustic distance (DTW). This is used to *authorize* a revision with a
   real acoustic comparison rather than an LLM's impression of the whole window.

Both are optional at runtime: when torchaudio/funasr are unavailable the
functions degrade to empty results instead of raising.
"""
from __future__ import annotations

import difflib
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

_PARAFORMER: Any = None
_PARAFORMER_LOCK = threading.Lock()
PARAFORMER_MODEL = os.environ.get(
    "ASR_ACOUSTIC_MODEL",
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)


def _load_mono(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def _get_paraformer() -> Any:
    global _PARAFORMER
    with _PARAFORMER_LOCK:
        if _PARAFORMER is None:
            try:
                from funasr import AutoModel

                _PARAFORMER = AutoModel(model=PARAFORMER_MODEL, disable_update=True)
            except Exception:  # pragma: no cover - optional dependency
                _PARAFORMER = False
        return _PARAFORMER


def _transcribe_paraformer_detailed(audio_path: str | Path) -> tuple[str, list[dict[str, Any]]]:
    """Run paraformer once; return ``(text, char_confs)``.

    ``char_confs`` is ``[{"char": "呢", "conf": 0.58}, ...]`` in recognition
    order. Sharing one forward pass means disagreement detection and char-level
    confidence do not pay the model cost twice.
    """
    model = _get_paraformer()
    if not model:
        return "", []
    try:
        import torch
        from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

        paraformer = model.model
        kwargs = model.kwargs
        frontend = kwargs.get("frontend")
        tokenizer = kwargs.get("tokenizer")
        device = kwargs.get("device", "cpu")
        fs = kwargs.get("fs", 16000)

        audio_sample_list = load_audio_text_image_video(
            str(audio_path),
            fs=frontend.fs,
            audio_fs=fs,
            data_type="sound",
            tokenizer=tokenizer,
        )
        speech, speech_lengths = extract_fbank(
            audio_sample_list, data_type="sound", frontend=frontend
        )
        speech = speech.to(device=device)
        speech_lengths = speech_lengths.to(device=device)

        with torch.no_grad():
            encoder_out, encoder_out_lens = paraformer.encode(speech, speech_lengths)
            if isinstance(encoder_out, tuple):
                encoder_out = encoder_out[0]
            predictor_outs = paraformer.calc_predictor(encoder_out, encoder_out_lens)
            pre_token_length = predictor_outs[1].round().long()
            if torch.max(pre_token_length) < 1:
                return "", []
            decoder_outs = paraformer.cal_decoder_with_predictor(
                encoder_out, encoder_out_lens, predictor_outs[0], pre_token_length
            )
            am_scores = decoder_outs[0][0, : pre_token_length[0], :]
            probs = torch.softmax(am_scores, dim=-1)
            yseq = am_scores.argmax(dim=-1).tolist()
            confs = probs.max(dim=-1)[0].tolist()

        kept_ids: list[int] = []
        char_confs: list[dict[str, Any]] = []
        for token_id, conf in zip(yseq, confs):
            if token_id in (paraformer.sos, paraformer.eos, paraformer.blank_id):
                continue
            kept_ids.append(token_id)
            token = str(tokenizer.ids2tokens([token_id])[0] or "")
            if token and (token.isalnum() or "\u4e00" <= token <= "\u9fff"):
                char_confs.append({"char": token, "conf": round(float(conf), 4)})
        text = tokenizer.tokens2text(tokenizer.ids2tokens(kept_ids))
        return str(text or "").strip(), char_confs
    except Exception:  # pragma: no cover - optional dependency
        return "", []


def _transcribe_paraformer(audio_path: str | Path) -> str:
    return _transcribe_paraformer_detailed(audio_path)[0]


def _diff_texts(first_pass_text: str, second: str, min_span: int) -> list[dict[str, Any]]:
    sm = difflib.SequenceMatcher(None, first_pass_text, second, autojunk=False)
    spans: list[dict[str, Any]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        span_a = first_pass_text[i1:i2]
        span_b = second[j1:j2]
        # Only report meaningful (CJK/alnum) disagreements of a minimum length.
        a_clean = "".join(ch for ch in span_a if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        b_clean = "".join(ch for ch in span_b if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if len(a_clean) < min_span and len(b_clean) < min_span:
            continue
        spans.append(
            {
                "tag": tag,
                "span_a": span_a,
                "span_b": span_b,
                "offset_a": i1,
                "offset_b": j1,
            }
        )
    return spans


def acoustic_signals(
    audio_path: str | Path,
    first_pass_text: str,
    *,
    min_span: int = 2,
    low_conf_threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One paraformer forward pass → (disagreement_spans, low_conf_chars).

    This is the single acoustic entry point used by the streaming pipeline, so a
    chunk is never transcribed twice by the second ASR.
    """
    full = acoustic_signals_full(
        audio_path, first_pass_text, min_span=min_span, low_conf_threshold=low_conf_threshold
    )
    return full["disagreements"], full["low_conf_chars"]


def acoustic_signals_full(
    audio_path: str | Path,
    first_pass_text: str,
    *,
    min_span: int = 2,
    low_conf_threshold: float = 0.75,
) -> dict[str, Any]:
    """One paraformer forward pass → all acoustic signals in a single dict.

    ``paraformer_text`` and ``char_confs`` are kept so the agent can use the
    second ASR's own transcript (with per-character confidence) as a "second
    opinion" when re-listening to an acoustically doubtful window, rather than
    re-running the same first-pass model.
    """
    second, char_confs = _transcribe_paraformer_detailed(audio_path)
    disagreements: list[dict[str, Any]] = []
    if second and first_pass_text:
        disagreements = _diff_texts(first_pass_text, second, min_span)
    low_conf = [c for c in char_confs if c["conf"] < low_conf_threshold]
    return {
        "disagreements": disagreements,
        "low_conf_chars": low_conf,
        "paraformer_text": second,
        "char_confs": char_confs,
    }


def _keep_chars(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _pinyin_similar(a: str, b: str) -> bool:
    """True when two multi-char spans are near-homophones (transliteration variants).

    A proper noun in a session is usually one of several near-homophone
    transliterations (e.g. an in-game hero name). A paraformer "correction" that
    swaps one transliteration for another is not fixing a mis-hearing — it is
    just introducing a *second* spelling of the same entity, breaking session
    consistency. We therefore decline such replacements: if the first pass and
    paraformer spans share most syllables, keep the first pass spelling.

    Single-character spans return False here; callers expand single-character
    diffs with surrounding context before judging (see ``high_conf_correction``).
    """
    if not a or not b or a == b:
        return False
    if len(a) < 2 or len(b) < 2:
        return False
    try:
        from pypinyin import lazy_pinyin

        pa = [p for p in lazy_pinyin(a)]
        pb = [p for p in lazy_pinyin(b)]
    except Exception:  # pragma: no cover - optional dependency
        return False
    if not pa or not pb:
        return False
    if pa == pb:
        return True  # identical syllables over multiple chars → transliteration variant
    if len(pa) == len(pb):
        same = sum(1 for x, y in zip(pa, pb) if x == y)
        return same / len(pa) >= 0.5
    return False


def high_conf_correction(
    first_pass_text: str,
    paraformer_text: str,
    char_confs: list[dict[str, Any]],
    *,
    min_conf: float = 0.85,
) -> str:
    """Char-level correction using the second ASR's high-confidence disagreement.

    Diff the first pass against the paraformer transcript and apply only the
    differences where paraformer's own decoder confidence is >= ``min_conf``.
    Low-confidence paraformer characters are *not* trusted (the model itself is
    unsure there), and deletions (chars paraformer dropped) are kept from the
    first pass to avoid introducing new omissions. Multi-character near-homophone
    replacements (transliteration variants) are declined to keep one spelling of
    an entity throughout the session. Edits are applied at exact character
    positions, so the first pass's punctuation is preserved.
    """
    if not paraformer_text or not char_confs:
        return first_pass_text
    pf_clean = _keep_chars(paraformer_text)
    fp_clean = _keep_chars(first_pass_text)
    if not pf_clean or not fp_clean:
        return first_pass_text
    # char_confs must align 1:1 with pf_clean; otherwise bail out safely.
    if len(char_confs) != len(pf_clean) or any(
        c["char"] != ch for c, ch in zip(char_confs, pf_clean)
    ):
        return first_pass_text
    confs = [float(c["conf"]) for c in char_confs]

    def _high(j1: int, j2: int) -> bool:
        return j2 > j1 and all(confs[j] >= min_conf for j in range(j1, j2))

    def _is_transliteration_variant(i1: int, i2: int, j1: int, j2: int) -> bool:
        span_a = fp_clean[i1:i2]
        span_b = pf_clean[j1:j2]
        if _pinyin_similar(span_a, span_b):
            return True
        # SequenceMatcher often splits a multi-character transliteration into a
        # one-character replacement surrounded by equal characters. Compare a
        # one-character difference with one shared character on either side so
        # "卡兹克" → "卡斯克" remains one protected entity-level variant.
        if len(span_a) == len(span_b) == 1:
            left_a = fp_clean[max(0, i1 - 1):i1]
            left_b = pf_clean[max(0, j1 - 1):j1]
            right_a = fp_clean[i2:i2 + 1]
            right_b = pf_clean[j2:j2 + 1]
            if left_a == left_b and right_a == right_b and left_a and right_a:
                return _pinyin_similar(left_a + span_a + right_a, left_b + span_b + right_b)
        return False

    # Original positions of the CJK/alnum characters in first_pass_text.
    fp_pos = [i for i, ch in enumerate(first_pass_text) if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]

    sm = difflib.SequenceMatcher(None, fp_clean, pf_clean, autojunk=False)
    edits: list[tuple[int, int, str]] = []  # (start, end, replacement)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and _high(j1, j2):
            span_a = fp_clean[i1:i2]
            span_b = pf_clean[j1:j2]
            if _is_transliteration_variant(i1, i2, j1, j2):
                continue  # transliteration variant — keep the first pass spelling
            start = fp_pos[i1]
            end = fp_pos[i2 - 1] + 1 if i2 > i1 else fp_pos[i1]
            edits.append((start, end, span_b))
        elif tag == "insert" and _high(j1, j2):
            # Insert after the i1-th character (or at the very start).
            pos = 0 if i1 == 0 else fp_pos[i1 - 1] + 1
            edits.append((pos, pos, pf_clean[j1:j2]))
        elif tag == "delete":
            pass  # keep first pass chars paraformer dropped
    if not edits:
        return first_pass_text
    # Apply from right to left so earlier positions stay valid.
    chars = list(first_pass_text)
    for start, end, repl in sorted(edits, reverse=True):
        chars[start:end] = repl
    return "".join(chars)


def detect_asr_disagreement(
    audio_path: str | Path,
    first_pass_text: str,
    *,
    min_span: int = 2,
) -> list[dict[str, Any]]:
    """Return spans where a second ASR (paraformer) disagrees with the first pass.

    Each item: {tag, span_a (first pass), span_b (paraformer), offset_a, offset_b}.
    Only CJK/alnum runs of length >= min_span are reported, so punctuation-only
    or single-character differences are ignored.
    """
    second = _transcribe_paraformer(audio_path)
    if not second or not first_pass_text:
        return []
    return _diff_texts(first_pass_text, second, min_span)


def char_level_confidence(audio_path: str | Path) -> list[dict[str, Any]]:
    """Return per-character acoustic confidence from the paraformer decoder.

    Reuses the in-process paraformer (``_get_paraformer``) but reads the decoder
    logits directly, so we get a confidence value for every recognized character
    instead of just the final text. Low-confidence characters are the acoustic
    "doubt" signal for direction 2 (char-level acoustic confidence): they point
    at exactly which characters the model itself is unsure about, even when the
    sentence is semantically fluent.

    Returns ``[{"char": "呢", "conf": 0.58}, ...]`` in recognition order.
    Degrades to ``[]`` when funasr is unavailable or an error occurs.
    """
    return _transcribe_paraformer_detailed(audio_path)[1]


def extract_mfcc(
    audio: np.ndarray,
    sr: int,
    *,
    n_mfcc: int = 13,
    num_ceps: int = 13,
) -> np.ndarray:
    """Extract Kaldi-style MFCCs; returns (frames, n_mfcc) float32 array."""
    try:
        import torch
        import torchaudio
    except Exception:  # pragma: no cover - optional dependency
        return np.zeros((0, n_mfcc), dtype=np.float32)
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    try:
        feats = torchaudio.compliance.kaldi.mfcc(
            waveform,
            sample_frequency=sr,
            num_ceps=num_ceps,
            num_mel_bins=40,
            dither=0.0,
        )
    except Exception:
        return np.zeros((0, n_mfcc), dtype=np.float32)
    return feats.numpy().astype(np.float32)


def _crop(audio: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, int(start_sec * sr))
    end = min(len(audio), int(end_sec * sr))
    if end <= start:
        return np.zeros((0,), dtype=np.float32)
    return audio[start:end]


def mfcc_distance(
    audio_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    reference_audio: str | Path | None = None,
    reference_start: float = 0.0,
    reference_end: float = 0.0,
) -> float | None:
    """Acoustic distance between a query window and a reference window (DTW on MFCC).

    When ``reference_audio`` is None, the reference window is the first
    ``reference_end - reference_start`` seconds of the *same* audio (a crude
    self-reference; callers usually pass a known-good clip instead).

    Returns a non-negative distance, or None when features are unavailable.
    """
    try:
        query, sr = _load_mono(audio_path)
    except Exception:
        return None
    query_seg = _crop(query, sr, start_sec, end_sec)
    if reference_audio is not None:
        try:
            ref, _ = _load_mono(reference_audio)
        except Exception:
            return None
    else:
        ref = query
    ref_seg = _crop(ref, sr, reference_start, reference_end)
    if len(query_seg) < sr * 0.02 or len(ref_seg) < sr * 0.02:
        return None
    query_feat = extract_mfcc(query_seg, sr)
    ref_feat = extract_mfcc(ref_seg, sr)
    if query_feat.size == 0 or ref_feat.size == 0:
        return None
    return float(_dtw(query_feat, ref_feat))


def _dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric DTW distance between two feature sequences."""
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[n, m] / (n + m)

