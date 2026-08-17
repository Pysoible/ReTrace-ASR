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
    second, char_confs = _transcribe_paraformer_detailed(audio_path)
    disagreements: list[dict[str, Any]] = []
    if second and first_pass_text:
        disagreements = _diff_texts(first_pass_text, second, min_span)
    low_conf = [c for c in char_confs if c["conf"] < low_conf_threshold]
    return disagreements, low_conf


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

