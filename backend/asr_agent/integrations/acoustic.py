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


def _transcribe_paraformer(audio_path: str | Path) -> str:
    model = _get_paraformer()
    if not model:
        return ""
    try:
        result = model.generate(input=str(audio_path))
    except Exception:
        return ""
    if not result:
        return ""
    return str((result[0].get("text") or "") if isinstance(result, list) else "").strip()


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


def transcribe_window(audio_path: str | Path, start_sec: float, end_sec: float) -> str:
    """Re-transcribe one audio window with paraformer (the second acoustic ASR).

    Used as an *independent acoustic* re-scoring of a focused span — instead of
    asking the LLM (Qwen-Omni) to re-listen to the whole window, a classical
    acoustic ASR votes on what it hears at that exact position.
    """
    if end_sec <= start_sec:
        return ""
    try:
        audio, sr = _load_mono(audio_path)
    except Exception:
        return ""
    seg = _crop(audio, sr, start_sec, end_sec)
    if len(seg) < sr * 0.05:
        return ""
    import soundfile as sf
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        tmp = Path(handle.name)
        sf.write(str(tmp), seg, sr)
    try:
        return _transcribe_paraformer(tmp)
    finally:
        tmp.unlink(missing_ok=True)


def _pinyin_text(text: str) -> str:
    try:
        from pypinyin import lazy_pinyin

        return "".join(lazy_pinyin(text)).lower()
    except Exception:  # pragma: no cover - optional dependency
        return text


def acoustic_verify(
    audio_path: str | Path,
    start_sec: float,
    end_sec: float,
    candidates: list[str],
) -> dict[str, Any]:
    """Acoustic re-scoring of a focused window via a second ASR's transcript.

    The second ASR (paraformer) re-transcribes the window; each candidate is
    scored by pinyin similarity to that transcript. Returns the same shape as
    ``verify_candidates`` (``{"ok": True, "scores": {...}}``) so it can be used
    as an alternative / additional acoustic authorizer.
    """
    unique = list(dict.fromkeys(str(item).strip() for item in candidates if item and str(item).strip()))
    if len(unique) < 2:
        return {"ok": False, "error": "need at least two candidates"}
    second = transcribe_window(audio_path, start_sec, end_sec)
    if not second:
        return {"ok": False, "error": "paraformer re-transcription unavailable"}
    second_py = _pinyin_text(second)
    if not second_py:
        return {"ok": False, "error": "pinyin unavailable"}
    scores: dict[str, float] = {}
    for candidate in unique:
        cand_py = _pinyin_text(candidate)
        if not cand_py:
            scores[candidate] = 0.0
            continue
        scores[candidate] = difflib.SequenceMatcher(None, cand_py, second_py).ratio()
    total = sum(scores.values())
    if total <= 0:
        return {"ok": False, "error": "no acoustic similarity"}
    return {"ok": True, "scores": {candidate: value / total for candidate, value in scores.items()}}

