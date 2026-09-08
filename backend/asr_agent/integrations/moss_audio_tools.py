"""MOSS-only focused audio re-observation tools.

These tools deliberately reuse MOSS for a small crop.  They are useful as a
focused second decode, but are not represented as independent acoustic evidence.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import numpy as np
import soundfile as sf
from Levenshtein import distance as levenshtein_distance
from pypinyin import Style, lazy_pinyin

from asr_agent.integrations import moss_asr

DELETE_CANDIDATE = "[DELETE]"


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", str(text or "")).lower()


def _tokens(text: str) -> list[str]:
    normalized = _normalize(text)
    return [str(item) for item in lazy_pinyin(normalized, style=Style.NORMAL, errors="default")]


def _best_similarity(reference: list[str], candidate: list[str]) -> float:
    if not candidate:
        return 0.0
    if not reference:
        return 0.0
    widths = range(max(1, len(candidate) - 2), min(len(reference), len(candidate) + 2) + 1)
    best = 0.0
    for width in widths:
        for start in range(0, len(reference) - width + 1):
            observed = reference[start : start + width]
            denominator = max(len(observed), len(candidate), 1)
            best = max(best, 1.0 - levenshtein_distance(observed, candidate) / denominator)
    return max(0.0, min(1.0, best))


def _candidate_similarity(transcript: str, candidate: str) -> float:
    observed_chars = list(_normalize(transcript))
    candidate_chars = list(_normalize(candidate))
    char_score = _best_similarity(observed_chars, candidate_chars)
    pinyin_score = _best_similarity(_tokens(transcript), _tokens(candidate))
    # Characters remain decisive for homophones; pinyin tolerates nearby ASR
    # substitutions without pretending that two spellings are identical.
    return 0.75 * char_score + 0.25 * pinyin_score


def _crop_mono_16k(audio_path: str, start_sec: float, end_sec: float) -> Path:
    source = Path(audio_path)
    if not source.exists():
        raise FileNotFoundError(f"audio is unavailable: {audio_path}")
    if end_sec <= start_sec:
        raise ValueError("need a valid audio window")
    with sf.SoundFile(source) as handle:
        source_rate = int(handle.samplerate)
        start_frame = max(0, int(start_sec * source_rate))
        end_frame = min(len(handle), int(end_sec * source_rate))
        if end_frame <= start_frame:
            raise ValueError("audio window is empty")
        handle.seek(start_frame)
        samples = handle.read(end_frame - start_frame, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    target_rate = 16000
    if source_rate != target_rate and len(mono) > 1:
        target_length = max(1, round(len(mono) * target_rate / source_rate))
        source_axis = np.arange(len(mono), dtype=np.float64)
        target_axis = np.arange(target_length, dtype=np.float64) * source_rate / target_rate
        mono = np.interp(target_axis, source_axis, mono).astype(np.float32)
    temp = NamedTemporaryFile(suffix=".wav", delete=False)
    temp.close()
    output = Path(temp.name)
    sf.write(output, mono, target_rate, subtype="PCM_16")
    return output


def _transcribe_crop(path: Path) -> str:
    payload = moss_asr._post_transcription(path)
    raw = str(payload.get("text") or payload.get("transcript") or "")
    return str(moss_asr.parse_moss_transcript(raw).get("text") or "").strip()


def verify_candidates(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    candidates: list[str],
    *,
    transcriber: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Re-decode one crop with MOSS and score exactly the supplied set."""
    unique = list(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))
    if len(unique) < 2:
        return {"ok": False, "error": "need at least two candidates and a valid audio window"}
    clipped: Path | None = None
    try:
        clipped = _crop_mono_16k(audio_path, start_sec, end_sec)
        transcript = (transcriber or _transcribe_crop)(clipped).strip()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"MOSS focused verification failed: {exc}"}
    finally:
        if clipped is not None:
            clipped.unlink(missing_ok=True)
    if not transcript:
        return {"ok": False, "error": "MOSS focused verification returned empty text"}

    similarities = {
        candidate: (
            0.05
            if candidate == DELETE_CANDIDATE
            else _candidate_similarity(transcript, candidate)
        )
        for candidate in unique
    }
    # A softmax produces normalized evidence while preserving uncertainty when
    # the focused decode does not clearly contain any supplied candidate.
    weights = {candidate: math.exp(6.0 * score) for candidate, score in similarities.items()}
    total = sum(weights.values())
    return {
        "ok": True,
        "scores": {candidate: value / total for candidate, value in weights.items()},
        "focused_transcript": transcript,
        "method": "same_model_focused_reobservation",
        "independent_acoustic_evidence": False,
        "audio_path": audio_path,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def retranscribe_window(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    *,
    transcriber: Callable[[Path], str] | None = None,
    domain_hints: list[str] | None = None,
    recover_coverage: bool = False,
) -> dict[str, Any]:
    """Open MOSS re-decode of a local crop; hints are intentionally ignored."""
    del domain_hints, recover_coverage
    clipped: Path | None = None
    try:
        clipped = _crop_mono_16k(audio_path, start_sec, end_sec)
        transcript = (transcriber or _transcribe_crop)(clipped).strip()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"MOSS focused retranscription failed: {exc}"}
    finally:
        if clipped is not None:
            clipped.unlink(missing_ok=True)
    if not transcript:
        return {"ok": False, "error": "MOSS focused retranscription returned empty text"}
    return {
        "ok": True,
        "text": transcript,
        "method": "same_model_focused_reobservation",
        "independent_acoustic_evidence": False,
        "audio_path": audio_path,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }
