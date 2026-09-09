"""MOSS-only focused audio re-observation tools.

These tools deliberately reuse MOSS for a small crop.  They are useful as a
focused second decode, but are not represented as independent acoustic evidence.
"""
from __future__ import annotations

import math
import os
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
        candidate: _candidate_similarity(transcript, candidate)
        for candidate in unique
        if candidate != DELETE_CANDIDATE
    }
    if DELETE_CANDIDATE in unique:
        # The first item is the observed span. Deletion support means a focused
        # re-decode produced substantive surrounding speech but no trace of that
        # span. A fixed near-zero DELETE score made insertion repair impossible.
        observed_span = next((item for item in unique if item != DELETE_CANDIDATE), "")
        observed_score = similarities.get(observed_span, 0.0)
        transcript_chars = len(_normalize(transcript))
        minimum_context = max(4, len(_normalize(observed_span)) * 2)
        similarities[DELETE_CANDIDATE] = (
            max(0.0, 1.0 - observed_score)
            if transcript_chars >= minimum_context
            else 0.05
        )
    non_delete_scores = [
        score for candidate, score in similarities.items() if candidate != DELETE_CANDIDATE
    ]
    minimum_match = float(os.getenv("MOSS_VERIFIER_MIN_ABSOLUTE_SIMILARITY", "0.60"))
    delete_supported = bool(
        DELETE_CANDIDATE in similarities
        and similarities[DELETE_CANDIDATE] >= float(os.getenv("MOSS_DELETE_MIN_ABSENCE_SCORE", "0.75"))
    )
    if (not non_delete_scores or max(non_delete_scores) < minimum_match) and not delete_supported:
        return {
            "ok": False,
            "failure_code": "no_closed_set_match",
            "error": "MOSS focused transcript does not match any supplied candidate",
            "focused_transcript": transcript,
            "absolute_similarities": similarities,
            "threshold": minimum_match,
            "method": "same_model_focused_reobservation",
            "independent_acoustic_evidence": False,
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
    """Open MOSS re-decode; split coverage-risk windows to avoid early EOS."""
    del domain_hints
    windows = [(start_sec, end_sec)]
    if recover_coverage and end_sec - start_sec > 5.5:
        windows = []
        cursor = start_sec
        while cursor < end_sec:
            window_end = min(end_sec, cursor + 5.0)
            windows.append((cursor, window_end))
            cursor = window_end
    parts: list[str] = []
    try:
        for window_start, window_end in windows:
            clipped: Path | None = None
            try:
                clipped = _crop_mono_16k(audio_path, window_start, window_end)
                text = (transcriber or _transcribe_crop)(clipped).strip()
                if text:
                    parts.append(text)
            finally:
                if clipped is not None:
                    clipped.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"MOSS focused retranscription failed: {exc}"}
    transcript = "".join(parts)
    if not transcript:
        return {"ok": False, "error": "MOSS focused retranscription returned empty text"}
    return {
        "ok": True,
        "text": transcript,
        "segmented": len(windows) > 1,
        "window_count": len(windows),
        "method": "same_model_focused_reobservation",
        "independent_acoustic_evidence": False,
        "audio_path": audio_path,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }
