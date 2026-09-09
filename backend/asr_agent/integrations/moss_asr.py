"""MOSS-Transcribe-Diarize adapter through an OpenAI-compatible endpoint."""
from __future__ import annotations

import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

from asr_agent.model_identity import MOSS_TRANSCRIBE_DIARIZE_MODEL

BACKEND = "moss-transcribe-diarize"
DEFAULT_MODEL_PATH = "/home/panyibo/models/openmoss/MOSS-Transcribe-Diarize"
DEFAULT_TRANSCRIBE_URL = "http://127.0.0.1:8010/v1/audio/transcriptions"

_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)


def _env_model_path() -> str:
    return os.getenv("MOSS_MODEL_PATH", DEFAULT_MODEL_PATH)


def _env_url() -> str:
    return os.getenv("MOSS_TRANSCRIBE_URL", DEFAULT_TRANSCRIBE_URL)


def _env_timeout() -> float:
    return float(os.getenv("MOSS_TRANSCRIBE_TIMEOUT", "7200"))


def _env_max_completion_tokens() -> str:
    return os.getenv(
        "MOSS_MAX_COMPLETION_TOKENS",
        os.getenv("MOSS_MAX_NEW_TOKENS", "32768"),
    )


def annotate_speaker_overlaps(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return copied chunks annotated with cross-speaker timestamp overlap.

    MOSS emits one primary ``speaker`` per segment.  Concurrent speakers appear
    as intersecting segments, so retain that primary label and derive the
    plural ``speakers``/overlap audit fields from the timestamps.  Short edge
    intersections are ignored to avoid treating timestamp jitter as speech
    overlap.
    """
    minimum_overlap = max(0.0, float(os.getenv("MOSS_OVERLAP_MIN_SEC", "0.25")))
    annotated = [dict(chunk) for chunk in chunks]
    for chunk in annotated:
        for field in ("speakers", "overlap", "overlap_duration_sec", "overlap_with_indices"):
            chunk.pop(field, None)
    overlap_ranges: dict[int, list[tuple[float, float]]] = {}
    overlap_indices: dict[int, set[int]] = {}
    overlap_speakers: dict[int, list[str]] = {}

    timed: list[tuple[float, float, int, str]] = []
    for position, chunk in enumerate(annotated):
        start, end = chunk.get("start_sec"), chunk.get("end_sec")
        speaker = str(chunk.get("speaker") or "").strip()
        if start is None or end is None or not speaker:
            continue
        start_value, end_value = float(start), float(end)
        if end_value > start_value:
            timed.append((start_value, end_value, position, speaker))
    timed.sort(key=lambda item: (item[0], item[1], item[2]))

    active: list[tuple[float, float, int, str]] = []
    for current in timed:
        current_start, current_end, current_position, current_speaker = current
        active = [item for item in active if item[1] - current_start >= minimum_overlap]
        for other_start, other_end, other_position, other_speaker in active:
            if other_speaker == current_speaker:
                continue
            overlap_start = max(current_start, other_start)
            overlap_end = min(current_end, other_end)
            if overlap_end <= overlap_start or overlap_end - overlap_start < minimum_overlap:
                continue
            for position, peer_position, peer_speaker in (
                (current_position, other_position, other_speaker),
                (other_position, current_position, current_speaker),
            ):
                overlap_ranges.setdefault(position, []).append((overlap_start, overlap_end))
                overlap_indices.setdefault(position, set()).add(peer_position)
                speakers = overlap_speakers.setdefault(position, [])
                if peer_speaker not in speakers:
                    speakers.append(peer_speaker)
        active.append(current)

    for position, ranges in overlap_ranges.items():
        ranges.sort()
        merged: list[list[float]] = []
        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        chunk = annotated[position]
        primary = str(chunk.get("speaker") or "").strip()
        chunk.update(
            {
                "speakers": [primary, *overlap_speakers.get(position, [])],
                "overlap": True,
                "overlap_duration_sec": round(sum(end - start for start, end in merged), 3),
                "overlap_with_indices": sorted(
                    int(annotated[index].get("index", index))
                    for index in overlap_indices.get(position, set())
                ),
            }
        )
    return annotated


def parse_moss_transcript(raw: str) -> dict[str, Any]:
    """Parse canonical MOSS ``[start][Sxx]text[end]`` output."""
    chunks_text: list[str] = []
    chunks: list[dict[str, Any]] = []
    for index, match in enumerate(_SEGMENT_RE.finditer(raw or "")):
        text = re.sub(r"\s+", " ", match.group("text")).strip()
        if not text:
            continue
        chunks_text.append(text)
        chunks.append(
            {
                "index": index,
                "start_sec": float(match.group("start")),
                "end_sec": float(match.group("end")),
                "speaker": match.group("speaker"),
            }
        )
    if chunks_text:
        chunks = annotate_speaker_overlaps(chunks)
        return {"text": "".join(chunks_text), "chunks_text": chunks_text, "chunks": chunks}
    text = re.sub(r"\s+", " ", raw or "").strip()
    return {
        "text": text,
        "chunks_text": [text] if text else [],
        "chunks": [{"index": 0}] if text else [],
    }


def _response_duration(payload: dict[str, Any]) -> float | None:
    direct = payload.get("duration") or payload.get("duration_sec")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    value = direct or usage.get("seconds")
    return float(value) if value is not None else None


def transcript_completeness(
    raw: str,
    chunks: list[dict[str, Any]],
    duration: float | None,
) -> dict[str, Any]:
    covered = max(
        (
            float(item["end_sec"])
            for item in chunks
            if item.get("end_sec") is not None
        ),
        default=0.0,
    )
    ratio = min(1.0, covered / duration) if duration and duration > 0 else None
    matches = list(_SEGMENT_RE.finditer(raw or ""))
    tail = (raw[matches[-1].end():] if matches else raw).strip()[:160]
    threshold = float(os.getenv("MOSS_MIN_COVERAGE_RATIO", "0.98"))
    return {
        "covered_until_sec": covered,
        "coverage_ratio": ratio,
        "threshold": threshold,
        "truncated": ratio is not None and ratio < threshold,
        "parse_tail": tail,
    }


def segment_coverage_uncertainties(
    chunks_text: list[str],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build conservative, GT-free coverage signals for MOSS turns.

    MOSS exposes no token confidence. A long timestamp span with very few
    recognized characters can still route the Agent to re-segment/re-decode;
    the signal never supplies replacement text or authorizes a revision.
    """
    threshold = float(os.getenv("MOSS_MIN_SEGMENT_CHAR_DENSITY", "2.0"))
    minimum_duration = float(os.getenv("MOSS_COVERAGE_MIN_SEGMENT_SEC", "6.0"))
    output: list[dict[str, Any]] = []
    for index, text in enumerate(chunks_text):
        chunk = chunks[index] if index < len(chunks) else {}
        start, end = chunk.get("start_sec"), chunk.get("end_sec")
        duration = max(0.0, float(end) - float(start)) if start is not None and end is not None else 0.0
        char_count = len(re.findall(r"[0-9A-Za-z\u4e00-\u9fff]", text or ""))
        density = char_count / duration if duration > 0 else None
        suspect = bool(duration >= minimum_duration and density is not None and density < threshold)
        overlap_detected = bool(chunk.get("overlap"))
        output.append({
            "coverage": {
                "detector": "moss_segment_char_density",
                "speech_window_sec": round(duration, 3),
                "char_count": char_count,
                "char_density": round(density, 3) if density is not None else None,
                "threshold": threshold,
                "truncated": suspect,
                "reasons": ["low_transcript_density"] if suspect else [],
                "recommended_actions": ["RESEGMENT", "REDECODE"] if suspect else [],
            },
            "overlap": {
                "detector": "moss_cross_speaker_timestamp_overlap",
                "detected": overlap_detected,
                "duration_sec": float(chunk.get("overlap_duration_sec") or 0.0),
                "speakers": list(chunk.get("speakers") or ([chunk["speaker"]] if chunk.get("speaker") else [])),
                "with_indices": list(chunk.get("overlap_with_indices") or []),
                "automatic_revision_allowed": False,
                "recommended_actions": (
                    ["AUDIT_OVERLAP", "GUIDED_SOURCE_SEPARATION"] if overlap_detected else []
                ),
            },
        })
    return output


def _multipart_form(fields: dict[str, str], file_field: str, file_path: Path) -> tuple[bytes, str]:
    boundary = f"----retrace-moss-{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(str(value).encode())
        parts.append(b"\r\n")
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
    )
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def _post_transcription(audio_path: Path) -> dict[str, Any]:
    fields = {
        "model": os.getenv("MOSS_MODEL_NAME", MOSS_TRANSCRIBE_DIARIZE_MODEL),
        "response_format": os.getenv("MOSS_RESPONSE_FORMAT", "json"),
        "temperature": os.getenv("MOSS_TEMPERATURE", "0"),
        "max_completion_tokens": _env_max_completion_tokens(),
    }
    body, boundary = _multipart_form(fields, "file", audio_path)
    request = urllib.request.Request(
        _env_url(),
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_env_timeout()) as response:
        return json.loads(response.read().decode("utf-8"))


def asr_status() -> dict[str, Any]:
    model_path = _env_model_path()
    return {
        "enabled": os.getenv("ASR_BACKEND", "").strip().lower() == BACKEND,
        "backend": BACKEND,
        "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
        "model_path": model_path,
        "model_exists": Path(model_path).exists(),
        "transcribe_url": _env_url(),
    }


def preload_engine() -> dict[str, Any]:
    status = asr_status()
    status["ok"] = True
    status["loaded"] = False
    return status


def shutdown_engines() -> None:
    return None


def transcribe_audio(audio: str) -> dict[str, Any]:
    started = time.perf_counter()
    audio_path = Path((audio or "").strip()).expanduser()
    if not str(audio_path):
        return {"ok": False, "error": "需要音频路径 audio"}
    if not audio_path.exists():
        return {"ok": False, "error": f"找不到音频文件: {audio}"}
    try:
        payload = _post_transcription(audio_path)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"MOSS HTTP {exc.code}: {detail}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"MOSS 转写失败: {exc}"}
    raw_text = str(payload.get("text") or payload.get("transcript") or "")
    parsed = parse_moss_transcript(raw_text)
    duration = _response_duration(payload)
    completeness = transcript_completeness(raw_text, parsed["chunks"], duration)
    truncated = bool(completeness["truncated"])
    return {
        "ok": bool(parsed["chunks_text"]) and not truncated,
        "failure_code": "incomplete_first_pass" if truncated else None,
        "audio": str(audio_path),
        "final_text": parsed["text"],
        "chunks_text": parsed["chunks_text"],
        "uncertainties": segment_coverage_uncertainties(parsed["chunks_text"], parsed["chunks"]),
        "backend": BACKEND,
        "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
        "duration_sec": duration,
        "completeness": completeness,
        "chunked": True,
        "chunk_count": len(parsed["chunks_text"]),
        "chunks": parsed["chunks"],
        "raw_response": payload,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }


def stream_transcribe_audio(audio: str, on_chunk) -> dict[str, Any]:
    result = transcribe_audio(audio)
    if not result.get("ok"):
        raise RuntimeError(result.get("error") or "MOSS transcription failed")
    for index, text in enumerate(result.get("chunks_text") or []):
        chunks = result.get("chunks") or []
        chunk = chunks[index] if index < len(chunks) else {"index": index}
        uncertainties = result.get("uncertainties") or []
        uncertainty = uncertainties[index] if index < len(uncertainties) else {}
        on_chunk(index, text, uncertainty, chunk)
    return result
