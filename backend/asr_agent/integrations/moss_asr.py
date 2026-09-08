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


def _env_max_new_tokens() -> str:
    return os.getenv("MOSS_MAX_NEW_TOKENS", "65536")


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
        return {"text": "".join(chunks_text), "chunks_text": chunks_text, "chunks": chunks}
    text = re.sub(r"\s+", " ", raw or "").strip()
    return {
        "text": text,
        "chunks_text": [text] if text else [],
        "chunks": [{"index": 0}] if text else [],
    }


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
        "max_new_tokens": _env_max_new_tokens(),
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
    return {
        "ok": bool(parsed["chunks_text"]),
        "audio": str(audio_path),
        "final_text": parsed["text"],
        "chunks_text": parsed["chunks_text"],
        "uncertainties": [{} for _ in parsed["chunks_text"]],
        "backend": BACKEND,
        "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
        "duration_sec": payload.get("duration") or payload.get("duration_sec"),
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
        on_chunk(index, text, {}, chunk)
    return result
