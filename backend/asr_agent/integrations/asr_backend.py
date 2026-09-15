"""Select the active first-pass ASR backend."""
from __future__ import annotations

import os
from typing import Any, Callable

from asr_agent.integrations import moss_asr, qwen_asr


def _backend() -> str:
    return os.getenv("ASR_BACKEND", "qwen-omni-vllm").strip().lower()


def _module():
    backend = _backend()
    if backend in {"moss", moss_asr.BACKEND}:
        return moss_asr
    return qwen_asr


def read_asr_config() -> Any:
    return qwen_asr.read_asr_config()


def asr_status() -> dict[str, Any]:
    status = dict(_module().asr_status())
    backend = _backend()
    status["selected_backend"] = backend
    status.setdefault("backend", moss_asr.BACKEND if backend in {"moss", moss_asr.BACKEND} else "qwen-omni-vllm")
    if "model" not in status:
        status["model"] = os.path.basename(str(status.get("model_path") or "unknown"))
    return status


def preload_engine() -> dict[str, Any]:
    return _module().preload_engine()


def shutdown_engines() -> None:
    qwen_asr.shutdown_engines()
    moss_asr.shutdown_engines()


def transcribe_audio(audio: str) -> dict[str, Any]:
    return _module().transcribe_audio(audio)


def stream_transcribe_audio(
    audio: str,
    on_chunk: Callable[[int, str, dict[str, Any], dict[str, Any]], None],
) -> dict[str, Any]:
    return _module().stream_transcribe_audio(audio, on_chunk)
