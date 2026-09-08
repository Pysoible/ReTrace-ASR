"""Immutable, checksum-keyed first-pass ASR artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def artifact_key(
    audio_path: str | Path,
    *,
    identity: dict[str, Any],
    config: dict[str, Any],
) -> str:
    audio_hash = hashlib.sha256()
    with Path(audio_path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            audio_hash.update(block)
    contract = {
        "audio_sha256": audio_hash.hexdigest(),
        "backend": str(identity.get("backend") or "unknown"),
        "model": str(identity.get("model") or "unknown"),
        "config": config,
    }
    encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def inference_contract(identity: dict[str, Any]) -> dict[str, Any]:
    backend = str(identity.get("backend") or "").lower()
    if backend.startswith("moss"):
        return {
            "endpoint_model": os.getenv("MOSS_MODEL_NAME", str(identity.get("model") or "unknown")),
            "max_completion_tokens": int(
                os.getenv("MOSS_MAX_COMPLETION_TOKENS", os.getenv("MOSS_MAX_NEW_TOKENS", "32768"))
            ),
            "min_coverage_ratio": float(os.getenv("MOSS_MIN_COVERAGE_RATIO", "0.98")),
            "response_format": os.getenv("MOSS_RESPONSE_FORMAT", "json"),
            "temperature": float(os.getenv("MOSS_TEMPERATURE", "0")),
        }
    return {
        "model_path": os.getenv("QWEN_MODEL_PATH", str(identity.get("model") or "unknown")),
        "chunk_seconds": float(os.getenv("ASR_CHUNK_SECONDS", "15")),
    }


class FirstPassArtifactRepository:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, artifact_id: str) -> Path:
        if not artifact_id or any(char not in "0123456789abcdef" for char in artifact_id.lower()):
            raise ValueError("invalid artifact id")
        return self.root / f"{artifact_id.lower()}.json"

    @staticmethod
    def _validate(payload: dict[str, Any]) -> None:
        completeness = payload.get("completeness") or {}
        if not payload.get("ok") or completeness.get("truncated") is True:
            raise ValueError("incomplete first-pass artifacts cannot be cached")

    def load(self, artifact_id: str) -> dict[str, Any] | None:
        path = self._path(artifact_id)
        if not path.exists():
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("artifact_id") != artifact_id.lower() or not isinstance(value.get("payload"), dict):
            raise ValueError(f"invalid first-pass artifact: {path}")
        self._validate(value["payload"])
        return value

    def save(self, artifact_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._validate(payload)
        path = self._path(artifact_id)
        value = {"artifact_id": artifact_id.lower(), "payload": payload}
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if path.exists():
            existing = self.load(artifact_id)
            if existing != value:
                raise ValueError("immutable first-pass artifact already exists with different content")
            return existing
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{artifact_id}.", suffix=".tmp", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return value
