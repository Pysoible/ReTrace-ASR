"""Truthful model identities and deterministic model-scoped memory keys."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


def _clean(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return cleaned or "unknown"


@dataclass(frozen=True)
class ModelIdentity:
    backend: str
    model: str
    role: str

    @property
    def family(self) -> str:
        backend = self.backend.strip().lower()
        if backend == "moss" or backend.startswith("moss-"):
            return "moss"
        if backend in {"qwen-omni", "omni"} or backend.startswith(("qwen-omni-", "qwen3-omni-")):
            return "qwen-omni"
        return _clean(backend)

    def as_dict(self) -> dict[str, str]:
        return {"backend": self.backend, "model": self.model, "role": self.role}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelIdentity":
        return cls(
            backend=str(value.get("backend") or "unknown"),
            model=str(value.get("model") or "unknown"),
            role=str(value.get("role") or "unknown"),
        )

    @classmethod
    def from_asr_result(cls, value: dict[str, Any]) -> "ModelIdentity":
        return cls(
            backend=str(value.get("backend") or "unknown"),
            model=str(value.get("model") or "unknown"),
            role="first_pass",
        )


def model_memory_scope(identity: ModelIdentity, *, namespace: str = "default") -> str:
    """Return a stable scope shared by sessions of one concrete model.

    Role is deliberately excluded: first pass, Judge, and audio tools from the
    same concrete model belong to one experiment memory, while backend/model or
    namespace changes create an isolated scope.
    """

    raw = f"{namespace}\0{identity.backend}\0{identity.model}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"model--{_clean(namespace)}--{_clean(identity.backend)}--{_clean(identity.model)}--{digest}"
