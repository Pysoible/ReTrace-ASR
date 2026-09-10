"""Runtime-checkable boundaries for optional acoustic evidence tools."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .schemas import EvidenceHypothesis


class ToolFailure(RuntimeError):
    """An evidence tool failed without authorizing a transcript change."""


@runtime_checkable
class EvidenceASR(Protocol):
    model_id: str

    def transcribe(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> tuple[EvidenceHypothesis, ...]: ...


@runtime_checkable
class Separator(Protocol):
    backend_id: str

    def separate(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> tuple[Path, ...]: ...
