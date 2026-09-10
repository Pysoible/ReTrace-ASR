"""Generate immutable acoustic evidence independently from offline scoring."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Callable, Iterable, Mapping

import soundfile as sf

from .tools import EvidenceASR, ToolFailure


def _audio_duration(path: Path) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def expand_audio_windows(
    raw_rows: Iterable[Mapping[str, object]],
    *,
    padding_sec: float,
    duration_lookup: Callable[[Path], float] = _audio_duration,
) -> list[dict[str, object]]:
    """Add symmetric context without losing the original region boundaries."""
    if padding_sec < 0:
        raise ValueError("padding_sec must be non-negative")
    durations: dict[Path, float] = {}
    expanded: list[dict[str, object]] = []
    for row in raw_rows:
        path = Path(str(row["audio_path"]))
        if path not in durations:
            durations[path] = duration_lookup(path)
        core_start = float(row["start_sec"])
        core_end = float(row["end_sec"])
        expanded.append(
            {
                **row,
                "core_start_sec": core_start,
                "core_end_sec": core_end,
                "start_sec": max(0.0, core_start - padding_sec),
                "end_sec": min(durations[path], core_end + padding_sec),
            }
        )
    return expanded


def generate_evidence_rows(
    raw_rows: Iterable[Mapping[str, object]],
    adapter: EvidenceASR,
    *,
    view: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, str]], list[dict[str, object]]]:
    evidence: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    timings: list[dict[str, object]] = []
    for row in raw_rows:
        segment_id = str(row["segment_id"])
        started = time.perf_counter()
        try:
            hypotheses = adapter.transcribe(
                Path(str(row["audio_path"])),
                float(row["start_sec"]),
                float(row["end_sec"]),
            )
            for hypothesis in hypotheses:
                payload = {"segment_id": segment_id, **asdict(hypothesis)}
                if view is not None:
                    payload["view"] = view
                evidence.append(payload)
            status = "ok"
        except ToolFailure as error:
            failures.append({"segment_id": segment_id, "error": str(error)})
            status = "failed"
        timings.append(
            {
                "segment_id": segment_id,
                "status": status,
                "elapsed_sec": time.perf_counter() - started,
            }
        )
    return evidence, failures, timings
