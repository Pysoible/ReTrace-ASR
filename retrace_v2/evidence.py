"""Generate immutable acoustic evidence independently from offline scoring."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Iterable, Mapping

from .tools import EvidenceASR, ToolFailure


def generate_evidence_rows(
    raw_rows: Iterable[Mapping[str, object]],
    adapter: EvidenceASR,
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
                evidence.append({"segment_id": segment_id, **asdict(hypothesis)})
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
