"""Parse predicted RTTM and derive speaker-overlap intervals."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..schemas import SpeakerTurn


def parse_rttm(path: Path, *, expected_recording: str) -> tuple[SpeakerTurn, ...]:
    turns: list[SpeakerTurn] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            raise ValueError(f"invalid RTTM row at line {line_number}")
        recording_id = fields[1]
        if recording_id != expected_recording:
            raise ValueError(
                f"RTTM recording {recording_id!r} does not match {expected_recording!r}"
            )
        start = float(fields[3])
        duration = float(fields[4])
        if duration <= 0:
            raise ValueError(f"RTTM duration must be positive at line {line_number}")
        turns.append(
            SpeakerTurn(
                recording_id=recording_id,
                speaker=fields[7],
                start_sec=start,
                end_sec=start + duration,
            )
        )
    return tuple(sorted(turns, key=lambda turn: (turn.start_sec, turn.end_sec, turn.speaker)))


def overlap_windows(turns: Iterable[SpeakerTurn]) -> list[tuple[float, float]]:
    rows = tuple(turns)
    boundaries = sorted({value for turn in rows for value in (turn.start_sec, turn.end_sec)})
    overlaps: list[tuple[float, float]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2
        active_speakers = {
            turn.speaker for turn in rows if turn.start_sec < midpoint < turn.end_sec
        }
        if len(active_speakers) < 2:
            continue
        if overlaps and overlaps[-1][1] == start:
            overlaps[-1] = overlaps[-1][0], end
        else:
            overlaps.append((start, end))
    return overlaps
