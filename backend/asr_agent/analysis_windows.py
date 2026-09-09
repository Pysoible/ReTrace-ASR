"""Bounded grouping for context-judge analysis without changing ASR turns."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AnalysisWindowPolicy:
    max_turns: int = 20
    max_chars: int = 600
    max_audio_sec: float = 90.0

    def __post_init__(self) -> None:
        if self.max_turns <= 0 or self.max_chars <= 0 or self.max_audio_sec <= 0:
            raise ValueError("analysis window bounds must be positive")


def _audio_span(turns: list[dict[str, Any]]) -> float:
    if not turns:
        return 0.0
    start = float(turns[0].get("start_sec") or 0.0)
    end = float(turns[-1].get("end_sec") or start)
    return max(0.0, end - start)


def _exceeds(turns: list[dict[str, Any]], policy: AnalysisWindowPolicy) -> bool:
    return (
        len(turns) > policy.max_turns
        or sum(len(str(item.get("text") or "")) for item in turns) > policy.max_chars
        or _audio_span(turns) > policy.max_audio_sec
    )


def partition_turns(
    turns: list[dict[str, Any]],
    policy: AnalysisWindowPolicy,
) -> list[list[dict[str, Any]]]:
    """Partition ordered turns; a bound-crossing turn begins the next window."""
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for turn in turns:
        proposed = [*current, turn]
        if current and _exceeds(proposed, policy):
            groups.append(current)
            current = [turn]
        else:
            current = proposed
    if current:
        groups.append(current)
    return groups
