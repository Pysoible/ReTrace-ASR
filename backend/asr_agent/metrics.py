"""Reproducible outcome metrics for retrospective ASR revisions."""
from __future__ import annotations

from typing import Any


def evaluate_revisions(
    *,
    events: list[dict[str, Any]],
    reference_by_turn: dict[str, str],
    turn_order: list[str],
    ambiguous_turn_count: int,
) -> dict[str, float | int]:
    """Score only committed text revisions against turn-level references."""
    revisions = [event for event in events if event.get("action") == "REVISE_TEXT" and event.get("active", True)]
    correct = sum(
        1
        for event in revisions
        if reference_by_turn.get(str(event.get("target_turn_id"))) == event.get("after_text")
    )
    positions = {turn_id: index for index, turn_id in enumerate(turn_order)}
    latencies = [
        positions[str(event["source_turn_id"])] - positions[str(event["target_turn_id"])]
        for event in revisions
        if str(event.get("source_turn_id")) in positions and str(event.get("target_turn_id")) in positions
    ]
    total = len(revisions)
    return {
        "committed_revisions": total,
        "correct_revisions": correct,
        "revision_precision": correct / total if total else 0.0,
        "overcorrection_rate": (total - correct) / total if total else 0.0,
        "revision_coverage": total / ambiguous_turn_count if ambiguous_turn_count else 0.0,
        "mean_resolution_latency_turns": sum(latencies) / len(latencies) if latencies else 0.0,
    }
