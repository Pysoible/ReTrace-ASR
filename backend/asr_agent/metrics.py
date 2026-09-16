"""Reproducible outcome metrics for retrospective ASR revisions."""
from __future__ import annotations

from typing import Any


def evaluate_revisions(
    *,
    events: list[dict[str, Any]],
    reference_by_turn: dict[str, str],
    turn_order: list[str],
    ambiguous_turn_count: int,
) -> dict[str, float | int | None]:
    """Score committed revisions with route-stratified safety and latency.

    ``REVISE_HISTORY`` is the retrospective, later-evidence route. A revision
    whose source and target are the same turn is the contemporaneous acoustic
    route. Keeping both rows prevents aggregate CER or revision precision from
    crediting the context agent for a local second-ASR correction (or vice versa).
    """
    superseded = {
        str(event["supersedes_event_id"])
        for event in events
        if event.get("active", True) and event.get("supersedes_event_id")
    }
    revision_actions = {"REVISE_TEXT", "REVISE_CURRENT", "REVISE_HISTORY"}
    revisions = [
        event
        for event in events
        if event.get("action") in revision_actions
        and event.get("active", True)
        and str(event.get("event_id", "")) not in superseded
    ]
    positions = {turn_id: index for index, turn_id in enumerate(turn_order)}
    correct_events = [
        event
        for event in revisions
        if reference_by_turn.get(str(event.get("target_turn_id"))) == event.get("after_text")
    ]
    historical = [
        event
        for event in revisions
        if positions.get(str(event.get("source_turn_id")), -1) > positions.get(str(event.get("target_turn_id")), -1)
    ]
    current = [event for event in revisions if event not in historical]
    historical_correct = [event for event in historical if event in correct_events]
    current_correct = [event for event in current if event in correct_events]
    historical_latencies = [
        positions[str(event["source_turn_id"])] - positions[str(event["target_turn_id"])]
        for event in historical
        if str(event.get("source_turn_id")) in positions and str(event.get("target_turn_id")) in positions
    ]
    total = len(revisions)
    correct = len(correct_events)
    precision = correct / total if total else 0.0
    recall = correct / ambiguous_turn_count if ambiguous_turn_count else 0.0
    historical_precision = len(historical_correct) / len(historical) if historical else None
    current_precision = len(current_correct) / len(current) if current else None
    rollbacks = [event for event in events if event.get("action") == "ROLLBACK" and event.get("active", True)]
    rollback_successes = sum(
        reference_by_turn.get(str(event.get("target_turn_id"))) == event.get("after_text")
        for event in rollbacks
    )
    audio_evidence_revisions = sum(
        any("audio" in str(item).lower() for item in event.get("evidence", []))
        for event in revisions
    )
    acoustic_sentinel_revisions = sum(
        any("acoustic_disagreement" in str(item).lower() for item in event.get("evidence", []))
        for event in current
    )

    def f_beta(beta: float) -> float:
        denominator = beta * beta * precision + recall
        return (1 + beta * beta) * precision * recall / denominator if denominator else 0.0

    return {
        "committed_revisions": total,
        "correct_revisions": correct,
        "automatic_rollbacks": len(rollbacks),
        "revision_precision": precision,
        "revision_recall": recall,
        "revision_f2": f_beta(2.0),
        "revision_f3": f_beta(3.0),
        "overcorrection_rate": (total - correct) / total if total else 0.0,
        "revision_coverage": total / ambiguous_turn_count if ambiguous_turn_count else 0.0,
        "mean_resolution_latency_turns": sum(historical_latencies) / len(historical_latencies) if historical_latencies else 0.0,
        "historical_revision_count": len(historical),
        "current_revision_count": len(current),
        "historical_correct_revisions": len(historical_correct),
        "current_correct_revisions": len(current_correct),
        "historical_revision_precision": historical_precision,
        "current_revision_precision": current_precision,
        "historical_overcorrection_rate": 1.0 - historical_precision if historical_precision is not None else None,
        "current_overcorrection_rate": 1.0 - current_precision if current_precision is not None else None,
        "historical_revision_recall": len(historical_correct) / ambiguous_turn_count if ambiguous_turn_count else 0.0,
        "historical_resolution_latency_turns": sum(historical_latencies) / len(historical_latencies) if historical_latencies else 0.0,
        "audio_evidence_revision_count": audio_evidence_revisions,
        "acoustic_sentinel_current_revision_count": acoustic_sentinel_revisions,
        "rollback_success_rate": rollback_successes / len(rollbacks) if rollbacks else None,
    }
