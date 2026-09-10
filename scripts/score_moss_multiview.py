#!/usr/bin/env python3
"""Score a completed GT-free multiview artifact after inference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_aishell4_eval import metrics, parse_reference


def _reference_in_window(rows: list[dict[str, Any]], start: float, end: float) -> str:
    return "".join(
        row["text"]
        for row in rows
        if start <= (float(row["start"]) + float(row["end"])) / 2.0 < end
    )


def _reference_overlap_intervals(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    events = []
    for row in rows:
        events.extend([(float(row["start"]), 1), (float(row["end"]), -1)])
    active = 0
    start = None
    output = []
    for timestamp, delta in sorted(events, key=lambda item: (item[0], item[1])):
        previous = active
        active += delta
        if previous < 2 <= active:
            start = timestamp
        elif previous >= 2 > active and start is not None and timestamp > start:
            output.append((start, timestamp))
            start = None
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision-artifact", type=Path)
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    decisions = (
        json.loads(args.decision_artifact.read_text(encoding="utf-8"))
        if args.decision_artifact
        else None
    )
    decision_by_window = {
        (float(block["start_sec"]), float(block["end_sec"])): block
        for block in ((decisions or {}).get("blocks") or [])
    }
    _reference, rows = parse_reference(args.reference)
    overlap_intervals = _reference_overlap_intervals(rows)
    blocks = []
    medoid_delta = oracle_delta = policy_delta = 0
    policy_outcomes = {"improved": 0, "harmed": 0, "neutral": 0}
    closed_set_delta = 0
    closed_set_outcomes = {"improved": 0, "harmed": 0, "neutral": 0}
    for block in artifact.get("blocks") or []:
        start, end = float(block["start_sec"]), float(block["end_sec"])
        reference = _reference_in_window(rows, start, end)
        scores = {
            name: metrics(reference, text)
            for name, text in (block.get("hypotheses") or {}).items()
        }
        policy = block["coverage_recovery_policy"]
        if policy["selected_view"] == "coverage_recovery_proposal":
            scores["coverage_recovery_proposal"] = metrics(
                reference, policy["proposed_text"]
            )
        anchor_edits = scores["baseline_anchor"]["edits"]
        medoid_name = block["consensus_medoid"]
        oracle_name = min(scores, key=lambda name: scores[name]["edits"])
        policy_name = policy["selected_view"]
        medoid_delta += scores[medoid_name]["edits"] - anchor_edits
        oracle_delta += scores[oracle_name]["edits"] - anchor_edits
        policy_change = scores[policy_name]["edits"] - anchor_edits
        policy_delta += policy_change
        if policy_name != "baseline_anchor":
            outcome = "improved" if policy_change < 0 else "harmed" if policy_change > 0 else "neutral"
            policy_outcomes[outcome] += 1
        closed_set = decision_by_window.get((start, end))
        closed_set_change = 0
        if closed_set is not None:
            scores["closed_set_policy"] = metrics(reference, closed_set["proposed_text"])
            closed_set_change = scores["closed_set_policy"]["edits"] - anchor_edits
            closed_set_delta += closed_set_change
            if closed_set.get("applied_candidate_ids"):
                outcome = (
                    "improved" if closed_set_change < 0 else "harmed" if closed_set_change > 0 else "neutral"
                )
                closed_set_outcomes[outcome] += 1
        overlap_sec = sum(
            max(0.0, min(end, right) - max(start, left))
            for left, right in overlap_intervals
        )
        blocks.append(
            {
                "start_sec": start,
                "end_sec": end,
                "detector": block["detector"],
                "reference": reference,
                "scores": scores,
                "consensus_medoid": medoid_name,
                "oracle_view": oracle_name,
                "policy_view": policy_name,
                "medoid_edit_delta": scores[medoid_name]["edits"] - anchor_edits,
                "oracle_edit_delta": scores[oracle_name]["edits"] - anchor_edits,
                "policy_edit_delta": policy_change,
                "closed_set_edit_delta": closed_set_change,
                "closed_set_applied_candidate_ids": (
                    closed_set.get("applied_candidate_ids") if closed_set else []
                ),
                "gt_overlap_sec": round(overlap_sec, 3),
                "gt_overlap_ratio": round(overlap_sec / max(end - start, 1e-9), 6),
            }
        )
    result = {
        "schema": "moss_multiview_probe_score_v1",
        "purpose": "offline_gt_diagnostics_only",
        "inference_artifact": str(args.artifact),
        "reference": str(args.reference),
        "blocks": blocks,
        "aggregate": {
            "selected_windows": len(blocks),
            "windows_intersecting_gt_overlap": sum(block["gt_overlap_sec"] > 0 for block in blocks),
            "consensus_medoid_net_edit_delta": medoid_delta,
            "oracle_view_net_edit_delta": oracle_delta,
            "coverage_policy_net_edit_delta": policy_delta,
            "coverage_policy_outcomes": policy_outcomes,
            "coverage_policy_harmful_rate": (
                policy_outcomes["harmed"] / max(1, sum(policy_outcomes.values()))
            ),
            "passes_five_error_probe_gate": (
                policy_delta <= -5
                and policy_outcomes["harmed"] / max(1, sum(policy_outcomes.values())) <= 0.05
            ),
            "closed_set_policy_net_edit_delta": closed_set_delta,
            "closed_set_policy_outcomes": closed_set_outcomes,
            "closed_set_policy_harmful_rate": (
                closed_set_outcomes["harmed"] / max(1, sum(closed_set_outcomes.values()))
            ),
            "passes_closed_set_probe_gate": (
                closed_set_delta <= -2 and closed_set_outcomes["harmed"] == 0
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
