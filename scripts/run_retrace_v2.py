#!/usr/bin/env python3
"""Run reference-free ReTrace V2 inference from frozen ASR evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrace_v2.agent import ReTraceAgent, ThresholdPolicy
from retrace_v2.artifacts import ArtifactWriter
from retrace_v2.schemas import (
    EvidenceHypothesis,
    ModelRole,
    RunManifest,
    SuspiciousRegion,
)
from retrace_v2.tools import ToolFailure


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


class PrecomputedTools:
    def __init__(self, evidence: list[EvidenceHypothesis]) -> None:
        self.original = tuple(row for row in evidence if not row.view.startswith("separated"))
        self.separated = tuple(row for row in evidence if row.view.startswith("separated"))

    def relisten(self, region: SuspiciousRegion) -> tuple[EvidenceHypothesis, ...]:
        return self.original

    def separate(self, region: SuspiciousRegion) -> tuple[str, ...]:
        channels = tuple(dict.fromkeys(row.view for row in self.separated))
        if not channels:
            raise ToolFailure("no precomputed separated evidence")
        return channels

    def relisten_separated(
        self,
        region: SuspiciousRegion,
        channels: tuple[str, ...],
    ) -> tuple[EvidenceHypothesis, ...]:
        return self.separated


def _hypothesis(row: dict[str, object]) -> EvidenceHypothesis:
    return EvidenceHypothesis(
        hypothesis_id=str(row["hypothesis_id"]),
        model_id=str(row["model_id"]),
        view=str(row.get("view") or "original"),
        text=str(row.get("text") or ""),
        start_sec=float(row["start_sec"]),
        end_sec=float(row["end_sec"]),
        acoustic_score=float(row.get("acoustic_score") or 0.0),
        speaker=str(row["speaker"]) if row.get("speaker") is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen-evidence ReTrace V2 inference")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--predicted-rttm", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--commit-margin", type=float, default=0.25)
    args = parser.parse_args()

    raw_rows = _read_jsonl(args.raw)
    evidence_rows = _read_jsonl(args.evidence)
    baseline_ids = {str(row.get("model_id") or "") for row in raw_rows}
    evidence_ids = {str(row.get("model_id") or "") for row in evidence_rows}
    if len(baseline_ids) != 1 or "" in baseline_ids:
        raise ValueError("run requires exactly one baseline model")
    if len(evidence_ids) != 1 or "" in evidence_ids:
        raise ValueError("run requires exactly one evidence model")

    manifest = RunManifest.create(
        run_id=args.output.name,
        models={
            ModelRole.BASELINE_ASR: next(iter(baseline_ids)),
            ModelRole.EVIDENCE_ASR: next(iter(evidence_ids)),
        },
        rttm_source="predicted" if args.predicted_rttm else "none",
    )
    if args.predicted_rttm and not args.predicted_rttm.is_file():
        raise ValueError("predicted RTTM path does not exist")

    evidence_by_segment: dict[str, list[EvidenceHypothesis]] = defaultdict(list)
    for row in evidence_rows:
        evidence_by_segment[str(row["segment_id"])].append(_hypothesis(row))

    policy = ThresholdPolicy(
        commit_margin=args.commit_margin,
        max_tool_calls=args.max_tool_calls,
    )
    agent = ReTraceAgent(policy)
    results = []
    started = time.perf_counter()
    for row in raw_rows:
        segment_id = str(row["segment_id"])
        region = SuspiciousRegion(
            region_id=str(row.get("region_id") or segment_id),
            segment_id=segment_id,
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            raw_text=str(row.get("text") or ""),
            raw_acoustic_score=float(row.get("raw_acoustic_score") or 0.0),
            overlap_probability=float(row.get("overlap_probability") or 0.0),
            triggers=tuple(str(value) for value in row.get("triggers") or ()),
        )
        results.append(agent.run(region, PrecomputedTools(evidence_by_segment[segment_id])))
    elapsed = time.perf_counter() - started

    writer = ArtifactWriter(args.output)
    writer.write_manifest(
        {
            "run_id": manifest.run_id,
            "models": {role.value: identity for role, identity in manifest.models.items()},
            "rttm_source": manifest.rttm_source,
            "inputs": {"raw": str(args.raw), "evidence": str(args.evidence)},
            "policy": {
                "commit_margin": policy.commit_margin,
                "max_tool_calls": policy.max_tool_calls,
            },
        }
    )
    writer.write_jsonl("raw_moss.jsonl", raw_rows)
    writer.write_jsonl("acoustic_evidence.jsonl", evidence_rows)
    writer.write_jsonl(
        "actions.jsonl",
        (
            {
                "region_id": result.region_id,
                "step": index,
                "kind": action.kind,
                "reason": action.reason,
                "tool_cost": action.tool_cost,
            }
            for result in results
            for index, action in enumerate(result.actions)
        ),
    )
    writer.write_jsonl(
        "decisions.jsonl",
        (
            {
                "region_id": result.region_id,
                "decision": result.decision,
                "selected_candidate_id": result.selected_candidate_id,
            }
            for result in results
        ),
    )
    writer.write_jsonl(
        "final_hypothesis.jsonl",
        (
            {
                "segment_id": result.segment_id,
                "region_id": result.region_id,
                "raw_text": result.raw_text,
                "final_text": result.final_text,
                "candidates": [item.candidate_text for item in result.candidates],
            }
            for result in results
        ),
    )
    writer.write_json(
        "timings.json",
        {"agent_elapsed_sec": elapsed, "segment_count": len(results)},
    )
    print(f"wrote {len(results)} segments to {args.output}")


if __name__ == "__main__":
    main()
