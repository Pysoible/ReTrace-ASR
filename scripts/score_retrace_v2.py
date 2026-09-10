#!/usr/bin/env python3
"""Score a finalized ReTrace V2 run against an offline transcript."""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrace_v2.artifacts import ArtifactWriter
from retrace_v2.scoring import ErrorMetric, score_rows


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric_payload(metric: ErrorMetric) -> dict[str, object]:
    return {**asdict(metric), "edits": metric.edits, "cer": metric.cer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline scoring for a finalized V2 run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()

    hypotheses = _read_jsonl(args.run / "final_hypothesis.jsonl")
    references = {
        str(row["segment_id"]): str(row.get("reference_text") or "")
        for row in _read_jsonl(args.reference)
    }
    inference_ids = {str(row["segment_id"]) for row in hypotheses}
    if inference_ids != set(references):
        raise ValueError("inference and scoring segment IDs must match exactly")
    rows = [
        {
            **row,
            "reference_text": references[str(row["segment_id"])],
        }
        for row in hypotheses
    ]
    report = score_rows(rows)
    payload = {
        "raw": _metric_payload(report.raw),
        "final": _metric_payload(report.final),
        "oracle_candidate": _metric_payload(report.oracle_candidate),
        "committed_revisions": report.committed_revisions,
        "improving_revisions": report.improving_revisions,
        "harmful_revisions": report.harmful_revisions,
        "neutral_revisions": report.neutral_revisions,
        "revision_precision": report.revision_precision,
    }
    ArtifactWriter(args.run / "scoring").write_json("metrics.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
