#!/usr/bin/env python3
"""Generate Paraformer evidence for prepared ReTrace V2 regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrace_v2.adapters.paraformer import ParaformerAdapter
from retrace_v2.artifacts import ArtifactWriter
from retrace_v2.evidence import generate_evidence_rows


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate independent V2 acoustic evidence")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    raw_rows = _read_jsonl(args.raw)
    if args.limit > 0:
        raw_rows = raw_rows[: args.limit]
    adapter = ParaformerAdapter(args.model)
    evidence, failures, timings = generate_evidence_rows(raw_rows, adapter)
    writer = ArtifactWriter(args.output_dir)
    writer.write_jsonl("evidence.jsonl", evidence)
    writer.write_jsonl("failures.jsonl", failures)
    writer.write_json("timings.json", {"segments": timings})
    writer.write_manifest(
        {
            "purpose": "v2_acoustic_evidence",
            "evidence_model": args.model,
            "input": str(args.raw),
            "requested_segments": len(raw_rows),
            "successful_hypotheses": len(evidence),
            "failed_segments": len(failures),
        }
    )
    print(
        json.dumps(
            {"evidence": len(evidence), "failures": len(failures)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
