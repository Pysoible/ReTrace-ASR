#!/usr/bin/env python3
"""Build an offline scoring file by time-aligning TextGrid references."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrace_v2.dataset import assign_references, parse_textgrid


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _textgrid_for(directory: Path, recording_id: str) -> Path:
    for suffix in (".TextGrid", ".textgrid", ".TextGrid.txt"):
        candidate = directory / f"{recording_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise ValueError(f"TextGrid not found for {recording_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare offline V2 scoring references")
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording-output", type=Path)
    args = parser.parse_args()

    by_recording: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in _read_jsonl(args.raw):
        by_recording[str(row["recording_id"])].append(row)

    output: list[dict[str, str]] = []
    recording_output: list[dict[str, str]] = []
    for recording_id, raw_rows in sorted(by_recording.items()):
        spans = parse_textgrid(_textgrid_for(args.textgrid_dir, recording_id))
        output.extend(assign_references(raw_rows, spans))
        recording_output.append(
            {
                "recording_id": recording_id,
                "reference_text": "".join(span.text for span in spans),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    recording_path = args.recording_output or args.output.with_name(
        "recording_reference.jsonl"
    )
    recording_path.parent.mkdir(parents=True, exist_ok=True)
    recording_temporary = recording_path.with_suffix(recording_path.suffix + ".tmp")
    with recording_temporary.open("w", encoding="utf-8") as handle:
        for row in recording_output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    recording_temporary.replace(recording_path)
    print(json.dumps({"recordings": len(by_recording), "segments": len(output)}))


if __name__ == "__main__":
    main()
