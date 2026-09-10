#!/usr/bin/env python3
"""Export frozen Raw MOSS turns from existing evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrace_v2.artifacts import ArtifactWriter
from retrace_v2.dataset import extract_raw_rows


def _audio_for(wav_dir: Path, recording_id: str) -> Path:
    for suffix in (".wav", ".flac"):
        candidate = wav_dir / f"{recording_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise ValueError(f"audio not found for {recording_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare immutable V2 development input")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-model", default="MOSS-Transcribe-Diarize")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    artifacts = sorted(args.results_root.glob("*/retrace_result.json"))
    if args.limit > 0:
        artifacts = artifacts[: args.limit]
    if not artifacts:
        raise ValueError("no retrace_result.json artifacts found")

    rows: list[dict[str, object]] = []
    recordings = []
    for artifact in artifacts:
        recording_id = artifact.parent.name
        audio = _audio_for(args.wav_dir, recording_id)
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        rows.extend(
            extract_raw_rows(
                payload,
                recording_id=recording_id,
                audio_path=audio,
                baseline_model=args.baseline_model,
            )
        )
        recordings.append(recording_id)

    writer = ArtifactWriter(args.output_dir)
    writer.write_jsonl("raw_moss.jsonl", rows)
    writer.write_manifest(
        {
            "purpose": "v2_development_input",
            "baseline_model": args.baseline_model,
            "recordings": recordings,
            "segment_count": len(rows),
        }
    )
    print(json.dumps({"recordings": recordings, "segments": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
