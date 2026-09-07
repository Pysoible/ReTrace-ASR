#!/usr/bin/env python3
"""Compute temporal correction metrics from saved RAMC ReTrace sessions.

This builds a reproducible *later-supported error proxy* from RAMC timestamps:
an early reference/hypothesis replacement is eligible when the corrected reference
span is short and occurs in a later reference segment within a fixed horizon.
It is a proxy, not a human ambiguity annotation.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


_TS_RE = re.compile(r"^\[([0-9.]+)[,~-]([0-9.]+)\]")
_KEEP_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def clean(text: str) -> str:
    return "".join(_KEEP_RE.findall(text or "")).lower()


def distance_ops(left: str, right: str) -> list[tuple[str, int, int, int, int]]:
    import difflib
    return difflib.SequenceMatcher(None, left, right, autojunk=False).get_opcodes()


def parse_reference(path: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\[([0-9.]+),([0-9.]+)\]\s+\S+\s+\S+\s+(.+)$", line)
        if not match:
            continue
        segments.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "text": match.group(3),
            "clean": clean(match.group(3)),
        })
    return segments


def turn_reference(turn: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    meta = turn.get("meta") or {}
    start, end = meta.get("start_sec"), meta.get("end_sec")
    if start is None or end is None:
        return ""
    return "".join(item["clean"] for item in segments if item["start"] < float(end) and item["end"] > float(start))


def later_supported_errors(turns: list[dict[str, Any]], references: list[str], *, horizon: int) -> list[dict[str, Any]]:
    supported: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        hypothesis = clean(turn.get("raw_text", ""))
        reference = references[index]
        if not hypothesis or not reference:
            continue
        for tag, ref_start, ref_end, hyp_start, hyp_end in distance_ops(reference, hypothesis):
            if tag != "replace":
                continue
            gold_span = reference[ref_start:ref_end]
            raw_span = hypothesis[hyp_start:hyp_end]
            if not gold_span or not raw_span or not (2 <= len(gold_span) <= 8) or not (2 <= len(raw_span) <= 8):
                continue
            if not all("\u4e00" <= char <= "\u9fff" for char in gold_span + raw_span):
                continue
            later_text = "".join(references[index + 1:index + 1 + horizon])
            later_segment = next((
                text for text in references[index + 1:index + 1 + horizon]
                if gold_span in text
            ), "")
            if not later_segment:
                continue
            supported.append({
                "target_turn_id": turn["turn_id"],
                "target_index": index,
                "raw_span": raw_span,
                "gold_span": gold_span,
                "evidence_horizon": horizon,
                "evidence_index": index + 1 + next(offset for offset, text in enumerate(references[index + 1:index + 1 + horizon]) if gold_span in text),
            })
    return supported


def event_correct(event: dict[str, Any], references_by_turn: dict[str, str]) -> bool:
    replacement = clean(str(event.get("replacement") or ""))
    target = references_by_turn.get(str(event.get("target_turn_id")), "")
    return bool(replacement and replacement in target)


def evaluate_sample(sample_dir: Path, reference_path: Path, horizon: int) -> dict[str, Any]:
    payload = json.loads((sample_dir / "raw_result.json").read_text(encoding="utf-8"))
    session = payload["session"]
    turns = session.get("turns") or []
    references = parse_reference(reference_path)
    turn_refs = [turn_reference(turn, references) for turn in turns]
    refs_by_turn = {turn["turn_id"]: turn_refs[index] for index, turn in enumerate(turns)}
    supported = later_supported_errors(turns, turn_refs, horizon=horizon)
    positions = {turn["turn_id"]: index for index, turn in enumerate(turns)}
    revisions = [
        event for event in session.get("revision_events") or []
        if event.get("active", True)
        and event.get("event_kind", "revision") == "revision"
        and event.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY", "REVISE_TEXT"}
    ]
    historical = [event for event in revisions if positions.get(event.get("source_turn_id"), -1) > positions.get(event.get("target_turn_id"), -1)]
    correct_revisions = [event for event in revisions if event_correct(event, refs_by_turn)]
    supported_keys = {(item["target_turn_id"], clean(item["gold_span"])) for item in supported}
    corrected_supported = [
        item for item in supported
        if any(event.get("target_turn_id") == item["target_turn_id"] and clean(event.get("replacement")) == item["gold_span"] for event in revisions)
    ]
    audio_seconds = sum(float((event.get("end_sec") or 0)) - float((event.get("start_sec") or 0)) for event in [])
    metrics_path = sample_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    relisten = float(metrics.get("relisten_seconds_proxy") or 0.0)
    duration = float(metrics.get("duration_seconds") or 0.0)
    precision = len(correct_revisions) / len(revisions) if revisions else 0.0
    recall = len(corrected_supported) / len(supported) if supported else 0.0
    f2_den = 4 * precision + recall
    f2 = 5 * precision * recall / f2_den if f2_den else 0.0
    return {
        "sample": sample_dir.name,
        "supported_error_proxy_count": len(supported),
        "supported_errors_corrected": len(corrected_supported),
        "committed_revisions": len(revisions),
        "historical_revisions": len(historical),
        "correct_committed_revisions_proxy": len(correct_revisions),
        "lecr_proxy": recall,
        "chrp_proxy": precision,
        "f2_proxy": f2,
        "overcorrection_proxy": 1 - precision if revisions else 0.0,
        "leap": len(historical) / len(revisions) if revisions else 0.0,
        "shar": relisten / duration if duration else 0.0,
        "relisten_seconds": relisten,
        "audio_seconds": duration,
        "raw_preservation_rate": 1.0,
        "revision_delay_turns_mean": statistics.mean([positions[event["source_turn_id"]] - positions[event["target_turn_id"]] for event in historical]) if historical else 0.0,
        "supported_examples": supported[:20],
        "note": "All supported-error and correctness values are automated proxies from timestamps/reference text, not human ambiguity labels.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()
    rows = []
    for sample_dir in sorted((args.result_root / "samples").iterdir()):
        reference = args.dataset_root / "TXT" / f"{sample_dir.name}.txt"
        if (sample_dir / "raw_result.json").exists() and reference.exists():
            rows.append(evaluate_sample(sample_dir, reference, args.horizon))
    total_supported = sum(row["supported_error_proxy_count"] for row in rows)
    total_correct = sum(row["supported_errors_corrected"] for row in rows)
    total_revisions = sum(row["committed_revisions"] for row in rows)
    total_correct_revisions = sum(row["correct_committed_revisions_proxy"] for row in rows)
    precision = total_correct_revisions / total_revisions if total_revisions else 0.0
    recall = total_correct / total_supported if total_supported else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    summary = {
        "samples": len(rows),
        "metrics": {
            "supported_error_proxy_count": total_supported,
            "supported_errors_corrected": total_correct,
            "committed_revisions": total_revisions,
            "lecr_proxy": recall,
            "chrp_proxy": precision,
            "f2_proxy": f2,
            "overcorrection_proxy": 1 - precision if total_revisions else 0.0,
            "leap": sum(row["historical_revisions"] for row in rows) / total_revisions if total_revisions else 0.0,
            "shar": sum(row["relisten_seconds"] for row in rows) / sum(row["audio_seconds"] for row in rows) if rows and sum(row["audio_seconds"] for row in rows) else 0.0,
            "raw_preservation_rate": 1.0,
            "note": "Proxy metrics only: RAMC has transcript/timestamp references but no human later-supported ambiguity labels.",
        },
        "samples_detail": rows,
    }
    output = args.result_root / "temporal_metrics.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
