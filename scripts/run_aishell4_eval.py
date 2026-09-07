#!/usr/bin/env python3
"""Evaluate AI-SHELL-4 with durable results and offline-only GT diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from Levenshtein import editops as levenshtein_editops
from scripts.report_candidate_pipeline import candidate_pipeline_metrics

_ROW_RE = re.compile(r"^\s*([0-9.]+)\s+([0-9.]+)\s+(\S+)\s+(.+?)\s*$")


def clean(text: str) -> str:
    text = re.sub(r"\[\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?\]", "", text or "")
    text = re.sub(r"<[^>]+>", "", text)
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)).lower()


def edit_ops(reference: str, hypothesis: str) -> dict[str, int]:
    operations = Counter(item[0] for item in levenshtein_editops(reference, hypothesis))
    return {
        "correct": len(reference) - operations["replace"] - operations["delete"],
        "substitutions": operations["replace"],
        "deletions": operations["delete"],
        "insertions": operations["insert"],
    }


def metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref, hyp = clean(reference), clean(hypothesis)
    ops = edit_ops(ref, hyp)
    edits = ops["substitutions"] + ops["deletions"] + ops["insertions"]
    overlap = sum((Counter(ref) & Counter(hyp)).values())
    precision = overlap / len(hyp) if hyp else 0.0
    recall = overlap / len(ref) if ref else 0.0
    return {
        "cer": edits / len(ref) if ref else (1.0 if hyp else 0.0),
        "edits": edits, "reference_chars": len(ref), "hypothesis_chars": len(hyp), **ops,
        "char_precision": precision, "char_recall": recall,
        "char_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def parse_reference(path: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ROW_RE.match(line)
        if match:
            rows.append({"start": float(match[1]), "end": float(match[2]), "spk": match[3], "text": re.sub(r"<[^>]+>", "", match[4]).strip()})
    rows.sort(key=lambda row: (row["start"], row["end"]))
    return "".join(row["text"] for row in rows), rows


def retrace_miss_analysis(session: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Use GT only after inference; never feed reference text into ReTrace."""
    turns = []
    for index, turn in enumerate(session.get("turns") or []):
        meta = turn.get("meta") or {}
        if meta.get("start_sec") is not None and meta.get("end_sec") is not None:
            turns.append((index, float(meta["start_sec"]), float(meta["end_sec"]), turn))
    assigned = {index: [] for index, *_ in turns}
    for row in rows:
        overlaps = [(max(0.0, min(row["end"], end) - max(row["start"], start)), index) for index, start, end, _ in turns]
        overlaps = [(amount, index) for amount, index in overlaps if amount > 0]
        if overlaps:
            assigned[max(overlaps)[1]].append(row)

    categories, revision_outcomes = Counter(), Counter()
    diagnostics = []
    raw_edits = final_edits = 0
    for index, start, end, turn in turns:
        reference = "".join(row["text"] for row in sorted(assigned[index], key=lambda row: (row["start"], row["end"])))
        raw = str(turn.get("raw_text") or "")
        current = str(turn.get("current_text") or raw)
        changed = clean(raw) != clean(current)
        if not clean(reference) and not changed:
            continue
        raw_metric, final_metric = metrics(reference, raw), metrics(reference, current)
        raw_edits += raw_metric["edits"]
        final_edits += final_metric["edits"]
        meta, uncertainty = turn.get("meta") or {}, (turn.get("meta") or {}).get("uncertainty") or {}
        judgment = meta.get("context_judgment") or {}
        has_signal = bool(uncertainty.get("acoustic_disagreement") or uncertainty.get("low_conf_chars") or (uncertainty.get("quality") or {}).get("suspect") or judgment.get("outcome") in {"UNCERTAIN", "CONFLICT"})
        if final_metric["edits"] == 0:
            category = "correct"
        elif final_metric["edits"] < raw_metric["edits"]:
            category = "corrected_improved"
        elif final_metric["edits"] > raw_metric["edits"]:
            category = "revision_harmed"
        elif uncertainty.get("acoustic_error"):
            category = "missed_acoustic_failure"
        elif has_signal:
            category = "detected_not_revised"
        else:
            category = "missed_no_online_signal"
        categories[category] += 1
        if changed:
            revision_outcomes["improved" if final_metric["edits"] < raw_metric["edits"] else "harmed" if final_metric["edits"] > raw_metric["edits"] else "neutral"] += 1
        if category != "correct":
            diagnostics.append({
                "turn_id": turn.get("turn_id"), "start_sec": start, "end_sec": end,
                "category": category, "raw_edits": raw_metric["edits"], "final_edits": final_metric["edits"],
                "reference": reference, "raw_text": raw, "current_text": current,
                "online_signals": {"acoustic_disagreement": uncertainty.get("acoustic_disagreement") or [], "low_conf_chars": uncertainty.get("low_conf_chars") or [], "acoustic_error": uncertainty.get("acoustic_error"), "quality": uncertainty.get("quality") or {}, "judgment": judgment},
            })
    diagnostics.sort(key=lambda item: (-item["final_edits"], item["start_sec"]))
    evaluated = sum(revision_outcomes.values())
    return {
        "purpose": "offline_gt_diagnostics_only", "categories": dict(categories),
        "raw_edits": raw_edits, "final_edits": final_edits, "net_edits_removed": raw_edits - final_edits,
        "revision_quality": {**dict(revision_outcomes), "evaluated": evaluated, "improvement_rate": revision_outcomes["improved"] / evaluated if evaluated else 0.0, "harm_rate": revision_outcomes["harmed"] / evaluated if evaluated else 0.0},
        "candidate_pipeline": {
            key: value
            for key, value in candidate_pipeline_metrics(session).items()
            if key != "candidate_sources"
        },
        "worst_missed_turns": diagnostics[:30],
    }


def evaluate_sample(base_url: str, audio: Path, reference_path: Path, output_dir: Path, timeout: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path, metrics_path = output_dir / "result.json", output_dir / "metrics.json"
    if result_path.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    reference, rows = parse_reference(reference_path)
    started = time.time()
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        session_id = f"aishell4_full_{audio.stem}_{int(time.time())}"
        response = requests.post(f"{base_url.rstrip('/')}/api/sessions/{session_id}/audio", json={"audio": str(audio)}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    session = payload.get("session") or {}
    turns = sorted(session.get("turns") or [], key=lambda turn: float((turn.get("meta") or {}).get("start_sec", 0)))
    raw, current = "".join(str(turn.get("raw_text") or "") for turn in turns), "".join(str(turn.get("current_text") or "") for turn in turns)
    item = {
        "sample": audio.stem, "audio": str(audio), "reference": str(reference_path), "result": str(result_path),
        "metric_kind": "diagnostic_whole_corpus_character_metrics", "audio_duration_sec": payload.get("asr", {}).get("duration_sec"),
        "request_elapsed_sec": round(time.time() - started, 2), "asr_elapsed_sec": payload.get("asr", {}).get("elapsed_sec"),
        "reference_segments": len(rows), "reference_overlap_pairs": sum(left["start"] < right["end"] and right["start"] < left["end"] and left["spk"] != right["spk"] for index, left in enumerate(rows) for right in rows[index + 1:]),
        "asr_chunks": payload.get("asr", {}).get("chunk_count"), "valid_turns": len(turns),
        "empty_chunks": sum(not str(text).strip() for text in payload.get("asr", {}).get("chunks_text", [])),
        "degenerate_turns": sum(bool((turn.get("meta") or {}).get("degeneration", {}).get("detected")) for turn in turns),
        "changed_turns": sum(turn.get("raw_text") != turn.get("current_text") for turn in turns),
        "raw": metrics(reference, raw), "final": metrics(reference, current),
        "retrace": {"active_events": sum(event.get("active", True) for event in session.get("revision_events") or []), "committed_revisions": sum(event.get("active", True) and event.get("event_kind") != "audit" and event.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY", "ROLLBACK"} for event in session.get("revision_events") or [])},
        "retrace_miss_analysis": retrace_miss_analysis(session, rows),
    }
    metrics_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=Path, required=True); parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=7200); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); samples = sorted(args.wav_dir.glob("*.wav")); summary_path = args.output_dir / "summary.json"; args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {item["sample"]: item for item in json.loads(summary_path.read_text(encoding="utf-8"))} if args.resume and summary_path.exists() else {}
    for index, audio in enumerate(samples, 1):
        reference = args.reference_dir / f"{audio.stem}.txt"
        if not reference.exists(): summary[audio.stem] = {"sample": audio.stem, "error": f"missing reference: {reference}"}; continue
        print(f"[{index}/{len(samples)}] {audio.name}", flush=True)
        try:
            item = evaluate_sample(args.base_url, audio, reference, args.output_dir / audio.stem, args.timeout); summary[audio.stem] = item
            summary_path.write_text(json.dumps(list(summary.values()), ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"sample": audio.stem, "raw_cer": item["raw"]["cer"], "final_cer": item["final"]["cer"], "chunks": item.get("asr_chunks")}, ensure_ascii=False), flush=True)
        except Exception as exc:
            summary[audio.stem] = {"sample": audio.stem, "error": f"{type(exc).__name__}: {exc}"}; summary_path.write_text(json.dumps(list(summary.values()), ensure_ascii=False, indent=2), encoding="utf-8"); print(summary[audio.stem], flush=True)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
