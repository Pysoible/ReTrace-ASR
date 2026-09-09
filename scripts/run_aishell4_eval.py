#!/usr/bin/env python3
"""Evaluate AI-SHELL-4 with durable results and offline-only GT diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import difflib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from Levenshtein import editops as levenshtein_editops

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.report_candidate_pipeline import candidate_pipeline_metrics

_ROW_RE = re.compile(r"^\s*([0-9.]+)\s+([0-9.]+)\s+(\S+)\s+(.+?)\s*$")
_TEXTGRID_NAME_RE = re.compile(r'^\s*name\s*=\s*"(.*?)"\s*$')
_TEXTGRID_XMIN_RE = re.compile(r"^\s*xmin\s*=\s*([0-9.]+)\s*$")
_TEXTGRID_XMAX_RE = re.compile(r"^\s*xmax\s*=\s*([0-9.]+)\s*$")
_TEXTGRID_TEXT_RE = re.compile(r'^\s*text\s*=\s*"(.*?)"\s*$')


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


def _repeated_tail(text: str) -> bool:
    value = clean(text)
    for width in range(1, min(8, len(value) // 2) + 1):
        if value[-width:] == value[-2 * width:-width]:
            return True
    return False


def classify_gt_error(
    reference: str,
    hypothesis: str,
    *,
    assigned_rows: list[dict[str, Any]],
    later_reference: str,
) -> dict[str, Any]:
    """Classify a GT mismatch after inference and nominate bounded Agent actions.

    The labels are diagnostic opportunities, not oracle instructions. In
    particular, a deletion is only *audio-action eligible*: the online Agent
    still needs a coverage/VAD signal before it may re-decode the audio.
    """
    ref, hyp = clean(reference), clean(hypothesis)
    ops = edit_ops(ref, hyp)
    edits = ops["substitutions"] + ops["deletions"] + ops["insertions"]
    if not edits:
        return {
            "primary_error_type": "correct",
            "error_types": [],
            "eligibility": [],
            "recommended_actions": [],
            "later_supported_spans": [],
        }

    error_types: list[str] = []
    eligibility: list[str] = []
    actions: list[str] = []
    speakers = {str(row.get("spk") or "unknown") for row in assigned_rows}
    crosses_asr_turns = any(int(row.get("_asr_turn_overlap_count") or 1) > 1 for row in assigned_rows)
    if len(speakers) > 1 or crosses_asr_turns:
        error_types.append("segmentation_or_speaker_assignment")
        eligibility.append("alignment_only")
        actions.append("REALIGN_SPEAKER_BOUNDARIES")

    deletion_heavy = ops["deletions"] >= max(2, ops["substitutions"], ops["insertions"])
    severe_undercoverage = len(ref) >= max(len(hyp) + 4, int(len(hyp) * 1.35))
    if (not hyp and ref) or deletion_heavy or severe_undercoverage:
        error_types.append("omission_or_undercoverage")
        eligibility.append("audio_action_eligible")
        actions.extend(["EXPAND_WINDOW", "RESEGMENT", "REDECODE"])

    insertion_heavy = ops["insertions"] >= max(2, ops["substitutions"], ops["deletions"])
    if insertion_heavy or _repeated_tail(hypothesis):
        error_types.append("insertion_or_repetition")
        eligibility.append("audio_action_eligible")
        actions.append("VERIFY_TIMING_AND_REDECODE")

    if ops["substitutions"] and ops["substitutions"] >= max(ops["deletions"], ops["insertions"]):
        error_types.append("acoustic_substitution")
        eligibility.append("audio_action_eligible")
        actions.append("VERIFY_LOCAL_AUDIO")

    later_clean = clean(later_reference)
    later_supported: list[str] = []
    for tag, left_start, left_end, _right_start, _right_end in difflib.SequenceMatcher(
        None, ref, hyp, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        span = ref[left_start:left_end]
        if 2 <= len(span) <= 12 and span in later_clean:
            later_supported.append(span)
    later_supported = list(dict.fromkeys(later_supported))
    if later_supported:
        error_types.append("later_evidence_candidate")
        eligibility.append("context_action_candidate")
        actions.extend(["WAIT", "RETRIEVE_MEMORY", "VERIFY_LOCAL_AUDIO"])

    if not error_types:
        error_types.append("mixed_or_unresolved")
    priority = (
        "segmentation_or_speaker_assignment"
        if "segmentation_or_speaker_assignment" in error_types
        else "omission_or_undercoverage"
        if "omission_or_undercoverage" in error_types
        else "insertion_or_repetition"
        if "insertion_or_repetition" in error_types
        else "acoustic_substitution"
        if "acoustic_substitution" in error_types
        else "later_evidence_candidate"
        if "later_evidence_candidate" in error_types
        else "mixed_or_unresolved"
    )
    return {
        "primary_error_type": priority,
        "error_types": list(dict.fromkeys(error_types)),
        "eligibility": list(dict.fromkeys(eligibility)),
        "recommended_actions": list(dict.fromkeys(actions)),
        "later_supported_spans": later_supported,
    }


def _parse_text_reference(path: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _ROW_RE.match(line)
        if match:
            rows.append({"start": float(match[1]), "end": float(match[2]), "spk": match[3], "text": re.sub(r"<[^>]+>", "", match[4]).strip()})
    rows.sort(key=lambda row: (row["start"], row["end"]))
    return "".join(row["text"] for row in rows), rows


def _parse_textgrid_reference(path: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    speaker = "unknown"
    start: float | None = None
    end: float | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if name_match := _TEXTGRID_NAME_RE.match(line):
            speaker = name_match[1] or speaker
            continue
        if xmin_match := _TEXTGRID_XMIN_RE.match(line):
            start = float(xmin_match[1])
            continue
        if xmax_match := _TEXTGRID_XMAX_RE.match(line):
            end = float(xmax_match[1])
            continue
        if text_match := _TEXTGRID_TEXT_RE.match(line):
            text = re.sub(r"<[^>]+>", "", text_match[1]).strip()
            if text and start is not None and end is not None and end > start:
                rows.append({"start": start, "end": end, "spk": speaker, "text": text})
            start = end = None
    rows.sort(key=lambda row: (row["start"], row["end"]))
    return "".join(row["text"] for row in rows), rows


def parse_reference(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.name.lower().endswith((".textgrid", ".textgrid.txt")):
        return _parse_textgrid_reference(path)
    return _parse_text_reference(path)


def resolve_reference_path(reference_dir: Path, stem: str) -> Path | None:
    for suffix in (".txt", ".TextGrid", ".textgrid"):
        candidate = reference_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


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
            assigned[max(overlaps)[1]].append({
                **row,
                "_asr_turn_overlap_count": len(overlaps),
            })

    categories, revision_outcomes = Counter(), Counter()
    error_type_counts, eligibility_counts, action_counts = Counter(), Counter(), Counter()
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
        later_reference = "".join(
            row["text"] for row in rows if float(row["start"]) >= end
        )
        taxonomy = classify_gt_error(
            reference,
            current,
            assigned_rows=assigned[index],
            later_reference=later_reference,
        )
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
        error_type_counts.update(taxonomy["error_types"])
        eligibility_counts.update(taxonomy["eligibility"])
        action_counts.update(taxonomy["recommended_actions"])
        if changed:
            revision_outcomes["improved" if final_metric["edits"] < raw_metric["edits"] else "harmed" if final_metric["edits"] > raw_metric["edits"] else "neutral"] += 1
        if category != "correct":
            diagnostics.append({
                "turn_id": turn.get("turn_id"), "start_sec": start, "end_sec": end,
                "category": category, "raw_edits": raw_metric["edits"], "final_edits": final_metric["edits"],
                "reference": reference, "raw_text": raw, "current_text": current,
                "gt_error_taxonomy": taxonomy,
                "online_signals": {"acoustic_disagreement": uncertainty.get("acoustic_disagreement") or [], "low_conf_chars": uncertainty.get("low_conf_chars") or [], "acoustic_error": uncertainty.get("acoustic_error"), "quality": uncertainty.get("quality") or {}, "judgment": judgment},
            })
    diagnostics.sort(key=lambda item: (-item["final_edits"], item["start_sec"]))
    evaluated = sum(revision_outcomes.values())
    return {
        "purpose": "offline_gt_diagnostics_only", "categories": dict(categories),
        "gt_error_taxonomy": {
            "error_type_counts": dict(error_type_counts),
            "eligibility_counts": dict(eligibility_counts),
            "recommended_action_counts": dict(action_counts),
            "note": "Ground Truth is used only after inference; eligibility labels nominate an action class and do not authorize an online revision.",
        },
        "raw_edits": raw_edits, "final_edits": final_edits, "net_edits_removed": raw_edits - final_edits,
        "revision_quality": {**dict(revision_outcomes), "evaluated": evaluated, "improvement_rate": revision_outcomes["improved"] / evaluated if evaluated else 0.0, "harm_rate": revision_outcomes["harmed"] / evaluated if evaluated else 0.0},
        "candidate_pipeline": {
            key: value
            for key, value in candidate_pipeline_metrics(session).items()
            if key != "candidate_sources"
        },
        "worst_missed_turns": diagnostics[:30],
    }


def _turn_references(session: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    turns = []
    for turn in session.get("turns") or []:
        meta = turn.get("meta") or {}
        if meta.get("start_sec") is not None and meta.get("end_sec") is not None:
            turns.append((str(turn.get("turn_id")), float(meta["start_sec"]), float(meta["end_sec"])))
    assigned: dict[str, list[dict[str, Any]]] = {turn_id: [] for turn_id, *_ in turns}
    for row in rows:
        overlaps = [
            (max(0.0, min(float(row["end"]), end) - max(float(row["start"]), start)), turn_id)
            for turn_id, start, end in turns
        ]
        overlaps = [(amount, turn_id) for amount, turn_id in overlaps if amount > 0]
        if overlaps:
            assigned[max(overlaps)[1]].append(row)
    return {
        turn_id: "".join(item["text"] for item in sorted(items, key=lambda item: (item["start"], item["end"])))
        for turn_id, items in assigned.items()
    }


def _committed_revision_quality(session: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, int]:
    references = _turn_references(session, rows)
    outcomes = Counter()
    for event in session.get("revision_events") or []:
        if not event.get("active", True) or event.get("event_kind", "revision") != "revision":
            continue
        if event.get("action") not in {"REVISE_CURRENT", "REVISE_HISTORY"}:
            continue
        reference = references.get(str(event.get("target_turn_id")), "")
        before = metrics(reference, str(event.get("before_text") or ""))["edits"]
        after = metrics(reference, str(event.get("after_text") or ""))["edits"]
        outcomes["improved" if after < before else "harmed" if after > before else "neutral"] += 1
    return {
        "committed": sum(outcomes.values()),
        "improved": outcomes["improved"],
        "harmed": outcomes["harmed"],
        "neutral": outcomes["neutral"],
    }


def evaluate_pair_payloads(
    baseline_payload: dict[str, Any],
    retrace_payload: dict[str, Any],
    *,
    reference: str,
    rows: list[dict[str, Any]],
    integrations: dict[str, Any],
) -> dict[str, Any]:
    """Validate and score one baseline/ReTrace pair from one immutable first pass."""
    session = retrace_payload.get("session") or {}
    turns = sorted(
        session.get("turns") or [],
        key=lambda turn: float((turn.get("meta") or {}).get("start_sec", 0)),
    )
    baseline_text = str(baseline_payload.get("transcript") or "")
    raw_text = "".join(str(turn.get("raw_text") or "") for turn in turns)
    final_text = "".join(str(turn.get("current_text") or turn.get("raw_text") or "") for turn in turns)
    artifact_id = baseline_payload.get("first_pass_artifact_id")
    invalid_reasons: list[str] = []
    if not artifact_id or artifact_id != retrace_payload.get("first_pass_artifact_id"):
        invalid_reasons.append("first_pass_artifact_mismatch")
    for payload in (baseline_payload, retrace_payload):
        asr = payload.get("asr") or {}
        if not asr.get("ok") or (asr.get("completeness") or {}).get("truncated") is True:
            if "incomplete_first_pass" not in invalid_reasons:
                invalid_reasons.append("incomplete_first_pass")
    if not (integrations.get("deepseek") or {}).get("ready"):
        invalid_reasons.append("deepseek_not_ready")
    if clean(baseline_text) != clean(raw_text):
        invalid_reasons.append("raw_transcript_mismatch")
    memory_status = ((session.get("observability") or {}).get("memory_status") or {})
    if memory_status.get("error"):
        invalid_reasons.append("memory_error")

    baseline_metric = metrics(reference, baseline_text)
    raw_metric = metrics(reference, raw_text)
    final_metric = metrics(reference, final_text)
    revision_quality = _committed_revision_quality(session, rows)
    funnel = candidate_pipeline_metrics(session)
    effectiveness = None
    if not invalid_reasons:
        baseline_edits = int(baseline_metric["edits"])
        committed = revision_quality["committed"]
        validated = int(funnel["validated_candidates"])
        error_type_reduction = {
            key: {
                "net_removed": int(baseline_metric[key]) - int(final_metric[key]),
                "relative": (
                    (int(baseline_metric[key]) - int(final_metric[key])) / int(baseline_metric[key])
                    if int(baseline_metric[key]) else 0.0
                ),
            }
            for key in ("substitutions", "insertions", "deletions")
        }
        effectiveness = {
            "ecer": (baseline_edits - int(final_metric["edits"])) / baseline_edits if baseline_edits else 0.0,
            "revision_precision": revision_quality["improved"] / committed if committed else None,
            "candidate_to_correction_yield": revision_quality["improved"] / validated if validated else None,
            "error_type_reduction": error_type_reduction,
        }
    return {
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "first_pass_artifact_id": artifact_id,
        "baseline": baseline_metric,
        "raw": raw_metric,
        "final": final_metric,
        "effectiveness": effectiveness,
        "retrace": {
            "committed_revisions": revision_quality["committed"],
            "revision_quality": revision_quality,
            "candidate_funnel": funnel,
        },
        "stage_timings_ms": retrace_payload.get("stage_timings_ms") or {},
        "stage_call_counts": retrace_payload.get("stage_call_counts") or {},
        "baseline_stage_timings_ms": baseline_payload.get("stage_timings_ms") or {},
        "memory_scope": session.get("memory_scope"),
        "memory_status": memory_status,
        "completeness": (retrace_payload.get("asr") or {}).get("completeness") or {},
    }


def aggregate_results(items: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in items if item.get("valid") is True]
    invalid = [item for item in items if item.get("valid") is not True]
    baseline_edits = sum(int((item.get("baseline") or {}).get("edits") or 0) for item in valid)
    final_edits = sum(int((item.get("final") or {}).get("edits") or 0) for item in valid)
    baseline_by_type = {
        key: sum(int((item.get("baseline") or {}).get(key) or 0) for item in valid)
        for key in ("substitutions", "insertions", "deletions")
    }
    final_by_type = {
        key: sum(int((item.get("final") or {}).get(key) or 0) for item in valid)
        for key in ("substitutions", "insertions", "deletions")
    }
    committed = sum(int(((item.get("retrace") or {}).get("revision_quality") or {}).get("committed") or 0) for item in valid)
    improved = sum(int(((item.get("retrace") or {}).get("revision_quality") or {}).get("improved") or 0) for item in valid)
    validated = sum(int(((item.get("retrace") or {}).get("candidate_funnel") or {}).get("validated_candidates") or 0) for item in valid)
    stage_timings: Counter[str] = Counter()
    stage_calls: Counter[str] = Counter()
    for item in valid:
        stage_timings.update({str(key): float(value) for key, value in (item.get("stage_timings_ms") or {}).items()})
        stage_calls.update({str(key): int(value) for key, value in (item.get("stage_call_counts") or {}).items()})
    return {
        "valid_samples": len(valid),
        "invalid_samples": len(invalid),
        "pooled_counts": {
            "baseline_edits": baseline_edits,
            "final_edits": final_edits,
            "committed_revisions": committed,
            "improved_revisions": improved,
            "validated_candidates": validated,
            "baseline_by_error_type": baseline_by_type,
            "final_by_error_type": final_by_type,
        },
        "effectiveness": {
            "ecer": (baseline_edits - final_edits) / baseline_edits if baseline_edits else 0.0,
            "revision_precision": improved / committed if committed else None,
            "candidate_to_correction_yield": improved / validated if validated else None,
            "error_type_reduction": {
                key: {
                    "net_removed": baseline_by_type[key] - final_by_type[key],
                    "relative": (
                        (baseline_by_type[key] - final_by_type[key]) / baseline_by_type[key]
                        if baseline_by_type[key] else 0.0
                    ),
                }
                for key in ("substitutions", "insertions", "deletions")
            },
        },
        "stage_timings_ms": dict(stage_timings),
        "stage_call_counts": dict(stage_calls),
        "invalid": [
            {"sample": item.get("sample"), "reasons": item.get("invalid_reasons") or [item.get("error") or "unknown_error"]}
            for item in invalid
        ],
    }


def evaluate_sample(base_url: str, audio: Path, reference_path: Path, output_dir: Path, timeout: float) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / "baseline_result.json"
    retrace_path = output_dir / "retrace_result.json"
    metrics_path = output_dir / "metrics.json"
    if baseline_path.exists() and retrace_path.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    reference, rows = parse_reference(reference_path)
    started = time.time()
    def post_mode(mode: str, path: Path) -> dict[str, Any]:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        session_id = f"aishell4_{mode}_{audio.stem}_{int(time.time())}"
        response = requests.post(
            f"{base_url.rstrip('/')}/api/sessions/{session_id}/audio",
            json={"audio": str(audio), "experiment_mode": mode},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    baseline_payload = post_mode("baseline", baseline_path)
    retrace_payload = post_mode("retrace", retrace_path)
    status_response = requests.get(f"{base_url.rstrip('/')}/api/integrations/status", timeout=min(timeout, 30.0))
    status_response.raise_for_status()
    integrations = status_response.json()
    session = retrace_payload.get("session") or {}
    turns = session.get("turns") or []
    item = evaluate_pair_payloads(
        baseline_payload,
        retrace_payload,
        reference=reference,
        rows=rows,
        integrations=integrations,
    )
    item.update({
        "sample": audio.stem, "audio": str(audio), "reference": str(reference_path),
        "baseline_result": str(baseline_path), "retrace_result": str(retrace_path),
        "metric_kind": "diagnostic_whole_corpus_character_metrics",
        "audio_duration_sec": retrace_payload.get("asr", {}).get("duration_sec"),
        "request_elapsed_sec": round(time.time() - started, 2),
        "asr_elapsed_sec": retrace_payload.get("asr", {}).get("elapsed_sec"),
        "reference_segments": len(rows),
        "reference_overlap_pairs": sum(left["start"] < right["end"] and right["start"] < left["end"] and left["spk"] != right["spk"] for index, left in enumerate(rows) for right in rows[index + 1:]),
        "asr_chunks": retrace_payload.get("asr", {}).get("chunk_count"), "valid_turns": len(turns),
        "empty_chunks": sum(not str(text).strip() for text in retrace_payload.get("asr", {}).get("chunks_text", [])),
        "degenerate_turns": sum(bool((turn.get("meta") or {}).get("degeneration", {}).get("detected")) for turn in turns),
        "changed_turns": sum(turn.get("raw_text") != turn.get("current_text") for turn in turns),
        "retrace_miss_analysis": retrace_miss_analysis(session, rows),
    })
    metrics_path.write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=Path, required=True); parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=7200); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    samples = sorted(
        path
        for suffix in ("*.wav", "*.flac")
        for path in args.wav_dir.glob(suffix)
    )
    summary_path = args.output_dir / "summary.json"; args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {item["sample"]: item for item in json.loads(summary_path.read_text(encoding="utf-8"))} if args.resume and summary_path.exists() else {}
    for index, audio in enumerate(samples, 1):
        reference = resolve_reference_path(args.reference_dir, audio.stem)
        if reference is None: summary[audio.stem] = {"sample": audio.stem, "error": f"missing reference for stem {audio.stem} in {args.reference_dir}"}; continue
        print(f"[{index}/{len(samples)}] {audio.name}", flush=True)
        try:
            item = evaluate_sample(args.base_url, audio, reference, args.output_dir / audio.stem, args.timeout); summary[audio.stem] = item
            summary_path.write_text(json.dumps(list(summary.values()), ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"sample": audio.stem, "valid": item["valid"], "ecer": (item.get("effectiveness") or {}).get("ecer"), "raw_cer": item["raw"]["cer"], "final_cer": item["final"]["cer"], "chunks": item.get("asr_chunks")}, ensure_ascii=False), flush=True)
        except Exception as exc:
            summary[audio.stem] = {"sample": audio.stem, "error": f"{type(exc).__name__}: {exc}"}; summary_path.write_text(json.dumps(list(summary.values()), ensure_ascii=False, indent=2), encoding="utf-8"); print(summary[audio.stem], flush=True)
    aggregate_path = args.output_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate_results(list(summary.values())), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path} and {aggregate_path}")


if __name__ == "__main__":
    main()
