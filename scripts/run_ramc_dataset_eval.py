#!/usr/bin/env python3
"""Evaluate ReTrace-ASR on a MagicData-RAMC WAV/TXT directory.

The evaluator preserves one complete API response and one metrics file per WAV.
Revision precision/recall require span-level gold labels, which RAMC does not
provide; those fields are recorded as null rather than estimated as truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import difflib
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
_TIMESTAMP_RE = re.compile(r"^\s*\[[0-9.]+\s*[,~-]\s*[0-9.]+\]")
_MENLI_SCORER: Any = None
_EMBED_MODEL: Any = None
_EMBED_TOKENIZER: Any = None
_METRIC_ERRORS: dict[str, str] = {}


def norm(text: str) -> str:
    text = re.sub(r"\[[0-9.]+\s*[,~-]\s*[0-9.]+\]", "", text or "")
    return "".join(_CHAR_RE.findall(text)).lower()


def edit_distance(left: str, right: str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, 1):
        current = [i]
        for j, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_clean = norm(reference)
    hypothesis_clean = norm(hypothesis)
    edits = edit_distance(reference_clean, hypothesis_clean)
    overlap = sum((Counter(reference_clean) & Counter(hypothesis_clean)).values())
    precision = overlap / len(hypothesis_clean) if hypothesis_clean else 0.0
    recall = overlap / len(reference_clean) if reference_clean else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cer": edits / len(reference_clean) if reference_clean else 0.0,
        "edits": edits,
        "reference_chars": len(reference_clean),
        "hypothesis_chars": len(hypothesis_clean),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def token_f1(reference: str, hypothesis: str) -> float:
    ref_tokens, hyp_tokens = list(norm(reference)), list(norm(hypothesis))
    ref_counts: dict[str, int] = {}
    hyp_counts: dict[str, int] = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in hyp_tokens:
        hyp_counts[token] = hyp_counts.get(token, 0) + 1
    overlap = sum(min(count, hyp_counts.get(token, 0)) for token, count in ref_counts.items())
    precision = overlap / len(hyp_tokens) if hyp_tokens else 0.0
    recall = overlap / len(ref_tokens) if ref_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean_pool_embedding(text: str) -> list[float]:
    global _EMBED_MODEL, _EMBED_TOKENIZER
    from transformers import AutoModel, AutoTokenizer
    import torch

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    if _EMBED_MODEL is None:
        _EMBED_TOKENIZER = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache")
        _EMBED_MODEL = AutoModel.from_pretrained(model_name, cache_dir=".cache").eval()
    batch = _EMBED_TOKENIZER([text], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        hidden = _EMBED_MODEL(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        vector = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        vector = torch.nn.functional.normalize(vector, p=2, dim=1)[0].cpu().tolist()
    return [float(value) for value in vector]


def embedding_similarity(reference: str, hypothesis: str) -> float | None:
    if "embedding_similarity" in _METRIC_ERRORS:
        return None
    try:
        left, right = _mean_pool_embedding(norm(reference)), _mean_pool_embedding(norm(hypothesis))
        return sum(a * b for a, b in zip(left, right))
    except Exception as exc:
        _METRIC_ERRORS.setdefault("embedding_similarity", f"{type(exc).__name__}: {exc}")
        return None


def official_menli(reference: str, hypothesis: str) -> float | None:
    global _MENLI_SCORER
    if "menli_tacl2023" in _METRIC_ERRORS:
        return None
    try:
        import menli
        package_dir = str(Path(menli.__file__).parent)
        if package_dir not in sys.path:
            sys.path.insert(0, package_dir)
        from menli.MENLI import MENLI
        if _MENLI_SCORER is None:
            _MENLI_SCORER = MENLI(
                direction="avg",
                formula="e",
                nli_weight=1.0,
                combine_with="None",
                model="D",
                cross_lingual=True,
            )
        score = _MENLI_SCORER.score_all(srcs=[], refs=[norm(reference)], hyps=[norm(hypothesis)])
        return float(score[0]) if score else None
    except Exception as exc:
        _METRIC_ERRORS.setdefault("menli_tacl2023", f"{type(exc).__name__}: {exc}")
        return None


def reference_text(path: Path) -> str:
    parts: list[str] = []
    current: list[str] = []

    def flush() -> None:
        text = "".join(current).strip()
        if text and text != "[*]":
            parts.append(text)

    for line in path.read_text(encoding="utf-8").splitlines():
        if _TIMESTAMP_RE.match(line):
            flush()
            fields = line.split(maxsplit=3)
            current = [fields[3].strip()] if len(fields) == 4 else []
        elif current and line.strip():
            current.append(line.strip())
    flush()
    return "".join(parts)


def reference_segments(path: Path) -> list[dict[str, Any]]:
    """Read RAMC groundtruth segments with time and speaker metadata."""
    segments: list[dict[str, Any]] = []
    pattern = re.compile(r"^\s*\[([0-9.]+)\s*,\s*([0-9.]+)\]\s+\S+\s+\S+\s+(.+?)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        text = match.group(3).strip()
        if text in {"[*]", "[+]"}:
            continue
        segments.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "text": text,
            "clean": norm(text),
        })
    return segments


def reference_for_turn(turn: dict[str, Any], segments: list[dict[str, Any]]) -> str:
    meta = turn.get("meta") or {}
    start, end = meta.get("start_sec"), meta.get("end_sec")
    if start is None or end is None:
        return ""
    overlapping = [
        segment for segment in segments
        if segment["start"] < float(end) and segment["end"] > float(start)
    ]
    return "".join(segment["clean"] for segment in overlapping)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def join_turns(session: dict[str, Any], field: str) -> str:
    return "".join(str(turn.get(field) or "") for turn in session.get("turns") or [])


def _dedupe_append(cleaned: str, part: str, *, max_overlap: int = 64) -> str:
    """Append normalized chunk text while removing duplicated overlap text."""
    if not cleaned:
        return part
    limit = min(max_overlap, len(cleaned), len(part))
    for width in range(limit, 0, -1):
        if cleaned[-width:] == part[:width]:
            return cleaned + part[width:]
    return cleaned + part


def join_turns_for_metrics(session: dict[str, Any], field: str) -> str:
    """Join chunk transcripts for metrics without charging overlap twice."""
    joined = ""
    previous_end: float | None = None
    for turn in session.get("turns") or []:
        part = norm(str(turn.get(field) or ""))
        if not part:
            continue
        meta = turn.get("meta") or {}
        start, end = meta.get("start_sec"), meta.get("end_sec")
        overlaps_previous = (
            previous_end is not None
            and start is not None
            and float(start) < previous_end
        )
        joined = _dedupe_append(joined, part) if overlaps_previous else joined + part
        if end is not None:
            previous_end = max(previous_end or float(end), float(end))
    return joined


def _overlap_seconds(start: float, end: float, segment: dict[str, Any]) -> float:
    return max(0.0, min(end, float(segment["end"])) - max(start, float(segment["start"])))


def reference_by_assigned_turn(
    turns: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, int]]:
    """Assign each reference segment to one best-overlap turn for diagnostics."""
    assigned: dict[str, list[str]] = {str(turn.get("turn_id") or ""): [] for turn in turns}
    ambiguous_turns: set[str] = set()
    for segment in segments:
        candidates: list[tuple[float, str]] = []
        for turn in turns:
            turn_id = str(turn.get("turn_id") or "")
            meta = turn.get("meta") or {}
            start, end = meta.get("start_sec"), meta.get("end_sec")
            if start is None or end is None:
                continue
            amount = _overlap_seconds(float(start), float(end), segment)
            if amount > 0:
                candidates.append((amount, turn_id))
        if len(candidates) > 1:
            ambiguous_turns.update(turn_id for _amount, turn_id in candidates)
        if candidates:
            _amount, chosen = max(candidates, key=lambda item: item[0])
            assigned.setdefault(chosen, []).append(str(segment.get("clean") or norm(str(segment.get("text") or ""))))
    references = {turn_id: "".join(parts) for turn_id, parts in assigned.items()}
    eligible = sum(1 for turn_id in assigned if assigned[turn_id])
    unscorable = max(0, len(turns) - eligible)
    return references, {
        "eligible_turn_count": eligible,
        "ambiguous_turn_count": len(ambiguous_turns),
        "unscorable_turn_count": unscorable,
    }


def groundtruth_comparison(
    session: dict[str, Any],
    turn_references: dict[str, str],
    *,
    limit: int = 15,
) -> dict[str, Any]:
    """Expose the highest-error turn-level comparisons for human review."""
    rows: list[dict[str, Any]] = []
    for turn in session.get("turns") or []:
        turn_id = str(turn.get("turn_id") or "")
        reference = norm(turn_references.get(turn_id, ""))
        raw = norm(str(turn.get("raw_text") or ""))
        current = norm(str(turn.get("current_text") or ""))
        if not reference:
            continue
        raw_edits = edit_distance(reference, raw)
        current_edits = edit_distance(reference, current)
        meta = turn.get("meta") or {}
        rows.append({
            "turn_id": turn_id,
            "start_sec": meta.get("start_sec"),
            "end_sec": meta.get("end_sec"),
            "reference": reference,
            "raw": raw,
            "current": current,
            "raw_edits": raw_edits,
            "current_edits": current_edits,
            "edit_change": raw_edits - current_edits,
            "raw_length": len(raw),
            "reference_length": len(reference),
        })
    rows.sort(key=lambda row: row["current_edits"], reverse=True)
    return {
        "turns": rows[:limit],
        "note": "References are selected by timestamp overlap; inspect speaker-boundary overlap before attributing an error to the ASR model.",
    }


def event_stats(session: dict[str, Any]) -> dict[str, Any]:
    events = [event for event in session.get("revision_events") or [] if event.get("active")]
    audits = [event for event in events if _is_audit_event(event)]
    revisions = [event for event in events if _is_committed_revision(event)]
    audio_events = [
        event for event in revisions
        if "audio" in str(event.get("resolver", "")).lower()
        or any("audio" in str(item).lower() for item in event.get("evidence") or [])
    ]
    target_turns = {event.get("target_turn_id") for event in audio_events if event.get("target_turn_id")}
    turns = session.get("turns") or []
    sentinel_observed_turns = 0
    sentinel_signal_turns = 0
    for turn in turns:
        uncertainty = (turn.get("meta") or {}).get("uncertainty") or {}
        observed = any(key in uncertainty for key in ("paraformer_text", "paraformer_char_confs", "acoustic_disagreement", "low_conf_chars"))
        signaled = bool(uncertainty.get("acoustic_disagreement") or uncertainty.get("low_conf_chars"))
        sentinel_observed_turns += int(observed)
        sentinel_signal_turns += int(signaled)
    relisten_seconds = 0.0
    for turn in session.get("turns") or []:
        if turn.get("turn_id") not in target_turns:
            continue
        meta = turn.get("meta") or {}
        if meta.get("start_sec") is not None and meta.get("end_sec") is not None:
            relisten_seconds += max(0.0, float(meta["end_sec"]) - float(meta["start_sec"]))
    whole_turn_recovery_count = 0
    repeated_tail_cleanup_count = 0
    local_revision_count = 0
    for event in revisions:
        evidence_text = " ".join(str(item) for item in event.get("evidence") or [])
        event_text = " ".join(
            str(event.get(key) or "")
            for key in ("event_id", "resolver", "rationale", "reason")
        )
        is_whole_turn = bool(
            norm(str(event.get("span") or ""))
            and norm(str(event.get("span") or "")) == norm(str(event.get("before_text") or ""))
        )
        is_repeated_tail = "repeated_tail" in evidence_text or "repeated-tail" in event_text
        whole_turn_recovery_count += int(is_whole_turn)
        repeated_tail_cleanup_count += int(is_repeated_tail)
        local_revision_count += int(not is_whole_turn and event.get("action") != "ROLLBACK")
    return {
        "decision_events": len(events),
        "audit_events": len(audits),
        "revision_events": len(revisions),
        "committed_revisions": len(revisions),
        "changed_turns": sum(turn.get("raw_text") != turn.get("current_text") for turn in session.get("turns") or []),
        "audio_verified_events": len(audio_events),
        "audio_verified_turns": len(target_turns),
        "relisten_seconds_proxy": relisten_seconds,
        "rollback_count": sum(event.get("action") == "ROLLBACK" for event in events),
        "defer_count": sum(event.get("action") == "DEFER" for event in events),
        "acoustic_sentinel_observed_turns": sentinel_observed_turns,
        "acoustic_sentinel_signal_turns": sentinel_signal_turns,
        "acoustic_sentinel_time_seconds": None,
        "whole_turn_recovery_count": whole_turn_recovery_count,
        "local_revision_count": local_revision_count,
        "repeated_tail_cleanup_count": repeated_tail_cleanup_count,
    }


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text or "")


def _has_latin(text: str) -> bool:
    return any(("a" <= char.lower() <= "z") for char in text or "")


def language_mismatch_count(
    session: dict[str, Any],
    turn_references: dict[str, str],
) -> int:
    count = 0
    for turn in session.get("turns") or []:
        turn_id = str(turn.get("turn_id") or "")
        reference = turn_references.get(turn_id, "")
        current = str(turn.get("current_text") or "")
        if _has_latin(reference) and not _has_cjk(reference) and _has_cjk(current):
            count += 1
        elif _has_cjk(reference) and not _has_latin(reference) and _has_latin(current):
            count += 1
    return count


def _event_replacement(event: dict[str, Any]) -> str:
    if event.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY", "REVISE_TEXT"} and event.get("span"):
        return norm(str(event.get("replacement") or ""))
    return norm(str(event.get("replacement") or event.get("after_text") or ""))


def _is_audit_event(event: dict[str, Any]) -> bool:
    return bool(
        event.get("event_kind") in {"audit", "candidate_audit"}
        or (
            not str(event.get("span") or "").strip()
            and str(event.get("resolver") or "") == "context-judge"
        )
    )


def _is_committed_revision(event: dict[str, Any]) -> bool:
    return bool(
        event.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY", "REVISE_TEXT", "ROLLBACK"}
        and event.get("event_kind", "revision") == "revision"
    )


def _event_has_audio_evidence(event: dict[str, Any]) -> bool:
    return bool(
        "audio" in str(event.get("resolver", "")).lower()
        or any("audio" in str(item).lower() for item in event.get("evidence") or [])
    )


def _revision_matches_reference(before: str, span: str, replacement: str, reference: str) -> bool:
    before, span, replacement, reference = map(norm, (before, span, replacement, reference))
    if not before or not span or not reference:
        return False
    start = before.find(span)
    if start < 0:
        return False
    end = start + len(span)
    projected_parts: list[str] = []
    for tag, before_start, before_end, ref_start, ref_end in difflib.SequenceMatcher(
        None, before, reference, autojunk=False
    ).get_opcodes():
        if before_end <= start or before_start >= end:
            continue
        if tag == "equal":
            overlap_start = max(start, before_start)
            overlap_end = min(end, before_end)
            projected = reference[ref_start + overlap_start - before_start : ref_start + overlap_end - before_start]
        elif tag == "replace":
            projected = reference[ref_start:ref_end]
        else:
            projected = ""
        projected_parts.append(projected)
    return "".join(projected_parts) == replacement


def _near_variant_pairs(reference: str, hypothesis: str) -> list[tuple[str, str]]:
    """Extract short Chinese replacement pairs as a RAMC near-variant proxy."""
    ref, hyp = norm(reference), norm(hypothesis)
    pairs: list[tuple[str, str]] = []
    for tag, ref_start, ref_end, hyp_start, hyp_end in difflib.SequenceMatcher(
        None, ref, hyp, autojunk=False
    ).get_opcodes():
        if tag != "replace":
            continue
        gold, raw = ref[ref_start:ref_end], hyp[hyp_start:hyp_end]
        if 2 <= len(gold) <= 8 and 2 <= len(raw) <= 8:
            if all("\u4e00" <= char <= "\u9fff" for char in gold + raw):
                pairs.append((raw, gold))
    return pairs


def revision_metrics(
    session: dict[str, Any],
    reference: str,
    duration: float,
    turn_references: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute the requested metrics from a RAMC session.

    RAMC groundtruth is treated as the human-correct transcript. Each event is
    judged against the reference text overlapping its target turn; no global
    substring match is used.
    """
    turns = session.get("turns") or []
    events = [event for event in session.get("revision_events") or [] if event.get("active", True)]
    revisions = [event for event in events if _is_committed_revision(event)]
    turn_order = {turn.get("turn_id"): index for index, turn in enumerate(turns)}
    refs = turn_references or {}
    correct = []
    for event in revisions:
        target_ref = refs.get(str(event.get("target_turn_id")), norm(reference))
        replacement = _event_replacement(event)
        raw_span = norm(str(event.get("span") or ""))
        if (
            replacement != raw_span
            and _revision_matches_reference(
                str(event.get("before_text") or ""),
                raw_span,
                replacement,
                target_ref,
            )
        ):
            correct.append(event)
    historical = [
        event for event in revisions
        if turn_order.get(event.get("source_turn_id"), -1) > turn_order.get(event.get("target_turn_id"), -1)
    ]
    current = [event for event in revisions if event not in historical]
    raw = "".join(str(turn.get("raw_text") or "") for turn in turns)
    pairs: list[tuple[str, str]] = []
    for turn in turns:
        target_ref = refs.get(str(turn.get("turn_id")), norm(reference))
        pairs.extend(_near_variant_pairs(target_ref, str(turn.get("raw_text") or "")))
    gold_pairs = {(raw_span, gold_span) for raw_span, gold_span in pairs}
    predicted_pairs = {
        (norm(str(event.get("span") or "")), _event_replacement(event))
        for event in revisions
        if norm(str(event.get("span") or "")) and _event_replacement(event)
    }
    matched_pairs = gold_pairs & predicted_pairs
    nv_precision = len(matched_pairs) / len(predicted_pairs) if predicted_pairs else 0.0
    nv_recall = len(matched_pairs) / len(gold_pairs) if gold_pairs else 0.0
    nv_f1 = 2 * nv_precision * nv_recall / (nv_precision + nv_recall) if nv_precision + nv_recall else 0.0
    relisten_seconds = event_stats(session)["relisten_seconds_proxy"]
    revision_count = len(revisions)
    correct_count = len(correct)
    audio_revision_count = sum(_event_has_audio_evidence(event) for event in revisions)
    historical_correct_count = sum(event in correct for event in historical)
    current_correct_count = sum(event in correct for event in current)
    historical_precision = historical_correct_count / len(historical) if historical else None
    current_precision = current_correct_count / len(current) if current else None
    historical_delay = [
        turn_order[str(event["source_turn_id"])] - turn_order[str(event["target_turn_id"])]
        for event in historical
        if str(event.get("source_turn_id")) in turn_order and str(event.get("target_turn_id")) in turn_order
    ]
    rollbacks = [event for event in events if event.get("action") == "ROLLBACK"]
    rollback_successes = sum(
        norm(str(event.get("after_text") or "")) == refs.get(str(event.get("target_turn_id")), "")
        for event in rollbacks
    )
    stats = event_stats(session)
    return {
        "HRA": historical_precision,
        "Revision Precision": correct_count / revision_count if revision_count else 0.0,
        "False Revision Rate": (revision_count - correct_count) / revision_count if revision_count else 0.0,
        "CCR": len(matched_pairs) / len(gold_pairs) if gold_pairs else 0.0,
        "EGR": audio_revision_count / revision_count if revision_count else 0.0,
        "RTS": relisten_seconds / duration if duration else 0.0,
        "Near-Variant Proper Noun F1": nv_f1,
        "_detail": {
            "committed_revisions": revision_count,
            "correct_revisions_proxy": correct_count,
            "historical_revision_count": len(historical),
            "current_revision_count": len(current),
            "historical_correct_revisions_proxy": historical_correct_count,
            "current_correct_revisions_proxy": current_correct_count,
            "historical_revision_precision": historical_precision,
            "current_revision_precision": current_precision,
            "historical_overcorrection_rate": 1 - historical_precision if historical_precision is not None else None,
            "current_overcorrection_rate": 1 - current_precision if current_precision is not None else None,
            "historical_resolution_latency_turns": statistics.mean(historical_delay) if historical_delay else None,
            "rollback_success_rate": rollback_successes / len(rollbacks) if rollbacks else None,
            "audio_evidence_revisions": audio_revision_count,
            "acoustic_sentinel_observed_turns": stats["acoustic_sentinel_observed_turns"],
            "acoustic_sentinel_signal_turns": stats["acoustic_sentinel_signal_turns"],
            "acoustic_sentinel_time_seconds": None,
            "component_time_note": "The live endpoint reports end-to-end elapsed time; first-pass and sentinel wall-clock timings require server-side instrumentation and remain null.",
            "near_variant_gold_pairs_proxy": sorted(gold_pairs),
            "near_variant_predicted_pairs": sorted(predicted_pairs),
            "near_variant_matched_pairs": sorted(matched_pairs),
            "relisten_seconds": relisten_seconds,
            "definition_note": (
                "RAMC groundtruth is treated as human-correct. Turn references are "
                "selected by timestamp overlap; near-variant pairs are inferred from "
                "raw-vs-groundtruth replacements and are not separately type-labeled."
            ),
        },
    }


def evaluate_one(base_url: str, wav: Path, txt: Path, output_root: Path, timeout: float, session_tag: str) -> dict[str, Any]:
    stem = wav.stem
    sample_dir = output_root / "samples" / stem
    sample_dir.mkdir(parents=True, exist_ok=True)
    reference = reference_text(txt)
    started = time.time()
    result: dict[str, Any] = {"audio_id": stem, "audio": str(wav), "reference": str(txt)}
    try:
        payload = post_json(
            f"{base_url.rstrip('/')}/api/sessions/{session_tag}_{stem}/audio",
            {"audio": str(wav)},
            timeout,
        )
        session = payload.get("session") or {}
        raw_display = join_turns(session, "raw_text")
        current_display = join_turns(session, "current_text")
        raw = join_turns_for_metrics(session, "raw_text")
        current = join_turns_for_metrics(session, "current_text")
        duration = audio_duration(wav)
        elapsed = time.time() - started
        raw_metrics = cer(reference, raw)
        current_metrics = cer(reference, current)
        stats = event_stats(session)
        groundtruth_segments = reference_segments(txt)
        turn_references, reference_diagnostics = reference_by_assigned_turn(session.get("turns") or [], groundtruth_segments)
        requested_metrics = revision_metrics(session, reference, duration, turn_references)
        comparison = groundtruth_comparison(session, turn_references)
        mismatch_count = language_mismatch_count(session, turn_references)
        metrics = {
            "audio_id": stem,
            "metric_kind": "diagnostic_cer",
            "official_scoring": False,
            "duration_seconds": duration,
            "elapsed_seconds": round(elapsed, 3),
            "rtf": round(elapsed / duration, 4) if duration else None,
            "raw_asr": raw_metrics,
            "final": current_metrics,
            "final_wer": None,
            "final_char_f1": token_f1(reference, current),
            "final_embedding_similarity": embedding_similarity(reference, current),
            "final_menli_tacl2023": official_menli(reference, current),
            "metric_errors": dict(_METRIC_ERRORS),
            "requested_metrics": requested_metrics,
            "groundtruth_comparison": comparison,
            "cer_absolute_gain": round(raw_metrics["cer"] - current_metrics["cer"], 6),
            "cer_relative_gain": round((raw_metrics["cer"] - current_metrics["cer"]) / raw_metrics["cer"], 6) if raw_metrics["cer"] else 0.0,
            **stats,
            "revision_precision": None,
            "revision_recall": None,
            "f2": None,
            "overcorrection_rate": None,
            "rollback_success_rate": None,
            "revision_delay_turns": None,
            "language_mismatch_count": mismatch_count,
            **reference_diagnostics,
            "notes": [
                "RAMC TXT groundtruth is treated as the human-correct reference.",
                "Global transcript CER is diagnostic and de-duplicates adjacent overlapped ASR chunks.",
                "final_wer is null because this script does not compute word error rate.",
            ],
        }
        (sample_dir / "raw_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (sample_dir / "reference.txt").write_text(reference, encoding="utf-8")
        (sample_dir / "raw_asr.txt").write_text(raw_display, encoding="utf-8")
        (sample_dir / "final_transcript.txt").write_text(current_display, encoding="utf-8")
        (sample_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        result.update({"status": "ok", **metrics})
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "error", "error": str(exc), "elapsed_seconds": round(time.time() - started, 3)})
        (sample_dir / "error.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    def mean(path: tuple[str, str]) -> float | None:
        values = [float(row[path[0]][path[1]]) for row in valid]
        return sum(values) / len(values) if values else None
    def requested_mean(name: str) -> float | None:
        values = [
            float(value)
            for row in valid
            if (value := (row.get("requested_metrics") or {}).get(name)) is not None
        ]
        return sum(values) / len(values) if values else None
    def detail_mean(name: str) -> float | None:
        values = [
            float(value)
            for row in valid
            if (value := ((row.get("requested_metrics") or {}).get("_detail") or {}).get(name)) is not None
        ]
        return sum(values) / len(values) if values else None
    return {
        "samples_total": len(rows),
        "samples_ok": len(valid),
        "samples_error": len(rows) - len(valid),
        "raw_mean_cer": mean(("raw_asr", "cer")),
        "final_mean_cer": mean(("final", "cer")),
        "mean_cer_absolute_gain": sum(row.get("cer_absolute_gain", 0.0) for row in valid) / len(valid) if valid else None,
        "mean_cer_relative_gain": sum(row.get("cer_relative_gain", 0.0) for row in valid) / len(valid) if valid else None,
        "revision_events_total": sum(row.get("revision_events", 0) for row in valid),
        "audio_verified_events_total": sum(row.get("audio_verified_events", 0) for row in valid),
        "relisten_seconds_proxy_total": sum(row.get("relisten_seconds_proxy", 0.0) for row in valid),
        "rollback_count_total": sum(row.get("rollback_count", 0) for row in valid),
        "acoustic_sentinel_observed_turns_total": sum(row.get("acoustic_sentinel_observed_turns", 0) for row in valid),
        "acoustic_sentinel_signal_turns_total": sum(row.get("acoustic_sentinel_signal_turns", 0) for row in valid),
        "whole_turn_recovery_count_total": sum(row.get("whole_turn_recovery_count", 0) for row in valid),
        "local_revision_count_total": sum(row.get("local_revision_count", 0) for row in valid),
        "repeated_tail_cleanup_count_total": sum(row.get("repeated_tail_cleanup_count", 0) for row in valid),
        "eligible_turn_count_total": sum(row.get("eligible_turn_count", 0) for row in valid),
        "ambiguous_turn_count_total": sum(row.get("ambiguous_turn_count", 0) for row in valid),
        "unscorable_turn_count_total": sum(row.get("unscorable_turn_count", 0) for row in valid),
        "language_mismatch_count_total": sum(row.get("language_mismatch_count", 0) for row in valid),
        "mean_rtf": sum(row.get("rtf", 0.0) or 0.0 for row in valid) / len(valid) if valid else None,
        "gold_label_metrics": "not available from transcript-only RAMC annotations",
        "metric_errors": dict(_METRIC_ERRORS),
        "requested_metrics": {
            name: requested_mean(name)
            for name in (
                "HRA",
                "Revision Precision",
                "False Revision Rate",
                "CCR",
                "EGR",
                "RTS",
                "Near-Variant Proper Noun F1",
            )
        },
        "route_metrics": {
            name: detail_mean(name)
            for name in (
                "historical_revision_precision",
                "current_revision_precision",
                "historical_overcorrection_rate",
                "current_overcorrection_rate",
                "historical_resolution_latency_turns",
                "rollback_success_rate",
            )
        },
        "component_time_note": "Component-level Qwen and Paraformer timings are not emitted by the current live endpoint; only end-to-end RTF is comparable until server instrumentation is added.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stems", nargs="+", help="evaluate only these WAV stems")
    parser.add_argument("--session-tag", default="", help="unique session namespace for this run")
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    wavs = sorted((args.dataset_root / "WAV").glob("*.wav"))
    if args.stems:
        wanted = set(args.stems)
        wavs = [wav for wav in wavs if wav.stem in wanted]
    if args.limit:
        wavs = wavs[:args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_root / "per_file_metrics.jsonl"
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    if args.resume and jsonl.exists():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                completed.add(row["audio_id"])
    for index, wav in enumerate(wavs, 1):
        if wav.stem in completed:
            continue
        txt = args.dataset_root / "TXT" / f"{wav.stem}.txt"
        if not txt.exists():
            row = {"audio_id": wav.stem, "status": "error", "error": f"missing reference: {txt}"}
        else:
            print(f"[{index}/{len(wavs)}] {wav.name}", flush=True)
            session_tag = args.session_tag.strip() or args.output_root.name
            row = evaluate_one(args.base_url, wav, txt, args.output_root, args.timeout, session_tag)
            print(json.dumps({k: row.get(k) for k in ("audio_id", "status", "raw_asr", "final", "revision_events", "rtf", "error")}, ensure_ascii=False), flush=True)
        rows.append(row)
        with jsonl.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"dataset_root": str(args.dataset_root), "output_root": str(args.output_root), "metrics": aggregate(rows), "files": rows}
    (args.output_root / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_root / "dataset_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "audio_id", "status", "duration_seconds", "elapsed_seconds", "rtf",
            "raw_cer", "final_cer", "cer_absolute_gain", "cer_relative_gain",
            "revision_events", "audio_verified_events", "relisten_seconds_proxy", "rollback_count",
            "historical_revision_count", "current_revision_count",
            "historical_revision_precision", "current_revision_precision",
            "historical_overcorrection_rate", "current_overcorrection_rate",
            "historical_resolution_latency_turns", "rollback_success_rate",
            "acoustic_sentinel_observed_turns", "acoustic_sentinel_signal_turns",
            "whole_turn_recovery_count", "local_revision_count", "repeated_tail_cleanup_count",
            "eligible_turn_count", "ambiguous_turn_count", "unscorable_turn_count", "language_mismatch_count",
            "error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            detail = (row.get("requested_metrics") or {}).get("_detail") or {}
            writer.writerow({
                "audio_id": row.get("audio_id"), "status": row.get("status"), "duration_seconds": row.get("duration_seconds"), "elapsed_seconds": row.get("elapsed_seconds"), "rtf": row.get("rtf"),
                "raw_cer": (row.get("raw_asr") or {}).get("cer"), "final_cer": (row.get("final") or {}).get("cer"), "cer_absolute_gain": row.get("cer_absolute_gain"), "cer_relative_gain": row.get("cer_relative_gain"),
                "revision_events": row.get("revision_events"), "audio_verified_events": row.get("audio_verified_events"), "relisten_seconds_proxy": row.get("relisten_seconds_proxy"), "rollback_count": row.get("rollback_count"),
                "historical_revision_count": detail.get("historical_revision_count"), "current_revision_count": detail.get("current_revision_count"),
                "historical_revision_precision": detail.get("historical_revision_precision"), "current_revision_precision": detail.get("current_revision_precision"),
                "historical_overcorrection_rate": detail.get("historical_overcorrection_rate"), "current_overcorrection_rate": detail.get("current_overcorrection_rate"),
                "historical_resolution_latency_turns": detail.get("historical_resolution_latency_turns"), "rollback_success_rate": detail.get("rollback_success_rate"),
                "acoustic_sentinel_observed_turns": row.get("acoustic_sentinel_observed_turns"), "acoustic_sentinel_signal_turns": row.get("acoustic_sentinel_signal_turns"),
                "whole_turn_recovery_count": row.get("whole_turn_recovery_count"), "local_revision_count": row.get("local_revision_count"), "repeated_tail_cleanup_count": row.get("repeated_tail_cleanup_count"),
                "eligible_turn_count": row.get("eligible_turn_count"), "ambiguous_turn_count": row.get("ambiguous_turn_count"), "unscorable_turn_count": row.get("unscorable_turn_count"), "language_mismatch_count": row.get("language_mismatch_count"),
                "error": row.get("error", ""),
            })
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
