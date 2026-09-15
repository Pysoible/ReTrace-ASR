#!/usr/bin/env python3
"""Evaluate an AMI STM result with timestamp-aligned references.

AMI STM contains overlapping speakers, so concatenating all lines is invalid.
The default ``dominant`` mode selects the speaker with the most overlap in each
ASR turn. Use ``--speaker`` for a fixed-speaker report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
_TIME_RE = re.compile(r"^\[[0-9.]+\s*[-,~]\s*[0-9.]+\]")


def normalize(text: str) -> str:
    return "".join(_CHAR_RE.findall(_TIME_RE.sub("", text or ""))).lower()


def words(text: str) -> list[str]:
    return _WORD_RE.findall(_TIME_RE.sub("", text or "").lower())


def edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for index, value in enumerate(left, 1):
        current = [index]
        for right_index, other in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (value != other),
            ))
        previous = current
    return previous[-1]


def parse_stm(path: Path) -> list[dict[str, Any]]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        text = fields[5].strip()
        if not text or text in {"<ignore_time_segment_in_scoring>", "[*]", "[+]"}:
            continue
        segments.append({
            "recording": fields[0],
            "speaker": fields[2],
            "start": float(fields[3]),
            "end": float(fields[4]),
            "text": text,
        })
    return segments


def write_time_sorted_stm(source: Path, destination: Path) -> Path:
    """Write a chronological derived STM without modifying the source file."""
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) >= 6:
            lines.append((float(fields[3]), float(fields[4]), fields[2], line))
    lines.sort(key=lambda item: (item[0], item[1], item[2]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(item[3] for item in lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return destination


def parse_rttm(path: Path) -> list[dict[str, Any]]:
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0] != "SPEAKER":
            continue
        start = float(fields[3])
        segments.append({
            "recording": fields[1],
            "speaker": fields[7],
            "start": start,
            "end": start + float(fields[4]),
        })
    return segments


def run_official_asclite(
    reference_stm: Path,
    hypothesis_ctm: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run NIST asclite on official AMI STM/CTM files without approximation."""
    asclite = shutil.which("asclite")
    if asclite is None:
        raise RuntimeError(
            "official AMI multi-speaker scoring requires NIST asclite; install asclite and put it on PATH"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_root = output_dir / "sclite"
    command = [
        asclite,
        "-r", str(reference_stm), "stm",
        "-h", str(hypothesis_ctm), "ctm",
        "-i", "wsj",
        "-o", "all",
        "-O", str(output_dir),
        "-n", result_root.name,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"sclite failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    report = result_root.with_suffix(".sys")
    return {
        "tool": "asclite",
        "command": command,
        "report": str(report) if report.exists() else None,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def overlap(start: float, end: float, segment: dict[str, Any]) -> float:
    return max(0.0, min(end, segment["end"]) - max(start, segment["start"]))


def reference_for_turn(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
    speaker: str | None,
    mode: str,
    hypothesis: str = "",
) -> tuple[str, str | None]:
    active = [segment for segment in segments if overlap(start, end, segment) > 0]
    if speaker:
        selected = [segment for segment in active if segment["speaker"] == speaker]
        return " ".join(segment["text"] for segment in sorted(selected, key=lambda item: item["start"])), speaker
    if mode == "dominant":
        durations: Counter[str] = Counter()
        for segment in active:
            durations[segment["speaker"]] += overlap(start, end, segment)
        chosen = durations.most_common(1)[0][0] if durations else None
        selected = [segment for segment in active if segment["speaker"] == chosen]
        return " ".join(segment["text"] for segment in sorted(selected, key=lambda item: item["start"])), chosen
    if mode == "best_speaker":
        by_speaker: dict[str, list[dict[str, Any]]] = {}
        for segment in active:
            by_speaker.setdefault(segment["speaker"], []).append(segment)
        if not by_speaker:
            return "", None
        hypothesis_words = words(hypothesis)

        def speaker_distance(candidate: str) -> tuple[float, float]:
            candidate_words = words(" ".join(item["text"] for item in by_speaker[candidate]))
            if not candidate_words:
                return (float("inf"), float("inf"))
            error_rate = edit_distance(candidate_words, hypothesis_words) / len(candidate_words)
            length_ratio = abs(len(candidate_words) - len(hypothesis_words)) / max(len(candidate_words), len(hypothesis_words), 1)
            return (error_rate + 0.25 * length_ratio, error_rate)

        chosen = min(
            by_speaker,
            key=speaker_distance,
        )
        selected = by_speaker[chosen]
        return " ".join(segment["text"] for segment in sorted(selected, key=lambda item: item["start"])), chosen
    return " ".join(segment["text"] for segment in sorted(active, key=lambda item: item["start"])), None


def assign_segments_to_turns(
    segments: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> dict[int, set[int]]:
    """Assign each STM segment to one ASR turn to prevent overlap double-counting."""
    assignments: dict[int, set[int]] = {index: set() for index in range(len(turns))}
    for segment_index, segment in enumerate(segments):
        candidates = [
            (overlap(float((turn.get("meta") or {}).get("start_sec")), float((turn.get("meta") or {}).get("end_sec")), segment), turn_index)
            for turn_index, turn in enumerate(turns)
            if (turn.get("meta") or {}).get("start_sec") is not None
            and (turn.get("meta") or {}).get("end_sec") is not None
        ]
        candidates = [(amount, index) for amount, index in candidates if amount > 0]
        if candidates:
            _, turn_index = max(candidates)
            assignments[turn_index].add(segment_index)
    return assignments


def dominant_speaker(rttm: list[dict[str, Any]], start: float, end: float) -> str | None:
    durations: Counter[str] = Counter()
    for segment in rttm:
        durations[segment["speaker"]] += overlap(start, end, segment)
    return durations.most_common(1)[0][0] if durations else None


def speaker_attribution(rttm: list[dict[str, Any]], start: float, end: float) -> dict[str, Any]:
    """Attribute one ASR turn to RTTM speakers and expose ambiguity."""
    durations: Counter[str] = Counter()
    intervals: list[tuple[float, float]] = []
    for segment in rttm:
        amount = overlap(start, end, segment)
        if amount > 0:
            durations[segment["speaker"]] += amount
            intervals.append((max(start, segment["start"]), min(end, segment["end"])))
    covered = sum(durations.values())
    union_covered = 0.0
    merged_start = merged_end = None
    for interval_start, interval_end in sorted(intervals):
        if interval_end <= interval_start:
            continue
        if merged_start is None:
            merged_start, merged_end = interval_start, interval_end
        elif interval_start > merged_end:
            union_covered += merged_end - merged_start
            merged_start, merged_end = interval_start, interval_end
        else:
            merged_end = max(merged_end, interval_end)
    if merged_start is not None:
        union_covered += merged_end - merged_start
    ranked = durations.most_common()
    return {
        "speaker": ranked[0][0] if ranked else None,
        "overlap_seconds": round(float(covered), 3),
        "union_coverage_seconds": round(float(union_covered), 3),
        "coverage_ratio": round(float(union_covered / max(end - start, 1e-9)), 3) if end > start else 0.0,
        "parallel_overlap_ratio": round(float(max(0.0, covered - union_covered) / max(union_covered, 1e-9)), 3) if union_covered else 0.0,
        "speaker_overlap_seconds": {speaker: round(float(value), 3) for speaker, value in ranked},
        "ambiguous": len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.5,
    }


def ambiguous_reference_segments(segments: list[dict[str, Any]]) -> bool:
    """True when assigned reference text still contains overlapping speakers."""
    for left_index, left in enumerate(segments):
        for right in segments[left_index + 1:]:
            if left.get("speaker") == right.get("speaker"):
                continue
            if overlap(float(left["start"]), float(left["end"]), right) > 0:
                return True
    return False


def rttm_speakers(rttm: list[dict[str, Any]]) -> list[str]:
    return sorted({str(segment["speaker"]) for segment in rttm})


def score(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_chars, hyp_chars = normalize(reference), normalize(hypothesis)
    ref_words, hyp_words = words(reference), words(hypothesis)
    char_edits = edit_distance(ref_chars, hyp_chars)
    word_edits = edit_distance(ref_words, hyp_words)
    overlap_count = sum((Counter(ref_chars) & Counter(hyp_chars)).values())
    precision = overlap_count / len(hyp_chars) if hyp_chars else 0.0
    recall = overlap_count / len(ref_chars) if ref_chars else 0.0
    return {
        "cer": char_edits / len(ref_chars) if ref_chars else 0.0,
        "wer": word_edits / len(ref_words) if ref_words else 0.0,
        "char_edits": char_edits,
        "word_edits": word_edits,
        "reference_chars": len(ref_chars),
        "hypothesis_chars": len(hyp_chars),
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "char_precision": precision,
        "char_recall": recall,
        "char_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def cpwer(
    references: dict[str, list[str]],
    hypotheses: dict[str, list[str]],
) -> dict[str, Any]:
    """Compute speaker-attributed cpWER with an optimal reference permutation."""
    speakers = sorted(set(references) | set(hypotheses))
    reference_words = {speaker: words(" ".join(references.get(speaker, []))) for speaker in speakers}
    hypothesis_words = {speaker: words(" ".join(hypotheses.get(speaker, []))) for speaker in speakers}
    reference_count = sum(len(value) for value in reference_words.values())
    best_edits: int | None = None
    best_assignment: dict[str, str] = {}
    for permutation in permutations(speakers):
        assignment = dict(zip(speakers, permutation))
        edits = sum(
            edit_distance(reference_words[reference_speaker], hypothesis_words[hypothesis_speaker])
            for hypothesis_speaker, reference_speaker in assignment.items()
        )
        if best_edits is None or edits < best_edits:
            best_edits = edits
            best_assignment = assignment
    return {
        "cpwer": best_edits / reference_count if reference_count else 0.0,
        "word_edits": best_edits or 0,
        "reference_words": reference_count,
        "hypothesis_words": sum(len(value) for value in hypothesis_words.values()),
        "speakers": speakers,
        "hypothesis_to_reference": best_assignment,
    }


def evaluate(
    result_path: Path,
    stm_path: Path,
    mode: str,
    speaker: str | None,
    rttm_path: Path | None = None,
    sorted_stm_path: Path | None = None,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    session = result.get("session") or {}
    turns = session.get("turns") or []
    effective_stm_path = stm_path
    if mode == "time_sorted":
        effective_stm_path = sorted_stm_path or stm_path.with_name(f"{stm_path.stem}.time_sorted.stm")
        write_time_sorted_stm(stm_path, effective_stm_path)
    segments = parse_stm(effective_stm_path)
    rttm = parse_rttm(rttm_path) if rttm_path else []
    if mode == "multi_speaker" and not rttm:
        raise ValueError("multi_speaker mode requires --rttm")
    rows = []
    assignments = assign_segments_to_turns(segments, turns)
    eligible_turn_count = 0
    ambiguous_turn_count = 0
    unscorable_turn_count = 0
    raw_reference, current_reference = [], []
    raw_hypothesis, current_hypothesis = [], []
    raw_speaker_hypotheses: dict[str, list[str]] = {speaker_id: [] for speaker_id in rttm_speakers(rttm)}
    current_speaker_hypotheses: dict[str, list[str]] = {speaker_id: [] for speaker_id in rttm_speakers(rttm)}
    speaker_references: dict[str, list[str]] = {speaker_id: [] for speaker_id in rttm_speakers(rttm)}
    attribution_rows: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(turns):
        meta = turn.get("meta") or {}
        start, end = meta.get("start_sec"), meta.get("end_sec")
        if start is None or end is None:
            continue
        raw = str(turn.get("raw_text") or "")
        current = str(turn.get("current_text") or "")
        active_segments = [segments[index] for index in assignments[turn_index]]
        ambiguous = ambiguous_reference_segments(active_segments)
        if mode == "multi_speaker":
            for segment in sorted(active_segments, key=lambda item: item["start"]):
                speaker_references.setdefault(segment["speaker"], []).append(segment["text"])
        reference, selected_speaker = reference_for_turn(
            active_segments, float(start), float(end), speaker, mode, raw
        )
        raw_reference.append(reference)
        current_reference.append(reference)
        raw_hypothesis.append(raw)
        current_hypothesis.append(current)
        assigned_speaker = dominant_speaker(rttm, float(start), float(end)) if mode == "multi_speaker" else None
        attribution = speaker_attribution(rttm, float(start), float(end)) if mode == "multi_speaker" else {}
        if assigned_speaker:
            raw_speaker_hypotheses.setdefault(assigned_speaker, []).append(raw)
            current_speaker_hypotheses.setdefault(assigned_speaker, []).append(current)
        if mode == "multi_speaker":
            attribution_rows.append({"turn_id": turn.get("turn_id"), **attribution})
            ambiguous = ambiguous or bool(attribution.get("ambiguous")) or float(attribution.get("parallel_overlap_ratio") or 0.0) > 0.0
        if not reference:
            unscorable_turn_count += 1
        elif ambiguous:
            ambiguous_turn_count += 1
            unscorable_turn_count += 1
        else:
            eligible_turn_count += 1
        rows.append({
            "turn_id": turn.get("turn_id"),
            "start_sec": start,
            "end_sec": end,
            "speaker": selected_speaker,
            "ambiguous": ambiguous,
            "reference": reference,
            "raw": raw,
            "current": current,
            "raw_metrics": score(reference, raw),
            "current_metrics": score(reference, current),
        })
    raw_metrics = score(" ".join(raw_reference), " ".join(raw_hypothesis))
    current_metrics = score(" ".join(current_reference), " ".join(current_hypothesis))
    multi_raw = multi_current = None
    if mode == "multi_speaker":
        multi_raw = cpwer(speaker_references, raw_speaker_hypotheses)
        multi_current = cpwer(speaker_references, current_speaker_hypotheses)
    events = [event for event in session.get("revision_events") or [] if event.get("active", True)]
    revisions = [
        event for event in events
        if event.get("event_kind", "revision") == "revision"
        and event.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY", "REVISE_TEXT", "ROLLBACK"}
    ]
    return {
        "result": str(result_path),
        "reference": str(stm_path),
        "effective_reference": str(effective_stm_path),
        "reference_mode": mode,
        "metric_kind": "diagnostic_chunk_aligned",
        "official_scoring": False,
        "fixed_speaker": speaker,
        "rttm": str(rttm_path) if rttm_path else None,
        "turns_scored": len(rows),
        "eligible_turn_count": eligible_turn_count,
        "ambiguous_turn_count": ambiguous_turn_count,
        "unscorable_turn_count": unscorable_turn_count,
        "speakers_in_stm": sorted({segment["speaker"] for segment in segments}),
        "raw_asr": raw_metrics,
        "final": current_metrics,
        "multi_speaker_raw": multi_raw,
        "multi_speaker_final": multi_current,
        "speaker_attribution": attribution_rows if mode == "multi_speaker" else None,
        "cer_absolute_gain": raw_metrics["cer"] - current_metrics["cer"],
        "wer_absolute_gain": raw_metrics["wer"] - current_metrics["wer"],
        "revision_events": len(events),
        "committed_revisions": len(revisions),
        "changed_turns": sum(row["raw"] != row["current"] for row in rows),
        "turn_metrics": rows,
        "warning": (
            "RTTM-attributed cpWER with chunk-level speaker assignment; verify with official AMI scoring"
            if mode == "multi_speaker"
            else
            "time-sorted STM diagnostic with unique segment assignment; overlapping speakers still require official AMI scoring"
            if mode == "time_sorted"
            else
            "best_speaker uses hypothesis-guided oracle speaker selection and is not official AMI scoring"
            if mode == "best_speaker"
            else "timestamp-aligned diagnostic, not official AMI multi-speaker cpWER/ORC WER"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--stm", type=Path, required=True)
    parser.add_argument("--mode", choices=("dominant", "best_speaker", "multi_speaker", "time_sorted", "official", "all"), default="dominant")
    parser.add_argument("--rttm", type=Path, help="RTTM file required by multi_speaker mode")
    parser.add_argument("--ctm", type=Path, help="system CTM required by official mode")
    parser.add_argument("--official-output", type=Path, help="directory for asclite output")
    parser.add_argument("--sorted-stm-output", type=Path, help="derived STM path for time_sorted mode")
    parser.add_argument("--speaker")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.speaker and args.mode in {"all", "multi_speaker"}:
        parser.error("--speaker cannot be combined with --mode all or multi_speaker")
    if args.mode == "multi_speaker" and not args.rttm:
        parser.error("--rttm is required by --mode multi_speaker")
    if args.mode == "official":
        if not args.rttm or not args.ctm:
            parser.error("--rttm and --ctm are required by --mode official")
        report = run_official_asclite(
            args.stm,
            args.ctm,
            args.official_output or args.result.parent / "asclite",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    report = evaluate(args.result, args.stm, args.mode, args.speaker, args.rttm, args.sorted_stm_output)
    output = args.output or args.result.with_name("ami_aligned_metrics.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "turn_metrics"}, ensure_ascii=False, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
