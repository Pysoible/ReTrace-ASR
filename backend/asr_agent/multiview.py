"""GT-free acoustic window detection and closed-set transcript comparison."""
from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
import re

import numpy as np
import soundfile as sf
from Levenshtein import distance as edit_distance


def circular_delay_and_sum(
    samples: np.ndarray,
    sampling_rate: int,
    azimuth_degrees: float,
    *,
    radius_m: float = 0.05,
    sound_speed_mps: float = 343.0,
) -> np.ndarray:
    """Form one exploratory far-field beam for an evenly spaced circular array."""
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("delay-and-sum requires a samples-by-channels array")
    channel_count = samples.shape[1]
    microphone_angles = 2.0 * np.pi * np.arange(channel_count) / channel_count
    positions = radius_m * np.stack(
        [np.cos(microphone_angles), np.sin(microphone_angles)], axis=1
    )
    azimuth = np.deg2rad(float(azimuth_degrees))
    direction = np.asarray([np.cos(azimuth), np.sin(azimuth)])
    delays = positions @ direction / sound_speed_mps * sampling_rate
    timeline = np.arange(len(samples), dtype=float)
    aligned = [
        np.interp(timeline + delay, timeline, samples[:, channel], left=0.0, right=0.0)
        for channel, delay in enumerate(delays)
    ]
    return np.mean(aligned, axis=0).astype(np.float32)


def _merge_intervals(
    intervals: Iterable[tuple[float, float]], *, gap_sec: float = 0.0
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if not merged or start > merged[-1][1] + gap_sec:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _covered_at(time_sec: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start <= time_sec <= end for start, end in intervals)


def _group_masked_frames(
    frames: list[dict[str, float]],
    mask: list[bool],
    *,
    pad_sec: float,
    gap_sec: float,
    min_sec: float,
    duration_sec: float,
) -> list[tuple[float, float]]:
    intervals = [
        (max(0.0, frame["start_sec"] - pad_sec), min(duration_sec, frame["end_sec"] + pad_sec))
        for frame, selected in zip(frames, mask)
        if selected
    ]
    return [
        (start, end)
        for start, end in _merge_intervals(intervals, gap_sec=gap_sec)
        if end - start >= min_sec
    ]


def detect_acoustic_windows(
    audio_path: str | Path,
    asr_chunks: list[dict[str, Any]],
    *,
    frame_sec: float = 0.25,
    noise_percentile: float = 15.0,
    speech_margin_db: float = 6.0,
    spatial_percentile: float = 88.0,
    coverage_tolerance_sec: float = 0.20,
    max_windows: int = 8,
) -> dict[str, Any]:
    """Nominate suspicious windows from audio only, then compare ASR coverage.

    Speech activity is derived from robust frame energy.  The spatial score is
    the second-to-first eigenvalue ratio of the multichannel covariance: it is
    only a routing signal for possible multiple spatial sources, never a claim
    that overlap is present and never authority to revise text.
    """
    source = Path(audio_path)
    frames: list[dict[str, float]] = []
    with sf.SoundFile(source) as handle:
        sampling_rate = int(handle.samplerate)
        channels = int(handle.channels)
        frame_samples = max(1, round(frame_sec * sampling_rate))
        position = 0
        while True:
            samples = handle.read(frame_samples, dtype="float32", always_2d=True)
            if not len(samples):
                break
            start = position / sampling_rate
            end = (position + len(samples)) / sampling_rate
            position += len(samples)
            centered = samples - samples.mean(axis=0, keepdims=True)
            rms = float(np.sqrt(np.mean(centered * centered) + 1e-12))
            spatial = 0.0
            if channels > 1 and len(samples) > channels:
                covariance = centered.T @ centered / max(1, len(centered) - 1)
                eigenvalues = np.linalg.eigvalsh(covariance)
                largest = float(max(eigenvalues[-1], 1e-12))
                spatial = float(max(0.0, eigenvalues[-2]) / largest)
            frames.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "rms_db": 20.0 * float(np.log10(max(rms, 1e-12))),
                    "spatial_ratio": spatial,
                }
            )
        duration = float(len(handle) / sampling_rate)

    energies = np.asarray([frame["rms_db"] for frame in frames], dtype=float)
    noise_floor = float(np.percentile(energies, noise_percentile)) if len(energies) else -120.0
    speech_threshold = noise_floor + speech_margin_db
    speech_mask = [frame["rms_db"] >= speech_threshold for frame in frames]
    speech_spatial = [
        frame["spatial_ratio"] for frame, speech in zip(frames, speech_mask) if speech
    ]
    spatial_threshold = (
        float(np.percentile(speech_spatial, spatial_percentile))
        if speech_spatial
        else 1.0
    )

    coverage = _merge_intervals(
        (
            max(0.0, float(chunk["start_sec"]) - coverage_tolerance_sec),
            min(duration, float(chunk["end_sec"]) + coverage_tolerance_sec),
        )
        for chunk in asr_chunks
        if chunk.get("start_sec") is not None and chunk.get("end_sec") is not None
    )
    uncovered_mask = [
        speech and not _covered_at((frame["start_sec"] + frame["end_sec"]) / 2.0, coverage)
        for frame, speech in zip(frames, speech_mask)
    ]
    spatial_mask = [
        speech and frame["spatial_ratio"] >= spatial_threshold
        for frame, speech in zip(frames, speech_mask)
    ]
    uncovered = _group_masked_frames(
        frames,
        uncovered_mask,
        pad_sec=0.5,
        gap_sec=0.5,
        min_sec=0.5,
        duration_sec=duration,
    )
    spatial = _group_masked_frames(
        frames,
        spatial_mask,
        pad_sec=1.0,
        gap_sec=0.75,
        min_sec=1.5,
        duration_sec=duration,
    )

    candidates: list[dict[str, Any]] = []
    for kind, intervals in (("uncovered_speech", uncovered), ("spatial_conflict", spatial)):
        for start, end in intervals:
            selected = [
                frame
                for frame in frames
                if frame["end_sec"] > start and frame["start_sec"] < end
            ]
            overlapping_asr = [
                chunk
                for chunk in asr_chunks
                if chunk.get("start_sec") is not None
                and chunk.get("end_sec") is not None
                and float(chunk["end_sec"]) > start
                and float(chunk["start_sec"]) < end
            ]
            candidates.append(
                {
                    "detector": kind,
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "duration_sec": round(end - start, 3),
                    "max_spatial_ratio": round(
                        max((frame["spatial_ratio"] for frame in selected), default=0.0), 6
                    ),
                    "mean_rms_db": (
                        round(float(np.mean([frame["rms_db"] for frame in selected])), 3)
                        if selected
                        else None
                    ),
                    "moss_overlap_chunk_count": sum(
                        bool(chunk.get("overlap")) for chunk in overlapping_asr
                    ),
                    "moss_chunk_count": len(overlapping_asr),
                    "recognized_content_chars": sum(
                        len(_content_text(str(chunk.get("text") or "")))
                        for chunk in overlapping_asr
                    ),
                }
            )
    uncovered_candidates = sorted(
        (item for item in candidates if item["detector"] == "uncovered_speech"),
        key=lambda item: (-item["duration_sec"], -item["max_spatial_ratio"], item["start_sec"]),
    )
    spatial_candidates = sorted(
        (item for item in candidates if item["detector"] == "spatial_conflict"),
        key=lambda item: (
            item["moss_overlap_chunk_count"] == 0,
            -item["moss_overlap_chunk_count"],
            -item["duration_sec"],
            -item["max_spatial_ratio"],
            item["start_sec"],
        ),
    )
    dual_signal = [item for item in spatial_candidates if item["moss_overlap_chunk_count"] > 0]
    selected = dual_signal[:max_windows]
    if len(selected) < max_windows:
        selected.extend(uncovered_candidates[:max_windows - len(selected)])
    if len(selected) < max_windows:
        selected.extend(
            item
            for item in spatial_candidates
            if item not in selected
        )
    return {
        "kind": "gt_free_multichannel_energy_spatial_audit",
        "audio": str(source),
        "sampling_rate": sampling_rate,
        "channels": channels,
        "duration_sec": duration,
        "frame_sec": frame_sec,
        "noise_floor_db": round(noise_floor, 3),
        "speech_threshold_db": round(speech_threshold, 3),
        "spatial_threshold": round(spatial_threshold, 6),
        "speech_frame_count": sum(speech_mask),
        "uncovered_speech_frame_count": sum(uncovered_mask),
        "spatial_conflict_frame_count": sum(spatial_mask),
        "selection_policy": "spatial_conflict_and_moss_overlap_then_uncovered_speech",
        "windows": selected[:max_windows],
    }


def normalized_distance(left: str, right: str) -> float:
    return edit_distance(left, right) / max(1, len(left), len(right))


def _content_text(text: str) -> str:
    return "".join(re.findall(r"[0-9A-Za-z\u4e00-\u9fff]", text or "")).lower()


def _dedup_text(text: str) -> str:
    return "".join(character for character in _content_text(text) if character not in "嗯呃啊哦")


def _neighbor_duplicate_ratio(span: str, neighbor_text: str) -> float:
    value = _dedup_text(span)
    neighbor = _dedup_text(neighbor_text)
    if not value or not neighbor:
        return 0.0
    match = SequenceMatcher(None, value, neighbor, autojunk=False).find_longest_match(
        0, len(value), 0, len(neighbor)
    )
    return match.size / len(value)


def consensus_medoid(hypotheses: dict[str, str]) -> str:
    """Select the hypothesis with minimum total distance, without a reference."""
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    return min(
        hypotheses,
        key=lambda name: (
            sum(normalized_distance(hypotheses[name], text) for text in hypotheses.values()),
            name != "baseline_anchor",
            name,
        ),
    )


def coverage_recovery_choice(
    hypotheses: dict[str, str],
    *,
    neighbor_text: str = "",
    minimum_added_chars: int = 6,
    minimum_support: int = 3,
    agreement_distance: float = 0.08,
    maximum_neighbor_duplicate_ratio: float = 0.60,
) -> dict[str, Any]:
    """Choose a coverage proposal from mutually supporting acoustic views.

    The policy is deliberately one-sided: it can nominate missing-content
    recovery, but never shortening or arbitrary substitution.  The returned
    choice remains a proposal until a separate temporal de-duplication gate
    confirms that the extra content is not already present nearby.
    """
    if "baseline_anchor" not in hypotheses:
        raise ValueError("baseline_anchor is required")
    medoid = consensus_medoid(hypotheses)
    anchor = _content_text(hypotheses["baseline_anchor"])
    candidate = _content_text(hypotheses[medoid])
    non_anchor = {
        name: _content_text(text)
        for name, text in hypotheses.items()
        if name != "baseline_anchor"
    }
    support = sum(
        normalized_distance(candidate, text) <= agreement_distance
        for text in non_anchor.values()
    )
    added_chars = len(candidate) - len(anchor)
    opcodes = SequenceMatcher(None, anchor, candidate, autojunk=False).get_opcodes()
    insertion_rows = []
    forbidden_mutations = []
    for tag, left_start, left_end, right_start, right_end in opcodes:
        if tag == "equal":
            continue
        if tag != "insert":
            forbidden_mutations.append(
                {
                    "operation": tag,
                    "source_text": anchor[left_start:left_end],
                    "candidate_text": candidate[right_start:right_end],
                }
            )
            continue
        span = candidate[right_start:right_end]
        duplicate_ratio = _neighbor_duplicate_ratio(span, neighbor_text)
        insertion_rows.append(
            {
                "anchor_position": left_start,
                "candidate_text": span,
                "neighbor_duplicate_ratio": round(duplicate_ratio, 6),
                "accepted": duplicate_ratio < maximum_neighbor_duplicate_ratio,
            }
        )
    accepted_insertions = [row for row in insertion_rows if row["accepted"]]
    propose = (
        medoid != "baseline_anchor"
        and added_chars >= minimum_added_chars
        and support >= minimum_support
        and not forbidden_mutations
        and bool(accepted_insertions)
    )
    proposed_text = anchor
    if propose:
        for row in sorted(accepted_insertions, key=lambda item: item["anchor_position"], reverse=True):
            position = int(row["anchor_position"])
            proposed_text = proposed_text[:position] + str(row["candidate_text"]) + proposed_text[position:]
    return {
        "selected_view": "coverage_recovery_proposal" if propose else "baseline_anchor",
        "proposed_text": proposed_text if propose else hypotheses["baseline_anchor"],
        "medoid_view": medoid,
        "added_content_chars": added_chars,
        "supporting_views": support,
        "minimum_added_chars": minimum_added_chars,
        "minimum_support": minimum_support,
        "agreement_distance": agreement_distance,
        "maximum_neighbor_duplicate_ratio": maximum_neighbor_duplicate_ratio,
        "insertion_candidates": insertion_rows,
        "forbidden_mutations": forbidden_mutations,
        "decision": "PROPOSE_COVERAGE_RECOVERY" if propose else "KEEP_BASELINE",
        "automatic_commit_allowed": False,
    }


def closed_set_differences(
    anchor: str, hypotheses: dict[str, str], *, min_support: int = 2
) -> list[dict[str, Any]]:
    """Return supported insert/delete/replace candidates; never apply them."""
    proposed: Counter[tuple[str, int, int, str, str]] = Counter()
    sources: dict[tuple[str, int, int, str, str], list[str]] = {}
    for name, text in hypotheses.items():
        if name == "baseline_anchor" or text == anchor:
            continue
        for tag, left_start, left_end, right_start, right_end in SequenceMatcher(
            None, anchor, text, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                continue
            key = (tag, left_start, left_end, anchor[left_start:left_end], text[right_start:right_end])
            proposed[key] += 1
            sources.setdefault(key, []).append(name)
    output = []
    for key, support in proposed.items():
        if support < min_support:
            continue
        tag, start, end, source, candidate = key
        output.append(
            {
                "operation": tag,
                "anchor_start": start,
                "anchor_end": end,
                "source_text": source,
                "candidate_text": candidate,
                "support": support,
                "sources": sorted(sources[key]),
                "automatic_commit_allowed": False,
            }
        )
    return sorted(output, key=lambda item: (item["anchor_start"], item["anchor_end"], item["operation"]))


def filter_semantic_replacement_candidates(
    candidates: Iterable[dict[str, Any]],
    *,
    minimum_support: int = 3,
    maximum_span_chars: int = 4,
) -> list[dict[str, Any]]:
    """Keep only bounded CJK substitutions suitable for a closed semantic gate.

    Insertions and deletions are excluded because semantics alone cannot prove
    that speech was present or absent.  The output remains non-authoritative:
    a later acoustic confirmation is required before applying a replacement.
    """
    output: list[dict[str, Any]] = []
    for raw in candidates:
        source = str(raw.get("source_text") or "")
        candidate = str(raw.get("candidate_text") or "")
        sources = sorted({str(item) for item in raw.get("sources") or [] if str(item)})
        if raw.get("operation") != "replace":
            continue
        if not (1 <= len(source) <= maximum_span_chars and 1 <= len(candidate) <= maximum_span_chars):
            continue
        if not re.fullmatch(r"[\u4e00-\u9fff]+", source) or not re.fullmatch(
            r"[\u4e00-\u9fff]+", candidate
        ):
            continue
        support = int(raw.get("support") or len(sources))
        if support < minimum_support or len(sources) < minimum_support:
            continue
        start, end = int(raw.get("anchor_start", -1)), int(raw.get("anchor_end", -1))
        if start < 0 or end <= start:
            continue
        item = dict(raw)
        item.update(
            {
                "candidate_id": f"replace:{start}:{end}:{source}:{candidate}",
                "support": support,
                "sources": sources,
                "automatic_commit_allowed": False,
            }
        )
        output.append(item)
    return sorted(output, key=lambda item: (item["anchor_start"], item["anchor_end"], item["candidate_id"]))


def apply_closed_set_replacements(
    anchor: str,
    candidates: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Apply verified replacements at their audited offsets.

    Offsets and source text must both match the immutable anchor.  Applying
    from right to left preserves all earlier offsets and prevents replacing a
    different occurrence of the same word.
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    occupied: list[tuple[int, int]] = []
    for candidate in sorted(candidates, key=lambda item: int(item.get("anchor_start", -1))):
        candidate_id = str(candidate.get("candidate_id") or "")
        start, end = int(candidate.get("anchor_start", -1)), int(candidate.get("anchor_end", -1))
        source = str(candidate.get("source_text") or "")
        replacement = str(candidate.get("candidate_text") or "")
        reason = ""
        if candidate.get("operation") != "replace" or start < 0 or end <= start or end > len(anchor):
            reason = "invalid_replacement"
        elif anchor[start:end] != source:
            reason = "source_mismatch"
        elif not replacement:
            reason = "empty_replacement"
        elif any(start < other_end and end > other_start for other_start, other_end in occupied):
            reason = "overlapping_candidate"
        if reason:
            rejected.append({"candidate_id": candidate_id, "reason": reason})
            continue
        accepted.append(dict(candidate))
        occupied.append((start, end))

    text = anchor
    for candidate in sorted(accepted, key=lambda item: int(item["anchor_start"]), reverse=True):
        start, end = int(candidate["anchor_start"]), int(candidate["anchor_end"])
        text = text[:start] + str(candidate["candidate_text"]) + text[end:]
    return {
        "text": text,
        "applied_candidate_ids": [str(item.get("candidate_id") or "") for item in accepted],
        "rejected": rejected,
    }


def strong_multiview_semantic_decision(
    candidate: dict[str, Any],
    candidates: Iterable[dict[str, Any]],
    semantic_gate: dict[str, Any],
    *,
    minimum_support: int = 3,
    minimum_semantic_confidence: float = 0.80,
    minimum_competitor_margin: int = 2,
) -> dict[str, Any]:
    """Require diverse acoustic support plus a decisive closed semantic choice."""
    sources = {str(item) for item in candidate.get("sources") or []}
    families = set()
    if any(source.startswith("beam_") for source in sources):
        families.add("beam")
    if any(re.fullmatch(r"channel_\d+", source) for source in sources):
        families.add("channel")
    if any(source in {"multichannel", "channel_average"} for source in sources):
        families.add("mixture")
    support = int(candidate.get("support") or len(sources))
    competitors = [
        item
        for item in candidates
        if item.get("candidate_id") != candidate.get("candidate_id")
        and item.get("anchor_start") == candidate.get("anchor_start")
        and item.get("anchor_end") == candidate.get("anchor_end")
        and item.get("source_text") == candidate.get("source_text")
    ]
    strongest_competitor = max((int(item.get("support") or 0) for item in competitors), default=0)
    support_margin = support - strongest_competitor if competitors else None
    reasons = []
    if not semantic_gate.get("selected"):
        reasons.append("semantic_gate_kept_baseline")
    if float(semantic_gate.get("confidence") or 0.0) < minimum_semantic_confidence:
        reasons.append("semantic_confidence_below_threshold")
    if support < minimum_support:
        reasons.append("insufficient_view_support")
    if not {"beam", "channel"}.issubset(families):
        reasons.append("insufficient_view_family_diversity")
    if support_margin is not None and support_margin < minimum_competitor_margin:
        reasons.append("insufficient_support_margin")
    commit = not reasons
    return {
        "decision": "COMMIT_REPLACEMENT" if commit else "KEEP_BASELINE",
        "reasons": reasons,
        "support": support,
        "minimum_support": minimum_support,
        "view_families": sorted(families),
        "required_view_families": ["beam", "channel"],
        "support_margin": support_margin,
        "minimum_competitor_margin": minimum_competitor_margin,
        "semantic_confidence": float(semantic_gate.get("confidence") or 0.0),
        "minimum_semantic_confidence": minimum_semantic_confidence,
        "prompted_moss_is_audit_only": True,
    }
