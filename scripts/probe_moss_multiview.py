#!/usr/bin/env python3
"""Create a GT-free MOSS multiview artifact for a few acoustic windows."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from asr_agent.integrations import moss_asr
from asr_agent.multiview import (
    closed_set_differences,
    circular_delay_and_sum,
    consensus_medoid,
    coverage_recovery_choice,
    detect_acoustic_windows,
)


def _text_in_window(asr: dict[str, Any], start: float, end: float) -> str:
    output = []
    for chunk, text in zip(asr.get("chunks") or [], asr.get("chunks_text") or []):
        if chunk.get("start_sec") is None or chunk.get("end_sec") is None:
            continue
        midpoint = (float(chunk["start_sec"]) + float(chunk["end_sec"])) / 2.0
        if start <= midpoint < end:
            output.append(str(text))
    return "".join(output)


def _decode(path: Path) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = moss_asr.transcribe_audio(str(path))
    elapsed = time.perf_counter() - started
    partial_with_text = (
        result.get("failure_code") == "incomplete_first_pass"
        and bool(result.get("text") or result.get("final_text"))
    )
    if not result.get("ok") and not partial_with_text:
        raise RuntimeError(f"MOSS decode failed for {path}: {result.get('failure_code') or result}")
    if partial_with_text:
        result.setdefault("completeness", {})["accepted_for_local_probe"] = True
    return result, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--baseline-raw-payload", type=Path)
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--halo-sec", type=float, default=8.0)
    parser.add_argument("--channel-views", type=int, default=2)
    parser.add_argument("--beam-views", type=int, default=4)
    parser.add_argument("--array-radius-m", type=float, default=0.05)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.baseline_cache and args.baseline_cache.exists():
        cached = json.loads(args.baseline_cache.read_text(encoding="utf-8"))
        baseline = cached.get("asr") if isinstance(cached.get("asr"), dict) else cached
        baseline_elapsed = 0.0
        baseline_cache_hit = True
    elif args.baseline_raw_payload and args.baseline_raw_payload.exists():
        payload = json.loads(args.baseline_raw_payload.read_text(encoding="utf-8"))
        raw = str(payload.get("text") or payload.get("transcript") or "")
        parsed = moss_asr.parse_moss_transcript(raw)
        with sf.SoundFile(args.audio) as source:
            audio_duration = len(source) / source.samplerate
        completeness = moss_asr.transcript_completeness(
            raw, parsed["chunks"], audio_duration
        )
        baseline = {
            "ok": True,
            "backend": moss_asr.BACKEND,
            "model": moss_asr.MOSS_TRANSCRIBE_DIARIZE_MODEL,
            "text": parsed["text"],
            "final_text": parsed["text"],
            "chunks_text": parsed["chunks_text"],
            "chunks": parsed["chunks"],
            "completeness": {
                **completeness,
                "accepted_for_probe": True,
                "acceptance_basis": "independent_acoustic_coverage_audit_pending",
            },
        }
        baseline_elapsed = 0.0
        baseline_cache_hit = False
        if args.baseline_cache:
            args.baseline_cache.parent.mkdir(parents=True, exist_ok=True)
            args.baseline_cache.write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    else:
        baseline, baseline_elapsed = _decode(args.audio)
        baseline_cache_hit = False
        if args.baseline_cache:
            args.baseline_cache.parent.mkdir(parents=True, exist_ok=True)
            args.baseline_cache.write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    detector_chunks = [
        {**chunk, "text": text}
        for chunk, text in zip(
            baseline.get("chunks") or [], baseline.get("chunks_text") or []
        )
    ]
    detector = detect_acoustic_windows(
        args.audio, detector_chunks, max_windows=args.max_windows
    )
    with sf.SoundFile(args.audio) as source:
        sampling_rate = int(source.samplerate)
        duration = len(source) / sampling_rate
    blocks = []
    with tempfile.TemporaryDirectory(prefix="moss-multiview-") as temporary:
        temporary_dir = Path(temporary)
        for index, window in enumerate(detector["windows"]):
            core_start, core_end = float(window["start_sec"]), float(window["end_sec"])
            crop_start = max(0.0, core_start - args.halo_sec)
            crop_end = min(duration, core_end + args.halo_sec)
            with sf.SoundFile(args.audio) as source:
                source.seek(round(crop_start * sampling_rate))
                crop = source.read(
                    round((crop_end - crop_start) * sampling_rate),
                    dtype="float32",
                    always_2d=True,
                )
            channel_energy = np.sqrt(np.mean(crop * crop, axis=0) + 1e-12)
            ranked_channels = list(np.argsort(channel_energy)[::-1][: max(0, args.channel_views)])
            variants: dict[str, np.ndarray] = {
                "multichannel": crop,
                "channel_average": crop.mean(axis=1),
                **{f"channel_{int(channel)}": crop[:, int(channel)] for channel in ranked_channels},
            }
            beam_angles = [
                360.0 * beam / args.beam_views
                for beam in range(max(0, args.beam_views))
            ]
            variants.update(
                {
                    f"beam_{int(angle):03d}": circular_delay_and_sum(
                        crop,
                        sampling_rate,
                        angle,
                        radius_m=args.array_radius_m,
                    )
                    for angle in beam_angles
                }
            )
            relative_start, relative_end = core_start - crop_start, core_end - crop_start
            hypotheses = {
                "baseline_anchor": _text_in_window(baseline, core_start, core_end)
            }
            anchor_indices = [
                position
                for position, chunk in enumerate(baseline.get("chunks") or [])
                if chunk.get("start_sec") is not None
                and chunk.get("end_sec") is not None
                and core_start
                <= (float(chunk["start_sec"]) + float(chunk["end_sec"])) / 2.0
                < core_end
            ]
            neighbor_text = "".join(
                str(text)
                for position, (chunk, text) in enumerate(
                    zip(baseline.get("chunks") or [], baseline.get("chunks_text") or [])
                )
                if position not in anchor_indices
                and chunk.get("start_sec") is not None
                and chunk.get("end_sec") is not None
                and float(chunk["end_sec"]) > crop_start
                and float(chunk["start_sec"]) < crop_end
            )
            decode_timings: dict[str, float] = {}
            for name, audio in variants.items():
                path = temporary_dir / f"{index:02d}-{name}.wav"
                sf.write(path, audio, sampling_rate, subtype="PCM_16")
                decoded, elapsed = _decode(path)
                hypotheses[name] = _text_in_window(decoded, relative_start, relative_end)
                decode_timings[name] = round(elapsed, 6)
            medoid = consensus_medoid(hypotheses)
            recovery_policy = coverage_recovery_choice(
                hypotheses, neighbor_text=neighbor_text
            )
            blocks.append(
                {
                    **window,
                    "crop_start_sec": round(crop_start, 3),
                    "crop_end_sec": round(crop_end, 3),
                    "halo_sec": args.halo_sec,
                    "ranked_channels": [int(channel) for channel in ranked_channels],
                    "beamforming": {
                        "kind": "exploratory_circular_delay_and_sum",
                        "azimuth_degrees": beam_angles,
                        "assumed_radius_m": args.array_radius_m,
                        "geometry_verified": False,
                    },
                    "hypotheses": hypotheses,
                    "baseline_anchor_indices": anchor_indices,
                    "neighbor_text": neighbor_text,
                    "consensus_medoid": medoid,
                    "coverage_recovery_policy": recovery_policy,
                    "closed_set_candidates": closed_set_differences(
                        hypotheses["baseline_anchor"], hypotheses
                    ),
                    "decode_timings_sec": decode_timings,
                }
            )

    artifact = {
        "schema": "moss_multiview_probe_v1",
        "purpose": "gt_free_candidate_generation_only",
        "reference_was_read": False,
        "audio": str(args.audio.resolve()),
        "model_identity": {
            "backend": moss_asr.BACKEND,
            "model": moss_asr.MOSS_TRANSCRIBE_DIARIZE_MODEL,
        },
        "baseline_cache_hit": baseline_cache_hit,
        "baseline_elapsed_sec": round(baseline_elapsed, 6),
        "baseline": {
            "text": baseline.get("text") or baseline.get("final_text") or "",
            "chunks_text": baseline.get("chunks_text") or [],
            "chunks": baseline.get("chunks") or [],
            "completeness": baseline.get("completeness") or {},
        },
        "detector": detector,
        "blocks": blocks,
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "windows": len(blocks),
        "candidates": sum(len(block["closed_set_candidates"]) for block in blocks),
        "elapsed_sec": artifact["elapsed_sec"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
