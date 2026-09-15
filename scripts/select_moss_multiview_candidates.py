#!/usr/bin/env python3
"""GT-free closed-set semantic selection and prompted MOSS confirmation."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from asr_agent.integrations import deepseek, moss_asr
from asr_agent.multiview import (
    apply_closed_set_replacements,
    closed_set_differences,
    filter_semantic_replacement_candidates,
    strong_multiview_semantic_decision,
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


def _guided_decode(
    audio_path: Path,
    block: dict[str, Any],
    candidate: dict[str, Any],
    temporary_dir: Path,
) -> dict[str, Any]:
    crop_start = float(block["crop_start_sec"])
    crop_end = float(block["crop_end_sec"])
    with sf.SoundFile(audio_path) as source:
        sampling_rate = int(source.samplerate)
        source.seek(round(crop_start * sampling_rate))
        crop = source.read(
            round((crop_end - crop_start) * sampling_rate),
            dtype="float32",
            always_2d=True,
        )
    target = temporary_dir / f"guided-{str(candidate['candidate_id']).replace(':', '_')}.wav"
    sf.write(target, crop, sampling_rate, subtype="PCM_16")
    prompt = (
        "请忠实转写音频，不要补充音频中不存在的内容。"
        f"以下是待核对的闭集词形，仅在声学确实支持时采用：{candidate['source_text']} / "
        f"{candidate['candidate_text']}。"
    )
    started = time.perf_counter()
    decoded = moss_asr.transcribe_audio(str(target), prompt=prompt)
    elapsed = time.perf_counter() - started
    partial_with_timed_text = (
        decoded.get("failure_code") == "incomplete_first_pass"
        and bool(decoded.get("chunks"))
        and bool(decoded.get("chunks_text"))
    )
    if not decoded.get("ok") and not partial_with_timed_text:
        return {
            "ok": False,
            "elapsed_sec": round(elapsed, 6),
            "failure_code": decoded.get("failure_code"),
            "error": decoded.get("error"),
            "exact_candidate_supported": False,
        }
    relative_start = float(block["start_sec"]) - crop_start
    relative_end = float(block["end_sec"]) - crop_start
    guided_text = _text_in_window(decoded, relative_start, relative_end)
    differences = closed_set_differences(
        str(block["hypotheses"]["baseline_anchor"]),
        {
            "baseline_anchor": str(block["hypotheses"]["baseline_anchor"]),
            "guided_prompt": guided_text,
        },
        min_support=1,
    )
    exact = any(
        item["operation"] == "replace"
        and item["anchor_start"] == candidate["anchor_start"]
        and item["anchor_end"] == candidate["anchor_end"]
        and item["source_text"] == candidate["source_text"]
        and item["candidate_text"] == candidate["candidate_text"]
        for item in differences
    )
    return {
        "ok": True,
        "accepted_partial_with_timed_text": partial_with_timed_text,
        "elapsed_sec": round(elapsed, 6),
        "prompt_kind": "closed_set_contrastive_transcription",
        "guided_text": guided_text,
        "exact_candidate_supported": exact,
        "differences": differences,
        "model_identity": {
            "backend": decoded.get("backend"),
            "model": decoded.get("model"),
        },
        "completeness": decoded.get("completeness") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-support", type=int, default=3)
    parser.add_argument("--maximum-span-chars", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    blocks = []
    with tempfile.TemporaryDirectory(prefix="moss-closed-set-") as temporary:
        temporary_dir = Path(temporary)
        for block in artifact.get("blocks") or []:
            anchor = str((block.get("hypotheses") or {}).get("baseline_anchor") or "")
            candidates = filter_semantic_replacement_candidates(
                block.get("closed_set_candidates") or [],
                minimum_support=args.minimum_support,
                maximum_span_chars=args.maximum_span_chars,
            )
            audits = []
            verified = []
            for candidate in candidates:
                semantic = deepseek.select_closed_set_replacement(anchor, candidate)
                acoustic = None
                if semantic["eligible_for_acoustic_verification"]:
                    acoustic = _guided_decode(args.audio, block, candidate, temporary_dir)
                commit_policy = strong_multiview_semantic_decision(
                    candidate, candidates, semantic
                )
                if commit_policy["decision"] == "COMMIT_REPLACEMENT":
                    verified.append(candidate)
                audits.append(
                    {
                        "candidate": candidate,
                        "semantic_gate": semantic,
                        "prompted_acoustic_gate": acoustic,
                        "commit_policy": commit_policy,
                        "committed": commit_policy["decision"] == "COMMIT_REPLACEMENT",
                    }
                )
            applied = apply_closed_set_replacements(anchor, verified)
            blocks.append(
                {
                    "start_sec": block["start_sec"],
                    "end_sec": block["end_sec"],
                    "baseline_anchor": anchor,
                    "eligible_candidates": candidates,
                    "candidate_audits": audits,
                    "proposed_text": applied["text"],
                    "applied_candidate_ids": applied["applied_candidate_ids"],
                    "apply_rejections": applied["rejected"],
                }
            )

    output = {
        "schema": "moss_multiview_closed_set_v1",
        "purpose": "gt_free_closed_set_selection_and_acoustic_confirmation",
        "reference_was_read": False,
        "input_artifact": str(args.artifact.resolve()),
        "audio": str(args.audio.resolve()),
        "model_identities": {
            "asr": artifact.get("model_identity"),
            "semantic_gate": deepseek.context_judge_identity(),
        },
        "policy": {
            "operations": ["replace"],
            "minimum_support": args.minimum_support,
            "maximum_span_chars": args.maximum_span_chars,
            "prompted_moss_role": "audit_only_not_independent_veto",
            "requires_prompted_acoustic_confirmation": False,
            "exact_anchor_offsets_required": True,
        },
        "blocks": blocks,
        "elapsed_sec": round(time.perf_counter() - started, 6),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "eligible_candidates": sum(len(block["eligible_candidates"]) for block in blocks),
                "semantic_selected": sum(
                    audit["semantic_gate"]["selected"]
                    for block in blocks
                    for audit in block["candidate_audits"]
                ),
                "committed": sum(len(block["applied_candidate_ids"]) for block in blocks),
                "elapsed_sec": output["elapsed_sec"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
