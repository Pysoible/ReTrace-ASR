"""Dataset preparation helpers that preserve frozen first-pass evidence."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ReferenceSpan:
    start_sec: float
    end_sec: float
    speaker: str
    text: str


def parse_textgrid(path: Path) -> list[ReferenceSpan]:
    speaker = "unknown"
    start: float | None = None
    end: float | None = None
    spans: list[ReferenceSpan] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if match := re.fullmatch(r'name\s*=\s*"(.*)"', stripped):
            speaker = match.group(1) or "unknown"
        elif match := re.fullmatch(r"xmin\s*=\s*([0-9.]+)", stripped):
            start = float(match.group(1))
        elif match := re.fullmatch(r"xmax\s*=\s*([0-9.]+)", stripped):
            end = float(match.group(1))
        elif match := re.fullmatch(r'text\s*=\s*"(.*)"', stripped):
            text = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            if text and start is not None and end is not None and end > start:
                spans.append(ReferenceSpan(start, end, speaker, text))
            start = end = None
    return sorted(spans, key=lambda span: (span.start_sec, span.end_sec, span.speaker))


def extract_raw_rows(
    payload: Mapping[str, object],
    *,
    recording_id: str,
    audio_path: Path,
    baseline_model: str,
) -> list[dict[str, object]]:
    asr = payload.get("asr")
    asr_meta = asr if isinstance(asr, Mapping) else {}
    backend = str(asr_meta.get("backend") or "").casefold()
    payload_model = str(asr_meta.get("model") or "").casefold()
    expected = baseline_model.casefold()
    if "moss" in expected and (
        backend != "moss" or "moss" not in payload_model or "omni" in payload_model
    ):
        raise ValueError("payload model identity does not prove a MOSS baseline")

    session = payload.get("session")
    session_map = session if isinstance(session, Mapping) else {}
    turns = session_map.get("turns")
    if not isinstance(turns, list):
        raise ValueError("payload session.turns is required")

    output: list[dict[str, object]] = []
    for turn in turns:
        if not isinstance(turn, Mapping):
            continue
        meta = turn.get("meta")
        meta_map = meta if isinstance(meta, Mapping) else {}
        if meta_map.get("start_sec") is None or meta_map.get("end_sec") is None:
            continue
        start = float(meta_map["start_sec"])
        end = float(meta_map["end_sec"])
        if end <= start:
            continue
        turn_id = str(turn.get("turn_id") or len(output))
        overlap_probability = float(
            meta_map.get("overlap_probability")
            if meta_map.get("overlap_probability") is not None
            else 1.0 if meta_map.get("overlap") else 0.0
        )
        output.append(
            {
                "recording_id": recording_id,
                "segment_id": f"{recording_id}:{turn_id}",
                "region_id": f"{recording_id}:{turn_id}",
                "start_sec": start,
                "end_sec": end,
                "speaker": str(turn.get("speaker") or turn.get("source") or "unknown"),
                "text": str(turn.get("raw_text") or ""),
                "model_id": baseline_model,
                "audio_path": str(audio_path),
                "raw_acoustic_score": 0.0,
                "overlap_probability": overlap_probability,
                "triggers": [
                    "predicted_overlap" if overlap_probability > 0 else "evidence_probe"
                ],
            }
        )
    return sorted(output, key=lambda row: (float(row["start_sec"]), float(row["end_sec"])))


def assign_references(
    raw_rows: Sequence[Mapping[str, object]],
    spans: Sequence[ReferenceSpan],
) -> list[dict[str, str]]:
    assigned: list[list[ReferenceSpan]] = [[] for _ in raw_rows]
    for span in spans:
        overlaps = [
            max(
                0.0,
                min(float(row["end_sec"]), span.end_sec)
                - max(float(row["start_sec"]), span.start_sec),
            )
            for row in raw_rows
        ]
        if overlaps and max(overlaps) > 0:
            assigned[overlaps.index(max(overlaps))].append(span)
    return [
        {
            "segment_id": str(row["segment_id"]),
            "reference_text": "".join(
                span.text for span in sorted(values, key=lambda item: (item.start_sec, item.end_sec))
            ),
        }
        for row, values in zip(raw_rows, assigned)
    ]
