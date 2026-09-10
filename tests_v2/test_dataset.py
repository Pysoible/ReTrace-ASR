from pathlib import Path

import pytest

from retrace_v2.dataset import ReferenceSpan, assign_references, extract_raw_rows


def moss_payload() -> dict[str, object]:
    return {
        "asr": {"backend": "moss", "model": "MOSS-Transcribe-Diarize"},
        "session": {
            "turns": [
                {
                    "turn_id": "t2",
                    "raw_text": "后半",
                    "current_text": "被旧系统修改",
                    "meta": {"start_sec": 1.0, "end_sec": 2.0, "overlap": True},
                },
                {
                    "turn_id": "t1",
                    "raw_text": "前半",
                    "current_text": "前半",
                    "meta": {"start_sec": 0.0, "end_sec": 1.0},
                },
            ]
        },
    }


def test_extract_raw_rows_uses_immutable_raw_not_legacy_current_text() -> None:
    rows = extract_raw_rows(
        moss_payload(),
        recording_id="rec1",
        audio_path=Path("/data/rec1.wav"),
        baseline_model="MOSS-Transcribe-Diarize",
    )

    assert [row["text"] for row in rows] == ["前半", "后半"]
    assert rows[1]["overlap_probability"] == 1.0
    assert rows[1]["audio_path"] == "/data/rec1.wav"
    assert all("reference" not in key for row in rows for key in row)


def test_extract_raw_rows_rejects_omni_payload_for_moss_baseline() -> None:
    payload = moss_payload()
    payload["asr"] = {"backend": "moss", "model": "Qwen3_Omni_30B"}

    with pytest.raises(ValueError, match="identity"):
        extract_raw_rows(
            payload,
            recording_id="rec1",
            audio_path=Path("/data/rec1.wav"),
            baseline_model="MOSS-Transcribe-Diarize",
        )


def test_assign_references_uses_largest_time_overlap() -> None:
    raw_rows = extract_raw_rows(
        moss_payload(),
        recording_id="rec1",
        audio_path=Path("/data/rec1.wav"),
        baseline_model="MOSS-Transcribe-Diarize",
    )
    spans = [
        ReferenceSpan(0.2, 0.8, "a", "甲"),
        ReferenceSpan(0.9, 1.8, "b", "乙"),
    ]

    assigned = assign_references(raw_rows, spans)

    assert assigned == [
        {"segment_id": "rec1:t1", "reference_text": "甲"},
        {"segment_id": "rec1:t2", "reference_text": "乙"},
    ]
