import json
from pathlib import Path

import pytest

from retrace_v2.artifacts import ArtifactWriter
from retrace_v2.scoring import score_recordings, score_rows


def scored_row(
    reference: str,
    raw: str,
    final: str,
    candidates: list[str],
) -> dict[str, object]:
    return {
        "segment_id": "s1",
        "reference_text": reference,
        "raw_text": raw,
        "final_text": final,
        "candidates": candidates,
    }


def test_inference_manifest_rejects_reference_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reference"):
        ArtifactWriter(tmp_path).write_manifest(
            {"run_id": "r1", "inputs": {"reference_dir": "/secret"}}
        )


def test_artifact_writer_replaces_temporary_file_atomically(tmp_path: Path) -> None:
    writer = ArtifactWriter(tmp_path)

    writer.write_manifest({"run_id": "r1", "models": {"baseline_asr": "moss"}})

    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["run_id"] == "r1"
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_oracle_candidate_exposes_generator_headroom() -> None:
    report = score_rows(
        [scored_row(reference="方案", raw="方按", final="方按", candidates=["方按", "方案"])]
    )

    assert report.raw.edits == 1
    assert report.final.edits == 1
    assert report.oracle_candidate.edits == 0


def test_pooled_cer_uses_total_errors_over_total_reference_chars() -> None:
    report = score_rows(
        [
            scored_row("甲乙", "甲", "甲", ["甲"]),
            scored_row("丙", "", "", [""]),
        ]
    )

    assert report.raw.cer == pytest.approx(2 / 3)
    assert report.raw.deletions == 2


def test_revision_precision_counts_only_committed_changes() -> None:
    report = score_rows(
        [
            scored_row("方案", "方按", "方案", ["方按", "方案"]),
            scored_row("项目", "项目", "项木", ["项目", "项木"]),
            scored_row("保持", "保持", "保持", ["保持"]),
        ]
    )

    assert report.committed_revisions == 2
    assert report.improving_revisions == 1
    assert report.harmful_revisions == 1
    assert report.revision_precision == pytest.approx(0.5)


def test_recording_score_uses_whole_recording_order_not_turn_error_sum() -> None:
    rows = [
        {
            **scored_row("乙", "甲", "甲", ["甲"]),
            "segment_id": "s1",
            "recording_id": "rec",
            "start_sec": 0.0,
        },
        {
            **scored_row("甲", "乙", "乙", ["乙"]),
            "segment_id": "s2",
            "recording_id": "rec",
            "start_sec": 0.5,
        },
    ]

    report = score_recordings(rows, {"rec": "甲乙"})

    assert score_rows(rows).raw.edits == 2
    assert report.raw.edits == 0
