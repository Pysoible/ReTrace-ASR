from dataclasses import FrozenInstanceError

import pytest

from retrace_v2.schemas import ModelRole, RawSegment, RunManifest


def test_raw_segment_is_immutable() -> None:
    row = RawSegment("s1", 0.0, 1.0, "spk0", "你好", "moss")

    with pytest.raises(FrozenInstanceError):
        row.text = "再见"  # type: ignore[misc]


def test_raw_segment_requires_forward_time() -> None:
    with pytest.raises(ValueError, match="after start"):
        RawSegment("s1", 1.0, 1.0, "spk0", "你好", "moss")


def test_manifest_rejects_same_baseline_and_evidence_identity() -> None:
    with pytest.raises(ValueError, match="independent"):
        RunManifest.create(
            run_id="r1",
            models={
                ModelRole.BASELINE_ASR: "MOSS",
                ModelRole.EVIDENCE_ASR: "moss",
            },
            rttm_source="predicted",
        )


def test_manifest_rejects_composite_model_identity() -> None:
    with pytest.raises(ValueError, match="one model"):
        RunManifest.create(
            run_id="r1",
            models={
                ModelRole.BASELINE_ASR: "MOSS/Qwen3_Omni_30B",
                ModelRole.EVIDENCE_ASR: "paraformer-zh",
            },
            rttm_source="none",
        )
