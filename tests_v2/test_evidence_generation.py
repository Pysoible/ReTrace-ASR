from pathlib import Path

from retrace_v2.evidence import generate_evidence_rows
from retrace_v2.schemas import EvidenceHypothesis
from retrace_v2.tools import ToolFailure


class FakeAdapter:
    model_id = "paraformer-zh"

    def transcribe(self, audio_path: Path, start_sec: float, end_sec: float):
        if "bad" in audio_path.name:
            raise ToolFailure("unreadable")
        return (
            EvidenceHypothesis(
                hypothesis_id="h1",
                model_id=self.model_id,
                view="original",
                text="方案",
                start_sec=start_sec,
                end_sec=end_sec,
                acoustic_score=0.9,
            ),
        )


def raw_row(segment_id: str, audio: Path) -> dict[str, object]:
    return {
        "segment_id": segment_id,
        "audio_path": str(audio),
        "start_sec": 1.0,
        "end_sec": 2.0,
    }


def test_generate_evidence_records_identity_and_segment() -> None:
    evidence, failures, timings = generate_evidence_rows(
        [raw_row("s1", Path("good.wav"))], FakeAdapter()
    )

    assert failures == []
    assert evidence[0]["segment_id"] == "s1"
    assert evidence[0]["model_id"] == "paraformer-zh"
    assert evidence[0]["text"] == "方案"
    assert timings[0]["status"] == "ok"


def test_generate_evidence_audits_failure_without_fabricating_text() -> None:
    evidence, failures, timings = generate_evidence_rows(
        [raw_row("s1", Path("bad.wav"))], FakeAdapter()
    )

    assert evidence == []
    assert failures == [{"segment_id": "s1", "error": "unreadable"}]
    assert timings[0]["status"] == "failed"
