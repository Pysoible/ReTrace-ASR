from pathlib import Path

from retrace_v2.evidence import expand_audio_windows, generate_evidence_rows
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


def test_expand_audio_windows_clamps_to_recording_duration() -> None:
    row = raw_row("s1", Path("good.wav"))

    expanded = expand_audio_windows(
        [row],
        padding_sec=1.5,
        duration_lookup=lambda _: 2.25,
    )

    assert expanded[0]["start_sec"] == 0.0
    assert expanded[0]["end_sec"] == 2.25
    assert expanded[0]["core_start_sec"] == 1.0
    assert expanded[0]["core_end_sec"] == 2.0


def test_generate_evidence_can_label_an_independent_view() -> None:
    evidence, _, _ = generate_evidence_rows(
        [raw_row("s1", Path("good.wav"))],
        FakeAdapter(),
        view="expanded_0.8s",
    )

    assert evidence[0]["view"] == "expanded_0.8s"
