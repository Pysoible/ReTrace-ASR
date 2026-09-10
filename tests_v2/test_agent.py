from dataclasses import dataclass, field

from retrace_v2.agent import ReTraceAgent, ThresholdPolicy
from retrace_v2.schemas import EvidenceHypothesis, SuspiciousRegion
from retrace_v2.tools import ToolFailure


def hypothesis(
    text: str,
    *,
    score: float,
    view: str = "original",
) -> EvidenceHypothesis:
    return EvidenceHypothesis(
        hypothesis_id=f"{view}:{text}",
        model_id="paraformer-zh",
        view=view,
        text=text,
        start_sec=0.0,
        end_sec=1.0,
        acoustic_score=score,
    )


def region(raw: str, *, overlap: float = 0.0) -> SuspiciousRegion:
    return SuspiciousRegion(
        region_id="r1",
        segment_id="s1",
        start_sec=0.0,
        end_sec=1.0,
        raw_text=raw,
        raw_acoustic_score=0.5,
        overlap_probability=overlap,
        triggers=("asr_disagreement",),
    )


@dataclass
class FakeTools:
    original: tuple[EvidenceHypothesis, ...] = ()
    separated: tuple[EvidenceHypothesis, ...] = ()
    calls: list[str] = field(default_factory=list)

    def relisten(self, item: SuspiciousRegion) -> tuple[EvidenceHypothesis, ...]:
        self.calls.append("relisten")
        return self.original

    def separate(self, item: SuspiciousRegion) -> tuple[str, ...]:
        self.calls.append("separate")
        return ("channel-0", "channel-1")

    def relisten_separated(
        self,
        item: SuspiciousRegion,
        channels: tuple[str, ...],
    ) -> tuple[EvidenceHypothesis, ...]:
        self.calls.append("relisten_separated")
        return self.separated


class FailingTools(FakeTools):
    def relisten(self, item: SuspiciousRegion) -> tuple[EvidenceHypothesis, ...]:
        raise ToolFailure("evidence backend unavailable")


def test_agent_relistens_then_commits_supported_candidate() -> None:
    tools = FakeTools(original=(hypothesis("方案", score=0.92),))

    result = ReTraceAgent(ThresholdPolicy(commit_margin=0.2)).run(region("方按"), tools)

    assert [action.kind for action in result.actions] == [
        "RELISTEN",
        "COMPARE_EVIDENCE",
        "COMMIT",
    ]
    assert result.final_text == "方案"
    assert result.decision == "COMMIT"


def test_agent_uses_separation_after_unresolved_overlap() -> None:
    tools = FakeTools(
        original=(hypothesis("方按", score=0.51),),
        separated=(hypothesis("方案", score=0.95, view="separated:spk0"),),
    )

    result = ReTraceAgent(ThresholdPolicy(commit_margin=0.2)).run(
        region("方按", overlap=0.9), tools
    )

    assert [action.kind for action in result.actions] == [
        "RELISTEN",
        "COMPARE_EVIDENCE",
        "SEPARATE",
        "RELISTEN_SEPARATED",
        "COMPARE_EVIDENCE",
        "COMMIT",
    ]
    assert result.final_text == "方案"


def test_agent_keeps_raw_when_tool_fails() -> None:
    result = ReTraceAgent().run(region("不可改"), FailingTools())

    assert result.final_text == "不可改"
    assert result.decision == "KEEP"
    assert result.actions[-1].reason == "tool_failure"


def test_agent_does_not_separate_nonoverlap_without_speaker_conflict() -> None:
    tools = FakeTools(original=(hypothesis("别的文本", score=0.51),))

    result = ReTraceAgent(ThresholdPolicy(commit_margin=0.2)).run(region("原文"), tools)

    assert result.final_text == "原文"
    assert "SEPARATE" not in [action.kind for action in result.actions]
    assert result.decision == "KEEP"
