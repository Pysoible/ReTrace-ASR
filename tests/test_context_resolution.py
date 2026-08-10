import pytest

from asr_agent.calibration import DecisionPolicy, EvidenceFeatures
from asr_agent.context_judge import ContextJudgment, FocusProposal, normalize_judgment
from asr_agent.models import Session, Turn
from asr_agent.resolver import EvidenceResolver


def test_judgment_rejects_a_span_not_present_in_raw_turn():
    session = Session("s", turns=[Turn("t1", "图博士来了", "图博士来了")])
    judgment = ContextJudgment(
        outcome="CONFLICT",
        confidence=0.9,
        focus=[FocusProposal("t1", "不存在", "涂博士", ["不存在", "涂博士"])],
    )

    with pytest.raises(ValueError, match="span"):
        normalize_judgment(judgment, session)


def test_policy_requires_audio_and_a_margin_before_revision():
    policy = DecisionPolicy()
    no_audio = EvidenceFeatures(context_confidence=0.95, audio_confidence=0.0, memory_support=0.8, independent_sources=1)
    tied_audio = EvidenceFeatures(context_confidence=0.95, audio_confidence=0.55, audio_margin=0.01, memory_support=0.8, independent_sources=1)
    strong = EvidenceFeatures(context_confidence=0.95, audio_confidence=0.94, audio_margin=0.88, memory_support=0.8, independent_sources=1)

    assert policy.decide(no_audio, has_audio=False) == "DEFER"
    assert policy.decide(tied_audio, has_audio=True) == "DEFER"
    assert policy.decide(strong, has_audio=True) == "REVISE"


def test_resolver_accepts_only_the_proposed_closed_set_winner():
    target = Turn("t1", "图博士来了", "图博士来了", meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1})
    source = Turn("t2", "负责人涂博士到了", "负责人涂博士到了")
    session = Session("s", turns=[target, source])
    focus = FocusProposal("t1", "图博士", "涂博士", ["图博士", "涂博士"], ["t2"])
    resolver = EvidenceResolver(
        audio_verifier=lambda **_: {"ok": True, "scores": {"图博士": 0.1, "涂博士": 0.9}}
    )

    result = resolver.resolve(session, source, focus, context_confidence=0.95)

    assert result.action == "REVISE_HISTORY"
    assert result.replacement == "涂博士"
    assert result.audio_verified is True
