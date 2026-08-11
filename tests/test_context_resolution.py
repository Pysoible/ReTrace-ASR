import pytest

from asr_agent.calibration import CalibrationPoint, DecisionPolicy, EvidenceFeatures, select_revision_threshold
from asr_agent.context_judge import BeliefProposal, ContextJudgment, FocusProposal, normalize_judgment
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


def test_context_judgment_carries_grounded_working_beliefs():
    session = Session("s", turns=[Turn("t1", "负责人是涂博士", "负责人是涂博士")])
    judgment = ContextJudgment(
        "NOVEL",
        0.9,
        beliefs=[BeliefProposal("实验室", "负责人", "涂博士", confidence=0.88, evidence_turn_ids=["t1"])],
    )

    normalized = normalize_judgment(judgment, session)

    assert normalized.beliefs[0].value == "涂博士"


def test_normalize_judgment_accepts_null_arrays_and_label_alias():
    """DeepSeek sometimes returns focus/beliefs=null and uses label instead of outcome."""
    session = Session("s", turns=[Turn("t1", "今天和涂博士讨论泰康方案", "今天和涂博士讨论泰康方案")])
    normalized = normalize_judgment(
        {
            "label": "NOVEL",
            "confidence": 0.8,
            "focus": None,
            "beliefs": [
                {
                    "subject": "user",
                    "predicate": "讨论",
                    "value": "泰康方案",
                    "aliases": None,
                    "confidence": 0.8,
                    "evidence_turn_ids": ["t1"],
                }
            ],
        },
        session,
    )

    assert normalized.outcome == "NOVEL"
    assert normalized.focus == []
    assert normalized.beliefs[0].aliases == []


def test_context_judgment_rejects_ungrounded_belief_evidence():
    session = Session("s", turns=[Turn("t1", "测试", "测试")])
    judgment = ContextJudgment(
        "NOVEL",
        beliefs=[BeliefProposal("甲", "状态", "完成", evidence_turn_ids=["missing"])],
    )

    with pytest.raises(ValueError, match="belief references"):
        normalize_judgment(judgment, session)


def test_resolver_preserves_coexisting_interpretations_without_audio():
    target = Turn("t1", "小王负责甲组", "小王负责甲组")
    source = Turn("t2", "另一个小王负责乙组", "另一个小王负责乙组")
    session = Session("s", turns=[target, source])
    focus = FocusProposal(
        "t1",
        "小王",
        "另一个小王",
        ["小王", "另一个小王"],
        ["t2"],
        relationship="COEXIST",
    )

    result = EvidenceResolver(audio_verifier=lambda **_: (_ for _ in ()).throw(AssertionError())).resolve(
        session, source, focus, context_confidence=0.9
    )

    assert result.action == "COEXIST"


def test_development_threshold_selection_prefers_recall_under_error_cap():
    selected = select_revision_threshold(
        [
            CalibrationPoint(0.95, True),
            CalibrationPoint(0.80, True),
            CalibrationPoint(0.70, True),
            CalibrationPoint(0.65, False),
            CalibrationPoint(0.20, False),
        ],
        beta=2.0,
        max_overcorrection=0.30,
    )

    assert selected.threshold == 0.70
    assert selected.recall == 1.0
    assert selected.source == "development"
