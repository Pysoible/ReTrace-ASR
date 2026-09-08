import pytest

from asr_agent.calibration import CalibrationPoint, DecisionPolicy, EvidenceFeatures, select_revision_threshold
from asr_agent.context_judge import BeliefProposal, ContextJudgment, FocusProposal, normalize_judgment
from asr_agent.models import Session, Turn
from asr_agent.model_identity import ModelIdentity
from asr_agent.resolver import EvidenceResolver


def test_read_asr_config_auto_loads_project_env(monkeypatch, tmp_path):
    repo = tmp_path / "retrace_repo"
    repo.mkdir()
    (repo / ".env").write_text("ASR_AUDIO_ENABLED=1\nASR_MODEL_PATH=/tmp/custom_qwen\nASR_GPU=3\n", encoding="utf-8")

    monkeypatch.chdir(repo)
    monkeypatch.delenv("ASR_AUDIO_ENABLED", raising=False)
    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)
    monkeypatch.delenv("ASR_GPU", raising=False)

    from asr_agent.integrations import qwen_asr

    config = qwen_asr.read_asr_config()

    assert config.enabled is True
    assert config.model_path == "/tmp/custom_qwen"
    assert config.gpu == "3"


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


def test_moss_turn_cannot_call_qwen_verifier(tmp_path):
    called = False

    def qwen_verify(**_):
        nonlocal called
        called = True
        return {"ok": True, "scores": {"南庄": 0.1, "男装": 0.9}}

    resolver = EvidenceResolver(
        audio_verifier=qwen_verify,
        verifier_identity=ModelIdentity(
            "qwen-omni-vllm", "Qwen3_Omni_30B", "targeted_verifier"
        ),
    )
    target = Turn(
        "t1",
        "南庄",
        "南庄",
        source="moss",
        meta={
            "audio_path": str(tmp_path / "a.wav"),
            "start_sec": 0,
            "end_sec": 1,
            "first_pass_identity": ModelIdentity(
                "moss-transcribe-diarize", "MOSS-Transcribe-Diarize", "first_pass"
            ).as_dict(),
        },
    )
    focus = FocusProposal(
        "t1", "南庄", "男装", ["南庄", "男装"], ["t1"], source="semantic_open"
    )

    result = resolver.resolve(
        Session("s", turns=[target]), target, focus, context_confidence=0.99
    )

    assert result.action == "DEFER"
    assert result.verifier_attempted is False
    assert result.failure_code == "verifier_backend_mismatch"
    assert called is False


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


def test_resolver_refuses_a_name_swap_when_both_words_are_already_in_the_turn():
    """Guard against "时候" → "狮子狗" false positives: when the proposed
    replacement already appears elsewhere in the same raw turn (and the span is
    a normal word), a closed-set verifier listening to the whole window is
    misled — refuse the swap rather than risk an absurd revision."""
    target = Turn("t1", "狮子狗刚出的的时候，它不就是就是刚进去", "狮子狗刚出的的时候，它不就是就是刚进去",
                  meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1})
    source = Turn("t2", "狮子狗刚出的的时候", "狮子狗刚出的的时候")
    session = Session("s", turns=[target, source])
    focus = FocusProposal("t1", "时候", "狮子狗", ["时候", "狮子狗"], ["t2"])

    resolver = EvidenceResolver(
        audio_verifier=lambda **_: {"ok": True, "scores": {"时候": 0.04, "狮子狗": 0.96}}
    )

    result = resolver.resolve(session, source, focus, context_confidence=0.9)

    assert result.action == "DEFER"
    assert "already appears elsewhere" in result.rationale


def test_resolver_still_revises_when_proposed_is_not_already_in_the_turn():
    """The guard must not block legitimate revisions where the proposed word is
    genuinely new (e.g. 图博士 → 涂博士 when 涂博士 never occurred before)."""
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


def test_resolver_does_not_treat_replacement_substring_as_a_separate_occurrence():
    target = Turn(
        "t1",
        "像像这些问题",
        "像像这些问题",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 2},
    )
    focus = FocusProposal("t1", "像像这些", "像这些", ["像像这些", "像这些"], ["t1"])
    calls = []

    def verify(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "scores": {
                candidate: 0.96 if candidate == "像这些" else 0.02
                for candidate in kwargs["candidates"]
            },
        }

    result = EvidenceResolver(audio_verifier=verify).resolve(
        Session("s", turns=[target]),
        target,
        focus,
        context_confidence=0.95,
    )

    assert calls
    assert result.action == "REVISE_CURRENT"
    assert result.replacement == "像这些"


def test_resolver_verifies_same_pronunciation_context_candidate():
    target = Turn(
        "t1",
        "搞下去，让他红红火火的。我这南庄那说两句啊呃，南庄啊那个。",
        "搞下去，让他红红火火的。我这南庄那说两句啊呃，南庄啊那个。",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 14},
    )
    session = Session("s", turns=[target, Turn("t2", "男装女装", "男装女装")])
    focus = FocusProposal("t1", "南庄", "男装", ["南庄", "男装"], ["t2"])

    resolver = EvidenceResolver(
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {candidate: (0.95 if candidate == "男装" else 0.05) for candidate in kwargs["candidates"]},
        }
    )

    result = resolver.resolve(session, target, focus, context_confidence=0.9)

    assert result.action == "REVISE_CURRENT"
    assert result.replacement == "男装"
    assert "context_homophone:南庄->男装" in result.evidence


def test_history_candidate_revision_changes_only_the_resolved_occurrence(tmp_path):
    from asr_agent.context_judge import ContextJudgment
    from asr_agent.retrace import ReTraceService

    audio_path = tmp_path / "turn.wav"
    audio_path.touch()
    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("CONFLICT", 0.9),
        audio_verifier=lambda **_: {"ok": True, "scores": {"南庄": 0.0, "男装": 1.0}},
    )
    service.process_turn("s", "t1", "男装女装商场", source="text")
    result = service.process_turn(
        "s", "t2", "我这南庄那说两句啊呃，南庄啊那个。", source="text",
        meta={"audio_path": str(audio_path), "start_sec": 0, "end_sec": 2},
    )

    assert result["session"]["turns"][1]["current_text"] == "我这男装那说两句啊呃，南庄啊那个。"
    revision = next(event for event in result["revisions"] if event["action"] == "REVISE_CURRENT")
    assert "candidate_source:history_homophone" in revision["evidence"]


def test_direct_context_revision_keeps_text_without_audio_evidence(tmp_path):
    from asr_agent.context_judge import ContextJudgment
    from asr_agent.retrace import ReTraceService

    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("CONFLICT", 0.99),
    )
    service.process_turn("s", "t1", "男装女装商场", source="text")
    result = service.process_turn("s", "t2", "我这南庄那说两句。", source="text")

    assert result["session"]["turns"][1]["current_text"] == "我这南庄那说两句。"


def test_direct_context_revision_keeps_text_for_ambiguous_audio(tmp_path):
    from asr_agent.context_judge import ContextJudgment
    from asr_agent.retrace import ReTraceService

    audio_path = tmp_path / "turn.wav"
    audio_path.touch()
    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("CONFLICT", 0.99),
        audio_verifier=lambda **_: {"ok": True, "scores": {"南庄": 0.52, "男装": 0.48}},
    )
    service.process_turn("s", "t1", "男装女装商场", source="text")
    result = service.process_turn(
        "s", "t2", "我这南庄那说两句。", source="text",
        meta={"audio_path": str(audio_path), "start_sec": 0, "end_sec": 2},
    )

    assert result["session"]["turns"][1]["current_text"] == "我这南庄那说两句。"


def test_uncertain_acoustic_signal_can_commit_unique_context_homophone(tmp_path):
    from asr_agent.context_judge import ContextJudgment
    from asr_agent.retrace import ReTraceService

    audio_path = tmp_path / "turn.wav"
    audio_path.touch()
    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("UNCERTAIN", 0.9),
        audio_verifier=lambda **_: {"ok": True, "scores": {"南庄": 0.0, "男装": 1.0}},
    )
    service.process_turn("s", "t1", "男装女装商场", source="text")
    result = service.process_turn(
        "s", "t2", "我这南庄那说两句。", source="text",
        meta={
            "audio_path": str(audio_path),
            "start_sec": 0,
            "end_sec": 2,
            "uncertainty": {"acoustic_disagreement": [{"span_a": "南庄", "span_b": "男装"}]},
        },
    )

    assert result["session"]["turns"][1]["current_text"] == "我这男装那说两句。"


def test_semantic_open_candidate_with_decisive_audio_does_not_require_acoustic_disagreement(tmp_path):
    from asr_agent.context_judge import FocusProposal
    from asr_agent.resolver import EvidenceResolver
    from asr_agent.models import Session, Turn

    audio_path = tmp_path / "turn.wav"
    audio_path.touch()
    target = Turn(
        "t1",
        "这个绿植被品包括。",
        "这个绿植被品包括。",
        meta={"audio_path": str(audio_path), "start_sec": 0, "end_sec": 2},
    )
    focus = FocusProposal(
        target_turn_id="t1",
        span="绿植被品",
        proposed_text="绿植品种",
        alternatives=["绿植被品", "绿植品种"],
        evidence_turn_ids=["t1"],
        source="semantic_open",
    )
    resolver = EvidenceResolver(
        audio_verifier=lambda **_: {"ok": True, "scores": {"绿植被品": 0.05, "绿植品种": 0.95, "[DELETE]": 0.0}},
    )

    result = resolver.resolve(
        Session("s", turns=[target]),
        target,
        focus,
        context_confidence=0.95,
    )

    assert result.action == "REVISE_CURRENT"
    assert result.replacement == "绿植品种"


def test_semantic_open_candidate_commits_end_to_end_without_history_or_paraformer(tmp_path):
    from asr_agent.context_judge import ContextJudgment, FocusProposal
    from asr_agent.retrace import ReTraceService

    audio_path = tmp_path / "turn.wav"
    audio_path.touch()
    judgment = ContextJudgment(
        "CONFLICT",
        0.95,
        focus=[FocusProposal(
            "t1", "南庄", "男装", ["南庄", "男装"], ["t1"],
            rationale="当前正在讨论服装部门", source="semantic_open",
        )],
    )
    service = ReTraceService(
        tmp_path / "state",
        context_judge=lambda **_: judgment,
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {item: (0.95 if item == "男装" else 0.03 if item == "南庄" else 0.02) for item in kwargs["candidates"]},
        },
    )

    result = service.process_turn(
        "s", "t1", "我负责南庄部门",
        meta={"audio_path": str(audio_path), "start_sec": 0.0, "end_sec": 2.0},
    )

    assert result["session"]["turns"][0]["current_text"] == "我负责男装部门"
    assert "candidate_source:semantic_open" in result["revisions"][0]["evidence"]


def test_acoustic_focus_keeps_current_turn_as_evidence():
    from asr_agent.retrace import ReTraceService

    turn = Turn(
        "t1",
        "我这南庄那说两句。",
        "我这南庄那说两句。",
        meta={"uncertainty": {"acoustic_disagreement": [{"span_a": "南庄", "span_b": "男装"}]}},
    )

    focus = ReTraceService._acoustic_focus(turn)

    assert focus[0].target_turn_id == "t1"
    assert focus[0].evidence_turn_ids == ["t1"]


def test_acoustic_support_is_one_when_span_falls_in_disagreement_region():
    """A focused span inside an acoustic-disagreement region (where two ASRs
    disagreed) gets acoustic_support=1.0 — the audio is genuinely ambiguous
    there, so the revision is better grounded."""
    target = Turn(
        "t1",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        meta={
            "uncertainty": {
                "acoustic_disagreement": [
                    {"tag": "replace", "span_a": "加额，", "span_b": "家的", "offset_a": 2, "offset_b": 2},
                ]
            }
        },
    )

    assert EvidenceResolver._acoustic_support(target, "加额") == 1.0
    assert EvidenceResolver._acoustic_support(target, "连加额") == 1.0


def test_acoustic_support_is_zero_when_span_outside_disagreement():
    """A span outside the disagreement region has acoustic_support=0.0."""
    target = Turn(
        "t1",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        meta={
            "uncertainty": {
                "acoustic_disagreement": [
                    {"tag": "replace", "span_a": "加额，", "span_b": "家的", "offset_a": 2, "offset_b": 2},
                ]
            }
        },
    )

    assert EvidenceResolver._acoustic_support(target, "螳螂") == 0.0
    assert EvidenceResolver._acoustic_support(target, "英雄") == 0.0


def test_acoustic_support_is_zero_without_disagreement_meta():
    """No acoustic_disagreement meta → acoustic_support=0.0 (graceful)."""
    target = Turn("t1", "这个连加额，那这个英雄", "这个连加额，那这个英雄")

    assert EvidenceResolver._acoustic_support(target, "连加额") == 0.0


def test_acoustic_support_flows_into_revision_evidence():
    """When the span is acoustically supported, the revision evidence records
    the acoustic_disagreement marker."""
    target = Turn(
        "t1",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        meta={
            "audio_path": "/tmp/fake.wav",
            "start_sec": 0.0,
            "end_sec": 7.0,
            "uncertainty": {
                "acoustic_disagreement": [
                    {"tag": "replace", "span_a": "加额，", "span_b": "家的", "offset_a": 2, "offset_b": 2},
                ]
            },
        },
    )
    source = Turn("t2", "这个雷恩加尔", "这个雷恩加尔")
    session = Session("s", turns=[target, source])
    focus = FocusProposal("t1", "连加额", "雷恩加尔", ["连加额", "雷恩加尔"], ["t2"])

    resolver = EvidenceResolver(
        audio_verifier=lambda **_: {"ok": True, "scores": {"连加额": 0.05, "雷恩加尔": 0.95}}
    )

    result = resolver.resolve(session, source, focus, context_confidence=0.9)

    assert result.action == "REVISE_HISTORY"
    assert any("acoustic_disagreement" in item for item in result.evidence)
