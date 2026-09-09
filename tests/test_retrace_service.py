import json

from asr_agent.context_judge import BeliefProposal, ContextJudgment, FocusProposal
from asr_agent.model_identity import ModelIdentity
from asr_agent.models import MemoryBelief
from asr_agent.retrace import ReTraceService


def conflict_judge(**_: object) -> ContextJudgment:
    return ContextJudgment(
        outcome="CONFLICT",
        confidence=0.94,
        rationale="后文给出了更明确的自称",
        focus=[
            FocusProposal(
                target_turn_id="t1",
                span="图博士",
                proposed_text="涂博士",
                alternatives=["图博士", "涂博士"],
                evidence_turn_ids=["t2"],
            )
        ],
    )


def test_observation_keeps_raw_text_immutable(tmp_path):
    service = ReTraceService(tmp_path)

    result = service.observe_turn("s", "t1", "图博士来了")

    turn = result["session"]["turns"][0]
    assert result["status"] == "queued"
    assert turn["raw_text"] == "图博士来了"
    assert turn["current_text"] == "图博士来了"


def test_agent_retranscribes_an_obviously_degenerate_current_turn(tmp_path):
    calls = []

    def retranscribe(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "text": "我们下周一开始进行项目验收"}

    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("CONSISTENT", 0.9),
        audio_retranscriber=retranscribe,
    )

    result = service.process_turn(
        "s",
        "t1",
        "啊啊啊啊啊啊啊啊",
        source="qwen-omni",
        meta={"audio_path": "/tmp/meeting.wav", "start_sec": 10.0, "end_sec": 17.0},
    )

    turn = result["session"]["turns"][0]
    revision = result["revisions"][0]
    assert turn["raw_text"] == "啊啊啊啊啊啊啊啊"
    assert turn["current_text"] == "我们下周一开始进行项目验收"
    assert revision["action"] == "REVISE_CURRENT"
    assert revision["resolver"] == "audio-degeneration-recovery"
    assert calls == [{"audio_path": "/tmp/meeting.wav", "start_sec": 10.0, "end_sec": 17.0}]


def test_degenerate_turn_does_not_enter_context_memory_when_rereading_fails(tmp_path):
    judge_called = False

    def judge(**_):
        nonlocal judge_called
        judge_called = True
        return ContextJudgment("NOVEL", 0.9)

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_retranscriber=lambda **_: {"ok": False, "error": "model unavailable"},
    )

    result = service.process_turn(
        "s",
        "t1",
        "项目项目项目项目项目项目",
        source="qwen-omni",
        meta={"audio_path": "/tmp/meeting.wav", "start_sec": 0.0, "end_sec": 8.0},
    )

    assert judge_called is False
    assert result["status"] == "deferred"
    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "项目项目项目项目项目项目"


def test_agent_can_revise_history_when_new_context_and_audio_agree(tmp_path):
    service = ReTraceService(
        tmp_path,
        context_judge=conflict_judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"图博士": 0.06, "涂博士": 0.94}},
    )
    service.process_turn(
        "s",
        "t1",
        "图博士来了",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0.0, "end_sec": 1.0},
    )

    result = service.process_turn("s", "t2", "负责人涂博士已经到了")

    first = result["session"]["turns"][0]
    assert first["raw_text"] == "图博士来了"
    assert first["current_text"] == "涂博士来了"
    assert result["revisions"][0]["action"] == "REVISE_HISTORY"
    assert result["revisions"][0]["source_turn_id"] == "t2"


def test_revision_cannot_translate_english_to_chinese(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.99,
            focus=[FocusProposal("t1", "industrial designer", "工业设计师", ["industrial designer", "工业设计师"], ["t2"])],
        )

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"industrial designer": 0.01, "工业设计师": 0.99}},
    )
    service.process_turn(
        "s",
        "t1",
        "My role is industrial designer",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 2},
    )

    result = service.process_turn("s", "t2", "That is her role")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "My role is industrial designer"


def test_revision_cannot_translate_chinese_to_english(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.99,
            focus=[FocusProposal("t1", "工业设计师", "industrial designer", ["工业设计师", "industrial designer"], ["t2"])],
        )

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"工业设计师": 0.01, "industrial designer": 0.99}},
    )
    service.process_turn(
        "s",
        "t1",
        "她是工业设计师",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 2},
    )

    result = service.process_turn("s", "t2", "She is responsible for the design")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "她是工业设计师"


def test_cross_language_candidates_are_rejected_before_audio_verification(tmp_path):
    verified = False

    def verify(**_):
        nonlocal verified
        verified = True
        return {"ok": True, "scores": {"工业设计师": 0.5, "industrial designer": 0.5}}

    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.99,
            focus=[FocusProposal("t1", "工业设计师", "工业设计师", ["工业设计师", "industrial designer"], ["t2"])],
        )

    service = ReTraceService(tmp_path, context_judge=judge, audio_verifier=verify)
    service.process_turn(
        "s",
        "t1",
        "她是工业设计师",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 2},
    )

    result = service.process_turn("s", "t2", "She is responsible for the design")

    assert result["revisions"] == []
    assert verified is False
    assert result["session"]["turns"][0]["current_text"] == "她是工业设计师"


def test_text_context_alone_never_mutates_transcript(tmp_path):
    service = ReTraceService(tmp_path, context_judge=conflict_judge)
    service.process_turn("s", "t1", "图博士来了")

    result = service.process_turn("s", "t2", "负责人涂博士已经到了")

    assert result["revisions"] == []
    assert result["session"]["analysis_status"] == "deferred"
    assert result["session"]["turns"][0]["current_text"] == "图博士来了"
    assert result["session"]["open_hypotheses"]


def test_near_variant_name_mismatch_can_be_revised_when_audio_is_clear_enough(tmp_path):
    def judge(*, current_turn, **_):
        if current_turn.turn_id == "t2":
            return ContextJudgment(
                "CONFLICT",
                0.92,
                focus=[FocusProposal("t1", "卡兹克", "卡兹个", ["卡兹克", "卡兹个"], ["t2"])],
            )
        return ContextJudgment("CONSISTENT", 0.8)

    def verify(*, candidates, **_):
        return {"ok": True, "scores": {"卡兹克": 0.06, "卡兹个": 0.94}}

    service = ReTraceService(tmp_path, context_judge=judge, audio_verifier=verify)
    service.process_turn(
        "s",
        "t1",
        "卡兹克来了",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0.0, "end_sec": 1.0},
    )

    result = service.process_turn("s", "t2", "卡兹个来了")

    assert result["revisions"][0]["action"] == "REVISE_HISTORY"
    assert result["session"]["turns"][0]["current_text"] == "卡兹个来了"


def test_strict_revision_defers_open_candidate_when_local_evidence_is_weak(tmp_path):
    def judge(*, current_turn, **_):
        if current_turn.turn_id == "t2":
            return ContextJudgment(
                "CONFLICT",
                0.99,
                focus=[FocusProposal("t1", "我们今天开会讨论", "他们昨晚开会讨论", ["我们今天开会讨论", "他们昨晚开会讨论"], ["t2"])],
            )
        return ContextJudgment("CONSISTENT", 0.8)

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"我们": 0.32, "他们": 0.68}},
    )
    service.process_turn(
        "s",
        "t1",
        "我们今天开会",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1},
    )

    result = service.process_turn("s", "t2", "继续讨论这个问题")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "我们今天开会"


def test_strict_revision_accepts_open_candidate_with_decisive_evidence(tmp_path):
    def judge(*, current_turn, **_):
        if current_turn.turn_id == "t2":
            return ContextJudgment(
                "CONFLICT",
                0.99,
                focus=[FocusProposal("t1", "我们", "他们", ["我们", "他们"], ["t2"])],
            )
        return ContextJudgment("CONSISTENT", 0.8)

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **kwargs: {"ok": True, "scores": {candidate: (0.95 if candidate == "他们" else 0.05) for candidate in kwargs["candidates"]}},
    )
    service.process_turn(
        "s",
        "t1",
        "我们今天开会讨论",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1},
    )

    result = service.process_turn("s", "t2", "他们负责记录")

    assert result["revisions"][0]["action"] == "REVISE_HISTORY"
    assert result["session"]["turns"][0]["current_text"] == "他们今天开会讨论"


def test_delete_focus_removes_an_audio_unsupported_span(tmp_path):
    def judge(**_):
        return {
            "outcome": "UNCERTAIN",
            "confidence": 0.99,
            "focus": [{
                "target_turn_id": "t1",
                "span": "清高的爱拉",
                "operation": "DELETE",
                "proposed_text": "",
                "alternatives": ["清高的爱拉"],
                "evidence_turn_ids": ["t1"],
            }],
        }

    def verify(*, candidates, **_):
        assert candidates == ["清高的爱拉", "[DELETE]"]
        return {"ok": True, "scores": {"清高的爱拉": 0.05, "[DELETE]": 0.95}}

    service = ReTraceService(tmp_path, context_judge=judge, audio_verifier=verify)
    result = service.process_turn(
        "s",
        "t1",
        "清高的爱拉是一个说法",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 2},
    )

    assert result["revisions"][0]["action"] == "REVISE_CURRENT"
    assert result["revisions"][0]["replacement"] == ""
    assert result["session"]["turns"][0]["current_text"] == "是一个说法"


def test_replace_focus_can_be_reclassified_as_delete_by_audio(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.99,
            focus=[FocusProposal("t1", "清高的爱拉", "清高的安拉", ["清高的爱拉", "清高的安拉"], ["t1"])],
        )

    def verify(*, candidates, **_):
        assert candidates == ["清高的爱拉", "清高的安拉", "[DELETE]"]
        return {
            "ok": True,
            "scores": {"清高的爱拉": 0.05, "清高的安拉": 0.05, "[DELETE]": 0.90},
        }

    service = ReTraceService(tmp_path, context_judge=judge, audio_verifier=verify)
    result = service.process_turn(
        "s",
        "t1",
        "清高的爱拉是一个说法",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 2},
    )

    assert result["revisions"][0]["replacement"] == ""
    assert result["session"]["turns"][0]["current_text"] == "是一个说法"


def test_replace_focus_retries_delete_pair_when_three_way_scores_are_invalid(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.99,
            focus=[FocusProposal("t1", "清高的爱拉", "清高的安拉", ["清高的爱拉", "清高的安拉"], ["t1"])],
        )

    def verify(*, candidates, **_):
        if candidates == ["清高的爱拉", "清高的安拉", "[DELETE]"]:
            return {"ok": True, "scores": {"清高的爱拉": 0.1, "清高的安拉": 0.9}}
        assert candidates == ["清高的爱拉", "[DELETE]"]
        return {"ok": True, "scores": {"清高的爱拉": 0.05, "[DELETE]": 0.95}}

    service = ReTraceService(tmp_path, context_judge=judge, audio_verifier=verify)
    result = service.process_turn(
        "s",
        "t1",
        "清高的爱拉是一个说法",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 2},
    )

    assert result["revisions"][0]["replacement"] == ""
    assert result["session"]["turns"][0]["current_text"] == "是一个说法"


def test_real_t063_delete_score_flows_through_service(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.99,
            focus=[FocusProposal("t1", "清高的爱拉", "清高的安拉", ["清高的爱拉", "清高的安拉"], ["t1"])],
        )

    def verify(*, candidates, **_):
        if candidates == ["清高的爱拉", "清高的安拉", "[DELETE]"]:
            return {"ok": True, "scores": {"清高的爱拉": 0.0, "[DELETE]": 1.0}}
        assert candidates == ["清高的爱拉", "[DELETE]"]
        return {"ok": True, "scores": {"清高的爱拉": 0.0, "[DELETE]": 1.0}}

    service = ReTraceService(tmp_path, context_judge=judge, audio_verifier=verify)
    result = service.process_turn(
        "s",
        "t1",
        "清高的爱拉他曾经说",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 1801.984, "end_sec": 1825.344},
    )

    assert result["revisions"][0]["action"] == "REVISE_CURRENT"
    assert result["revisions"][0]["replacement"] == ""
    assert result["session"]["turns"][0]["current_text"] == "他曾经说"


def test_empty_span_context_decision_is_classified_as_audit(tmp_path):
    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("CONSISTENT", 0.8),
    )
    result = service.process_turn("s", "t1", "正常文本")

    event = result["decisions"][0]
    assert event["action"] == "KEEP_OLD"
    assert event["span"] == ""
    assert event["event_kind"] == "audit"
    assert result["revisions"] == []


def test_audio_revision_rejects_unrelated_replacement_for_short_span(tmp_path):
    def judge(*, current_turn, **_):
        if current_turn.turn_id == "t2":
            return ContextJudgment(
                "CONFLICT",
                0.98,
                focus=[FocusProposal("t1", "是", "国济公", ["是", "国济公"], ["t2"])],
            )
        return ContextJudgment("CONSISTENT", 0.8)

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"是": 0.01, "国济公": 0.99}},
    )
    service.process_turn(
        "s",
        "t1",
        "是正确的",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1},
    )

    result = service.process_turn("s", "t2", "确认是国济公")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "是正确的"


def test_open_retranscription_without_span_alignment_is_deferred(tmp_path, monkeypatch):
    def judge(*, current_turn, **_):
        if current_turn.turn_id == "t2":
            return ContextJudgment(
                "CONFLICT",
                0.98,
                focus=[FocusProposal("t1", "图博士", "涂博士", ["图博士", "涂博士"], ["t2"])],
            )
        return ContextJudgment("CONSISTENT", 0.8)

    monkeypatch.setattr(
        "asr_agent.resolver.retranscribe_window",
        lambda *args, **kwargs: {"ok": True, "text": "其他内容涂博士其他内容"},
    )
    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": False},
    )
    service.process_turn(
        "s",
        "t1",
        "图博士来了",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1},
    )

    result = service.process_turn("s", "t2", "负责人涂博士已经到了")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "图博士来了"


def test_default_fallback_does_not_invent_proper_nouns(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn("s", "t1", "比如泰信商业保险，比如泰康的保险")

    result = service.process_turn("s", "t2", "继续讨论保险方案")

    assert result["revisions"] == []
    assert [turn["current_text"] for turn in result["session"]["turns"]] == [
        "比如泰信商业保险，比如泰康的保险",
        "继续讨论保险方案",
    ]


def test_duplicate_turn_id_is_rejected(tmp_path):
    service = ReTraceService(tmp_path)
    service.observe_turn("s", "t1", "第一次")

    try:
        service.observe_turn("s", "t1", "第二次")
    except ValueError as exc:
        assert "duplicate turn_id" in str(exc)
    else:
        raise AssertionError("duplicate turn_id must be rejected")


def test_later_evidence_can_automatically_rollback_a_bad_revision(tmp_path):
    def judge(*, current_turn, **_):
        if current_turn.turn_id == "t2":
            return ContextJudgment(
                "CONFLICT",
                0.95,
                focus=[FocusProposal("t1", "图博士", "涂博士", ["图博士", "涂博士"], ["t2"])],
            )
        if current_turn.turn_id == "t3":
            return ContextJudgment(
                "CONFLICT",
                0.97,
                focus=[FocusProposal("t1", "涂博士", "图博士", ["涂博士", "图博士"], ["t3"])],
            )
        return ContextJudgment("CONSISTENT", 0.8)

    calls = iter(["涂博士", "图博士"])

    def verify(*, candidates, **_):
        winner = next(calls)
        return {"ok": True, "scores": {item: (0.95 if item == winner else 0.05) for item in candidates}}

    service = ReTraceService(tmp_path, context_judge=judge, audio_verifier=verify)
    service.process_turn("s", "t1", "图博士来了", meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1})
    service.process_turn("s", "t2", "有人确认是涂博士")

    result = service.process_turn("s", "t3", "最新材料确认原来是图博士")

    assert result["revisions"][0]["action"] == "ROLLBACK"
    assert result["session"]["turns"][0]["current_text"] == "图博士来了"
    assert result["session"]["turns"][0]["raw_text"] == "图博士来了"


def test_novel_information_updates_short_term_memory_and_ledger(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "NOVEL",
            0.91,
            beliefs=[BeliefProposal("实验室", "负责人", "涂博士", confidence=0.89, evidence_turn_ids=["t1"])],
        )

    result = ReTraceService(tmp_path, context_judge=judge).process_turn("s", "t1", "实验室负责人是涂博士")

    beliefs = list(result["session"]["working_beliefs"].values())
    assert beliefs[0]["value"] == "涂博士"
    assert beliefs[0]["source_session_ids"] == ["s"]
    assert result["session"]["revision_events"][0]["action"] == "ACCEPT_NEW"
    assert result["session"]["turns"][0]["current_text"] == "实验室负责人是涂博士"


def test_coexist_decision_is_audited_without_rewriting_text(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.9,
            focus=[FocusProposal("t1", "小王", "另一个小王", ["小王", "另一个小王"], ["t2"], relationship="COEXIST")],
        )

    service = ReTraceService(tmp_path, context_judge=judge)
    service.process_turn("s", "t1", "小王负责甲组")
    result = service.process_turn("s", "t2", "另一个小王负责乙组")

    assert result["decisions"][0]["action"] == "COEXIST"
    assert result["session"]["turns"][0]["current_text"] == "小王负责甲组"


def test_same_observed_version_is_idempotent(tmp_path):
    service = ReTraceService(tmp_path)
    observed = service.observe_turn("s", "t1", "测试")
    first = service.analyze_turn("s", "t1", observed_version=observed["observed_version"])

    second = service.analyze_turn("s", "t1", observed_version=observed["observed_version"])

    assert second["session"]["version"] == first["session"]["version"]
    assert len(second["session"]["revision_events"]) == len(first["session"]["revision_events"])


def test_stale_analysis_rejudges_against_latest_session(tmp_path):
    seen_turn_counts: list[int] = []

    def judge(*, session, **_):
        seen_turn_counts.append(len(session.turns))
        return ContextJudgment("CONSISTENT", 0.8)

    service = ReTraceService(tmp_path, context_judge=judge)
    observed = service.observe_turn("s", "t1", "第一句")
    service.observe_turn("s", "t2", "第二句")

    result = service.analyze_turn("s", "t1", observed_version=observed["observed_version"])

    assert seen_turn_counts == [2]
    assert result["revalidated"] is True


def test_session_exposes_realtime_observability_snapshot(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "NOVEL",
            0.91,
            rationale="新信息补充了负责人",
            beliefs=[BeliefProposal("实验室", "负责人", "涂博士", confidence=0.89, evidence_turn_ids=["t1"])],
        )

    service = ReTraceService(tmp_path, context_judge=judge)
    service.long_term_memory.save("default", [
        MemoryBelief(
            belief_id="long-1",
            subject="实验室",
            predicate="负责人",
            value="涂博士",
            confidence=0.96,
            status="stable",
            source_turn_ids=["old-t1"],
            source_session_ids=["old-session"],
        ),
    ])

    service.process_turn("s", "t1", "实验室负责人是涂博士")
    state = service.get_session("s")
    trace = state["observability"]

    assert trace["latest_analysis"] == {
        "turn_id": "t1",
        "outcome": "NOVEL",
        "confidence": 0.91,
        "rationale": "新信息补充了负责人",
        "observed_version": 1,
        "analyzed_version": 2,
        "revalidated": False,
    }
    assert [turn["turn_id"] for turn in trace["short_term"]["recent_turns"]] == ["t1"]
    assert trace["short_term"]["working_beliefs"][0]["value"] == "涂博士"
    assert trace["long_term"]["beliefs"][0]["belief_id"] == "long-1"


def test_hypothesis_observability_keeps_focus_relationship(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.9,
            focus=[FocusProposal(
                "t1",
                "旧负责人",
                "新负责人",
                ["旧负责人", "新负责人"],
                ["t2"],
                relationship="TEMPORAL_CHANGE",
            )],
        )

    service = ReTraceService(tmp_path, context_judge=judge)
    service.process_turn("s", "t1", "旧负责人参加会议")
    service.process_turn("s", "t2", "新负责人今天接任")

    state = service.get_session("s")
    hypothesis = state["observability"]["short_term"]["open_hypotheses"][0]
    assert hypothesis["relationship"] == "TEMPORAL_CHANGE"


def test_service_restart_replays_projection_from_raw_and_ledger(tmp_path):
    service = ReTraceService(
        tmp_path,
        context_judge=conflict_judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"图博士": 0.05, "涂博士": 0.95}},
    )
    service.process_turn(
        "s",
        "t1",
        "图博士来了",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0, "end_sec": 1},
    )
    service.process_turn("s", "t2", "负责人涂博士到了")
    path = tmp_path / "s.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["turns"][0]["current_text"] = "被污染的投影"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    restored = ReTraceService(tmp_path).get_session("s")

    assert restored["turns"][0]["raw_text"] == "图博士来了"
    assert restored["turns"][0]["current_text"] == "涂博士来了"


def test_uncertain_relisten_is_rejected_when_hotword_biases_the_model(tmp_path):
    """A relisten that merely differs (led astray by a stale domain hotword like
    CS:GO) must NOT overwrite the first pass — it is the same ASR model and can
    produce an even worse transcript (e.g. "C S go连加二" for a LoL clip)."""
    def judge(**_):
        return ContextJudgment("UNCERTAIN", 0.5)

    retranscribe_calls = []

    def retranscribe(**kwargs):
        retranscribe_calls.append(kwargs)
        return {"ok": True, "text": "C S go连加二，那这个英雄跟那个跟螳螂还是有有渊源呐，是吧？"}

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_retranscriber=retranscribe,
    )

    result = service.process_turn(
        "s",
        "t1",
        "[0.0-7.0] 这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        source="qwen-omni",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0.0, "end_sec": 7.0},
    )

    turn = result["session"]["turns"][0]
    assert turn["raw_text"] == "[0.0-7.0] 这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？"
    assert turn["current_text"] == "[0.0-7.0] 这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？"
    assert not [r for r in result["revisions"] if r["resolver"] == "audio-uncertainty-relisten"]
    assert turn["meta"]["relisten_uncertain"]["rejected"]


def test_uncertain_relisten_does_not_replace_a_healthy_turn(tmp_path):
    """A remembered entity is evidence, not permission to overwrite a healthy turn."""
    def judge(*, current_turn, **_):
        if current_turn.turn_id == "t1":
            return ContextJudgment(
                "NOVEL",
                0.91,
                beliefs=[BeliefProposal("speaker", "mentions", "雷恩加尔", confidence=0.9, evidence_turn_ids=["t1"])],
            )
        return ContextJudgment("UNCERTAIN", 0.5)

    def retranscribe(**kwargs):
        return {"ok": True, "text": "对，这个雷恩加尔，那这个英雄跟那个螳螂还是很有渊源的，是吧？"}

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_retranscriber=retranscribe,
    )
    service.process_turn("s", "t1", "我们聊一下雷恩加尔这个英雄")

    result = service.process_turn(
        "s",
        "t2",
        "[0.0-7.0] 这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        source="qwen-omni",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0.0, "end_sec": 7.0},
    )

    turn = result["session"]["turns"][1]
    assert turn["current_text"] == "[0.0-7.0] 这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？"
    assert not [r for r in result["revisions"] if r["resolver"] == "audio-uncertainty-relisten"]
    assert turn["meta"]["relisten_uncertain"]["candidate_count"] >= 1


def test_healthy_turn_relisten_difference_uses_candidate_resolver(tmp_path):
    verifier_calls = []

    def verify(**kwargs):
        verifier_calls.append(kwargs)
        return {
            "ok": True,
            "scores": {item: (0.96 if item == "男装" else 0.03 if item == "南庄" else 0.01) for item in kwargs["candidates"]},
        }

    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("UNCERTAIN", 0.95),
        audio_retranscriber=lambda **_: {"ok": True, "text": "我负责男装部门"},
        audio_verifier=verify,
    )

    result = service.process_turn(
        "s", "t1", "我负责南庄部门", source="qwen-omni",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0.0, "end_sec": 2.0},
    )

    assert verifier_calls
    assert result["session"]["turns"][0]["current_text"] == "我负责男装部门"
    assert "candidate_source:relisten_open" in result["revisions"][0]["evidence"]


def test_candidate_outcome_is_persisted_as_structured_audit(tmp_path):
    audio = tmp_path / "turn.wav"
    audio.touch()
    service = ReTraceService(
        tmp_path / "state",
        context_judge=lambda **_: ContextJudgment(
            "CONFLICT", 0.95,
            focus=[FocusProposal("t1", "南庄", "男装", ["南庄", "男装"], ["t1"], source="semantic_open")],
        ),
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {item: (0.96 if item == "男装" else 0.03 if item == "南庄" else 0.01) for item in kwargs["candidates"]},
        },
    )

    result = service.process_turn("s", "t1", "我负责南庄部门", meta={"audio_path": str(audio), "start_sec": 0.0, "end_sec": 2.0})
    audit = next(event for event in result["session"]["revision_events"] if event["event_kind"] == "candidate_audit")

    assert audit["candidate_id"]
    assert audit["candidate_stage"] == "committed"
    assert "candidate_source:semantic_open" in audit["evidence"]
    assert "verifier_attempted:true" in audit["evidence"]
    assert any(item.startswith("audio_verified:") for item in audit["evidence"])


def test_moss_turn_cannot_call_qwen_open_relistener(tmp_path):
    called = False

    def qwen_relisten(**_):
        nonlocal called
        called = True
        return {"ok": True, "text": "男装"}

    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("UNCERTAIN", 0.95),
        audio_retranscriber=qwen_relisten,
        relistener_identity=ModelIdentity(
            "qwen-omni-vllm", "Qwen3_Omni_30B", "open_relistener"
        ),
    )

    result = service.process_turn(
        "s",
        "t1",
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

    turn_meta = result["session"]["turns"][0]["meta"]
    assert called is False
    assert turn_meta["relisten_uncertain"]["failure_code"] == "relistener_backend_mismatch"


def test_retrace_records_component_timings_and_call_counts(tmp_path):
    identity = ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "first_pass")
    verifier = ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "targeted_verifier")
    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment(
            "CONFLICT", 0.99,
            focus=[FocusProposal("t1", "南庄", "男装", ["南庄", "男装"], ["t1"])],
        ),
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {item: (0.99 if item == "男装" else 0.01) for item in kwargs["candidates"]},
        },
        verifier_identity=verifier,
    )

    result = service.process_turn(
        "s", "t1", "南庄", source="qwen-omni",
        meta={"audio_path": "/tmp/a.wav", "start_sec": 0, "end_sec": 1,
              "first_pass_identity": identity.as_dict()},
    )

    meta = result["session"]["turns"][0]["meta"]
    timings = meta["stage_timings_ms"]
    calls = meta["stage_call_counts"]
    for stage in ("context_judge", "candidate_build", "targeted_verifier", "open_relisten", "ledger_commit", "end_to_end"):
        assert timings[stage] >= 0
        assert calls[stage] >= 0
    assert calls["context_judge"] == 1
    assert calls["targeted_verifier"] == 1
    assert calls["open_relisten"] == 0


def test_coverage_risk_uses_segmented_relisten_even_when_judge_is_consistent(tmp_path):
    """A fluent but severely under-covered turn is an omission recovery task."""
    calls = []

    def retranscribe(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "text": "设置的呀我觉得这个游戏的人机设计是每一局规定数量的"}

    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("CONSISTENT", 0.92),
        audio_retranscriber=retranscribe,
    )
    result = service.process_turn(
        "s",
        "t1",
        "[0.0-12.0] 设置的呀。",
        source="qwen-omni",
        meta={
            "audio_path": "/tmp/fake.wav",
            "start_sec": 0.0,
            "end_sec": 12.0,
            "uncertainty": {"coverage": {"truncated": True, "speech_sec": 10.0, "char_density": 0.4}},
        },
    )

    turn = result["session"]["turns"][0]
    assert calls[0]["recover_coverage"] is True
    assert "人机设计" in turn["current_text"]
    assert turn["meta"]["relisten_uncertain"]["coverage_risk"] is True
    assert any("acoustic:coverage-risk" in item for item in result["revisions"][0]["evidence"])


def test_coverage_relisten_does_not_replace_turn_for_trivial_text_growth(tmp_path):
    def retranscribe(**_kwargs):
        return {"ok": True, "text": "今年年初的疫情呢"}

    service = ReTraceService(
        tmp_path,
        context_judge=lambda **_: ContextJudgment("CONSISTENT", 0.92),
        audio_retranscriber=retranscribe,
    )
    result = service.process_turn(
        "s",
        "t1",
        "今年年初的疫情",
        source="moss-transcribe-diarize",
        meta={
            "audio_path": "/tmp/fake.wav",
            "start_sec": 0.0,
            "end_sec": 8.0,
            "uncertainty": {"coverage": {"truncated": True, "char_density": 1.5}},
        },
    )

    turn = result["session"]["turns"][0]
    assert turn["current_text"] == "今年年初的疫情"
    assert not [event for event in result["revisions"] if event["event_kind"] == "revision"]
    assert turn["meta"]["relisten_uncertain"]["changed"] is False


def test_judge_tolerates_single_object_focus_from_model(tmp_path):
    """A model returning "focus": {...} (single dict) instead of a list must not
    fail the whole judgment — it should be treated as a one-element list."""
    from asr_agent.context_judge import normalize_judgment

    service = ReTraceService(tmp_path)
    service.process_turn("s", "t1", "图博士来了")
    service.process_turn("s", "t2", "负责人涂博士到了")

    raw = {
        "outcome": "CONFLICT",
        "confidence": 0.9,
        "rationale": "后文确认",
        "focus": {
            "target_turn_id": "t2",
            "span": "涂博士",
            "proposed_text": "图博士",
            "alternatives": ["涂博士", "图博士"],
            "evidence_turn_ids": ["t2"],
            "relationship": "MUTUALLY_EXCLUSIVE",
        },
        "beliefs": [],
    }
    session = service.repository.load("s")
    judgment = normalize_judgment(raw, session)
    assert judgment.outcome == "CONFLICT"
    assert len(judgment.focus) == 1
    assert judgment.focus[0].proposed_text == "图博士"


def test_session_summary_exposes_key_health_and_decision_metrics(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.9,
            focus=[FocusProposal("t1", "图博士", "涂博士", ["图博士", "涂博士"], ["t2"])],
        )

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"图博士": 0.05, "涂博士": 0.95}},
    )

    service.process_turn("s", "t1", "图博士来了")
    service.process_turn("s", "t2", "负责人涂博士到了")

    summary = service.summarize_session("s")

    assert summary["turn_count"] == 2
    assert summary["latest_outcome"] == "CONFLICT"
    assert summary["open_hypotheses"] == 1
    assert summary["revision_count"] >= 1
    assert summary["active_hypotheses"] >= 0
    assert summary["memory_scope"] == "default"
    assert summary["hypothesis_status_counts"]["resolved"] >= 0


def test_hypothesis_lifecycle_counts_distinguish_active_vs_resolved(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.9,
            focus=[FocusProposal("t1", "图博士", "涂博士", ["图博士", "涂博士"], ["t2"])],
        )

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"图博士": 0.05, "涂博士": 0.95}},
    )
    service.process_turn(
        "s",
        "t1",
        "图博士来了",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0.0, "end_sec": 1.0},
    )
    service.process_turn("s", "t2", "负责人涂博士到了")

    summary = service.summarize_session("s")
    state = service.get_session("s")

    assert summary["hypothesis_status_counts"]["resolved"] >= 1
    assert summary["hypothesis_status_counts"]["active"] == 0
    assert summary["hypothesis_status_counts"]["pending"] == 0
    assert state["observability"]["decision_state"]["status_counts"]["resolved"] >= 1


def test_session_tracks_formal_decision_state_lifecycle(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "NOVEL",
            0.91,
            beliefs=[BeliefProposal("实验室", "负责人", "涂博士", confidence=0.89, evidence_turn_ids=["t1"])],
        )

    service = ReTraceService(tmp_path, context_judge=judge)
    service.process_turn("s", "t1", "实验室负责人是涂博士")

    session = service.repository.load("s")
    assert session.decision_state.accepted_facts[0]["value"] == "涂博士"
    assert session.decision_state.pending_hypotheses == []
    assert session.decision_state.status_counts["resolved"] == 0
    assert session.decision_state.status_counts["active"] == 0


def test_observability_distinguishes_pending_hypotheses_from_accepted_facts(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "NOVEL",
            0.91,
            beliefs=[BeliefProposal("实验室", "负责人", "涂博士", confidence=0.89, evidence_turn_ids=["t1"])],
        )

    service = ReTraceService(tmp_path, context_judge=judge)
    service.process_turn("s", "t1", "实验室负责人是涂博士")

    state = service.get_session("s")
    decision_state = state["observability"]["decision_state"]

    assert decision_state["pending_hypotheses"] == []
    assert decision_state["accepted_facts"][0]["value"] == "涂博士"
    assert decision_state["accepted_facts"][0]["subject"] == "实验室"


def test_resolved_and_rejected_hypotheses_are_not_counted_as_open(tmp_path):
    def judge(**_):
        return ContextJudgment(
            "CONFLICT",
            0.9,
            focus=[FocusProposal("t1", "图博士", "涂博士", ["图博士", "涂博士"], ["t2"])],
        )

    service = ReTraceService(
        tmp_path,
        context_judge=judge,
        audio_verifier=lambda **_: {"ok": True, "scores": {"图博士": 0.05, "涂博士": 0.95}},
    )
    service.process_turn(
        "s",
        "t1",
        "图博士来了",
        meta={"audio_path": "/tmp/fake.wav", "start_sec": 0.0, "end_sec": 1.0},
    )
    result = service.process_turn("s", "t2", "负责人涂博士到了")

    summary = service.summarize_session("s")
    state = service.get_session("s")
    history = next(iter(state["open_hypotheses"].values()))

    assert summary["open_hypotheses"] == 0
    assert summary["active_hypotheses"] == 0
    assert state["observability"]["decision_state"]["pending_hypotheses"] == []
    assert history["status"] == "resolved"
    assert result["revisions"][0]["action"] == "REVISE_HISTORY"


def test_decision_engine_uses_a_single_evidence_gate_for_revision(tmp_path):
    from asr_agent.decision_engine import DecisionEngine, EvidenceBundle

    engine = DecisionEngine()
    assert engine.decide(EvidenceBundle(audio_confidence=0.9, audio_margin=0.25, context_confidence=0.9, memory_support=0.8, has_audio=True)) == "REVISE"
    assert engine.decide(EvidenceBundle(audio_confidence=0.4, audio_margin=0.1, context_confidence=0.9, memory_support=0.8, has_audio=True)) == "DEFER"
    assert engine.summarize(EvidenceBundle(audio_confidence=0.9, audio_margin=0.25, context_confidence=0.9, memory_support=0.8, has_audio=True))["decision"] == "REVISE"
