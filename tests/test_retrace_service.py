import json

from asr_agent.context_judge import BeliefProposal, ContextJudgment, FocusProposal
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


def test_text_context_alone_never_mutates_transcript(tmp_path):
    service = ReTraceService(tmp_path, context_judge=conflict_judge)
    service.process_turn("s", "t1", "图博士来了")

    result = service.process_turn("s", "t2", "负责人涂博士已经到了")

    assert result["revisions"] == []
    assert result["session"]["analysis_status"] == "deferred"
    assert result["session"]["turns"][0]["current_text"] == "图博士来了"
    assert result["session"]["open_hypotheses"]


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
