from asr_agent.context_judge import ContextJudgment, FocusProposal
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
