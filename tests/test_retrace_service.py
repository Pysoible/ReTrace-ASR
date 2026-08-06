from asr_agent.retrace import EntityProfile, ReTraceService


def test_future_evidence_revises_text_and_promotes_entity(tmp_path):
    service = ReTraceService(tmp_path)
    service.upsert_entities("s", [EntityProfile("lead", "涂博士", attributes={"role": "负责人", "org": "实验室"})])
    service.process_turn("s", "t1", "图博士让我交报告。", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]}, entity_candidate_ids={"图博士": ["lead"]})
    result = service.process_turn("s", "t2", "实验室负责人下午要汇报。")
    turn = result["session"]["turns"][0]
    assert turn["raw_text"] == "图博士让我交报告。"
    assert turn["current_text"] == "涂博士让我交报告。"
    assert result["revisions"][0]["action"] == "REVISE_TEXT"
    assert result["session"]["verified_memory"]["lead"]["name"] == "涂博士"


def test_tied_evidence_stays_in_quarantine(tmp_path):
    service = ReTraceService(tmp_path)
    service.upsert_entities("s", [EntityProfile("a", "涂博士", attributes={"role": "成员"}), EntityProfile("b", "涂博士", attributes={"role": "成员"})])
    service.process_turn("s", "t1", "涂博士发来材料。", confidence={"涂博士": 0.2}, entity_candidate_ids={"涂博士": ["a", "b"]})
    result = service.process_turn("s", "t2", "那位成员下午补充材料。")
    assert result["revisions"] == []
    assert result["session"]["turns"][0]["hypotheses"][0]["action"] == "DEFER"
    assert set(result["session"]["quarantine_memory"]) == {"a", "b"}


def test_revision_event_is_persisted_and_undo_restores_display_text(tmp_path):
    service = ReTraceService(tmp_path)
    service.upsert_entities("s", [EntityProfile("lead", "涂博士", attributes={"role": "负责人"})])
    service.process_turn(
        "s", "t1", "图博士让我交报告。", confidence={"图博士": 0.2},
        text_candidates={"图博士": ["图博士", "涂博士"]}, entity_candidate_ids={"图博士": ["lead"]},
    )
    revised = service.process_turn("s", "t2", "负责人下午要汇报。")

    event = revised["session"]["revision_events"][0]
    assert event["action"] == "REVISE_TEXT"
    assert event["active"] is True
    assert service.get_session("s")["revision_events"][0]["event_id"] == event["event_id"]

    undone = service.undo_revision("s", event["event_id"], reason="manual verification")
    assert undone["session"]["turns"][0]["raw_text"] == "图博士让我交报告。"
    assert undone["session"]["turns"][0]["current_text"] == "图博士让我交报告。"
    assert undone["session"]["revision_events"][-1]["action"] == "UNDO_REVISION"
    assert undone["session"]["revision_events"][-1]["reverted_event_id"] == event["event_id"]


def test_deepseek_evidence_result_is_constrained_to_known_candidates(tmp_path):
    def scorer(**_: object) -> dict[str, object]:
        return {"action": "REVISE_TEXT", "candidate": "杜博士", "entity_id": None, "score": 0.99, "evidence": ["后续提及负责人"], "rationale": "model choice"}

    service = ReTraceService(tmp_path, evidence_scorer=scorer)
    service.process_turn("s", "t1", "图博士来了", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    result = service.process_turn("s", "t2", "负责人到了", use_llm=True)

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "图博士来了"
