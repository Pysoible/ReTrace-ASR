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


def test_entity_candidates_are_recalled_from_text_candidates_and_later_evidence_is_aggregated(tmp_path):
    evidence_seen: list[str] = []

    def scorer(**payload: object) -> dict[str, object]:
        evidence_seen.append(str(payload["evidence_text"]))
        return {"action": "REVISE_TEXT", "candidate": "涂博士", "entity_id": "lead", "score": 0.9, "evidence": ["负责人", "实验室"], "rationale": "two later turns"}

    service = ReTraceService(tmp_path, evidence_scorer=scorer)
    service.upsert_entities("s", [EntityProfile("lead", "涂博士", attributes={"role": "未出现属性"})])
    first = service.process_turn("s", "t1", "图博士来了", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    assert first["session"]["turns"][0]["hypotheses"][0]["entity_candidate_ids"] == ["lead"]
    service.process_turn("s", "t2", "下午需要汇报", use_llm=False)
    result = service.process_turn("s", "t3", "实验室负责人已经到了", use_llm=True)

    assert result["session"]["turns"][0]["current_text"] == "涂博士来了"
    assert "下午需要汇报" in evidence_seen[-1]
    assert "实验室负责人已经到了" in evidence_seen[-1]


def test_undo_replays_entity_memory_back_to_quarantine(tmp_path):
    service = ReTraceService(tmp_path)
    service.upsert_entities("s", [EntityProfile("lead", "涂博士", attributes={"role": "负责人"})])
    service.process_turn("s", "t1", "图博士来了", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    revised = service.process_turn("s", "t2", "负责人到了")
    event_id = revised["revisions"][0]["event_id"]

    undone = service.undo_revision("s", event_id)
    assert "lead" not in undone["session"]["verified_memory"]
    assert undone["session"]["quarantine_memory"]["lead"]["status"] == "candidate"


def test_temporary_keep_does_not_close_a_hypothesis_to_later_counterevidence(tmp_path):
    decisions = iter([
        {"action": "KEEP", "candidate": "图博士", "score": 0.6, "evidence": ["weak"], "rationale": "not enough"},
        {"action": "REVISE_TEXT", "candidate": "涂博士", "score": 0.92, "evidence": ["explicit role"], "rationale": "later proof"},
    ])
    service = ReTraceService(tmp_path, evidence_scorer=lambda **_: next(decisions))
    service.process_turn("s", "t1", "图博士来了", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    service.process_turn("s", "t2", "有人到了", use_llm=True)
    result = service.process_turn("s", "t3", "负责人涂博士到了", use_llm=True)

    assert result["session"]["turns"][0]["current_text"] == "涂博士来了"


def test_entity_profile_recalls_nearby_asr_name_when_qwen_omits_candidates(tmp_path):
    service = ReTraceService(tmp_path)
    service.upsert_entities("s", [EntityProfile("lead", "涂博士", attributes={"role": "负责人"})])
    result = service.process_turn("s", "t1", "图博士到了")

    hypothesis = result["session"]["turns"][0]["hypotheses"][0]
    assert hypothesis["span"] == "图博士"
    assert hypothesis["text_candidates"] == ["图博士", "涂博士"]
    assert hypothesis["entity_candidate_ids"] == ["lead"]
