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
