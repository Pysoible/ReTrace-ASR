from asr_agent.retrace import EntityProfile, ReTraceService
from asr_agent.uncertainty import detect_suspicious_spans


def test_detect_suspicious_span_combines_asr_signals():
    spans = detect_suspicious_spans(
        "请图博士审批合同",
        confidence={"图博士": 0.31},
        nbest=["请涂博士审批合同"],
    )

    assert spans[0].text == "图博士"
    assert {"low_confidence", "nbest_disagreement"} <= set(spans[0].reasons)


def test_high_entropy_hypothesis_waits_without_mutating_subtitle(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn(
        "s", "t1", "请图博士审批",
        confidence={"图博士": 0.2},
        text_candidates={"图博士": ["图博士", "涂博士", "屠博士"]},
    )

    state = service.get_session("s")["turns"][0]["hypotheses"][0]
    assert state["decision"] == "WAIT"
    assert len(state["candidates"]) == 3
    assert service.get_session("s")["turns"][0]["current_text"] == "请图博士审批"


def test_high_risk_hypothesis_requests_user_before_commit(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn(
        "s", "t1", "请图博士审批",
        confidence={"图博士": 0.2},
        text_candidates={"图博士": ["图博士", "涂博士"]},
        risk="high",
    )

    assert service.get_session("s")["turns"][0]["hypotheses"][0]["decision"] == "ASK_USER"


def test_reassessment_passes_only_later_raw_text_to_scorer(tmp_path):
    seen: list[str] = []
    service = ReTraceService(
        tmp_path,
        evidence_scorer=lambda **payload: seen.append(payload["evidence_text"]) or {"action": "DEFER"},
    )
    service.process_turn("s", "t1", "图博士来了", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    service.process_turn("s", "t2", "负责人到了")
    session = service._load("s")
    session.turns[1].current_text = "已修订的伪证据"
    service._save(session)

    service.process_turn("s", "t3", "实验室确认", use_llm=True)

    assert seen == ["负责人到了 实验室确认"]


def test_high_risk_hypothesis_never_auto_commits_from_later_evidence(tmp_path):
    service = ReTraceService(tmp_path)
    service.upsert_entities("s", [EntityProfile("lead", "涂博士", attributes={"role": "负责人"})])
    service.process_turn(
        "s", "t1", "图博士来了", risk="high",
        confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]}, entity_candidate_ids={"图博士": ["lead"]},
    )
    result = service.process_turn("s", "t2", "负责人到了")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["hypotheses"][0]["decision"] == "ASK_USER"


def test_high_risk_reflection_requires_confirmation_even_with_a_valid_quote(tmp_path):
    def reflector(**_: object) -> list[dict[str, object]]:
        return [{
            "target_turn_id": "t1", "before_text": "图博士", "after_text": "涂博士", "score": 0.91,
            "evidence": [{"turn_id": "t2", "quote": "负责人涂博士"}],
        }]

    service = ReTraceService(tmp_path, reflector=reflector)
    service.process_turn("s", "t1", "图博士来了", risk="high", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    result = service.process_turn("s", "t2", "负责人涂博士来了")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["hypotheses"][0]["decision"] == "ASK_USER"


def test_keep_original_appends_an_audit_event(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn("s", "t1", "图博士来了", risk="high", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})

    result = service.confirm_hypothesis("s", "t1", "图博士", "图博士", reason="operator checked audio")

    assert result["event"]["action"] == "KEEP_ORIGINAL"
    assert result["session"]["turns"][0]["current_text"] == "图博士来了"


def test_llm_revision_requires_a_verbatim_later_raw_quote(tmp_path):
    service = ReTraceService(
        tmp_path,
        evidence_scorer=lambda **_: {
            "action": "REVISE_TEXT", "candidate": "涂博士", "score": 0.95,
            "evidence": [{"turn_id": "t2", "quote": "不存在的引文"}],
        },
    )
    service.process_turn("s", "t1", "图博士来了", confidence={"图博士": 0.2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    result = service.process_turn("s", "t2", "负责人到了", use_llm=True)

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "图博士来了"


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
        return {"action": "REVISE_TEXT", "candidate": "涂博士", "entity_id": "lead", "score": 0.9, "evidence": [{"turn_id": "t2", "quote": "下午需要汇报"}, {"turn_id": "t3", "quote": "实验室负责人"}], "rationale": "two later turns"}

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
        {"action": "KEEP", "candidate": "图博士", "score": 0.6, "evidence": [], "rationale": "not enough"},
        {"action": "REVISE_TEXT", "candidate": "涂博士", "score": 0.92, "evidence": [{"turn_id": "t3", "quote": "负责人涂博士"}], "rationale": "later proof"},
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
