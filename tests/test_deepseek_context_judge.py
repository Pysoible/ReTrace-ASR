from asr_agent.integrations import deepseek
from asr_agent.memory import MemoryPacket
from asr_agent.models import Session, Turn


def test_deepseek_context_judge_returns_validated_focus(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        deepseek,
        "_chat_json",
        lambda *_, **__: {
            "outcome": "CONFLICT",
            "confidence": 0.93,
            "rationale": "后文自称更明确",
            "beliefs": [{
                "subject": "实验室",
                "predicate": "负责人",
                "value": "涂博士",
                "confidence": 0.9,
                "evidence_turn_ids": ["t2"],
            }],
            "focus": [{
                "target_turn_id": "t1",
                "span": "图博士",
                "proposed_text": "涂博士",
                "alternatives": ["图博士", "涂博士"],
                "evidence_turn_ids": ["t2"],
                "relationship": "MUTUALLY_EXCLUSIVE",
            }],
        },
    )
    session = Session("s", turns=[Turn("t1", "图博士来了", "图博士来了"), Turn("t2", "我是涂博士", "我是涂博士")])

    result = deepseek.judge_context(session=session, current_turn=session.turns[-1], memory=MemoryPacket())

    assert result.outcome == "CONFLICT"
    assert result.focus[0].proposed_text == "涂博士"
    assert result.beliefs[0].subject == "实验室"


def test_conflict_with_one_homophone_candidate_creates_focus(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {"outcome": "CONFLICT", "confidence": 0.9})
    monkeypatch.setattr(
        deepseek,
        "_homophone_candidates",
        lambda *_: [{"span": "南庄", "candidate": "男装", "evidence_turn_ids": ["t0"]}],
    )
    turn = Turn("t1", "我这南庄那说两句", "我这南庄那说两句")
    result = deepseek.judge_context(
        session=Session("s", turns=[Turn("t0", "男装女装", "男装女装"), turn]),
        current_turn=turn,
        memory=MemoryPacket(),
    )

    assert result.outcome == "CONFLICT"
    assert result.focus[0].span == "南庄"
    assert result.focus[0].proposed_text == "男装"
    assert result.focus[0].evidence_turn_ids == ["t0"]
    assert result.focus[0].source == "history_homophone"


def test_homophone_candidates_keep_turn_boundaries_and_source_evidence():
    current = Turn("t3", "我这南庄那说两句", "我这南庄那说两句")
    session = Session(
        "s",
        turns=[
            Turn("t1", "男", "男"),
            Turn("t2", "装", "装"),
            Turn("t3", current.raw_text, current.current_text),
        ],
    )

    candidates = deepseek._homophone_candidates(session, current)

    assert not any(item["candidate"] == "男装" for item in candidates)

    session.turns[0].current_text = "男装女装"
    candidates = deepseek._homophone_candidates(session, current)

    match = next(item for item in candidates if item["span"] == "南庄" and item["candidate"] == "男装")
    assert match["evidence_turn_ids"] == ["t1"]


def test_judge_accepts_semantic_open_candidate_not_found_in_history(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        deepseek,
        "_chat_json",
        lambda *_, **__: {
            "outcome": "CONFLICT",
            "confidence": 0.9,
            "candidates": [{
                "target_turn_id": "t1",
                "span": "南庄",
                "candidate": "男装",
                "source": "semantic_open",
                "evidence_turn_ids": ["t1"],
                "rationale": "当前正在讨论服装部门",
            }],
        },
    )
    turn = Turn("t1", "我负责南庄部门", "我负责南庄部门")

    result = deepseek.judge_context(
        session=Session("s", turns=[turn]),
        current_turn=turn,
        memory=MemoryPacket(),
    )

    assert result.focus[0].span == "南庄"
    assert result.focus[0].proposed_text == "男装"
    assert result.focus[0].source == "semantic_open"


def test_judge_accepts_top_level_delete_candidate(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(
        deepseek,
        "_chat_json",
        lambda *_, **__: {
            "outcome": "CONFLICT",
            "confidence": 0.95,
            "candidates": [{
                "target_turn_id": "t1",
                "span": "啊啊",
                "candidate": "",
                "operation": "DELETE",
                "source": "semantic_open",
                "evidence_turn_ids": ["t1"],
                "rationale": "局部音频确认该片段是幻觉",
            }],
        },
    )
    turn = Turn("t1", "开始啊啊继续", "开始啊啊继续")

    result = deepseek.judge_context(
        session=Session("s", turns=[turn]),
        current_turn=turn,
        memory=MemoryPacket(),
    )

    assert result.focus[0].operation == "DELETE"
    assert result.focus[0].span == "啊啊"
    assert result.focus[0].proposed_text == ""


def test_focus_candidate_absent_from_history_is_marked_semantic_open(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {
        "outcome": "CONFLICT",
        "confidence": 0.9,
        "focus": [{
            "target_turn_id": "t1",
            "span": "南庄",
            "proposed_text": "男装",
            "alternatives": ["南庄", "男装"],
            "evidence_turn_ids": ["t1"],
        }],
    })
    turn = Turn("t1", "我负责南庄部门", "我负责南庄部门")

    result = deepseek.judge_context(session=Session("s", turns=[turn]), current_turn=turn, memory=MemoryPacket())

    assert result.focus[0].source == "semantic_open"


def test_focus_and_candidate_pool_are_deduplicated(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    proposal = {
        "target_turn_id": "t1", "span": "南庄", "proposed_text": "男装",
        "alternatives": ["南庄", "男装"], "evidence_turn_ids": ["t1"],
    }
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {
        "outcome": "CONFLICT", "confidence": 0.9,
        "focus": [proposal], "candidates": [{**proposal, "candidate": "男装"}],
    })
    turn = Turn("t1", "我负责南庄部门", "我负责南庄部门")

    result = deepseek.judge_context(session=Session("s", turns=[turn]), current_turn=turn, memory=MemoryPacket())

    assert len(result.focus) == 1
    assert result.focus[0].source == "semantic_open"


def test_context_judgment_accepts_common_focus_field_aliases():
    session = Session("s", turns=[Turn("t1", "伊卡拉来了", "伊卡拉来了"), Turn("t2", "安拉在经文中", "安拉在经文中")])
    normalized = deepseek.normalize_judgment({
        "outcome": "CONFLICT", "confidence": 0.9,
        "focus": [{"target_turn": "t1", "current_text": "伊卡拉", "replacement": "安拉", "closed_set": ["伊卡拉", "安拉"], "evidence_turn_id": "t2", "relation": "MUTUALLY_EXCLUSIVE"}],
    }, session)
    assert normalized.focus[0].target_turn_id == "t1"
    assert normalized.focus[0].span == "伊卡拉"
    assert normalized.focus[0].proposed_text == "安拉"
    assert normalized.focus[0].evidence_turn_ids == ["t2"]


def test_consistent_model_label_keeps_valid_focus_for_audio_gate():
    session = Session("s", turns=[Turn("t1", "爱拉来了", "爱拉来了"), Turn("t2", "安拉在经文中", "安拉在经文中")])
    normalized = deepseek.normalize_judgment({"outcome": "CONSISTENT", "confidence": 0.9, "focus": [{"target_turn_id": "t1", "span": "爱拉", "proposed_text": "安拉", "alternatives": ["爱拉", "安拉"], "evidence_turn_ids": ["t2"]}]}, session)
    assert normalized.outcome == "UNCERTAIN"
    assert normalized.focus[0].proposed_text == "安拉"


def test_prompt_requires_canonical_alias_to_become_focus():
    assert "不要把 alias 当作已经正确" in deepseek._CONTEXT_JUDGE_SYSTEM
    assert "必须提出该历史 turn 的 focus" in deepseek._CONTEXT_JUDGE_SYSTEM


def test_normalize_repairs_grounded_focus_missing_alternatives():
    from asr_agent.context_judge import normalize_judgment
    session = Session("s", turns=[Turn("t1", "攒生节来了", "攒生节来了")])
    normalized = normalize_judgment({"outcome": "UNCERTAIN", "confidence": 0.6, "focus": [{"target_turn_id": "t1", "span": "攒生节", "proposed_text": "宰牲节"}]}, session)
    assert normalized.focus[0].alternatives == ["攒生节", "宰牲节"]
    assert normalized.focus[0].evidence_turn_ids == ["t1"]


def test_deepseek_context_judge_safely_defers_malformed_output(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {"outcome": "CONFLICT", "focus": [{"bad": "shape"}]})
    turn = Turn("t1", "测试", "测试")

    result = deepseek.judge_context(session=Session("s", turns=[turn]), current_turn=turn, memory=MemoryPacket())

    assert result.outcome == "UNCERTAIN"
    assert result.focus == []


def test_malformed_focus_uses_independent_audio_diff_when_candidates_are_missing(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {"outcome": "CONFLICT", "focus": [{"bad": "shape"}]})
    monkeypatch.setattr(
        deepseek,
        "retranscribe_window",
        lambda *_, **__: {"ok": True, "text": "涂博士来了"},
    )
    turn = Turn(
        "t1",
        "图博士来了",
        "图博士来了",
        meta={"audio_path": "/tmp/sample.wav", "start_sec": 0.0, "end_sec": 1.0},
    )

    result = deepseek.judge_context(session=Session("s", turns=[turn]), current_turn=turn, memory=MemoryPacket())

    assert result.outcome == "UNCERTAIN"
    assert result.focus[0].span == "图"
    assert result.focus[0].proposed_text == "涂"


def test_audio_diff_fallback_ignores_punctuation_only_changes(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {"outcome": "CONFLICT", "focus": [{"bad": "shape"}]})
    monkeypatch.setattr(deepseek, "retranscribe_window", lambda *_, **__: {"ok": True, "text": "图博士来了。"})
    turn = Turn("t1", "图博士来了，", "图博士来了，", meta={"audio_path": "/tmp/sample.wav", "start_sec": 0.0, "end_sec": 1.0})

    result = deepseek.judge_context(session=Session("s", turns=[turn]), current_turn=turn, memory=MemoryPacket())

    assert result.focus == []


def _belief(value: str):
    from asr_agent.models import MemoryBelief

    return MemoryBelief(
        belief_id="b",
        subject="speaker",
        predicate="mentions_champion",
        value=value,
        confidence=0.9,
        status="provisional",
        source_turn_ids=["t1"],
        source_session_ids=["s"],
    )


def test_domain_entities_exclude_canonical_text():
    """canonical_text beliefs record span corrections, not domain proper nouns;
    their values (e.g. "是说"/"是设") must not be surfaced as domain entities
    to the LLM judge."""
    from asr_agent.models import MemoryBelief

    memory = MemoryPacket(working_beliefs=[
        MemoryBelief(
            belief_id="c1", subject="turn:x:canonical_span", predicate="canonical_text",
            value="是说", aliases=["是设"], confidence=0.9, status="provisional",
            source_turn_ids=["t1"], source_session_ids=["s"],
        ),
        MemoryBelief(
            belief_id="c2", subject="turn:x:canonical_span", predicate="canonical_text",
            value="是设", aliases=["是什"], confidence=0.9, status="provisional",
            source_turn_ids=["t1"], source_session_ids=["s"],
        ),
        _belief("皇子"),
    ])

    entities = deepseek._domain_entities(memory)

    assert "是说" not in entities
    assert "是设" not in entities
    assert "皇子" in entities


def test_canonical_entities_are_exposed_separately_with_audio_provenance():
    from asr_agent.models import MemoryBelief

    memory = MemoryPacket(canonical_entities=[
        MemoryBelief(
            belief_id="e1", subject="turn:t1:canonical_entity", predicate="canonical_entity",
            value="卡兹克", aliases=["卡斯克"], confidence=0.9, status="provisional",
            source_turn_ids=["t1"], source_session_ids=["s"], evidence_kinds=["audio_verified"],
        )
    ])

    assert deepseek._canonical_entities(memory) == [{
        "canonical": "卡兹克", "aliases": ["卡斯克"], "confidence": 0.9, "evidence_turn_ids": ["t1"],
    }]
