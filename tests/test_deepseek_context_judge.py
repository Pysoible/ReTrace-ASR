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


def test_deepseek_context_judge_safely_defers_malformed_output(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {"outcome": "CONFLICT", "focus": [{"bad": "shape"}]})
    turn = Turn("t1", "测试", "测试")

    result = deepseek.judge_context(session=Session("s", turns=[turn]), current_turn=turn, memory=MemoryPacket())

    assert result.outcome == "UNCERTAIN"
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
