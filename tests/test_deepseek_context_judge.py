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
            "focus": [{
                "target_turn_id": "t1",
                "span": "图博士",
                "proposed_text": "涂博士",
                "alternatives": ["图博士", "涂博士"],
                "evidence_turn_ids": ["t2"],
            }],
        },
    )
    session = Session("s", turns=[Turn("t1", "图博士来了", "图博士来了"), Turn("t2", "我是涂博士", "我是涂博士")])

    result = deepseek.judge_context(session=session, current_turn=session.turns[-1], memory=MemoryPacket())

    assert result.outcome == "CONFLICT"
    assert result.focus[0].proposed_text == "涂博士"


def test_deepseek_context_judge_safely_defers_malformed_output(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek, "_chat_json", lambda *_, **__: {"outcome": "CONFLICT", "focus": [{"bad": "shape"}]})
    turn = Turn("t1", "测试", "测试")

    result = deepseek.judge_context(session=Session("s", turns=[turn]), current_turn=turn, memory=MemoryPacket())

    assert result.outcome == "UNCERTAIN"
    assert result.focus == []
