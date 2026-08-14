from pathlib import Path

from asr_agent.context_judge import ExplicitSignalFallbackJudge
from asr_agent.models import Session, Turn


def test_fallback_judge_only_uses_explicit_asr_signals():
    judge = ExplicitSignalFallbackJudge()
    session = Session("s", turns=[Turn("t1", "今天讨论保险方案", "今天讨论保险方案")])

    result = judge(session=session, current_turn=session.turns[0], memory=None)

    assert result.outcome == "CONSISTENT"
    assert result.focus == []


def test_fallback_judge_marks_low_confidence_as_uncertain_without_guessing_span():
    judge = ExplicitSignalFallbackJudge()
    turn = Turn(
        "t1",
        "图博士来了",
        "图博士来了",
        meta={"asr_signals": {"confidence": {"图博士": 0.2}, "nbest": ["图博士来了", "涂博士来了"]}},
    )
    session = Session("s", turns=[turn])

    result = judge(session=session, current_turn=turn, memory=None)

    assert result.outcome == "UNCERTAIN"
    assert result.focus == []


def test_fallback_judge_flags_acoustic_disagreement_as_uncertain():
    """A second independent ASR disagreeing with the first pass is an acoustic
    uncertainty signal — the fallback judge must mark it UNCERTAIN (triggering
    relisten) without itself nominating any span."""
    judge = ExplicitSignalFallbackJudge()
    turn = Turn(
        "t1",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        "这个连加额，那这个英雄跟那个跟螳螂还是有有渊源的是吧？",
        meta={
            "uncertainty": {
                "acoustic_disagreement": [
                    {"tag": "replace", "span_a": "加额，", "span_b": "家的", "offset_a": 2, "offset_b": 2},
                ]
            }
        },
    )
    session = Session("s", turns=[turn])

    result = judge(session=session, current_turn=turn, memory=None)

    assert result.outcome == "UNCERTAIN"
    assert result.focus == []


def test_old_global_ngram_nomination_is_removed():
    source = Path("backend/asr_agent/retrace.py").read_text(encoding="utf-8")
    assert "_COMMON_BIGRAMS" not in source
    assert "_GENERIC_ENTITY_BLOCK" not in source
    assert "ngram" not in source.lower()
