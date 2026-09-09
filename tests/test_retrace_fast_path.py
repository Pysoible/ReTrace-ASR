from asr_agent.context_judge import ContextJudgment
from asr_agent.retrace import ReTraceService


def test_moss_normal_turn_calls_context_judge_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ASR_FAST_NORMAL_TURNS", raising=False)
    calls = []

    def context_judge(**kwargs):
        calls.append(kwargs)
        return ContextJudgment("CONSISTENT", 0.9)

    service = ReTraceService(tmp_path, context_judge=context_judge)
    result = service.process_turn("s", "t1", "这是一段正常的中文转写文本。", source="moss")

    assert len(calls) == 1
    assert result["status"] == "idle"
    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "这是一段正常的中文转写文本。"
    assert result["session"]["turns"][0]["meta"]["context_judgment"]["outcome"] == "CONSISTENT"


def test_normal_turn_uses_fast_keep_path_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_FAST_NORMAL_TURNS", "1")

    def context_judge(**_kwargs):
        raise AssertionError("explicit fast mode should not call the remote context judge")

    service = ReTraceService(tmp_path, context_judge=context_judge)
    result = service.process_turn("s", "t1", "这是一段正常的中文转写文本。", source="moss")

    assert result["status"] == "idle"
    assert result["judgment"]["rationale"] == "fast normal-turn path"


def test_signal_turn_still_calls_context_judge(tmp_path):
    calls = []

    def context_judge(**kwargs):
        calls.append(kwargs)
        from asr_agent.context_judge import ContextJudgment

        return ContextJudgment("UNCERTAIN", 0.5)

    service = ReTraceService(tmp_path, context_judge=context_judge)
    service.process_turn(
        "s",
        "t1",
        "疑似低覆盖文本。",
        source="moss",
        meta={"uncertainty": {"low_conf_chars": [{"char": "疑", "conf": 0.2}]}},
    )

    assert len(calls) == 1
