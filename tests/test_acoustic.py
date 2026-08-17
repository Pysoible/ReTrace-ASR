from asr_agent.integrations import acoustic


def test_char_level_confidence_returns_detailed_confidences(monkeypatch):
    monkeypatch.setattr(
        acoustic,
        "_transcribe_paraformer_detailed",
        lambda _p: (
            "你对这个游戏有什么看法呢",
            [{"char": "你", "conf": 0.9}, {"char": "呢", "conf": 0.58}],
        ),
    )
    confs = acoustic.char_level_confidence("/tmp/x.wav")
    assert confs == [{"char": "你", "conf": 0.9}, {"char": "呢", "conf": 0.58}]


def test_acoustic_signals_splits_disagreement_and_low_conf(monkeypatch):
    monkeypatch.setattr(
        acoustic,
        "_transcribe_paraformer_detailed",
        lambda _p: (
            "我是张三",
            [{"char": "张", "conf": 0.55}, {"char": "三", "conf": 0.4}],
        ),
    )
    disagreements, low_conf = acoustic.acoustic_signals(
        "/tmp/x.wav", "我是李四", low_conf_threshold=0.75
    )
    # "李四" (first pass) vs "张三" (paraformer) disagree on a 2-char span.
    assert any("李四" in d["span_a"] and "张三" in d["span_b"] for d in disagreements)
    # Low-confidence characters below 0.75 are surfaced.
    assert [c["char"] for c in low_conf] == ["张", "三"]


def test_acoustic_signals_empty_second_pass(monkeypatch):
    monkeypatch.setattr(acoustic, "_transcribe_paraformer_detailed", lambda _p: ("", []))
    disagreements, low_conf = acoustic.acoustic_signals("/tmp/x.wav", "你好")
    assert disagreements == []
    assert low_conf == []


def test_detect_asr_disagreement_uses_shared_diff(monkeypatch):
    monkeypatch.setattr(acoustic, "_transcribe_paraformer", lambda _p: "我是张三")
    spans = acoustic.detect_asr_disagreement("/tmp/x.wav", "我是李四")
    assert any("张三" in s["span_b"] or "李四" in s["span_a"] for s in spans)
