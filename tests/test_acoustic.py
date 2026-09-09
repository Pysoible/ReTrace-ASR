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


def test_high_conf_correction_replaces_only_high_conf_disagreement():
    # "吗" (first pass) vs "呢" (paraformer, high conf) → replace.
    fp = "你对这个游戏有什么看法吗"
    pf = "你对这个游戏有什么看法呢"
    confs = [
        {"char": "你", "conf": 0.9}, {"char": "对", "conf": 0.9}, {"char": "这", "conf": 0.9},
        {"char": "个", "conf": 0.9}, {"char": "游", "conf": 0.9}, {"char": "戏", "conf": 0.9},
        {"char": "有", "conf": 0.9}, {"char": "什", "conf": 0.9}, {"char": "么", "conf": 0.9},
        {"char": "看", "conf": 0.9}, {"char": "法", "conf": 0.9}, {"char": "呢", "conf": 0.92},
    ]
    assert acoustic.high_conf_correction(fp, pf, confs) == "你对这个游戏有什么看法呢"


def test_high_conf_correction_keeps_low_conf_first_pass_char():
    # paraformer's "呢" is low-confidence → keep first-pass "吗".
    fp = "你对这个游戏有什么看法吗"
    pf = "你对这个游戏有什么看法呢"
    confs = [
        {"char": "你", "conf": 0.9}, {"char": "对", "conf": 0.9}, {"char": "这", "conf": 0.9},
        {"char": "个", "conf": 0.9}, {"char": "游", "conf": 0.9}, {"char": "戏", "conf": 0.9},
        {"char": "有", "conf": 0.9}, {"char": "什", "conf": 0.9}, {"char": "么", "conf": 0.9},
        {"char": "看", "conf": 0.9}, {"char": "法", "conf": 0.9}, {"char": "呢", "conf": 0.55},
    ]
    assert acoustic.high_conf_correction(fp, pf, confs) == "你对这个游戏有什么看法吗"


def test_high_conf_correction_inserts_high_conf_omitted_chars():
    # paraformer has extra chars with high confidence → insert (fix omission).
    fp = "我吃"
    pf = "我吃饭"
    confs = [
        {"char": "我", "conf": 0.9}, {"char": "吃", "conf": 0.9},
        {"char": "饭", "conf": 0.88},
    ]
    assert acoustic.high_conf_correction(fp, pf, confs) == "我吃饭"


def test_high_conf_correction_bails_on_misaligned_confs():
    fp = "我吃"
    pf = "我吃饭"
    confs = [{"char": "我", "conf": 0.9}]  # length mismatch → bail out
    assert acoustic.high_conf_correction(fp, pf, confs) == "我吃"


def test_high_conf_correction_preserves_punctuation():
    # "吗" (first pass) vs "呢" (paraformer, high conf) with punctuation around.
    fp = "你对这个游戏有什么看法吗？"
    pf = "你对这个游戏有什么看法呢"
    confs = [
        {"char": "你", "conf": 0.9}, {"char": "对", "conf": 0.9}, {"char": "这", "conf": 0.9},
        {"char": "个", "conf": 0.9}, {"char": "游", "conf": 0.9}, {"char": "戏", "conf": 0.9},
        {"char": "有", "conf": 0.9}, {"char": "什", "conf": 0.9}, {"char": "么", "conf": 0.9},
        {"char": "看", "conf": 0.9}, {"char": "法", "conf": 0.9}, {"char": "呢", "conf": 0.92},
    ]
    assert acoustic.high_conf_correction(fp, pf, confs) == "你对这个游戏有什么看法呢？"


def test_high_conf_correction_preserves_punctuation_on_insert():
    fp = "我吃。"
    pf = "我吃饭"
    confs = [
        {"char": "我", "conf": 0.9}, {"char": "吃", "conf": 0.9},
        {"char": "饭", "conf": 0.88},
    ]
    assert acoustic.high_conf_correction(fp, pf, confs) == "我吃饭。"


def test_high_conf_correction_returns_original_when_no_change():
    fp = "你好，世界。"
    pf = "你好世界"
    confs = [{"char": "你", "conf": 0.9}, {"char": "好", "conf": 0.9},
             {"char": "世", "conf": 0.9}, {"char": "界", "conf": 0.9}]
    assert acoustic.high_conf_correction(fp, pf, confs) == "你好，世界。"


def _fake_pypinyin(monkeypatch):
    """Inject a fake pypinyin with a small syllable map for test characters."""
    import sys
    import types

    syl = {
        "卡": "ka", "兹": "zi", "斯": "si", "克": "ke",
        "它": "ta", "他": "ta", "吗": "ma", "呢": "ne",
        "你": "ni", "好": "hao", "世": "shi", "界": "jie",
        "对": "dui", "这": "zhe", "个": "ge", "游": "you",
        "戏": "xi", "有": "you", "什": "shen", "么": "me",
        "看": "kan", "法": "fa", "我": "wo", "吃": "chi", "饭": "fan",
    }
    mod = types.ModuleType("pypinyin")
    mod.lazy_pinyin = lambda s: [syl.get(ch, ch) for ch in s]
    monkeypatch.setitem(sys.modules, "pypinyin", mod)
    return mod


def test_high_conf_correction_declines_transliteration_variant(monkeypatch):
    _fake_pypinyin(monkeypatch)
    # "卡兹克" vs "卡斯克" are near-homophone transliterations → keep "卡兹克".
    fp = "把卡兹克杀了的话"
    pf = "把卡斯克杀了的话"
    confs = [
        {"char": "把", "conf": 0.9}, {"char": "卡", "conf": 0.9}, {"char": "斯", "conf": 0.9},
        {"char": "克", "conf": 0.9}, {"char": "杀", "conf": 0.9}, {"char": "了", "conf": 0.9},
        {"char": "的", "conf": 0.9}, {"char": "话", "conf": 0.9},
    ]
    assert acoustic.high_conf_correction(fp, pf, confs) == "把卡兹克杀了的话"


def test_high_conf_correction_allows_single_char_homophone(monkeypatch):
    _fake_pypinyin(monkeypatch)
    # Single-char homophones ("它"/"他") are still corrected.
    fp = "它充了一千"
    pf = "他充了一千"
    confs = [
        {"char": "他", "conf": 0.9}, {"char": "充", "conf": 0.9}, {"char": "了", "conf": 0.9},
        {"char": "一", "conf": 0.9}, {"char": "千", "conf": 0.9},
    ]
    assert acoustic.high_conf_correction(fp, pf, confs) == "他充了一千"
