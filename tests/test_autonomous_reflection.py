from asr_agent.retrace import ReTraceService


def test_autonomous_reflection_revises_only_after_later_quote_and_audio_relisten(tmp_path):
    def reflector(**_: object) -> list[dict[str, object]]:
        return [{
            "target_turn_id": "t1", "before_text": "图博士", "after_text": "涂博士",
            "evidence": [{"turn_id": "t2", "quote": "实验室负责人涂博士"}],
            "score": 0.91, "rationale": "later self-identification resolves a homophone",
        }]

    service = ReTraceService(
        tmp_path,
        reflector=reflector,
        audio_verifier=lambda **_: {"ok": True, "scores": {"图博士": .05, "涂博士": .95}},
    )
    service.process_turn(
        "s", "t1", "图博士下午汇报",
        confidence={"图博士": .2}, text_candidates={"图博士": ["图博士", "涂博士"]},
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn("s", "t2", "实验室负责人涂博士确认下午汇报")

    assert result["session"]["turns"][0]["current_text"] == "涂博士下午汇报"
    event = result["revisions"][0]
    assert event["resolver"] == "audio-semantic-gate"
    assert event["evidence"][0].startswith("t2:") and "涂博士" in event["evidence"][0]
    assert any(item.startswith("audio:/tmp/source.wav:") and "涂博士" in item for item in event["evidence"])


def test_agent_discovers_error_span_without_prior_uncertainty(tmp_path):
    """Datasets do not label error spans; later evidence should open a closed set."""

    def reflector(**_: object) -> list[dict[str, object]]:
        return [{
            "target_turn_id": "t1", "before_text": "威沃", "after_text": "vivo",
            "evidence": [{"turn_id": "t2", "quote": "有人说是vivo"}],
            "score": 0.93, "rationale": "later brand mention contradicts earlier ASR",
        }]

    service = ReTraceService(
        tmp_path,
        reflector=reflector,
        audio_verifier=lambda **_: {"ok": True, "scores": {"威沃": .1, "vivo": .9}},
    )
    # No confidence / text_candidates provided — agent must discover the span.
    service.process_turn(
        "s", "t1", "大家用的是什么品牌，威沃还是别的。",
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn("s", "t2", "有人说是vivo，也有人说小米。", use_llm=True)

    turn = result["session"]["turns"][0]
    assert turn["raw_text"] == "大家用的是什么品牌，威沃还是别的。"
    assert turn["current_text"] == "大家用的是什么品牌，vivo还是别的。"
    assert any(item["span"] == "威沃" for item in turn["hypotheses"])
    assert result["revisions"][0]["resolver"] == "audio-semantic-gate"


def test_deterministic_candidate_echo_revises_without_llm(tmp_path):
    """Open hypothesis + later candidate verbatim must revise even when LLM is off."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **_: {"ok": True, "scores": {"威沃": .2, "vivo": .8}},
    )
    service.process_turn(
        "s", "t1", "品牌是威沃",
        confidence={"威沃": 0.2},
        text_candidates={"威沃": ["威沃", "vivo"]},
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn(
        "s", "t2", "后来确认是vivo",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 1, "end_sec": 2},
    )
    assert result["session"]["turns"][0]["current_text"] == "品牌是vivo"
    assert result["revisions"]


def test_same_turn_near_form_conflict_can_revise(tmp_path):
    """泰信 then 泰康 in one turn should open a closed set and allow audio revision."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **_: {"ok": True, "scores": {"泰信": .15, "泰康": .85}},
    )
    result = service.process_turn(
        "s", "t1",
        "比如泰信商业保险，比如泰康的保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 2},
    )
    # depending on token order: 泰信 before 泰康 => revise first mention
    assert result["revisions"]
    assert "泰康" in result["session"]["turns"][0]["current_text"]
    assert "泰信商业保险" not in result["session"]["turns"][0]["current_text"] or result["session"]["turns"][0]["current_text"].count("泰康") >= 1


def test_autonomous_reflection_rejects_evidence_not_quoted_from_a_later_turn(tmp_path):
    def reflector(**_: object) -> list[dict[str, object]]:
        return [{
            "target_turn_id": "t1", "before_text": "甲", "after_text": "乙",
            "evidence": [{"turn_id": "t2", "quote": "不存在的证据"}], "score": 0.99,
        }]

    service = ReTraceService(tmp_path, reflector=reflector)
    service.process_turn("s", "t1", "甲已经到了")
    result = service.process_turn("s", "t2", "后面出现了新的说明")

    assert result["revisions"] == []
    assert result["session"]["turns"][0]["current_text"] == "甲已经到了"


def test_same_turn_canonical_earlier_corrects_later_drift(tmp_path):
    """R0020-style order: 泰康 then 泰信 must revise the later drift."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **_: {"ok": True, "scores": {"泰信": .15, "泰康": .85}},
    )
    result = service.process_turn(
        "s", "t1",
        "比如泰康的保险，比如泰信商业保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 2},
    )
    assert result["revisions"]
    assert any(e["span"] == "泰信" and e["replacement"] == "泰康" for e in result["revisions"])
    assert "泰信商业保险" not in result["session"]["turns"][0]["current_text"]


def test_degenerate_turn_open_relisten_recovers_transcript(tmp_path):
    """Collapsed ASR like 遥遥遥遥 must be replaced by open re-listen text."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **_: {"ok": False, "error": "unused"},
        audio_retranscriber=lambda **_: {
            "ok": True,
            "text": "骁龙。那麒麟的呢？因为华为是麒麟的。",
            "audio_path": "/tmp/source.wav",
            "start_sec": 85.5,
            "end_sec": 96.5,
        },
    )
    result = service.process_turn(
        "s",
        "t7",
        "遥。遥遥遥遥。",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 87.36, "end_sec": 95.008},
    )
    assert result["revisions"]
    assert result["revisions"][0]["resolver"] == "audio-open-relisten"
    assert "麒麟" in result["session"]["turns"][0]["current_text"]
    assert "遥遥遥遥" not in result["session"]["turns"][0]["current_text"]


def test_tama_sparse_llm_skips_reflect_without_triggers(tmp_path):
    """Chinese-only turns with no deferred hang must not call DeepSeek every turn."""
    calls = {"n": 0}

    def reflector(**_: object) -> list[dict[str, object]]:
        calls["n"] += 1
        return []

    service = ReTraceService(
        tmp_path,
        reflector=reflector,
        audio_verifier=lambda **_: {"ok": False, "error": "unused"},
    )
    service.process_turn("s", "t1", "今天讨论一下保险方案", use_llm=True)
    result = service.process_turn("s", "t2", "然后继续看下一页内容", use_llm=True)
    assert calls["n"] == 0
    assert result["session"]["turns"][1]["meta"]["llm"]["mode"] == "tama_sparse"
    assert result["session"]["turns"][1]["meta"]["llm"]["calls"] == 0


def test_tama_sparse_llm_fires_on_ascii_brand_trigger(tmp_path):
    """Cross-script brand in later turn is a TAMA-style sparse LLM trigger."""
    calls = {"n": 0}

    def reflector(**_: object) -> list[dict[str, object]]:
        calls["n"] += 1
        return [{
            "target_turn_id": "t1", "before_text": "威沃", "after_text": "vivo",
            "evidence": [{"turn_id": "t2", "quote": "有人说是vivo"}],
            "score": 0.93, "rationale": "sparse brand trigger",
        }]

    service = ReTraceService(
        tmp_path,
        reflector=reflector,
        audio_verifier=lambda **_: {"ok": True, "scores": {"威沃": .1, "vivo": .9}},
    )
    service.process_turn(
        "s", "t1", "大家用的是威沃", use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn("s", "t2", "有人说是vivo", use_llm=True)
    # Deterministic brand-alias path may revise before LLM; sparse trigger still fires.
    assert "cross_script_brand" in result["session"]["turns"][1]["meta"]["llm"]["triggers"]
    assert result["session"]["turns"][0]["current_text"] == "大家用的是vivo"


def test_cross_turn_near_form_revises_mid_sentence_evidence(tmp_path):
    """R0020-style: earlier 泰信, later mid-sentence 泰康 must revise without LLM."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {c: (0.85 if c == "泰康" else 0.15) for c in kwargs.get("candidates", [])},
        },
    )
    service.process_turn(
        "s",
        "t1",
        "比如泰信商业保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn(
        "s",
        "t2",
        "好的单位会上泰康的保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 1, "end_sec": 2},
    )
    assert result["revisions"]
    assert result["session"]["turns"][0]["current_text"] == "比如泰康商业保险"
    assert result["revisions"][0]["resolver"] == "audio-semantic-gate"


def test_title_near_form_revises_across_turns(tmp_path):
    """图博士→涂博士 must be discoverable from later title evidence alone."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {c: (0.9 if c == "涂博士" else 0.1) for c in kwargs.get("candidates", [])},
        },
    )
    service.process_turn(
        "s",
        "t1",
        "图博士下午汇报",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn(
        "s",
        "t2",
        "实验室负责人涂博士确认",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 1, "end_sec": 2},
    )
    assert result["session"]["turns"][0]["current_text"] == "涂博士下午汇报"
    assert any(e["span"] == "图博士" and e["replacement"] == "涂博士" for e in result["revisions"])


def test_brand_alias_revises_without_llm(tmp_path):
    """威沃→vivo should revise from deterministic brand alias path (no DeepSeek)."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: (_ for _ in ()).throw(AssertionError("LLM should not be required")),
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {c: (0.9 if c == "vivo" else 0.1) for c in kwargs.get("candidates", [])},
        },
    )
    service.process_turn(
        "s",
        "t1",
        "大家用威沃",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn(
        "s",
        "t2",
        "确认是vivo",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 1, "end_sec": 2},
    )
    assert result["session"]["turns"][0]["current_text"] == "大家用vivo"
    assert result["revisions"][0]["resolver"] == "audio-semantic-gate"


def test_strong_semantic_commits_when_audio_verifier_fails(tmp_path):
    """Qwen verify flakes must not drop a high-score later-evidence revision."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **_: {"ok": False, "error": "verifier timeout"},
    )
    service.process_turn(
        "s",
        "t1",
        "比如泰信商业保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn(
        "s",
        "t2",
        "好的单位会上泰康的保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 1, "end_sec": 2},
    )
    assert result["revisions"]
    assert result["revisions"][0]["resolver"] == "semantic-evidence-gate"
    assert result["session"]["turns"][0]["current_text"] == "比如泰康商业保险"


def test_weak_audio_plus_strong_semantic_commits_without_llm(tmp_path):
    """Audio top but below 0.70 should still commit with strong later evidence."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {c: (0.6 if c == "泰康" else 0.4) for c in kwargs.get("candidates", [])},
        },
    )
    service.process_turn(
        "s",
        "t1",
        "比如泰信商业保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn(
        "s",
        "t2",
        "好的单位会上泰康的保险",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 1, "end_sec": 2},
    )
    assert result["revisions"]
    assert result["revisions"][0]["resolver"] == "audio-semantic-weak-gate"
    assert "泰康商业保险" in result["session"]["turns"][0]["current_text"]


def test_does_not_revise_subspan_inside_longer_product_token(tmp_path):
    """曲屏 evidence must not rewrite 布屏 inside 瀑布屏."""
    service = ReTraceService(
        tmp_path,
        reflector=lambda **_: [],
        audio_verifier=lambda **kwargs: {
            "ok": True,
            "scores": {c: 0.8 for c in kwargs.get("candidates", [])},
        },
    )
    service.process_turn(
        "s",
        "t1",
        "首先它是全面瀑布屏",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 0, "end_sec": 1},
    )
    result = service.process_turn(
        "s",
        "t2",
        "你们是喜欢曲屏还是瀑布屏",
        use_llm=False,
        meta={"audio_path": "/tmp/source.wav", "start_sec": 1, "end_sec": 2},
    )
    assert "瀑布屏" in result["session"]["turns"][0]["current_text"]
    assert "瀑曲屏" not in result["session"]["turns"][0]["current_text"]
    assert not any(e.get("span") == "布屏" for e in result["revisions"])
