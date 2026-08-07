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
    assert event["evidence"][0] == "t2:实验室负责人涂博士"
    assert event["evidence"][1].startswith("audio:/tmp/source.wav:0-1:涂博士")


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
