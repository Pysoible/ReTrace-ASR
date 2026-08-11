from asr_agent.integrations.qwen_asr import parse_observation, parse_uncertainty_tags


def test_parse_observation_accepts_json_with_uncertain_spans():
    raw = '{"text":"威沃和小米","uncertain_spans":[{"span":"威沃","candidates":["威沃","vivo"],"confidence":0.2}]}'
    result = parse_observation(raw)
    assert result["text"] == "威沃和小米"
    assert result["uncertainty"]["confidence"]["威沃"] == 0.2
    assert result["uncertainty"]["text_candidates"]["威沃"] == ["威沃", "vivo"]


def test_parse_observation_falls_back_to_plain_text():
    result = parse_observation("大家好我想先知道品牌")
    assert result["text"] == "大家好我想先知道品牌"
    assert result["uncertainty"] == {}


def test_parse_observation_rejects_diarization_json_dump():
    raw = '[{"start_time": "0.0", "end_time": "1.0", "label": "说话人1"}]'
    result = parse_observation(raw)
    assert result["text"] == ""


def test_parse_observation_preserves_degenerate_loop_for_auditable_recovery():
    result = parse_observation('{"text":"啊啊啊啊啊啊啊啊","uncertain_spans":[]}')

    assert result["text"] == "啊啊啊啊啊啊啊啊"


def test_parse_uncertainty_tags_against_fixed_text():
    text = "首先知道大家用的是什么品牌威沃"
    raw = '{"uncertain_spans":[{"span":"威沃","candidates":["威沃","vivo"],"confidence":0.3}]}'
    tagged = parse_uncertainty_tags(raw, text)
    assert tagged["confidence"]["威沃"] == 0.3
    assert "vivo" in tagged["text_candidates"]["威沃"]
