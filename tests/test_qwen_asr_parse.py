from asr_agent.integrations.qwen_asr import (
    _coverage_signal,
    _is_truncated,
    _speech_duration_sec,
    parse_observation,
    parse_uncertainty_tags,
)


def test_is_truncated_flags_low_density_transcript(monkeypatch):
    """A chunk with 10s of speech but only 4 chars is under-transcribed."""
    monkeypatch.setattr("asr_agent.integrations.qwen_asr._speech_duration_sec", lambda _p: 10.0)

    assert _is_truncated(None, "M四还好") is True  # 4 chars / 10s = 0.4
    assert _is_truncated(None, "") is True  # empty transcript


def test_is_truncated_accepts_normal_density(monkeypatch):
    monkeypatch.setattr("asr_agent.integrations.qwen_asr._speech_duration_sec", lambda _p: 10.0)

    # 29 chars / 10s = 2.9 chars/sec — normal conversational Mandarin.
    assert _is_truncated(None, "这个游戏的话我觉得就是怎么说呢就是它这个机制还是不错的") is False


def test_is_truncated_skips_short_speech(monkeypatch):
    monkeypatch.setattr("asr_agent.integrations.qwen_asr._speech_duration_sec", lambda _p: 2.0)

    # Too little voiced time (< 3s) → not judged, even if density is low.
    assert _is_truncated(None, "嗯") is False


def test_coverage_signal_is_json_serializable_with_numpy_duration(monkeypatch):
    import json
    import numpy as np

    monkeypatch.setattr("asr_agent.integrations.qwen_asr._speech_duration_sec", lambda _p: np.float64(10.0))

    signal = _coverage_signal(None, "设置的呀")

    assert signal["truncated"] is True
    assert isinstance(signal["speech_sec"], float)
    assert isinstance(signal["char_density"], float)
    assert json.loads(json.dumps(signal))["truncated"] is True


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


def test_streaming_asr_enriches_uncertainty_before_publishing(monkeypatch, tmp_path):
    from asr_agent.integrations import qwen_asr

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"placeholder")
    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"placeholder")
    published = []

    monkeypatch.setattr(qwen_asr, "read_asr_config", lambda: type("Config", (), {"audio_dir": str(tmp_path), "gpu_ids": ["0"]})())
    monkeypatch.setattr(qwen_asr, "_resolve_audio", lambda *_: audio)
    monkeypatch.setattr(qwen_asr, "_engines", lambda: [object()])
    monkeypatch.setattr(qwen_asr, "split_audio_file", lambda *_: {"chunks": [{"index": 0, "path": str(chunk), "start_sec": 0.0, "end_sec": 2.0}], "chunk_count": 1, "duration_sec": 2.0, "chunked": True})
    monkeypatch.setattr(qwen_asr, "cleanup_chunks", lambda *_: None)
    monkeypatch.setattr(qwen_asr, "_infer_chunks_streaming", lambda _paths, _prompt, callback: callback(0, "威沃和小米"))
    monkeypatch.setattr(qwen_asr, "_enrich_uncertainty", lambda *_args: {"confidence": {"威沃": 0.3}, "text_candidates": {"威沃": ["威沃", "vivo"]}})
    monkeypatch.setattr(qwen_asr, "_coverage_signal", lambda *_args, **_kwargs: {"truncated": False})
    monkeypatch.setattr(qwen_asr, "_attach_acoustic_disagreement", lambda _path, _text, uncertainty: uncertainty)

    qwen_asr.stream_transcribe_audio(str(audio), lambda *args: published.append(args))

    assert published[0][2]["text_candidates"] == {"威沃": ["威沃", "vivo"]}
