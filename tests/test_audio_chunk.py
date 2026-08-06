import numpy as np

from asr_agent.integrations.audio_chunk import ChunkConfig, join_transcripts, plan_chunks
from asr_agent.integrations.qwen_asr import parse_observation


def test_short_audio_is_single_chunk():
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    cfg = ChunkConfig(max_sec=15.0, overlap_sec=0.3, min_sec=0.4, top_db=30.0, target_sr=sr)
    spans = plan_chunks(len(audio), sr, cfg, audio)
    assert spans == [(0, len(audio))]


def test_long_audio_is_split_under_max_sec():
    sr = 16000
    # 45s of low-level noise so silence split may fail and hard-split kicks in.
    rng = np.random.default_rng(0)
    audio = (rng.normal(0, 0.01, sr * 45)).astype(np.float32)
    cfg = ChunkConfig(max_sec=15.0, overlap_sec=0.3, min_sec=0.4, top_db=30.0, target_sr=sr)
    spans = plan_chunks(len(audio), sr, cfg, audio)
    assert len(spans) >= 3
    assert all((end - start) / sr <= cfg.max_sec + 1e-6 for start, end in spans)


def test_join_transcripts_concatenates_chinese():
    assert join_transcripts(["施工电梯", "报验完成"]) == "施工电梯报验完成"


def test_qwen_observation_preserves_only_valid_uncertainty_candidates():
    observation = parse_observation('{"text":"图博士到了","uncertain_spans":[{"span":"图博士","candidates":["图博士","涂博士"],"confidence":0.2}]}')
    assert observation["text"] == "图博士到了"
    assert observation["uncertainty"]["confidence"] == {"图博士": 0.2}
    assert observation["uncertainty"]["text_candidates"] == {"图博士": ["图博士", "涂博士"]}


def test_qwen_observation_preserves_nbest_alternatives_for_retrace():
    observation = parse_observation(
        '{"text":"图博士到了","alternatives":["涂博士到了","图博士到了"],'
        '"uncertain_spans":[]}'
    )

    assert observation["nbest"] == ["图博士到了", "涂博士到了"]
