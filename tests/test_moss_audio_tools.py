import numpy as np
import soundfile as sf

from asr_agent.integrations import moss_audio_tools


def test_moss_verifier_crops_downmixes_and_scores_only_supplied_candidates(tmp_path):
    audio = tmp_path / "stereo.wav"
    samples = np.zeros((32000, 2), dtype=np.float32)
    sf.write(audio, samples, 16000)
    seen = {}

    def transcribe(path):
        data, sample_rate = sf.read(path, always_2d=True)
        seen["shape"] = data.shape
        seen["sample_rate"] = sample_rate
        return "今天请涂博士介绍实验"

    result = moss_audio_tools.verify_candidates(
        str(audio),
        0.25,
        1.25,
        ["图博士", "涂博士"],
        transcriber=transcribe,
    )

    assert result["ok"] is True
    assert set(result["scores"]) == {"图博士", "涂博士"}
    assert result["scores"]["涂博士"] > result["scores"]["图博士"]
    assert seen == {"shape": (16000, 1), "sample_rate": 16000}


def test_moss_retranscriber_returns_focused_observation(tmp_path):
    audio = tmp_path / "mono.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)

    result = moss_audio_tools.retranscribe_window(
        str(audio),
        0.0,
        0.5,
        transcriber=lambda _path: "局部转写",
        domain_hints=["不会发送给MOSS"],
    )

    assert result["ok"] is True
    assert result["text"] == "局部转写"
    assert result["method"] == "same_model_focused_reobservation"
