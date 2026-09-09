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


def test_moss_coverage_retranscriber_splits_long_window(tmp_path):
    audio = tmp_path / "mono.wav"
    sf.write(audio, np.zeros(12 * 16000, dtype=np.float32), 16000)
    calls = []

    def transcribe(path):
        calls.append(sf.info(path).duration)
        return f"第{len(calls)}段"

    result = moss_audio_tools.retranscribe_window(
        str(audio),
        0.0,
        12.0,
        transcriber=transcribe,
        recover_coverage=True,
    )

    assert result["ok"] is True
    assert result["text"] == "第1段第2段第3段"
    assert result["segmented"] is True
    assert result["window_count"] == 3
    assert calls == [5.0, 5.0, 2.0]


def test_moss_verifier_rejects_when_focused_transcript_matches_no_candidate(tmp_path):
    audio = tmp_path / "mono.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)

    result = moss_audio_tools.verify_candidates(
        str(audio),
        0.0,
        0.5,
        ["像像这些", "像这些"],
        transcriber=lambda _path: "这这样子是怎么着",
    )

    assert result["ok"] is False
    assert result["failure_code"] == "no_closed_set_match"
    assert result["focused_transcript"] == "这这样子是怎么着"


def test_moss_verifier_can_support_deleting_an_absent_inserted_span(tmp_path):
    audio = tmp_path / "mono.wav"
    sf.write(audio, np.zeros(32000, dtype=np.float32), 16000)

    result = moss_audio_tools.verify_candidates(
        str(audio),
        0.0,
        1.5,
        ["清高的爱拉", moss_audio_tools.DELETE_CANDIDATE],
        transcriber=lambda _path: "他曾经说这是一个说法",
    )

    assert result["ok"] is True
    assert result["scores"][moss_audio_tools.DELETE_CANDIDATE] > result["scores"]["清高的爱拉"]


def test_moss_verifier_does_not_delete_when_focused_observation_is_too_short(tmp_path):
    audio = tmp_path / "mono.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)

    result = moss_audio_tools.verify_candidates(
        str(audio),
        0.0,
        0.5,
        ["清高的爱拉", moss_audio_tools.DELETE_CANDIDATE],
        transcriber=lambda _path: "他说",
    )

    assert result["ok"] is False
    assert result["failure_code"] == "no_closed_set_match"
