from asr_agent.integrations.audio_verifier import verify_candidates


def test_audio_verifier_normalizes_only_closed_set_scores(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"placeholder")

    result = verify_candidates(
        str(audio), 0.0, 1.0, ["图博士", "涂博士"],
        runner=lambda **_: {"scores": {"图博士": 1.0, "涂博士": 3.0}},
    )

    assert result["ok"] is True
    assert result["scores"] == {"图博士": 0.25, "涂博士": 0.75}


def test_audio_verifier_rejects_unknown_candidate_scores(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"placeholder")

    result = verify_candidates(
        str(audio), 0.0, 1.0, ["图博士", "涂博士"],
        runner=lambda **_: {"scores": {"图博士": 0.5, "杜博士": 0.5}},
    )

    assert result["ok"] is False


def test_audio_verifier_accepts_selected_candidate_index(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"placeholder")

    result = verify_candidates(
        str(audio), 0.0, 1.0, ["清高的爱拉", "[DELETE]"],
        runner=lambda **_: {"choice": 1},
    )

    assert result["ok"] is True
    assert result["scores"]["[DELETE]"] == 1.0


def test_audio_verifier_accepts_selected_candidate_text(tmp_path):
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"placeholder")

    result = verify_candidates(
        str(audio), 0.0, 1.0, ["清高的爱拉", "[DELETE]"],
        runner=lambda **_: {"selected": "[DELETE]"},
    )

    assert result["ok"] is True
    assert result["scores"]["[DELETE]"] == 1.0
