import pytest

from asr_agent.asr_artifacts import FirstPassArtifactRepository, artifact_key


def test_artifact_key_changes_with_audio_or_inference_contract(tmp_path):
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"same audio")
    identity = {"backend": "moss-transcribe-diarize", "model": "MOSS-Transcribe-Diarize"}

    first = artifact_key(audio, identity=identity, config={"max_completion_tokens": 32768})
    second = artifact_key(audio, identity=identity, config={"max_completion_tokens": 5120})

    assert first != second


def test_repository_refuses_incomplete_artifact(tmp_path):
    repository = FirstPassArtifactRepository(tmp_path)

    with pytest.raises(ValueError, match="incomplete"):
        repository.save("key", {"ok": False, "completeness": {"truncated": True}})


def test_repository_round_trips_complete_artifact(tmp_path):
    repository = FirstPassArtifactRepository(tmp_path)
    payload = {
        "ok": True,
        "backend": "moss-transcribe-diarize",
        "model": "MOSS-Transcribe-Diarize",
        "chunks_text": ["完整转写"],
        "completeness": {"truncated": False, "coverage_ratio": 0.99},
    }

    saved = repository.save("abc123", payload)
    loaded = repository.load("abc123")

    assert saved == loaded
    assert loaded["artifact_id"] == "abc123"
    assert loaded["payload"] == payload
