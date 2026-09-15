from asr_agent.integrations import asr_backend


def test_asr_backend_routes_moss_without_calling_qwen(monkeypatch):
    called = {"moss": False, "qwen": False}

    monkeypatch.setenv("ASR_BACKEND", "moss-transcribe-diarize")
    monkeypatch.setattr(
        asr_backend.moss_asr,
        "transcribe_audio",
        lambda audio: called.__setitem__("moss", True) or {"ok": True, "audio": audio},
    )
    monkeypatch.setattr(
        asr_backend.qwen_asr,
        "transcribe_audio",
        lambda _audio: called.__setitem__("qwen", True) or {"ok": False},
    )

    assert asr_backend.transcribe_audio("/tmp/a.wav") == {"ok": True, "audio": "/tmp/a.wav"}
    assert called == {"moss": True, "qwen": False}
