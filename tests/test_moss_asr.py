import json
from pathlib import Path

from asr_agent.integrations import moss_asr


def test_parse_moss_transcript_into_retrace_chunks():
    raw = "[0.48][S01]大家好[1.66][2.10][S02]今天开会[3.25]"

    parsed = moss_asr.parse_moss_transcript(raw)

    assert parsed == {
        "text": "大家好今天开会",
        "chunks_text": ["大家好", "今天开会"],
        "chunks": [
            {"index": 0, "start_sec": 0.48, "end_sec": 1.66, "speaker": "S01"},
            {"index": 1, "start_sec": 2.1, "end_sec": 3.25, "speaker": "S02"},
        ],
    }


def test_transcribe_audio_posts_to_moss_endpoint(monkeypatch, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"text": "[0.00][S01]测试文本[1.00]"}).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        body = request.data.decode("utf-8", errors="ignore")
        assert "MOSS-Transcribe-Diarize" in body
        assert "response_format" in body
        return Response()

    monkeypatch.setenv("MOSS_TRANSCRIBE_URL", "http://127.0.0.1:8010/v1/audio/transcriptions")
    monkeypatch.setattr(moss_asr.urllib.request, "urlopen", fake_urlopen)

    result = moss_asr.transcribe_audio(str(audio))

    assert seen["url"] == "http://127.0.0.1:8010/v1/audio/transcriptions"
    assert result["ok"] is True
    assert result["backend"] == "moss-transcribe-diarize"
    assert result["model"] == "MOSS-Transcribe-Diarize"
    assert result["chunks_text"] == ["测试文本"]
    assert result["chunks"][0]["speaker"] == "S01"
