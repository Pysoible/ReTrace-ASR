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


def test_moss_request_uses_transcription_completion_parameter(monkeypatch, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    seen = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "text": "[0.00][S01]完整文本[9.90]",
                    "usage": {"type": "duration", "seconds": 10.0},
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        seen["body"] = request.data.decode("utf-8", errors="ignore")
        return Response()

    monkeypatch.setenv("MOSS_MAX_COMPLETION_TOKENS", "32768")
    monkeypatch.setattr(moss_asr.urllib.request, "urlopen", fake_urlopen)

    result = moss_asr.transcribe_audio(str(audio))

    assert 'name="max_completion_tokens"' in seen["body"]
    assert 'name="max_new_tokens"' not in seen["body"]
    assert result["duration_sec"] == 10.0
    assert result["completeness"]["coverage_ratio"] == 0.99
    assert result["completeness"]["truncated"] is False


def test_moss_marks_silent_http_200_truncation(monkeypatch, tmp_path):
    payload = {
        "text": "[0.00][S01]只有前半段[37.00][8",
        "usage": {"type": "duration", "seconds": 100.0},
    }
    monkeypatch.setattr(moss_asr, "_post_transcription", lambda _: payload)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    result = moss_asr.transcribe_audio(str(audio))

    assert result["ok"] is False
    assert result["failure_code"] == "incomplete_first_pass"
    assert result["completeness"]["coverage_ratio"] == 0.37
    assert result["completeness"]["parse_tail"] == "[8"
