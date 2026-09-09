import json
from pathlib import Path

from asr_agent.integrations import moss_asr
from asr_agent.integrations import gss_overlap


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


def test_parse_moss_transcript_marks_cross_speaker_timestamp_overlap(monkeypatch):
    monkeypatch.setenv("MOSS_OVERLAP_MIN_SEC", "0.25")
    raw = (
        "[0.00][S06]主说话人持续发言[3.00]"
        "[1.00][S02]另一位说话人插话[2.00]"
        "[3.10][S03]后续发言[4.00]"
    )

    parsed = moss_asr.parse_moss_transcript(raw)

    first, second, third = parsed["chunks"]
    assert first == {
        "index": 0,
        "start_sec": 0.0,
        "end_sec": 3.0,
        "speaker": "S06",
        "speakers": ["S06", "S02"],
        "overlap": True,
        "overlap_duration_sec": 1.0,
        "overlap_with_indices": [1],
    }
    assert second["speakers"] == ["S02", "S06"]
    assert second["overlap"] is True
    assert second["overlap_duration_sec"] == 1.0
    assert second["overlap_with_indices"] == [0]
    assert third == {"index": 2, "start_sec": 3.1, "end_sec": 4.0, "speaker": "S03"}


def test_parse_moss_transcript_ignores_short_cross_speaker_overlap(monkeypatch):
    monkeypatch.setenv("MOSS_OVERLAP_MIN_SEC", "0.25")
    raw = "[0.00][S01]第一句[1.10][1.00][S02]第二句[2.00]"

    parsed = moss_asr.parse_moss_transcript(raw)

    assert all("overlap" not in chunk for chunk in parsed["chunks"])


def test_moss_segment_coverage_routes_sparse_long_turn_to_redecode(monkeypatch):
    monkeypatch.setenv("MOSS_MIN_SEGMENT_CHAR_DENSITY", "2.0")

    uncertainties = moss_asr.segment_coverage_uncertainties(
        ["设置的呀", "这是一段正常速度的转录文本"],
        [
            {"start_sec": 0.0, "end_sec": 12.0},
            {"start_sec": 12.0, "end_sec": 17.0},
        ],
    )

    sparse = uncertainties[0]["coverage"]
    assert sparse["truncated"] is True
    assert sparse["reasons"] == ["low_transcript_density"]
    assert sparse["recommended_actions"] == ["RESEGMENT", "REDECODE"]
    assert uncertainties[1]["coverage"]["truncated"] is False


def test_moss_overlap_is_a_separate_audit_signal_not_coverage_truncation(monkeypatch):
    monkeypatch.setenv("MOSS_OVERLAP_MIN_SEC", "0.25")
    chunks = moss_asr.parse_moss_transcript(
        "[0.00][S01]第一位发言[2.00][1.00][S02]第二位插话[1.80]"
    )["chunks"]

    uncertainties = moss_asr.segment_coverage_uncertainties(
        ["第一位发言", "第二位插话"],
        chunks,
    )

    assert uncertainties[0]["coverage"]["truncated"] is False
    assert uncertainties[0]["overlap"] == {
        "detector": "moss_cross_speaker_timestamp_overlap",
        "detected": True,
        "duration_sec": 0.8,
        "speakers": ["S01", "S02"],
        "with_indices": [1],
        "automatic_revision_allowed": False,
        "recommended_actions": ["AUDIT_OVERLAP", "GUIDED_SOURCE_SEPARATION"],
    }


def test_gss_overlap_candidates_keep_only_local_substitutions():
    candidates = moss_asr.extract_gss_substitution_candidates(
        "宣传代业，代业，哦对，前期就做这个",
        "宣传单页，单页，哦对，前期就要做这个。宣传宣传单页，对。嗯。",
    )

    assert candidates == [
        {
            "span": "代业",
            "candidate": "单页",
            "source": "gss_overlap",
        }
    ]


def test_gss_overlap_candidates_reject_language_switch_and_insertions():
    assert moss_asr.extract_gss_substitution_candidates("对对对", "Good evening.") == []
    assert moss_asr.extract_gss_substitution_candidates("今天开会", "今天下午开会") == []


def test_moss_uncertainty_exposes_gss_span_candidates_without_authorizing_revision():
    uncertainties = moss_asr.segment_coverage_uncertainties(
        ["宣传代业，啊对"],
        [
            {
                "speaker": "S08",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "overlap": True,
                "speakers": ["S08", "S02"],
                "overlap_duration_sec": 0.8,
                "overlap_with_indices": [2],
                "gss_text": "宣传单页啊，对。",
            }
        ],
    )

    assert uncertainties[0]["text_candidates"] == {"代业": ["代业", "单页"]}
    assert uncertainties[0]["overlap"]["substitution_candidates"] == [
        {"span": "代业", "candidate": "单页", "source": "gss_overlap"}
    ]
    assert uncertainties[0]["overlap"]["automatic_revision_allowed"] is False


def test_gss_selects_largest_overlap_components():
    chunks = [
        {"index": 0, "overlap": True, "overlap_duration_sec": 2.0, "overlap_with_indices": [1]},
        {"index": 1, "overlap": True, "overlap_duration_sec": 2.0, "overlap_with_indices": [0]},
        {"index": 2, "overlap": True, "overlap_duration_sec": 0.5, "overlap_with_indices": [3]},
        {"index": 3, "overlap": True, "overlap_duration_sec": 0.5, "overlap_with_indices": [2]},
    ]

    positions, components = gss_overlap._selected_positions(chunks, 1)

    assert positions == [0, 1]
    assert components == [{"positions": [0, 1], "overlap_duration_sec": 2.0}]


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
