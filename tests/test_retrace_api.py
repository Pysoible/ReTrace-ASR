from fastapi.testclient import TestClient
from pathlib import Path

from asr_agent import server
from asr_agent.context_judge import ContextJudgment
from asr_agent.retrace import ReTraceService
from asr_agent.server import create_app


def test_turn_api_returns_versioned_session_and_audit(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/sessions/s/turns",
            json={"turn_id": "t1", "text": "图博士来了", "confidence": {"图博士": 0.2}},
        )
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        assert response.json()["session"]["turns"][0]["turn_id"] == "t1"
        assert client.get("/api/sessions/s").status_code == 200


def test_integrations_status_endpoint(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/integrations/status")
    assert response.status_code == 200
    body = response.json()
    assert "qwen_asr" in body
    assert "enabled" in body["qwen_asr"]
    assert body["retrace_policy"] == {
        "fast_normal_turns": False,
        "strict_revision": True,
        "acoustic_disagreement": True,
        "homophone_discovery_available": True,
    }


def test_moss_backend_result_marks_audio_turn_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        server,
        "transcribe_audio",
        lambda _audio: {
            "ok": True,
            "backend": "moss-transcribe-diarize",
            "chunks_text": ["测试文本"],
            "chunks": [{"start_sec": 0.0, "end_sec": 1.0}],
            "uncertainties": [{}],
        },
    )

    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/sessions/s/audio", json={"audio": "/tmp/moss.wav"})

    assert response.status_code == 200
    assert response.json()["session"]["turns"][0]["source"] == "moss"


def test_moss_backend_uses_canonical_model_name_without_qwen_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_MODEL_PATH", "/models/Qwen3_Omni_30B")
    monkeypatch.setattr(
        server,
        "transcribe_audio",
        lambda _audio: {
            "ok": True,
            "backend": "moss-transcribe-diarize",
            "chunks_text": ["测试文本"],
            "chunks": [{"start_sec": 0.0, "end_sec": 1.0}],
            "uncertainties": [{}],
        },
    )

    with TestClient(create_app(tmp_path)) as client:
        body = client.post("/api/sessions/s/audio", json={"audio": "/tmp/moss.wav"}).json()

    expected = {
        "backend": "moss-transcribe-diarize",
        "model": "MOSS-Transcribe-Diarize",
        "role": "first_pass",
    }
    assert body["provenance"]["first_pass"] == expected
    assert body["session"]["pipeline_provenance"]["first_pass"] == expected
    assert body["session"]["turns"][0]["meta"]["first_pass_identity"] == expected


def test_moss_analysis_uses_bounded_windows_without_merging_turns(tmp_path, monkeypatch):
    calls = []

    def judge(*, current_turn, **_):
        calls.append(
            {
                "turn_id": current_turn.turn_id,
                "window": list(current_turn.meta.get("analysis_window_turn_ids") or []),
                "session_complete": bool(current_turn.meta.get("session_complete")),
            }
        )
        return ContextJudgment("CONSISTENT", 0.9)

    chunks = [
        {"start_sec": float(index), "end_sec": float(index + 1), "speaker": "S01"}
        for index in range(21)
    ]
    monkeypatch.setenv("ASR_JUDGE_WINDOW_MAX_TURNS", "10")
    monkeypatch.setenv("ASR_JUDGE_WINDOW_MAX_CHARS", "10000")
    monkeypatch.setenv("ASR_JUDGE_WINDOW_MAX_AUDIO_SEC", "10000")
    monkeypatch.setattr(
        server,
        "transcribe_audio",
        lambda _audio: {
            "ok": True,
            "backend": "moss-transcribe-diarize",
            "model": "MOSS-Transcribe-Diarize",
            "chunks_text": [f"第{index}句" for index in range(21)],
            "chunks": chunks,
            "uncertainties": [{} for _ in chunks],
        },
    )
    service = ReTraceService(tmp_path / "service", context_judge=judge)

    with TestClient(create_app(tmp_path / "app", service=service)) as client:
        body = client.post("/api/sessions/s/audio", json={"audio": "/tmp/moss.wav"}).json()

    assert len(body["session"]["turns"]) == 21
    assert [len(item["window"]) for item in calls] == [10, 10, 1]
    assert calls[-1]["session_complete"] is True


def test_runtime_coverage_is_derived_for_old_cached_moss_payload(monkeypatch):
    monkeypatch.setenv("MOSS_MIN_SEGMENT_CHAR_DENSITY", "2.0")
    payload = {
        "ok": True,
        "backend": "moss-transcribe-diarize",
        "chunks_text": ["设置的呀"],
        "chunks": [{"start_sec": 0.0, "end_sec": 12.0}],
        "uncertainties": [{}],
    }

    derived = server._derive_runtime_asr_signals(payload)

    assert "coverage" not in payload["uncertainties"][0]
    assert derived["uncertainties"][0]["coverage"]["truncated"] is True
    assert derived["uncertainties"][0]["coverage"]["detector"] == "moss_segment_char_density"


def test_moss_coverage_anomaly_inside_window_gets_its_own_agent_action(tmp_path, monkeypatch):
    calls = []

    def judge(*, current_turn, **_):
        calls.append((current_turn.turn_id, list(current_turn.meta.get("analysis_window_turn_ids") or [])))
        return ContextJudgment("CONSISTENT", 0.9)

    chunks = [
        {"start_sec": float(index * 7), "end_sec": float(index * 7 + 7), "speaker": "S01"}
        for index in range(12)
    ]
    chunk_texts = ["这是正常速度的一整段完整转录文本内容" for _ in chunks]
    chunk_texts[4] = "设置的呀"
    uncertainties = [{} for _ in chunks]
    uncertainties[4] = {"coverage": {"truncated": True, "char_density": 1.0}}
    monkeypatch.setenv("ASR_JUDGE_WINDOW_MAX_TURNS", "10")
    monkeypatch.setenv("ASR_JUDGE_WINDOW_MAX_CHARS", "10000")
    monkeypatch.setenv("ASR_JUDGE_WINDOW_MAX_AUDIO_SEC", "10000")
    monkeypatch.setattr(
        server,
        "transcribe_audio",
        lambda _audio: {
            "ok": True,
            "backend": "moss-transcribe-diarize",
            "model": "MOSS-Transcribe-Diarize",
            "chunks_text": chunk_texts,
            "chunks": chunks,
            "uncertainties": uncertainties,
        },
    )
    service = ReTraceService(
        tmp_path / "service",
        context_judge=judge,
        audio_retranscriber=lambda **_: {"ok": True, "text": "第四句正常文本"},
    )

    with TestClient(create_app(tmp_path / "app", service=service)) as client:
        client.post("/api/sessions/s/audio", json={"audio": "/tmp/moss.wav"}).raise_for_status()

    assert calls == [
        ("t005", ["t005"]),
        ("t010", [f"t{index:03d}" for index in range(1, 11)]),
        ("t012", ["t011", "t012"]),
    ]


def test_default_moss_service_uses_only_moss_audio_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "moss-transcribe-diarize")

    with TestClient(create_app(tmp_path)) as client:
        service = client.app.state.service

    assert service.resolver.verifier_identity.family == "moss"
    assert service.relistener_identity.family == "moss"
    assert service.resolver.audio_verifier.__module__.endswith("moss_audio_tools")
    assert service.audio_retranscriber.__module__.endswith("moss_audio_tools")


def test_moss_retrace_provenance_names_moss_audio_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_BACKEND", "moss-transcribe-diarize")
    monkeypatch.setattr(
        server,
        "transcribe_audio",
        lambda _audio: {
            "ok": True,
            "backend": "moss-transcribe-diarize",
            "model": "MOSS-Transcribe-Diarize",
            "chunks_text": ["完整转写"],
            "chunks": [{"start_sec": 0.0, "end_sec": 1.0}],
            "uncertainties": [{}],
        },
    )

    with TestClient(create_app(tmp_path)) as client:
        body = client.post("/api/sessions/s/audio", json={"audio": "/tmp/missing.wav"}).json()

    assert body["provenance"]["first_pass"]["model"] == "MOSS-Transcribe-Diarize"
    assert body["provenance"]["targeted_verifier"]["model"] == "MOSS-Transcribe-Diarize"
    assert body["provenance"]["open_relistener"]["model"] == "MOSS-Transcribe-Diarize"
    assert body["provenance"]["context_judge"]["backend"] == "deepseek-api"


def test_baseline_and_retrace_reuse_same_complete_first_pass_artifact(tmp_path, monkeypatch):
    calls = []
    audio = tmp_path / "sample.flac"
    audio.write_bytes(b"test audio bytes")

    def transcribe(path):
        calls.append(path)
        return {
            "ok": True,
            "backend": "moss-transcribe-diarize",
            "model": "MOSS-Transcribe-Diarize",
            "chunks_text": ["第一句", "第二句"],
            "chunks": [
                {"start_sec": 0.0, "end_sec": 1.0},
                {"start_sec": 1.0, "end_sec": 2.0},
            ],
            "uncertainties": [{}, {}],
            "completeness": {"truncated": False, "coverage_ratio": 1.0},
        }

    monkeypatch.setenv("ASR_BACKEND", "moss-transcribe-diarize")
    monkeypatch.setattr(server, "transcribe_audio", transcribe)
    service = ReTraceService(
        tmp_path / "service",
        context_judge=lambda **_: ContextJudgment("CONSISTENT", 0.9),
    )

    with TestClient(create_app(tmp_path / "app", service=service)) as client:
        baseline = client.post(
            "/api/sessions/baseline/audio",
            json={"audio": str(audio), "experiment_mode": "baseline"},
        ).json()
        retrace = client.post(
            "/api/sessions/retrace/audio",
            json={"audio": str(audio), "experiment_mode": "retrace"},
        ).json()

    assert len(calls) == 1
    assert baseline["first_pass_artifact_id"] == retrace["first_pass_artifact_id"]
    assert baseline["first_pass_cache_hit"] is False
    assert retrace["first_pass_cache_hit"] is True
    assert "".join(turn["raw_text"] for turn in retrace["session"]["turns"]) == baseline["transcript"]


def test_audio_sessions_share_memory_by_model_and_isolate_other_models(tmp_path, monkeypatch):
    results = iter([
        {"ok": True, "backend": "qwen-omni-vllm", "model": "Qwen3_Omni_30B",
         "chunks_text": ["甲"], "chunks": [{}], "uncertainties": [{}]},
        {"ok": True, "backend": "qwen-omni-vllm", "model": "Qwen3_Omni_30B",
         "chunks_text": ["乙"], "chunks": [{}], "uncertainties": [{}]},
        {"ok": True, "backend": "moss-transcribe-diarize", "model": "MOSS-Transcribe-Diarize",
         "chunks_text": ["丙"], "chunks": [{}], "uncertainties": [{}]},
    ])
    monkeypatch.setenv("ASR_EXPERIMENT_NAMESPACE", "scope-test")
    monkeypatch.setattr(server, "transcribe_audio", lambda _audio: next(results))

    with TestClient(create_app(tmp_path)) as client:
        qwen_a = client.post("/api/sessions/a/audio", json={"audio": "/tmp/a.wav"}).json()
        qwen_b = client.post("/api/sessions/b/audio", json={"audio": "/tmp/b.wav"}).json()
        moss = client.post("/api/sessions/c/audio", json={"audio": "/tmp/c.wav"}).json()

    assert qwen_a["session"]["memory_scope"] == qwen_b["session"]["memory_scope"]
    assert qwen_a["session"]["memory_scope"] != moss["session"]["memory_scope"]
    assert qwen_a["session"]["memory_scope"].startswith("model--scope-test--")


def test_baseline_mode_never_invokes_retrace_or_memory(tmp_path, monkeypatch):
    class ForbiddenService:
        def __getattr__(self, name):
            raise AssertionError(f"baseline invoked ReTrace service method: {name}")

    monkeypatch.setattr(
        server,
        "transcribe_audio",
        lambda _audio: {
            "ok": True,
            "backend": "qwen-omni-vllm",
            "model": "Qwen3_Omni_30B",
            "final_text": "第一句",
            "chunks_text": ["第一句"],
            "chunks": [{"start_sec": 0.0, "end_sec": 1.0}],
            "uncertainties": [{}],
        },
    )

    with TestClient(create_app(tmp_path, service=ForbiddenService())) as client:
        body = client.post(
            "/api/sessions/base/audio",
            json={"audio": "/tmp/a.wav", "experiment_mode": "baseline"},
        ).json()

    assert body["experiment_mode"] == "baseline"
    assert body["transcript"] == "第一句"
    assert body["revisions"] == []
    assert set(body["provenance"]) == {"first_pass"}
    assert body["stage_timings_ms"]["first_pass_asr"] >= 0
    assert body["stage_timings_ms"]["request_total"] >= body["stage_timings_ms"]["first_pass_asr"]
    assert not list(tmp_path.rglob("*.json"))


def test_text_only_evidence_does_not_create_an_undoable_revision(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        client.post("/api/sessions/s/turns", json={"turn_id": "t1", "text": "图博士来了", "confidence": {"图博士": 0.2}})
        later = client.post("/api/sessions/s/turns", json={"turn_id": "t2", "text": "负责人到了"}).json()
        assert later["status"] == "queued"
        state = client.get("/api/sessions/s").json()["session"]
        assert state["turns"][0]["raw_text"] == "图博士来了"
        assert state["turns"][0]["current_text"] == "图博士来了"


def test_audio_upload_endpoint_rejects_empty_file(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.post(
        "/api/sessions/s/audio/upload",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_manual_confirmation_endpoint_is_not_exposed(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.post(
        "/api/sessions/s/turns",
        json={
            "turn_id": "t1", "text": "图博士到了", "risk": "high",
            "confidence": {"图博士": 0.2},
            "text_candidates": {"图博士": ["图博士", "涂博士"]},
        },
    )

    response = client.post("/api/sessions/s/hypotheses/t1/confirm", json={"span": "图博士", "candidate": "涂博士"})
    assert response.status_code in {404, 405}


def test_manual_undo_endpoint_is_not_exposed():
    source = Path("backend/asr_agent/server.py").read_text(encoding="utf-8")
    assert '"/api/sessions/{session_id}/revisions/{event_id}/undo"' not in source
