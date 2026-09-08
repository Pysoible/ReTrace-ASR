from fastapi.testclient import TestClient
from pathlib import Path

from asr_agent import server
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


def test_moss_backend_never_acquires_qwen_model_name(tmp_path, monkeypatch):
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

    expected = {"backend": "moss-transcribe-diarize", "model": "unknown", "role": "first_pass"}
    assert body["provenance"]["first_pass"] == expected
    assert body["session"]["pipeline_provenance"]["first_pass"] == expected
    assert body["session"]["turns"][0]["meta"]["first_pass_identity"] == expected


def test_audio_sessions_share_memory_by_model_and_isolate_other_models(tmp_path, monkeypatch):
    results = iter([
        {"ok": True, "backend": "qwen-omni-vllm", "model": "Qwen3_Omni_30B",
         "chunks_text": ["甲"], "chunks": [{}], "uncertainties": [{}]},
        {"ok": True, "backend": "qwen-omni-vllm", "model": "Qwen3_Omni_30B",
         "chunks_text": ["乙"], "chunks": [{}], "uncertainties": [{}]},
        {"ok": True, "backend": "moss-transcribe-diarize", "model": "MOSS-Audio-7B",
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
