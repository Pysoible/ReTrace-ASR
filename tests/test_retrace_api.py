from fastapi.testclient import TestClient
from pathlib import Path

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
        data={"turn_id": "t1", "two_pass": "true", "use_llm": "false", "correct_with_llm": "false"},
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
