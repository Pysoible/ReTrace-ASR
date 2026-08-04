from fastapi.testclient import TestClient

from asr_agent.server import create_app


def test_turn_api_returns_versioned_session_and_audit(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.put("/api/sessions/s/entities", json={"entities": [{"entity_id": "lead", "name": "涂博士", "attributes": {"role": "负责人"}}]})
    response = client.post("/api/sessions/s/turns", json={"turn_id": "t1", "text": "图博士来了", "confidence": {"图博士": 0.2}, "text_candidates": {"图博士": ["图博士", "涂博士"]}, "entity_candidate_ids": {"图博士": ["lead"]}})
    assert response.status_code == 200
    assert response.json()["session"]["turns"][0]["turn_id"] == "t1"
    assert client.get("/api/sessions/s").status_code == 200
