from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr_agent.retrace import EntityProfile, ReTraceService


class EntityRequest(BaseModel):
    entities: list[dict[str, Any]]


class TurnRequest(BaseModel):
    turn_id: str
    text: str
    confidence: dict[str, float] = Field(default_factory=dict)
    text_candidates: dict[str, list[str]] = Field(default_factory=dict)
    entity_candidate_ids: dict[str, list[str]] = Field(default_factory=dict)


def create_app(workspace: Path | None = None) -> FastAPI:
    root = workspace or Path.cwd() / "retrace_state"
    service = ReTraceService(root / "sessions")
    app = FastAPI(title="ReTrace-ASR")
    app.state.service = service

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "ReTrace-ASR"}

    @app.put("/api/sessions/{session_id}/entities")
    def upsert_entities(session_id: str, request: EntityRequest) -> dict[str, Any]:
        return {"session": service.upsert_entities(session_id, [EntityProfile.from_dict(item) for item in request.entities])}

    @app.post("/api/sessions/{session_id}/turns")
    def process_turn(session_id: str, request: TurnRequest) -> dict[str, Any]:
        try:
            return service.process_turn(session_id, request.turn_id, request.text, confidence=request.confidence, text_candidates=request.text_candidates, entity_candidate_ids=request.entity_candidate_ids)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return {"session": service.get_session(session_id)}

    frontend = Path(__file__).parents[2] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
