from __future__ import annotations

import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr_agent.integrations.deepseek import deepseek_status
from asr_agent.integrations.qwen_asr import asr_status, preload_engine, transcribe_audio
from asr_agent.realtime import RealtimeAnalysisCoordinator
from asr_agent.retrace import EntityProfile, ReTraceService

_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".pcm"}


class EntityRequest(BaseModel):
    entities: list[dict[str, Any]]


class TurnRequest(BaseModel):
    turn_id: str
    text: str
    confidence: dict[str, float] = Field(default_factory=dict)
    text_candidates: dict[str, list[str]] = Field(default_factory=dict)
    entity_candidate_ids: dict[str, list[str]] = Field(default_factory=dict)
    use_llm: bool = False
    nbest: list[str] = Field(default_factory=list)
    risk: str = "medium"
    speaker: str | None = None
    audio_path: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None


class AudioTurnRequest(BaseModel):
    turn_id: str
    audio: str
    use_llm: bool = True


def _turn_meta(request: TurnRequest) -> dict[str, Any] | None:
    meta: dict[str, Any] = {}
    if request.speaker:
        meta["speaker"] = request.speaker
    if request.audio_path:
        meta["audio_path"] = request.audio_path
    if request.start_sec is not None:
        meta["start_sec"] = request.start_sec
    if request.end_sec is not None:
        meta["end_sec"] = request.end_sec
    return meta or None


def _safe_filename(name: str) -> str:
    base = Path(name or "audio.wav").name
    cleaned = re.sub(r"[^A-Za-z0-9._\u4e00-\u9fff-]+", "_", base).strip("._") or "audio.wav"
    if Path(cleaned).suffix.lower() not in _AUDIO_SUFFIXES:
        cleaned = f"{cleaned}.wav"
    return cleaned


def _audio_session_id(audio_path: str, requested: str | None = None) -> str:
    """One long audio file maps to one session id (derived from the filename)."""
    requested = (requested or "").strip()
    if requested and requested.lower() not in {"auto", "demo", "default"}:
        return requested
    name = Path(audio_path).name
    # uploads look like "<uuid>_<original>"; prefer the original stem
    if "_" in name and re.match(r"^[0-9a-f]{16,}_", name):
        name = name.split("_", 1)[1]
    stem = Path(name).stem or "audio"
    safe = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff-]+", "_", stem).strip("_") or "audio"
    return f"audio_{safe[:96]}"


def create_app(
    workspace: Path | None = None,
    *,
    service: ReTraceService | None = None,
    coordinator: RealtimeAnalysisCoordinator | None = None,
) -> FastAPI:
    root = workspace or Path.cwd() / "retrace_state"
    service = service or ReTraceService(root / "sessions")
    coordinator = coordinator or RealtimeAnalysisCoordinator(service)
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        coordinator.close()

    app = FastAPI(title="ReTrace-ASR", lifespan=lifespan)
    app.state.service = service
    app.state.coordinator = coordinator
    app.state.upload_dir = upload_dir

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "ReTrace-ASR"}

    @app.get("/api/integrations/status")
    def integrations_status() -> dict[str, Any]:
        return {"qwen_asr": asr_status(), "deepseek": deepseek_status()}

    @app.post("/api/integrations/qwen/preload")
    def qwen_preload() -> dict[str, Any]:
        try:
            return preload_engine()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.put("/api/sessions/{session_id}/entities")
    def upsert_entities(session_id: str, request: EntityRequest) -> dict[str, Any]:
        return {"session": service.upsert_entities(session_id, [EntityProfile.from_dict(item) for item in request.entities])}

    @app.post("/api/sessions/{session_id}/turns", status_code=202)
    def process_turn(session_id: str, request: TurnRequest) -> dict[str, Any]:
        try:
            observed = service.observe_turn(
                session_id,
                request.turn_id,
                request.text,
                confidence=request.confidence,
                text_candidates=request.text_candidates,
                entity_candidate_ids=request.entity_candidate_ids,
                use_llm=request.use_llm,
                source="text",
                risk=request.risk,
                nbest=request.nbest,
                meta=_turn_meta(request),
            )
            coordinator.submit(session_id, request.turn_id)
            return observed
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def _build_session_from_asr(
        session_id: str,
        *,
        audio_path: str,
        asr: dict[str, Any],
        use_llm: bool,
        source: str = "qwen-omni",
        mode: str = "audio-session",
    ) -> dict[str, Any]:
        """One long audio → one session; each ASR chunk → one turn in that session."""
        if not asr.get("ok"):
            raise HTTPException(status_code=503, detail=asr.get("error") or "Qwen ASR failed")

        parts = list(asr.get("chunks_text") or [])
        if not parts:
            text = str(asr.get("final_text") or "").strip()
            if text:
                parts = [text]
        if not parts:
            raise HTTPException(status_code=502, detail="Qwen ASR returned empty text")

        chunk_meta = list(asr.get("chunks") or [])
        bound_session = _audio_session_id(audio_path, session_id)
        service.reset_session(bound_session)

        uncertainties = list(asr.get("uncertainties") or [])
        nbest_by_chunk = list(asr.get("nbest") or [])
        revisions: list[dict[str, Any]] = []
        for index, text in enumerate(parts):
            turn_id = f"t{index + 1:03d}"
            chunk = chunk_meta[index] if index < len(chunk_meta) else {}
            uncertainty = uncertainties[index] if index < len(uncertainties) else {}
            nbest = nbest_by_chunk[index] if index < len(nbest_by_chunk) else []
            start = chunk.get("start_sec")
            end = chunk.get("end_sec")
            display = text.strip()
            if start is not None and end is not None and display:
                display = f"[{float(start):.1f}-{float(end):.1f}] {display}"
            try:
                result = service.process_turn(
                    bound_session,
                    turn_id,
                    display,
                    confidence=dict(uncertainty.get("confidence") or {}),
                    text_candidates=dict(uncertainty.get("text_candidates") or {}),
                    entity_candidate_ids=dict(uncertainty.get("entity_candidate_ids") or {}),
                    use_llm=use_llm and index > 0,
                    source=source,
                    nbest=list(nbest) if isinstance(nbest, list) else [],
                    meta={
                        "audio_path": audio_path,
                        "chunk_index": index,
                        "start_sec": start,
                        "end_sec": end,
                        "uncertainty": uncertainty,
                    },
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            revisions.extend(result.get("revisions") or [])

        out: dict[str, Any] = {
            "session_id": bound_session,
            "session": service.get_session(bound_session),
            "revisions": revisions,
            "asr": asr,
            "turn_count": len(parts),
            "mode": mode,
        }
        return out

    def _run_audio_session(
        session_id: str,
        *,
        audio_path: str,
        use_llm: bool,
    ) -> dict[str, Any]:
        asr = transcribe_audio(audio_path)
        return _build_session_from_asr(
            session_id,
            audio_path=audio_path,
            asr=asr,
            use_llm=use_llm,
        )

    @app.post("/api/sessions/{session_id}/audio")
    def process_audio_turn(session_id: str, request: AudioTurnRequest) -> dict[str, Any]:
        return _run_audio_session(
            session_id,
            audio_path=request.audio,
            use_llm=request.use_llm,
        )

    @app.post("/api/sessions/{session_id}/audio/upload")
    async def process_audio_upload(
        session_id: str,
        file: UploadFile = File(...),
        turn_id: str = Form(""),  # unused: chunks become turns automatically
        use_llm: bool = Form(True),
    ) -> dict[str, Any]:
        del turn_id  # kept for form compatibility with older frontend
        filename = _safe_filename(file.filename or "audio.wav")
        dest = upload_dir / f"{uuid.uuid4().hex}_{filename}"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty audio upload")
        dest.write_bytes(content)
        return _run_audio_session(
            session_id,
            audio_path=str(dest),
            use_llm=use_llm,
        )

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return {"session": service.get_session(session_id)}

    frontend = Path(__file__).parents[2] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
