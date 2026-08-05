from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr_agent.integrations.deepseek import correct_text_with_deepseek, deepseek_status
from asr_agent.integrations.domainterms import _load_env
from asr_agent.integrations.evolve import evolve_audio
from asr_agent.integrations.qwen_asr import asr_status, preload_engine, transcribe_audio
from asr_agent.retrace import EntityProfile, ReTraceService

# Load project .env (ASR_AUDIO_ENABLED, ASR_MODEL_PATH, ...) before reading status.
_load_env(Path(__file__).resolve().parents[2] / ".env")

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
    correct_with_llm: bool = False


class AudioTurnRequest(BaseModel):
    turn_id: str
    audio: str
    two_pass: bool = True
    domain: str | None = None
    top_k: int | None = None
    confidence: dict[str, float] = Field(default_factory=dict)
    text_candidates: dict[str, list[str]] = Field(default_factory=dict)
    entity_candidate_ids: dict[str, list[str]] = Field(default_factory=dict)
    use_llm: bool = True
    correct_with_llm: bool = False
    evolve: bool = False
    max_iters: int = 3
    dry_run_dict: bool = False


class CorrectRequest(BaseModel):
    text: str
    terms: list[str] = Field(default_factory=list)
    domain: str | None = None


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


def create_app(workspace: Path | None = None) -> FastAPI:
    root = workspace or Path.cwd() / "retrace_state"
    service = ReTraceService(root / "sessions")
    upload_dir = root / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="ReTrace-ASR")
    app.state.service = service
    app.state.upload_dir = upload_dir

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "ReTrace-ASR"}

    @app.get("/api/integrations/status")
    def integrations_status() -> dict[str, Any]:
        return {"deepseek": deepseek_status(), "qwen_asr": asr_status()}

    @app.post("/api/integrations/qwen/preload")
    def qwen_preload() -> dict[str, Any]:
        try:
            return preload_engine()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/integrations/deepseek/correct")
    def deepseek_correct(request: CorrectRequest) -> dict[str, Any]:
        return correct_text_with_deepseek(request.text, terms=request.terms, domain=request.domain)

    @app.put("/api/sessions/{session_id}/entities")
    def upsert_entities(session_id: str, request: EntityRequest) -> dict[str, Any]:
        return {"session": service.upsert_entities(session_id, [EntityProfile.from_dict(item) for item in request.entities])}

    @app.post("/api/sessions/{session_id}/turns")
    def process_turn(session_id: str, request: TurnRequest) -> dict[str, Any]:
        try:
            return service.process_turn(
                session_id,
                request.turn_id,
                request.text,
                confidence=request.confidence,
                text_candidates=request.text_candidates,
                entity_candidate_ids=request.entity_candidate_ids,
                use_llm=request.use_llm,
                correct_with_llm=request.correct_with_llm,
                source="text",
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def _build_session_from_asr(
        session_id: str,
        *,
        audio_path: str,
        asr: dict[str, Any],
        use_llm: bool,
        correct_with_llm: bool,
        source: str = "qwen-omni",
        mode: str = "audio-session",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One long audio → one session; each ASR chunk → one turn in that session."""
        if not asr.get("ok"):
            raise HTTPException(status_code=503, detail=asr.get("error") or "Qwen ASR failed")

        parts = list(asr.get("pass2_chunks") or asr.get("pass1_chunks") or [])
        if not parts:
            text = str(asr.get("final_text") or asr.get("pass1_text") or "").strip()
            if text:
                parts = [text]
        if not parts:
            raise HTTPException(status_code=502, detail="Qwen ASR returned empty text")

        chunk_meta = list(asr.get("chunks") or [])
        bound_session = _audio_session_id(audio_path, session_id)
        service.reset_session(bound_session)

        pass1_parts = list(asr.get("pass1_chunks") or [])
        revisions: list[dict[str, Any]] = []
        for index, text in enumerate(parts):
            turn_id = f"t{index + 1:03d}"
            chunk = chunk_meta[index] if index < len(chunk_meta) else {}
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
                    use_llm=use_llm and index > 0,
                    correct_with_llm=correct_with_llm,
                    source=source,
                    meta={
                        "audio_path": audio_path,
                        "chunk_index": index,
                        "start_sec": start,
                        "end_sec": end,
                        "pass1_text": pass1_parts[index] if index < len(pass1_parts) else None,
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
        if extra:
            out.update(extra)
        return out

    def _run_audio_session(
        session_id: str,
        *,
        audio_path: str,
        two_pass: bool,
        domain: str | None,
        top_k: int | None,
        use_llm: bool,
        correct_with_llm: bool,
        evolve: bool = False,
        max_iters: int = 3,
        dry_run_dict: bool = False,
    ) -> dict[str, Any]:
        if evolve:
            evolved = evolve_audio(
                audio_path,
                max_iters=max_iters,
                two_pass=two_pass,
                domain=domain,
                top_k=top_k,
                dry_run_dict=dry_run_dict,
            )
            if not evolved.get("ok"):
                raise HTTPException(status_code=503, detail=evolved.get("error") or "evolve failed")
            asr = evolved.get("asr") or {}
            return _build_session_from_asr(
                session_id,
                audio_path=audio_path,
                asr=asr,
                use_llm=use_llm,
                correct_with_llm=correct_with_llm,
                source="qwen-omni-evolve",
                mode="audio-session-evolve",
                extra={
                    "evolve": {
                        "iterations": evolved.get("iterations"),
                        "max_iters": evolved.get("max_iters"),
                        "evolved": evolved.get("evolved"),
                        "history": evolved.get("history") or [],
                        "final_text": evolved.get("final_text") or "",
                    }
                },
            )

        asr = transcribe_audio(audio_path, two_pass=two_pass, domain=domain, top_k=top_k)
        return _build_session_from_asr(
            session_id,
            audio_path=audio_path,
            asr=asr,
            use_llm=use_llm,
            correct_with_llm=correct_with_llm,
        )

    @app.post("/api/sessions/{session_id}/audio")
    def process_audio_turn(session_id: str, request: AudioTurnRequest) -> dict[str, Any]:
        return _run_audio_session(
            session_id,
            audio_path=request.audio,
            two_pass=request.two_pass,
            domain=request.domain,
            top_k=request.top_k,
            use_llm=request.use_llm,
            correct_with_llm=request.correct_with_llm,
            evolve=request.evolve,
            max_iters=request.max_iters,
            dry_run_dict=request.dry_run_dict,
        )

    @app.post("/api/sessions/{session_id}/audio/upload")
    async def process_audio_upload(
        session_id: str,
        file: UploadFile = File(...),
        turn_id: str = Form(""),  # unused: chunks become turns automatically
        two_pass: bool = Form(True),
        use_llm: bool = Form(True),
        correct_with_llm: bool = Form(False),
        domain: str | None = Form(None),
        evolve: bool = Form(False),
        max_iters: int = Form(3),
        dry_run_dict: bool = Form(False),
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
            two_pass=two_pass,
            domain=domain or None,
            top_k=None,
            use_llm=use_llm,
            correct_with_llm=correct_with_llm,
            evolve=evolve,
            max_iters=max_iters,
            dry_run_dict=dry_run_dict,
        )

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return {"session": service.get_session(session_id)}

    frontend = Path(__file__).parents[2] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
