from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr_agent.integrations.deepseek import deepseek_status
from asr_agent.integrations.qwen_asr import (
    asr_status,
    preload_engine,
    shutdown_engines,
    stream_transcribe_audio,
    transcribe_audio,
)
from asr_agent.realtime import RealtimeAnalysisCoordinator
from asr_agent.retrace import ReTraceService

_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".pcm"}


class TurnRequest(BaseModel):
    turn_id: str
    text: str
    confidence: dict[str, float] = Field(default_factory=dict)
    text_candidates: dict[str, list[str]] = Field(default_factory=dict)
    nbest: list[str] = Field(default_factory=list)
    speaker: str | None = None
    audio_path: str | None = None
    start_sec: float | None = None
    end_sec: float | None = None


class AudioTurnRequest(BaseModel):
    audio: str


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


# --- Streaming audio-session events (Server-Sent Events) -------------------

_EVENT_STREAMS: dict[str, "queue.Queue[dict[str, Any] | None]"] = {}
_EVENT_STREAMS_LOCK = threading.Lock()


def _new_event_stream(session_id: str) -> "queue.Queue[dict[str, Any] | None]":
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    with _EVENT_STREAMS_LOCK:
        _EVENT_STREAMS[session_id] = events
    return events


def _drop_event_stream(session_id: str) -> None:
    with _EVENT_STREAMS_LOCK:
        _EVENT_STREAMS.pop(session_id, None)


def _publish(session_id: str, event: dict[str, Any] | None) -> None:
    with _EVENT_STREAMS_LOCK:
        events = _EVENT_STREAMS.get(session_id)
    if events is not None:
        events.put(event)


class _OrderedChunkProcessor:
    """Consume transcribed chunks strictly in index order from a worker thread.

    Transcriptions arrive concurrently from both GPUs; the processor buffers
    out-of-order results and emits turn `(index, payload)` in ascending order so
    the session timeline stays consistent.
    """

    def __init__(self, on_ready: Any) -> None:
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: dict[int, Any] = {}
        self._next = 0
        self._done = False

    def submit(self, index: int, payload: Any) -> None:
        with self._cond:
            self._pending[index] = payload
            self._cond.notify_all()

    def finish(self) -> None:
        with self._cond:
            self._done = True
            self._cond.notify_all()

    def run(self) -> None:
        while True:
            with self._cond:
                while self._next not in self._pending and not self._done:
                    self._cond.wait()
                if self._next in self._pending:
                    index = self._next
                    payload = self._pending.pop(index)
                    self._next += 1
                elif self._done:
                    break
                else:
                    continue
            self._on_ready(index, payload)


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
        shutdown_engines()
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

    @app.post("/api/sessions/{session_id}/turns", status_code=202)
    def process_turn(session_id: str, request: TurnRequest) -> dict[str, Any]:
        try:
            observed = service.observe_turn(
                session_id,
                request.turn_id,
                request.text,
                confidence=request.confidence,
                text_candidates=request.text_candidates,
                source="text",
                nbest=request.nbest,
                meta=_turn_meta(request),
            )
            coordinator.submit(session_id, request.turn_id, observed_version=observed["observed_version"])
            return observed
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    def _build_session_from_asr(
        session_id: str,
        *,
        audio_path: str,
        asr: dict[str, Any],
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
        # Each audio session keeps its own durable long-term memory file
        # (scope == session id), so switching sessions never loses or mixes
        # previously consolidated beliefs.
        service.reset_session(bound_session, memory_scope=bound_session)

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
    ) -> dict[str, Any]:
        asr = transcribe_audio(audio_path)
        return _build_session_from_asr(
            session_id,
            audio_path=audio_path,
            asr=asr,
        )

    def _stream_audio_session_background(
        session_id: str,
        *,
        audio_path: str,
        source: str = "qwen-omni",
    ) -> None:
        """Stream a long audio into a session turn-by-turn, publishing SSE events.

        Runs on a worker thread: chunks are transcribed in parallel across GPUs and
        the ordered processor analyses them one by one, pushing a ``turn`` event to
        the frontend as soon as each turn's judgment is ready.
        """
        # The event stream key must match the bound session id that the frontend
        # connects to (the upload request may have used a generic "demo"/"auto"
        # session id, while the derived bound_session is the real identifier).
        bound_session = _audio_session_id(audio_path, session_id)
        events = _new_event_stream(bound_session)
        try:
            service.reset_session(bound_session, memory_scope=bound_session)
            _publish(bound_session, {"type": "session", "session_id": bound_session})
        except Exception as exc:
            _publish(bound_session, {"type": "error", "message": f"初始化会话失败: {exc}"})
            events.put(None)
            _drop_event_stream(bound_session)
            return

        def _analyze_chunk(index: int, payload: Any) -> None:
            text, uncertainty, chunk = payload
            turn_id = f"t{index + 1:03d}"
            start = chunk.get("start_sec")
            end = chunk.get("end_sec")
            display = (text or "").strip()
            if start is not None and end is not None and display:
                display = f"[{float(start):.1f}-{float(end):.1f}] {display}"
            if not display:
                _publish(bound_session, {"type": "turn", "index": index, "turn_id": turn_id, "text": "", "session": service.get_session(bound_session)})
                return
            try:
                result = service.process_turn(
                    bound_session,
                    turn_id,
                    display,
                    confidence=dict(uncertainty.get("confidence") or {}),
                    text_candidates=dict(uncertainty.get("text_candidates") or {}),
                    source=source,
                    nbest=[],
                    meta={
                        "audio_path": audio_path,
                        "chunk_index": index,
                        "start_sec": start,
                        "end_sec": end,
                        "uncertainty": uncertainty,
                    },
                )
            except Exception as exc:
                _publish(bound_session, {"type": "error", "message": f"分析 turn {turn_id} 失败: {exc}"})
                return
            _publish(
                bound_session,
                {
                    "type": "turn",
                    "index": index,
                    "turn_id": turn_id,
                    "text": display,
                    "judgment": result.get("judgment"),
                    "revisions": result.get("revisions") or [],
                    "session": service.get_session(bound_session),
                },
            )

        processor = _OrderedChunkProcessor(_analyze_chunk)
        worker = threading.Thread(
            target=processor.run,
            name=f"retrace-order-{bound_session}",
            daemon=True,
        )
        worker.start()

        def _on_chunk(index: int, text: str, uncertainty: dict[str, Any], chunk: dict[str, Any]) -> None:
            processor.submit(index, (text, uncertainty, chunk))

        try:
            summary = stream_transcribe_audio(audio_path, _on_chunk)
            _publish(bound_session, {"type": "asr_done", "chunk_count": summary.get("chunk_count"), "duration_sec": summary.get("duration_sec")})
        except Exception as exc:
            _publish(bound_session, {"type": "error", "message": f"音频转写失败: {exc}"})
        finally:
            processor.finish()
            worker.join(timeout=120)
            _publish(bound_session, {"type": "done", "session_id": bound_session, "session": service.get_session(bound_session)})
            events.put(None)
            _drop_event_stream(bound_session)

    def _start_stream_audio_session(session_id: str, *, audio_path: str) -> dict[str, Any]:
        bound_session = _audio_session_id(audio_path, session_id)
        thread = threading.Thread(
            target=_stream_audio_session_background,
            args=(session_id,),
            kwargs={"audio_path": audio_path},
            name=f"retrace-audio-{bound_session}",
            daemon=True,
        )
        thread.start()
        return {
            "session_id": bound_session,
            "status": "processing",
            "stream": f"/api/sessions/{bound_session}/events",
        }

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(session_id: str) -> StreamingResponse:
        with _EVENT_STREAMS_LOCK:
            events = _EVENT_STREAMS.get(session_id)

        async def event_generator():
            try:
                while True:
                    item = await asyncio.to_thread(events.get)
                    if item is None:
                        break
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            finally:
                _drop_event_stream(session_id)

        if events is None:
            raise HTTPException(status_code=404, detail="没有进行中的音频会话")
        return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/api/sessions/{session_id}/audio")
    def process_audio_turn(session_id: str, request: AudioTurnRequest) -> dict[str, Any]:
        return _run_audio_session(
            session_id,
            audio_path=request.audio,
        )

    @app.post("/api/sessions/{session_id}/audio/upload")
    async def process_audio_upload(
        session_id: str,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        filename = _safe_filename(file.filename or "audio.wav")
        dest = upload_dir / f"{uuid.uuid4().hex}_{filename}"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="empty audio upload")
        dest.write_bytes(content)
        # Streaming mode: return immediately; the frontend follows the SSE stream
        # and renders each turn as soon as its judgment is ready.
        return _start_stream_audio_session(
            session_id,
            audio_path=str(dest),
        )

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        return {"session": service.get_session(session_id)}

    frontend = Path(__file__).parents[2] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
