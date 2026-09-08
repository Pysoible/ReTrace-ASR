from __future__ import annotations

import asyncio
import importlib.util
import json
import os
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

from asr_agent.integrations.audio_verifier import verify_candidates
from asr_agent.integrations.deepseek import deepseek_status
from asr_agent.integrations.qwen_asr import (
    asr_status,
    preload_engine,
    read_asr_config,
    shutdown_engines,
    stream_transcribe_audio,
    transcribe_audio,
)
from asr_agent.model_identity import ModelIdentity, model_memory_scope
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


class AudioVerificationRequest(BaseModel):
    audio: str
    start_sec: float
    end_sec: float
    candidates: list[str]


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


def _groundtruth_roots(root: Path) -> list[Path]:
    """Return local reference roots, including the sibling AMI dataset."""
    return [root, root.parent, root.parent.parent]


def _groundtruth_hints(session_id: str, turns: list[dict[str, Any]]) -> set[str]:
    hints = {Path(str((turn.get("meta") or {}).get("audio_path") or "")).stem.lower() for turn in turns}
    hints.discard("")
    hints.add(session_id.lower())
    hints.update(
        hint.split("_", 1)[1]
        for hint in list(hints)
        if re.match(r"^[0-9a-f]{16,}_", hint)
    )
    return hints


def _groundtruth_audio_stems(turns: list[dict[str, Any]]) -> set[str]:
    stems: set[str] = set()
    for turn in turns:
        audio_path = str((turn.get("meta") or {}).get("audio_path") or "")
        if not audio_path:
            continue
        stem = Path(audio_path).stem.lower()
        stems.add(stem)
        if re.match(r"^[0-9a-f]{16,}_", stem):
            stems.add(stem.split("_", 1)[1])
    return stems


def _groundtruth_recording_hints(session_id: str, turns: list[dict[str, Any]]) -> set[str]:
    hints = _groundtruth_hints(session_id, turns)
    for hint in list(hints):
        if "_" in hint:
            hints.add(hint.split("_", 1)[0])
    return {hint for hint in hints if hint}


def _groundtruth_match_keys(value: str) -> set[str]:
    value = value.lower().strip()
    keys = {value}
    # AliMeeting references may be generated for a different clip duration,
    # e.g. R0015_M0135_3min.ref.json for a R0015_M0135_5min.wav upload.
    keys.update(re.sub(r"_(?:\d+(?:\.\d+)?)(?:min|sec|s)$", "", value) for _ in [0])
    return {key for key in keys if key}


def _parse_stm_reference(path: Path) -> list[dict[str, Any]]:
    utterances = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        text = re.sub(r"<sil>", "", fields[5], flags=re.IGNORECASE).strip()
        if not text or text in {"<ignore_time_segment_in_scoring>", "[*]", "[+]"}:
            continue
        utterances.append({
            "start": float(fields[3]),
            "end": float(fields[4]),
            "spk": fields[2],
            "text": text,
        })
    return utterances


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
    qwen_model = Path(read_asr_config().model_path).name
    qwen_first_pass_identity = ModelIdentity("qwen-omni-vllm", qwen_model, "first_pass")
    if service is None:
        service = ReTraceService(
            root / "sessions",
            verifier_identity=ModelIdentity(
                "qwen-omni-vllm", qwen_model, "targeted_verifier"
            ),
            relistener_identity=ModelIdentity(
                "qwen-omni-vllm", qwen_model, "open_relistener"
            ),
        )
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
        disabled = {"0", "false", "no", "off"}
        return {
            "qwen_asr": asr_status(),
            "deepseek": deepseek_status(),
            "retrace_policy": {
                "fast_normal_turns": os.getenv("ASR_FAST_NORMAL_TURNS", "0").strip().lower() not in disabled,
                "strict_revision": os.getenv("ASR_STRICT_REVISION", "1").strip().lower() not in disabled,
                "acoustic_disagreement": os.getenv("ASR_ACOUSTIC_DISAGREEMENT", "1").strip().lower() not in disabled,
                "homophone_discovery_available": importlib.util.find_spec("pypinyin") is not None,
            },
        }

    @app.post("/api/integrations/qwen/preload")
    def qwen_preload() -> dict[str, Any]:
        try:
            return preload_engine()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/integrations/qwen/verify")
    def qwen_verify(request: AudioVerificationRequest) -> dict[str, Any]:
        result = verify_candidates(
            request.audio,
            request.start_sec,
            request.end_sec,
            request.candidates,
        )
        if not result.get("ok"):
            raise HTTPException(status_code=502, detail=result.get("error") or "audio verification failed")
        return result

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
        first_pass = ModelIdentity.from_asr_result(asr)
        provenance = {"first_pass": first_pass.as_dict()}
        memory_scope = model_memory_scope(
            first_pass,
            namespace=os.getenv("ASR_EXPERIMENT_NAMESPACE", "default"),
        )
        service.reset_session(
            bound_session,
            memory_scope=memory_scope,
            pipeline_provenance=provenance,
        )

        uncertainties = list(asr.get("uncertainties") or [])
        nbest_by_chunk = list(asr.get("nbest") or [])
        routed_by_chunk: dict[int, list[dict[str, Any]]] = {}
        for item in asr.get("routed_speaker_segments") or []:
            try:
                routed_by_chunk.setdefault(int(item.get("chunk_index", -1)), []).append(item)
            except (TypeError, ValueError):
                continue
        revisions: list[dict[str, Any]] = []
        processed_turn_ids: list[str] = []
        for index, text in enumerate(parts):
            turn_id = f"t{index + 1:03d}"
            chunk = chunk_meta[index] if index < len(chunk_meta) else {}
            uncertainty = uncertainties[index] if index < len(uncertainties) else {}
            nbest = nbest_by_chunk[index] if index < len(nbest_by_chunk) else []
            start = chunk.get("start_sec")
            end = chunk.get("end_sec")
            display = text.strip()
            if not display:
                continue
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
                        "speaker_segments": uncertainty.get("speaker_segments") or [],
                        "speakers": chunk.get("speakers") or [],
                        "overlap": bool(chunk.get("overlap")),
                        "routing": chunk.get("routing"),
                        "routed_speaker_segments": routed_by_chunk.get(index, []),
                        "first_pass_identity": first_pass.as_dict(),
                    },
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            processed_turn_ids.append(turn_id)
            revisions.extend(result.get("revisions") or [])

        # The batch path has the complete session available. One bounded final
        # audit exposes later evidence without running another ASR pass.
        if processed_turn_ids:
            final_audit = service.analyze_turn(
                bound_session,
            processed_turn_ids[-1],
                session_complete=True,
            )
            revisions.extend(final_audit.get("revisions") or [])

        out: dict[str, Any] = {
            "session_id": bound_session,
            "session": service.get_session(bound_session),
            "revisions": revisions,
            "asr": asr,
            "turn_count": len(processed_turn_ids),
            "mode": mode,
            "provenance": provenance,
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
            source="moss" if asr.get("backend") == "moss-transcribe-diarize" else "qwen-omni",
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
            provenance = {"first_pass": qwen_first_pass_identity.as_dict()}
            memory_scope = model_memory_scope(
                qwen_first_pass_identity,
                namespace=os.getenv("ASR_EXPERIMENT_NAMESPACE", "default"),
            )
            service.reset_session(
                bound_session,
                memory_scope=memory_scope,
                pipeline_provenance=provenance,
            )
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
                        "speaker_segments": uncertainty.get("speaker_segments") or [],
                        "routed_speaker_segments": uncertainty.get("routed_speaker_segments") or [],
                        "speakers": chunk.get("speakers") or [],
                        "overlap": bool(chunk.get("overlap")),
                        "routing": chunk.get("routing"),
                        "first_pass_identity": qwen_first_pass_identity.as_dict(),
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
            from asr_agent.integrations.qwen_asr import _infer_speaker_segments, _truthy, parse_rttm, read_asr_config
            if summary.get("rttm") and _truthy(os.getenv("ASR_STREAM_SPEAKER_ROUTING", "0")):
                _publish(bound_session, {"type": "speaker_asr_started", "session": service.get_session(bound_session)})
                processor.finish()
                worker.join(timeout=120)
                intervals = parse_rttm(Path(summary["rttm"]))
                routed = _infer_speaker_segments(Path(audio_path), intervals, read_asr_config())
                for index, turn in enumerate(service.get_session(bound_session).get("turns") or []):
                    start = float((turn.get("meta") or {}).get("start_sec") or 0)
                    end = float((turn.get("meta") or {}).get("end_sec") or 0)
                    segments = [item for item in routed if start <= item["start_sec"] < end]
                    if segments:
                        service.update_turn_metadata(bound_session, str(turn["turn_id"]), {"routed_speaker_segments": segments})
                        _publish(bound_session, {"type": "speaker_update", "turn_id": turn["turn_id"], "session": service.get_session(bound_session)})
                    _publish(bound_session, {"type": "speaker_asr_finished", "session": service.get_session(bound_session)})
        except Exception as exc:
            _publish(bound_session, {"type": "error", "message": f"音频转写失败: {exc}"})
        finally:
            processor.finish()
            worker.join(timeout=120)
            try:
                completed_session = service.get_session(bound_session)
                completed_turns = completed_session.get("turns") or []
                if completed_turns:
                    service.analyze_turn(
                        bound_session,
                        str(completed_turns[-1]["turn_id"]),
                        session_complete=True,
                    )
            except Exception as exc:
                _publish(bound_session, {"type": "error", "message": f"完整 session 审计失败: {exc}"})
            _publish(bound_session, {"type": "done", "session_id": bound_session, "session": service.get_session(bound_session)})
            events.put(None)
            _drop_event_stream(bound_session)

    def _start_stream_audio_session(session_id: str, *, audio_path: str) -> dict[str, Any]:
        # Uploaded files can be submitted repeatedly with the same original
        # filename. Keep each upload isolated so an older background stream
        # cannot append late chunks into the new session's transcript.
        bound_session = f"{_audio_session_id(audio_path, session_id)}_{uuid.uuid4().hex[:8]}"
        thread = threading.Thread(
            target=_stream_audio_session_background,
            args=(bound_session,),
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

    @app.get("/api/sessions/{session_id}/groundtruth")
    def get_groundtruth(session_id: str) -> dict[str, Any]:
        """Return time-aligned reference utterances for frontend inspection."""
        current = service.get_session(session_id)
        turns = current.get("turns") or []
        hints = _groundtruth_hints(session_id, turns)
        audio_stems = _groundtruth_audio_stems(turns)
        recording_hints = _groundtruth_recording_hints(session_id, turns)
        hint_keys = {key for hint in hints | recording_hints for key in _groundtruth_match_keys(hint)}
        candidates = [
            path
            for base in _groundtruth_roots(root)
            for directory in (base / "alimeeting_eval", base / "retrace_state" / "alimeeting_eval")
            if directory.is_dir()
            for path in directory.glob("*.ref.json")
        ]
        ai_shell_reference_dirs = [
            Path(os.getenv("ASR_AISHELL4_REFERENCE_DIR", "")) if os.getenv("ASR_AISHELL4_REFERENCE_DIR") else None,
            Path("/home/ma-user/work/dataset/sjk_data/sdr_data/AI-SHELL-4/test/textgrid2txt"),
        ]
        candidates.extend(
            path
            for directory in ai_shell_reference_dirs
            if directory is not None and directory.is_dir()
            for path in directory.glob("*.txt")
            if path.is_file()
        )
        exact_audio_candidates = [
            path for path in sorted(set(candidates))
            if path.stem.lower() in audio_stems
        ]
        reference_path = exact_audio_candidates[0] if exact_audio_candidates else next(
            (path for path in sorted(set(candidates)) if any(
                path.stem.lower() in hint or hint in path.stem.lower() for hint in hint_keys
            )),
            None,
        )
        if reference_path is not None:
            try:
                payload = json.loads(reference_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            utterances = payload.get("utterances") if isinstance(payload, dict) else None
            if isinstance(utterances, list):
                return {"available": True, "source": str(reference_path), "source_type": "AliMeeting_JSON", "stem": payload.get("stem"), "utterances": utterances}

        if reference_path is not None and reference_path.suffix.lower() == ".txt":
            utterances = []
            for line in reference_path.read_text(encoding="utf-8").splitlines():
                fields = line.split("\t", 3)
                if len(fields) < 4:
                    continue
                try:
                    start, end = float(fields[0]), float(fields[1])
                except ValueError:
                    continue
                text = re.sub(r"<[^>]+>", "", fields[3]).strip()
                if text:
                    utterances.append({"start": start, "end": end, "spk": fields[2], "text": text})
            return {"available": True, "source": str(reference_path), "source_type": "AI-SHELL-4_TextGrid", "stem": reference_path.stem, "utterances": utterances}

        stm_candidates = [
            path
            for base in _groundtruth_roots(root)
            for path in (base / "AMI" / "STM").glob("*.stm")
            if path.is_file()
        ]
        stm_path = next((path for path in sorted(set(stm_candidates)) if path.stem.lower() in hint_keys), None)
        if stm_path is None:
            return {"available": False, "source": None, "utterances": []}
        return {
            "available": True,
            "source": str(stm_path),
            "source_type": "AMI_STM",
            "stem": stm_path.stem,
            "utterances": _parse_stm_reference(stm_path),
        }

    frontend = Path(__file__).parents[2] / "frontend" / "dist"
    if frontend.exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app


app = create_app()
