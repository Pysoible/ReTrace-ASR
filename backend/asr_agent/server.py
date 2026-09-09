from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from asr_agent.analysis_windows import AnalysisWindowPolicy, partition_turns
from asr_agent.asr_artifacts import FirstPassArtifactRepository, artifact_key, inference_contract
from asr_agent.integrations.audio_verifier import verify_candidates
from asr_agent.integrations.deepseek import context_judge_identity, deepseek_status
from asr_agent.integrations.asr_backend import (
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


def _derive_runtime_asr_signals(asr: dict[str, Any]) -> dict[str, Any]:
    """Attach code-versioned signals without mutating immutable ASR evidence."""
    derived = dict(asr)
    if derived.get("backend") != "moss-transcribe-diarize":
        return derived
    from asr_agent.integrations.moss_asr import segment_coverage_uncertainties

    coverage = segment_coverage_uncertainties(
        list(derived.get("chunks_text") or []),
        list(derived.get("chunks") or []),
    )
    existing = list(derived.get("uncertainties") or [])
    derived["uncertainties"] = [
        {
            **(existing[index] if index < len(existing) else {}),
            **signal,
        }
        for index, signal in enumerate(coverage)
    ]
    return derived


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
    experiment_mode: Literal["baseline", "retrace"] = "retrace"


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
    # Load project .env once during app construction. `read_asr_config()` is
    # intentionally retained here because it already implements the repo-local
    # dotenv fallback used by the existing Qwen path; MOSS and DeepSeek then see
    # the same environment without depending on the Qwen backend at inference.
    read_asr_config()
    configured_first_pass = ModelIdentity.from_asr_result(asr_status())
    verifier_identity = None
    relistener_identity = None
    audio_verifier = None
    audio_retranscriber = None
    if configured_first_pass.family == "qwen-omni":
        verifier_identity = ModelIdentity(
            configured_first_pass.backend,
            configured_first_pass.model,
            "targeted_verifier",
        )
        relistener_identity = ModelIdentity(
            configured_first_pass.backend,
            configured_first_pass.model,
            "open_relistener",
        )
    elif configured_first_pass.family == "moss":
        from asr_agent.integrations.moss_audio_tools import (
            retranscribe_window as moss_retranscribe_window,
            verify_candidates as moss_verify_candidates,
        )

        audio_verifier = moss_verify_candidates
        audio_retranscriber = moss_retranscribe_window
        verifier_identity = ModelIdentity(
            configured_first_pass.backend,
            configured_first_pass.model,
            "targeted_verifier",
        )
        relistener_identity = ModelIdentity(
            configured_first_pass.backend,
            configured_first_pass.model,
            "open_relistener",
        )
    if service is None:
        service = ReTraceService(
            root / "sessions",
            audio_verifier=audio_verifier,
            audio_retranscriber=audio_retranscriber,
            verifier_identity=verifier_identity,
            relistener_identity=relistener_identity,
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
    artifact_repository = FirstPassArtifactRepository(root / "first_pass_artifacts")
    app.state.first_pass_artifacts = artifact_repository

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "ReTrace-ASR"}

    @app.get("/api/integrations/status")
    def integrations_status() -> dict[str, Any]:
        disabled = {"0", "false", "no", "off"}
        first_pass_identity = ModelIdentity.from_asr_result(asr_status())
        return {
            "qwen_asr": asr_status(),
            "asr": asr_status(),
            "deepseek": deepseek_status(),
            "audio_tools": {
                "first_pass": first_pass_identity.as_dict(),
                "targeted_verifier": (
                    service.resolver.verifier_identity.as_dict()
                    if service.resolver.verifier_identity is not None else None
                ),
                "open_relistener": (
                    service.relistener_identity.as_dict()
                    if service.relistener_identity is not None else None
                ),
            },
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
        if mode != "baseline":
            provenance["context_judge"] = context_judge_identity()
            if (
                service.resolver.verifier_identity is not None
                and service.resolver.verifier_identity.family == first_pass.family
            ):
                provenance["targeted_verifier"] = service.resolver.verifier_identity.as_dict()
            if (
                service.relistener_identity is not None
                and service.relistener_identity.family == first_pass.family
            ):
                provenance["open_relistener"] = service.relistener_identity.as_dict()
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
        prepared_turns: list[dict[str, Any]] = []
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
            prepared_turns.append({
                "turn_id": turn_id,
                "text": display,
                "index": index,
                "chunk": chunk,
                "uncertainty": uncertainty,
                "nbest": nbest,
                "start_sec": start,
                "end_sec": end,
            })

        moss_windows: list[list[dict[str, Any]]] = []
        moss_window_ids_by_trigger: dict[str, list[str]] = {}
        if first_pass.family == "moss":
            moss_windows = partition_turns(
                prepared_turns,
                AnalysisWindowPolicy(
                    max_turns=int(os.getenv("ASR_JUDGE_WINDOW_MAX_TURNS", "20")),
                    max_chars=int(os.getenv("ASR_JUDGE_WINDOW_MAX_CHARS", "600")),
                    max_audio_sec=float(os.getenv("ASR_JUDGE_WINDOW_MAX_AUDIO_SEC", "90")),
                ),
            )
            moss_window_ids_by_trigger = {
                str(window[-1]["turn_id"]): [str(item["turn_id"]) for item in window]
                for window in moss_windows
            }

        for prepared in prepared_turns:
            turn_id = str(prepared["turn_id"])
            index = int(prepared["index"])
            chunk = dict(prepared["chunk"] or {})
            uncertainty = dict(prepared["uncertainty"] or {})
            nbest = prepared["nbest"]
            turn_meta = {
                "audio_path": audio_path,
                "chunk_index": index,
                "start_sec": prepared["start_sec"],
                "end_sec": prepared["end_sec"],
                "uncertainty": uncertainty,
                "speaker_segments": uncertainty.get("speaker_segments") or [],
                "speakers": chunk.get("speakers") or [],
                "overlap": bool(chunk.get("overlap")),
                "routing": chunk.get("routing"),
                "routed_speaker_segments": routed_by_chunk.get(index, []),
                "first_pass_identity": first_pass.as_dict(),
            }
            if turn_id in moss_window_ids_by_trigger:
                turn_meta["analysis_window_turn_ids"] = moss_window_ids_by_trigger[turn_id]
            elif bool((uncertainty.get("coverage") or {}).get("truncated")):
                # Coverage anomalies are sparse and must not be hidden merely
                # because they occur in the middle of a batched Judge window.
                # Keep this audit local so it adds one targeted re-listen, not a
                # return to expensive per-turn semantic judging.
                turn_meta["analysis_window_turn_ids"] = [turn_id]
            try:
                if first_pass.family == "moss":
                    service.observe_turn(
                        bound_session,
                        turn_id,
                        str(prepared["text"]),
                        confidence=dict(uncertainty.get("confidence") or {}),
                        text_candidates=dict(uncertainty.get("text_candidates") or {}),
                        source=source,
                        nbest=list(nbest) if isinstance(nbest, list) else [],
                        meta=turn_meta,
                    )
                    result = {"revisions": []}
                else:
                    result = service.process_turn(
                        bound_session,
                        turn_id,
                        str(prepared["text"]),
                        confidence=dict(uncertainty.get("confidence") or {}),
                        text_candidates=dict(uncertainty.get("text_candidates") or {}),
                        source=source,
                        nbest=list(nbest) if isinstance(nbest, list) else [],
                        meta=turn_meta,
                    )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            processed_turn_ids.append(turn_id)
            revisions.extend(result.get("revisions") or [])

        if first_pass.family == "moss":
            window_trigger_ids = {
                str(window[-1]["turn_id"])
                for window in moss_windows
            }
            coverage_trigger_ids = {
                str(prepared["turn_id"])
                for prepared in prepared_turns
                if bool(((prepared.get("uncertainty") or {}).get("coverage") or {}).get("truncated"))
            }
            ordered_trigger_ids = [
                str(prepared["turn_id"])
                for prepared in prepared_turns
                if str(prepared["turn_id"]) in window_trigger_ids | coverage_trigger_ids
            ]
            for trigger_id in ordered_trigger_ids:
                audit = service.analyze_turn(
                    bound_session,
                    trigger_id,
                    session_complete=trigger_id == str(prepared_turns[-1]["turn_id"]),
                )
                revisions.extend(audit.get("revisions") or [])
        # Qwen's existing streaming-style per-turn analysis is retained, with
        # one final audit that exposes later evidence to the last turn.
        elif processed_turn_ids:
            final_audit = service.analyze_turn(
                bound_session,
                processed_turn_ids[-1],
                session_complete=True,
            )
            revisions.extend(final_audit.get("revisions") or [])

        session_payload = service.get_session(bound_session)
        aggregate_timings: dict[str, float] = {}
        aggregate_calls: dict[str, int] = {}
        for turn in session_payload.get("turns") or []:
            meta = turn.get("meta") or {}
            for key, value in (meta.get("stage_timings_ms") or {}).items():
                aggregate_timings[str(key)] = aggregate_timings.get(str(key), 0.0) + float(value)
            for key, value in (meta.get("stage_call_counts") or {}).items():
                aggregate_calls[str(key)] = aggregate_calls.get(str(key), 0) + int(value)
        out: dict[str, Any] = {
            "session_id": bound_session,
            "session": session_payload,
            "revisions": revisions,
            "asr": asr,
            "turn_count": len(processed_turn_ids),
            "mode": mode,
            "experiment_mode": mode,
            "provenance": provenance,
            "stage_timings_ms": aggregate_timings,
            "stage_call_counts": aggregate_calls,
        }
        return out

    def _run_audio_session(
        session_id: str,
        *,
        audio_path: str,
        experiment_mode: Literal["baseline", "retrace"] = "retrace",
    ) -> dict[str, Any]:
        request_started = time.perf_counter()
        asr_started = time.perf_counter()
        selected_identity = ModelIdentity.from_asr_result(asr_status())
        first_pass_artifact_id: str | None = None
        first_pass_cache_hit = False
        audio_file = Path(audio_path).expanduser()
        if audio_file.exists():
            first_pass_artifact_id = artifact_key(
                audio_file,
                identity=selected_identity.as_dict(),
                config=inference_contract(selected_identity.as_dict()),
            )
            artifact = artifact_repository.load(first_pass_artifact_id)
            if artifact is not None:
                asr = dict(artifact["payload"])
                first_pass_cache_hit = True
            else:
                asr = transcribe_audio(audio_path)
                if asr.get("ok") and not (asr.get("completeness") or {}).get("truncated"):
                    artifact_repository.save(first_pass_artifact_id, asr)
        else:
            asr = transcribe_audio(audio_path)
        # Cached first-pass artifacts intentionally remain immutable. Coverage
        # routing is a derived Agent signal, so recompute it under the current
        # code/config even when the transcript itself comes from an older cache.
        asr = _derive_runtime_asr_signals(asr)
        first_pass_ms = (time.perf_counter() - asr_started) * 1000.0
        if experiment_mode == "baseline":
            if not asr.get("ok"):
                raise HTTPException(status_code=503, detail=asr.get("error") or "ASR failed")
            identity = ModelIdentity.from_asr_result(asr)
            parts = [str(item).strip() for item in (asr.get("chunks_text") or []) if str(item).strip()]
            transcript = str(asr.get("final_text") or "").strip() or "".join(parts)
            if not transcript:
                raise HTTPException(status_code=502, detail="ASR returned empty text")
            return {
                "session_id": _audio_session_id(audio_path, session_id),
                "experiment_mode": "baseline",
                "mode": "baseline",
                "transcript": transcript,
                "turn_count": len(parts) or 1,
                "revisions": [],
                "asr": asr,
                "first_pass_artifact_id": first_pass_artifact_id,
                "first_pass_cache_hit": first_pass_cache_hit,
                "provenance": {"first_pass": identity.as_dict()},
                "stage_timings_ms": {
                    "first_pass_asr": first_pass_ms,
                    "request_total": (time.perf_counter() - request_started) * 1000.0,
                },
                "stage_call_counts": {"first_pass_asr": 1, "request_total": 1},
            }
        result = _build_session_from_asr(
            session_id,
            audio_path=audio_path,
            asr=asr,
            source="moss" if asr.get("backend") == "moss-transcribe-diarize" else "qwen-omni",
            mode="retrace",
        )
        result["stage_timings_ms"]["first_pass_asr"] = first_pass_ms
        result["stage_timings_ms"]["request_total"] = (time.perf_counter() - request_started) * 1000.0
        result["stage_call_counts"]["first_pass_asr"] = 1
        result["stage_call_counts"]["request_total"] = 1
        result["first_pass_artifact_id"] = first_pass_artifact_id
        result["first_pass_cache_hit"] = first_pass_cache_hit
        return result

    def _stream_audio_session_background(
        session_id: str,
        *,
        audio_path: str,
        source: str = "qwen-omni",
        experiment_mode: Literal["baseline", "retrace"] = "retrace",
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
        baseline_turns: list[dict[str, Any]] = []
        try:
            first_pass_identity = ModelIdentity.from_asr_result(asr_status())
            provenance = {"first_pass": first_pass_identity.as_dict()}
            if experiment_mode == "retrace":
                memory_scope = model_memory_scope(
                    first_pass_identity,
                    namespace=os.getenv("ASR_EXPERIMENT_NAMESPACE", "default"),
                )
                service.reset_session(
                    bound_session,
                    memory_scope=memory_scope,
                    pipeline_provenance=provenance,
                )
            _publish(bound_session, {
                "type": "session",
                "session_id": bound_session,
                "experiment_mode": experiment_mode,
                "provenance": provenance,
            })
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
            if experiment_mode == "baseline":
                if display:
                    baseline_turns.append({
                        "turn_id": turn_id,
                        "raw_text": display,
                        "current_text": display,
                        "source": source,
                        "meta": {"start_sec": start, "end_sec": end, "chunk_index": index},
                    })
                _publish(bound_session, {
                    "type": "turn", "index": index, "turn_id": turn_id,
                    "text": display, "experiment_mode": "baseline", "revisions": [],
                })
                return
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
                        "first_pass_identity": first_pass_identity.as_dict(),
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
            speaker_routing_enabled = os.getenv("ASR_STREAM_SPEAKER_ROUTING", "0").strip().lower() in {"1", "true", "yes", "on"}
            if (
                experiment_mode == "retrace"
                and first_pass_identity.family == "qwen-omni"
                and summary.get("rttm")
                and speaker_routing_enabled
            ):
                from asr_agent.integrations.qwen_asr import _infer_speaker_segments, parse_rttm, read_asr_config

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
            if experiment_mode == "baseline":
                transcript = "".join(turn["raw_text"] for turn in baseline_turns)
                _publish(bound_session, {
                    "type": "done",
                    "session_id": bound_session,
                    "experiment_mode": "baseline",
                    "transcript": transcript,
                    "turns": baseline_turns,
                    "revisions": [],
                    "provenance": provenance,
                })
                events.put(None)
                _drop_event_stream(bound_session)
                return
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

    def _start_stream_audio_session(
        session_id: str,
        *,
        audio_path: str,
        experiment_mode: Literal["baseline", "retrace"] = "retrace",
    ) -> dict[str, Any]:
        # Uploaded files can be submitted repeatedly with the same original
        # filename. Keep each upload isolated so an older background stream
        # cannot append late chunks into the new session's transcript.
        bound_session = f"{_audio_session_id(audio_path, session_id)}_{uuid.uuid4().hex[:8]}"
        thread = threading.Thread(
            target=_stream_audio_session_background,
            args=(bound_session,),
            kwargs={"audio_path": audio_path, "experiment_mode": experiment_mode},
            name=f"retrace-audio-{bound_session}",
            daemon=True,
        )
        thread.start()
        return {
            "session_id": bound_session,
            "status": "processing",
            "stream": f"/api/sessions/{bound_session}/events",
            "experiment_mode": experiment_mode,
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
            experiment_mode=request.experiment_mode,
        )

    @app.post("/api/sessions/{session_id}/audio/upload")
    async def process_audio_upload(
        session_id: str,
        file: UploadFile = File(...),
        experiment_mode: Literal["baseline", "retrace"] = Form("retrace"),
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
            experiment_mode=experiment_mode,
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
