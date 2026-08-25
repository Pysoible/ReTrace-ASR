"""Qwen-Omni / Qwen-audio single-pass ASR observation adapter."""
from __future__ import annotations

import atexit
import json
import logging
import multiprocessing as mp
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from asr_agent.integrations.audio_chunk import (
    cleanup_chunks,
    join_transcripts,
    read_chunk_config,
    split_audio_file,
)

_log = logging.getLogger("asr_agent.qwen_asr")

DEFAULT_MODEL_PATH = "/home/ma-user/work/dataset/sjk_data/sjk/model_demo/checkpoint-793-merged"
DEFAULT_AUDIO_DIR = "/home/ma-user/work/dataset/sjk_data/ASR_audio"

_ENGINE: Any = None
_ENGINE_POOL: list[Any] | None = None
_POOL_LOCK = threading.Lock()
_RR_LOCK = threading.Lock()
_RR_COUNTER = 0

_PLAIN_PROMPT = "转写这段中文语音。只输出转写文本，不要输出 JSON、标签或解释。"
_OBS_PROMPT = (
    '转写这段中文语音。严格只输出 JSON：'
    '{"text":"完整转写","uncertain_spans":[{"span":"原片段","candidates":["原片段","同音候选"],"confidence":0.0}]}。'
    "只在确有不确定时给出 uncertain_spans；候选必须包含原片段。"
)
_RAMC_PLAIN_PROMPT = (
    "请将上述音频转写为文字。尽量逐字保留中文口语、重复、语气词和停顿，"
    "不要总结、改写或补全没有听到的内容。只输出转写文本，不要输出 JSON、时间戳、标签或解释。"
)
_TAG_PROMPT_TMPL = (
    "下面是一段中文 ASR 转写。请标出可能听错的专有名词、品牌、人名、术语（同音/近音不确定处）。"
    '严格只输出 JSON：{{"uncertain_spans":[{{"span":"原片段","candidates":["原片段","同音候选"],"confidence":0.0}}]}}。'
    "候选必须包含原片段；拿不准才标注；没有不确定则输出 {{\"uncertain_spans\":[]}}。\n"
    "转写：{text}"
)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _observation_prompt() -> str:
    mode = os.getenv("ASR_PROMPT_MODE", "json").strip().lower()
    if mode in {"plain", "ramc", "ramc_plain"}:
        return _RAMC_PLAIN_PROMPT
    return _OBS_PROMPT


def parse_gpu_ids(gpu: str | None) -> list[str]:
    """Parse ASR_GPU like '0' or '0,1' into concrete device ids."""
    raw = (gpu or "0").replace(";", ",")
    ids = [part.strip() for part in raw.split(",") if part.strip()]
    return ids or ["0"]


def _load_project_env() -> None:
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def shard_indices(n_items: int, n_workers: int) -> list[list[int]]:
    """Round-robin index shards for multi-GPU chunk inference."""
    workers = max(1, n_workers)
    shards: list[list[int]] = [[] for _ in range(workers)]
    for index in range(max(0, n_items)):
        shards[index % workers].append(index)
    return shards


@dataclass(frozen=True)
class AsrConfig:
    enabled: bool
    model_path: str
    audio_dir: str
    gpu: str
    max_model_len: int
    max_tokens: int
    gpu_memory_utilization: float
    max_num_seqs: int
    enforce_eager: bool

    @property
    def gpu_ids(self) -> list[str]:
        return parse_gpu_ids(self.gpu)


def read_asr_config() -> AsrConfig:
    _load_project_env()
    return AsrConfig(
        enabled=_truthy(os.getenv("ASR_AUDIO_ENABLED")),
        model_path=os.getenv("ASR_MODEL_PATH", DEFAULT_MODEL_PATH),
        audio_dir=os.getenv("ASR_AUDIO_DIR", DEFAULT_AUDIO_DIR),
        gpu=os.getenv("ASR_GPU", "0"),
        # ASR needs headroom for multimodal activations; 8192@0.9 OOMs on 80GB.
        max_model_len=int(os.getenv("ASR_MAX_MODEL_LEN", "4096")),
        max_tokens=int(os.getenv("ASR_MAX_TOKENS", "512")),
        gpu_memory_utilization=float(os.getenv("ASR_GPU_MEM_UTIL", "0.85")),
        max_num_seqs=int(os.getenv("ASR_MAX_NUM_SEQS", "1")),
        enforce_eager=_truthy(os.getenv("ASR_ENFORCE_EAGER", "1")),
    )


def asr_status() -> dict[str, Any]:
    cfg = read_asr_config()
    chunk = read_chunk_config()
    pool = _ENGINE_POOL or ([] if _ENGINE is None else [_ENGINE])
    return {
        "enabled": cfg.enabled,
        "loaded": bool(pool),
        "workers": len(pool),
        "gpus": cfg.gpu_ids,
        "model_path": cfg.model_path,
        "model_exists": Path(cfg.model_path).exists(),
        "audio_dir": cfg.audio_dir,
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "max_num_seqs": cfg.max_num_seqs,
        "enforce_eager": cfg.enforce_eager,
        "chunk_max_sec": chunk.max_sec,
        "chunk_overlap_sec": chunk.overlap_sec,
    }


def preload_engine() -> dict[str, Any]:
    """Eagerly load the vLLM engine pool so the first /audio request is not cold."""
    _engines()
    status = asr_status()
    status["ok"] = True
    return status


def shutdown_engines() -> None:
    """Release local/remote ASR workers (best-effort)."""
    global _ENGINE, _ENGINE_POOL
    with _POOL_LOCK:
        pool = list(_ENGINE_POOL or [])
        _ENGINE_POOL = None
        _ENGINE = None
    for worker in pool:
        close = getattr(worker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


atexit.register(shutdown_engines)


class _InferEngine(Protocol):
    def infer_one(self, audio_path: Path | str, prompt: str) -> str: ...

    def infer_many(self, audio_paths: list[Path | str], prompt: str) -> list[str]: ...


class _LocalEngine:
    """In-process vLLM engine bound to one visible GPU."""

    def __init__(self, engine: Any, request_config: Any, gpu: str) -> None:
        self.engine = engine
        self.request_config = request_config
        self.gpu = gpu
        self._lock = threading.Lock()

    def infer_one(self, audio_path: Path | str, prompt: str) -> str:
        with self._lock:
            return _infer_one(self.engine, self.request_config, Path(audio_path), prompt)

    def infer_many(self, audio_paths: list[Path | str], prompt: str) -> list[str]:
        with self._lock:
            return [_infer_one(self.engine, self.request_config, Path(path), prompt) for path in audio_paths]

    def close(self) -> None:
        return None


class _RemoteEngine:
    """Spawned worker process with its own CUDA_VISIBLE_DEVICES."""

    def __init__(self, gpu: str, cfg: AsrConfig) -> None:
        ctx = mp.get_context("spawn")
        self.parent_conn, child_conn = ctx.Pipe(duplex=True)
        self.gpu = gpu
        self._lock = threading.Lock()
        self.proc = ctx.Process(
            target=_gpu_worker_main,
            args=(child_conn, gpu, _cfg_to_dict(cfg)),
            daemon=True,
            name=f"retrace-asr-gpu{gpu}",
        )
        self.proc.start()
        child_conn.close()
        status, payload = self.parent_conn.recv()
        if status != "ready":
            self.close()
            raise RuntimeError(f"ASR worker on GPU {gpu} failed to start: {payload}")

    def infer_one(self, audio_path: Path | str, prompt: str) -> str:
        with self._lock:
            self.parent_conn.send(("infer", str(audio_path), prompt))
            status, payload = self.parent_conn.recv()
        if status != "ok":
            raise RuntimeError(str(payload))
        return str(payload or "")

    def infer_many(self, audio_paths: list[Path | str], prompt: str) -> list[str]:
        with self._lock:
            self.parent_conn.send(("infer_many", [str(path) for path in audio_paths], prompt))
            status, payload = self.parent_conn.recv()
        if status != "ok_many":
            raise RuntimeError(str(payload))
        return [str(item or "") for item in payload]

    def close(self) -> None:
        try:
            if self.proc.is_alive():
                self.parent_conn.send(("shutdown",))
                self.proc.join(timeout=15)
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout=5)
        except Exception:
            pass
        try:
            self.parent_conn.close()
        except Exception:
            pass


def _cfg_to_dict(cfg: AsrConfig) -> dict[str, Any]:
    return {
        "model_path": cfg.model_path,
        "max_model_len": cfg.max_model_len,
        "max_tokens": cfg.max_tokens,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "max_num_seqs": cfg.max_num_seqs,
        "enforce_eager": cfg.enforce_eager,
    }


def _gpu_worker_main(conn: Any, gpu: str, cfg: dict[str, Any]) -> None:
    """Child process entry: bind one GPU, load engine, serve infer requests."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MAX_PIXELS", "1003520")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
    os.environ.setdefault("FPS_MAX_FRAMES", "12")
    os.environ.setdefault("VLLM_USE_V1", os.environ.get("VLLM_USE_V1", "0"))
    try:
        from swift.llm import InferRequest, RequestConfig, VllmEngine  # noqa: WPS433

        engine = VllmEngine(
            cfg["model_path"],
            max_model_len=int(cfg["max_model_len"]),
            gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
            max_num_seqs=int(cfg["max_num_seqs"]),
            enforce_eager=bool(cfg["enforce_eager"]),
            limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
        )
        request_config = RequestConfig(
            max_tokens=int(cfg["max_tokens"]),
            temperature=0,
            stream=False,
            seed=42,
        )
        conn.send(("ready", None))
        while True:
            message = conn.recv()
            op = message[0]
            if op == "shutdown":
                break
            if op == "infer":
                _, audio_path, prompt = message
                try:
                    req = InferRequest(
                        messages=[{"role": "user", "content": prompt}],
                        audios=[str(audio_path)],
                    )
                    resp = engine.infer([req], request_config)
                    text = ""
                    if resp and resp[0].choices:
                        text = (resp[0].choices[0].message.content or "").replace("\n", " ").strip()
                    conn.send(("ok", text))
                except Exception as exc:  # noqa: BLE001
                    conn.send(("err", str(exc)))
                continue
            if op == "infer_many":
                _, audio_paths, prompt = message
                try:
                    texts: list[str] = []
                    for audio_path in audio_paths:
                        req = InferRequest(
                            messages=[{"role": "user", "content": prompt}],
                            audios=[str(audio_path)],
                        )
                        resp = engine.infer([req], request_config)
                        text = ""
                        if resp and resp[0].choices:
                            text = (resp[0].choices[0].message.content or "").replace("\n", " ").strip()
                        texts.append(text)
                    conn.send(("ok_many", texts))
                except Exception as exc:  # noqa: BLE001
                    conn.send(("err", str(exc)))
                continue
            conn.send(("err", f"unknown op: {op}"))
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("err", str(exc)))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _build_local_engine(cfg: AsrConfig, gpu: str) -> _LocalEngine:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("MAX_PIXELS", "1003520")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
    os.environ.setdefault("FPS_MAX_FRAMES", "12")
    try:
        from swift.llm import RequestConfig, VllmEngine  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(f"未安装 ms-swift / vLLM：{exc}") from exc
    if not Path(cfg.model_path).exists():
        raise RuntimeError(f"模型路径不存在: {cfg.model_path}（用 ASR_MODEL_PATH 覆盖）")
    engine = VllmEngine(
        cfg.model_path,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        enforce_eager=cfg.enforce_eager,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
    )
    request_config = RequestConfig(max_tokens=cfg.max_tokens, temperature=0, stream=False, seed=42)
    return _LocalEngine(engine, request_config, gpu)


def _engines() -> list[Any]:
    """Return one ASR worker per configured GPU (data-parallel chunk inference)."""
    global _ENGINE, _ENGINE_POOL
    with _POOL_LOCK:
        if _ENGINE_POOL is not None:
            return _ENGINE_POOL

        cfg = read_asr_config()
        if not cfg.enabled:
            raise RuntimeError(
                "音频推理未启用。在 GPU 机器上设置 ASR_AUDIO_ENABLED=1，并确保 "
                "ASR_MODEL_PATH 指向 Qwen-Omni / Qwen-audio merged checkpoint。"
            )
        if not Path(cfg.model_path).exists():
            raise RuntimeError(f"模型路径不存在: {cfg.model_path}（用 ASR_MODEL_PATH 覆盖）")

        gpu_ids = cfg.gpu_ids
        if len(gpu_ids) == 1:
            local = _build_local_engine(cfg, gpu_ids[0])
            _ENGINE_POOL = [local]
            _ENGINE = (local.engine, local.request_config)
            return _ENGINE_POOL

        # Multi-GPU: one spawned replica per device so chunk ASR can run in parallel.
        workers: list[Any] = []
        try:
            for gpu in gpu_ids:
                workers.append(_RemoteEngine(gpu, cfg))
        except Exception:
            for worker in workers:
                close = getattr(worker, "close", None)
                if callable(close):
                    close()
            raise
        _ENGINE_POOL = workers
        _ENGINE = workers[0]
        return _ENGINE_POOL


def _engine():
    """Backward-compatible handle: first worker, or (engine, request_config) for local mode."""
    workers = _engines()
    first = workers[0]
    if isinstance(first, _LocalEngine):
        return first.engine, first.request_config
    return first


def _pick_worker() -> Any:
    global _RR_COUNTER
    workers = _engines()
    with _RR_LOCK:
        index = _RR_COUNTER % len(workers)
        _RR_COUNTER += 1
    return workers[index]


def _infer_one_audio(audio_path: Path | str, prompt: str) -> str:
    """Run one audio clip on the next worker (round-robin across GPUs)."""
    worker = _pick_worker()
    if isinstance(worker, tuple):
        engine, request_config = worker
        return _infer_one(engine, request_config, Path(audio_path), prompt)
    return worker.infer_one(audio_path, prompt)


def _resolve_audio(audio: str, cfg: AsrConfig) -> Path | None:
    path = Path(audio).expanduser()
    if path.is_absolute() and path.exists():
        return path
    for base in (Path(cfg.audio_dir), Path.cwd()):
        candidate = (base / audio).resolve()
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def _infer_chunks(engine, request_config, chunk_paths: list[Path], prompt: str) -> list[str]:
    """Infer many chunks; when multiple GPUs are configured, run shards in parallel."""
    del engine, request_config  # pool-aware path below
    workers = _engines()
    if len(workers) == 1:
        worker = workers[0]
        if isinstance(worker, _LocalEngine):
            return worker.infer_many(chunk_paths, prompt)
        return worker.infer_many(chunk_paths, prompt)

    shards = shard_indices(len(chunk_paths), len(workers))
    texts: list[str] = [""] * len(chunk_paths)

    def _run(worker: Any, indices: list[int]) -> list[tuple[int, str]]:
        if not indices:
            return []
        paths = [chunk_paths[index] for index in indices]
        outputs = worker.infer_many(paths, prompt)
        return list(zip(indices, outputs))

    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = [pool.submit(_run, worker, indices) for worker, indices in zip(workers, shards)]
        for future in as_completed(futures):
            for index, text in future.result():
                texts[index] = text
    return texts


def _infer_one(engine, request_config, audio_path: Path, prompt: str) -> str:
    from swift.llm import InferRequest  # noqa: WPS433

    req = InferRequest(messages=[{"role": "user", "content": prompt}], audios=[str(audio_path)])
    resp = engine.infer([req], request_config)
    if resp and resp[0].choices:
        return (resp[0].choices[0].message.content or "").replace("\n", " ").strip()
    return ""


def _strip_code_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(raw: str) -> Any | None:
    text = _strip_code_fence(raw)
    if not text:
        return None
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _is_invalid_transcript(text: str) -> bool:
    """Reject empty/schema output while preserving loops for auditable recovery."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if cleaned.startswith("{") or cleaned.startswith("["):
        return True
    if "start_time" in cleaned and "end_time" in cleaned and "label" in cleaned:
        return True
    return False


def _parse_uncertain_spans(text: str, spans: Any) -> dict[str, Any]:
    confidence: dict[str, float] = {}
    candidates: dict[str, list[str]] = {}
    if not isinstance(spans, list):
        return {"confidence": confidence, "text_candidates": candidates}
    for item in spans:
        if not isinstance(item, dict):
            continue
        span = str(item.get("span") or "").strip()
        alternatives = [str(value).strip() for value in item.get("candidates") or [] if str(value).strip()]
        alternatives = list(dict.fromkeys(alternatives))
        score_raw = item.get("confidence")
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            continue
        if (
            span
            and 1 <= len(span) <= 8
            and re.fullmatch(r"[\w\u4e00-\u9fff]+", span)
            and span in text
            and alternatives
            and span in alternatives
            and len(alternatives) >= 2
            and 0 <= score < 0.65
        ):
            confidence[span] = score
            candidates[span] = alternatives
    return {"confidence": confidence, "text_candidates": candidates}


def parse_observation(raw: str) -> dict[str, Any]:
    """Parse a Qwen observation without inventing uncertainty on malformed output."""
    payload = _extract_json_object(raw)
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        text = payload["text"].strip()
        if _is_invalid_transcript(text):
            return {"text": "", "uncertainty": {}}
        uncertainty = _parse_uncertain_spans(text, payload.get("uncertain_spans"))
        return {"text": text, "uncertainty": uncertainty}

    text = _strip_code_fence(raw)
    if _is_invalid_transcript(text):
        return {"text": "", "uncertainty": {}}
    return {"text": text, "uncertainty": {}}


def parse_uncertainty_tags(raw: str, text: str) -> dict[str, Any]:
    """Parse a second-pass uncertain_spans-only JSON against an existing transcript."""
    payload = _extract_json_object(raw)
    if not isinstance(payload, dict):
        return {"confidence": {}, "text_candidates": {}}
    spans = payload.get("uncertain_spans")
    if spans is None and "span" in payload:
        spans = [payload]
    return _parse_uncertain_spans(text, spans)


def _enrich_uncertainty(engine, request_config, chunk_path: Path, text: str, uncertainty: dict[str, Any]) -> dict[str, Any]:
    """Discover bounded candidates after plain ASR without changing its text."""
    del engine, request_config
    result = dict(uncertainty or {})
    if os.getenv("ASR_PROMPT_MODE", "json").strip().lower() not in {"plain", "ramc", "ramc_plain"}:
        return result or {"confidence": {}, "text_candidates": {}}
    # Keep discovery selective: plain ASR is the cheap first pass; audio calls
    # happen only after the context judge nominates a concrete focus.
    if not _truthy(os.getenv("ASR_UNCERTAINTY_DISCOVERY", "1")) or not text:
        return result or {"confidence": {}, "text_candidates": {}}
    try:
        raw = _infer_one_audio(chunk_path, _TAG_PROMPT_TMPL.format(text=text))
        discovered = parse_uncertainty_tags(raw, text)
        result["confidence"] = {**result.get("confidence", {}), **discovered["confidence"]}
        result["text_candidates"] = {**result.get("text_candidates", {}), **discovered["text_candidates"]}
    except Exception:
        pass
    return result or {"confidence": {}, "text_candidates": {}}


def _acoustic_disagreement_enabled() -> bool:
    return _truthy(os.getenv("ASR_ACOUSTIC_DISAGREEMENT", "1"))


def _speech_duration_sec(chunk_path: Path) -> float:
    """Estimate the voiced duration (seconds) of a chunk via energy VAD.

    Used to detect truncation: a chunk with many seconds of speech but a tiny
    transcript is likely to have been under-transcribed by the first pass.
    """
    try:
        import librosa
        import soundfile as sf

        audio, sr = sf.read(str(chunk_path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if len(audio) < sr * 0.1:
            return 0.0
        intervals = librosa.effects.split(audio, top_db=30.0)
        return sum(ed - st for st, ed in intervals) / sr
    except Exception:
        return 0.0


def _coverage_signal(chunk_path: Path, text: str, *, min_chars_per_sec: float = 1.5) -> dict[str, Any]:
    """Return auditable speech-coverage evidence for agent routing.

    A fluent but tiny transcript is a different failure mode from a confused
    character: the agent must recover missing content with a segmented re-ASR,
    not apply a local character substitution.
    """
    # librosa/NumPy arithmetic may yield numpy.float64/numpy.bool_ values.
    # Coverage is persisted in session JSON and SSE payloads, so convert every
    # field at this boundary to a native Python JSON scalar.
    speech_sec = float(_speech_duration_sec(chunk_path))
    char_count = len(re.sub(r"[^\w\u4e00-\u9fff]", "", text or ""))
    density = char_count / max(speech_sec, 0.1) if speech_sec else 0.0
    truncated = not text
    if text and speech_sec >= 3.0:
        truncated = density < min_chars_per_sec
    return {
        "speech_sec": float(round(speech_sec, 3)),
        "char_count": int(char_count),
        "char_density": float(round(density, 3)),
        "truncated": bool(truncated),
    }


def _is_truncated(chunk_path: Path, text: str, *, min_chars_per_sec: float = 1.5) -> bool:
    """True when the transcript density is suspiciously low for the voiced time.

    Normal conversational Mandarin is ~3-5 chars/sec of speech; an under-
    transcribed chunk drops below ~1 char/sec. Empty transcripts are always
    considered truncated (the caller then re-listens with the plain prompt).
    """
    return bool(_coverage_signal(chunk_path, text, min_chars_per_sec=min_chars_per_sec)["truncated"])


def _attach_acoustic_disagreement(chunk_path: Path, text: str, uncertainty: dict[str, Any]) -> dict[str, Any]:
    """Run a second independent ASR (paraformer) once and attach acoustic
    uncertainty signals: (1) spans where the two acoustic models disagree,
    (2) characters whose paraformer decoder confidence is low, and (3) the
    paraformer transcript itself (with per-character confidence) as a "second
    opinion" the agent can use for correction — all independent of the LLM judge
    and of Qwen's self-reported confidence."""
    if not _acoustic_disagreement_enabled() or not text:
        return uncertainty
    try:
        from asr_agent.integrations import acoustic  # local import to avoid cycles

        signals = acoustic.acoustic_signals_full(chunk_path, text)
    except Exception:
        return uncertainty
    if signals["disagreements"] or signals["low_conf_chars"] or signals["paraformer_text"]:
        uncertainty = dict(uncertainty)
        if signals["disagreements"]:
            uncertainty["acoustic_disagreement"] = signals["disagreements"]
        if signals["low_conf_chars"]:
            uncertainty["low_conf_chars"] = signals["low_conf_chars"]
        if signals["paraformer_text"]:
            uncertainty["paraformer_text"] = signals["paraformer_text"]
        if signals["char_confs"]:
            uncertainty["paraformer_char_confs"] = signals["char_confs"]
    return uncertainty


def _infer_chunks_streaming(
    chunk_paths: list[Path],
    prompt: str,
    on_chunk: Callable[[int, str], None],
) -> None:
    """Transcribe chunks across GPUs, invoking on_chunk(index, raw) as soon as
    each chunk finishes so the caller can stream results incrementally."""
    workers = _engines()
    shards = shard_indices(len(chunk_paths), len(workers))

    def _run(worker: Any, indices: list[int]) -> None:
        for index in indices:
            try:
                raw = worker.infer_one(chunk_paths[index], prompt)
            except Exception:
                raw = ""
            on_chunk(index, raw)

    with ThreadPoolExecutor(max_workers=max(1, len(workers))) as pool:
        futures = [
            pool.submit(_run, worker, indices)
            for worker, indices in zip(workers, shards)
            if indices
        ]
        for future in as_completed(futures):
            future.result()


def stream_transcribe_audio(
    audio: str,
    on_chunk: Callable[[int, str, dict[str, Any], dict[str, Any]], None],
) -> dict[str, Any]:
    """Streaming Qwen-Omni ASR: split the audio, transcribe chunks across all GPUs
    and call ``on_chunk(index, text, uncertainty, chunk_meta)`` as soon as each
    chunk's transcript is ready (dual-GPU parallel)."""
    audio = (audio or "").strip()
    if not audio:
        raise RuntimeError("需要音频路径 audio")

    cfg = read_asr_config()
    path = _resolve_audio(audio, cfg)
    if path is None:
        raise RuntimeError(f"找不到音频文件: {audio}（绝对路径，或相对 ASR_AUDIO_DIR={cfg.audio_dir}）")

    _engines()
    chunk_info = split_audio_file(path, read_chunk_config())
    chunk_paths = [Path(item["path"]) for item in chunk_info["chunks"]]
    chunk_meta = [
        {
            "index": item["index"],
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
        }
        for item in chunk_info["chunks"]
    ]

    def _finish(index: int, raw: str) -> None:
        observation = parse_observation(raw)
        text = str(observation.get("text") or "")
        if not text:
            try:
                plain = _infer_one_audio(chunk_paths[index], _PLAIN_PROMPT)
                observation = parse_observation(plain)
            except Exception:
                pass
        text = str(observation.get("text") or "")
        # Truncation re-listen: when the first pass produced a suspiciously short
        # transcript for the voiced duration (under-transcription), re-listen once
        # with the plain prompt and adopt it only if it is substantially fuller.
        if text and _is_truncated(chunk_paths[index], text):
            try:
                plain = _infer_one_audio(chunk_paths[index], _PLAIN_PROMPT)
                plain_obs = parse_observation(plain)
                plain_text = str(plain_obs.get("text") or "")
                plain_trunc = _is_truncated(chunk_paths[index], plain_text) if plain_text else None
                print(
                    f"[truncation-relisten] chunk={index} first={text[:60]!r} "
                    f"plain={plain_text[:60]!r} plain_truncated={plain_trunc}",
                    flush=True,
                )
                if plain_text and not plain_trunc:
                    observation = plain_obs
                    text = plain_text
            except Exception as exc:  # noqa: BLE001
                print(f"[truncation-relisten] chunk={index} FAILED {exc!r}", flush=True)
        uncertainty = dict(observation.get("uncertainty") or {})
        uncertainty["coverage"] = _coverage_signal(chunk_paths[index], text)
        uncertainty = _attach_acoustic_disagreement(chunk_paths[index], text, uncertainty)
        on_chunk(index, text, uncertainty, chunk_meta[index])

    try:
        _infer_chunks_streaming(chunk_paths, _observation_prompt(), _finish)
    finally:
        cleanup_chunks(chunk_info)
    return {
        "ok": True,
        "audio": str(path),
        "chunk_count": int(chunk_info.get("chunk_count") or len(chunk_paths)),
        "duration_sec": chunk_info.get("duration_sec"),
        "chunked": bool(chunk_info.get("chunked")),
        "chunks": chunk_meta,
    }


def transcribe_audio(audio: str) -> dict[str, Any]:
    """Single-pass Qwen-Omni ASR for chunked conversational observations."""
    audio = (audio or "").strip()
    if not audio:
        return {"ok": False, "error": "需要音频路径 audio"}

    cfg = read_asr_config()
    path = _resolve_audio(audio, cfg)
    if path is None:
        return {"ok": False, "error": f"找不到音频文件: {audio}（绝对路径，或相对 ASR_AUDIO_DIR={cfg.audio_dir}）"}

    try:
        _engines()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    t0 = time.perf_counter()
    try:
        chunk_info = split_audio_file(path, read_chunk_config())
    except Exception as exc:
        return {"ok": False, "error": f"音频切分失败: {exc}"}

    chunk_paths = [Path(item["path"]) for item in chunk_info["chunks"]]
    chunk_meta = [
        {
            "index": item["index"],
            "start_sec": item["start_sec"],
            "end_sec": item["end_sec"],
        }
        for item in chunk_info["chunks"]
    ]

    try:
        try:
            raw_parts = _infer_chunks(None, None, chunk_paths, _observation_prompt())
        except Exception as exc:
            return {"ok": False, "error": f"Qwen 转写失败: {exc}", "chunks": chunk_meta}

        observations = [parse_observation(raw) for raw in raw_parts]

        # Fallback: empty/garbage chunks get a plain-text second try.
        for index, observation in enumerate(observations):
            if observation.get("text"):
                continue
            try:
                plain = _infer_one_audio(chunk_paths[index], _PLAIN_PROMPT)
            except Exception:
                continue
            recovered = parse_observation(plain)
            if recovered.get("text"):
                observations[index] = recovered

        # Enrich empty uncertainties with a tagging pass (keeps text fixed).
        for index, observation in enumerate(observations):
            observation["uncertainty"] = _enrich_uncertainty(
                None,
                None,
                chunk_paths[index],
                str(observation.get("text") or ""),
                dict(observation.get("uncertainty") or {}),
            )
            observation["uncertainty"] = _attach_acoustic_disagreement(
                chunk_paths[index],
                str(observation.get("text") or ""),
                observation["uncertainty"],
            )

        parts = [str(item.get("text") or "") for item in observations]
        final_text = join_transcripts([part for part in parts if part])
        out: dict[str, Any] = {
            "ok": True,
            "audio": str(path),
            "final_text": final_text,
            "chunks_text": parts,
            "uncertainties": [item.get("uncertainty") or {} for item in observations],
            "backend": "qwen-omni-vllm",
            "workers": len(_engines()),
            "gpus": read_asr_config().gpu_ids,
            "duration_sec": chunk_info.get("duration_sec"),
            "chunked": bool(chunk_info.get("chunked")),
            "chunk_count": int(chunk_info.get("chunk_count") or 1),
            "chunks": chunk_meta,
        }
        out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        return out
    finally:
        cleanup_chunks(chunk_info)
