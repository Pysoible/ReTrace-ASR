"""Qwen-Omni / Qwen-audio two-pass ASR — same stack as ASR_agent.audio_asr."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asr_agent.integrations.audio_chunk import (
    cleanup_chunks,
    join_transcripts,
    read_chunk_config,
    split_audio_file,
)
from asr_agent.integrations.domainterms import ensure_importable

DEFAULT_MODEL_PATH = "/home/ma-user/work/dataset/sjk_data/sjk/model_demo/checkpoint-793-merged"
DEFAULT_AUDIO_DIR = "/home/ma-user/work/dataset/sjk_data/ASR_audio"

_ENGINE: Any = None


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AsrConfig:
    enabled: bool
    model_path: str
    audio_dir: str
    gpu: str
    max_model_len: int
    max_tokens: int
    default_top_k: int
    gpu_memory_utilization: float
    max_num_seqs: int
    enforce_eager: bool


def read_asr_config() -> AsrConfig:
    return AsrConfig(
        enabled=_truthy(os.getenv("ASR_AUDIO_ENABLED")),
        model_path=os.getenv("ASR_MODEL_PATH", DEFAULT_MODEL_PATH),
        audio_dir=os.getenv("ASR_AUDIO_DIR", DEFAULT_AUDIO_DIR),
        gpu=os.getenv("ASR_GPU", "0"),
        # ASR needs headroom for multimodal activations; 8192@0.9 OOMs on 80GB.
        max_model_len=int(os.getenv("ASR_MAX_MODEL_LEN", "4096")),
        max_tokens=int(os.getenv("ASR_MAX_TOKENS", "512")),
        default_top_k=int(os.getenv("ASR_DEFAULT_TOPK", "30")),
        gpu_memory_utilization=float(os.getenv("ASR_GPU_MEM_UTIL", "0.85")),
        max_num_seqs=int(os.getenv("ASR_MAX_NUM_SEQS", "1")),
        enforce_eager=_truthy(os.getenv("ASR_ENFORCE_EAGER", "1")),
    )


def asr_status() -> dict[str, Any]:
    cfg = read_asr_config()
    chunk = read_chunk_config()
    return {
        "enabled": cfg.enabled,
        "loaded": _ENGINE is not None,
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
    """Eagerly load the vLLM engine so the first /audio request is not cold."""
    _engine()
    status = asr_status()
    status["ok"] = True
    return status


def _engine():
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    cfg = read_asr_config()
    if not cfg.enabled:
        raise RuntimeError(
            "音频推理未启用。在 GPU 机器上设置 ASR_AUDIO_ENABLED=1，并确保 "
            "ASR_MODEL_PATH 指向 Qwen-Omni / Qwen-audio merged checkpoint。"
        )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", cfg.gpu)
    os.environ.setdefault("MAX_PIXELS", "1003520")
    os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
    os.environ.setdefault("FPS_MAX_FRAMES", "12")

    try:
        from swift.llm import RequestConfig, VllmEngine  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(f"未安装 ms-swift / vLLM：{exc}") from exc

    if not Path(cfg.model_path).exists():
        raise RuntimeError(f"模型路径不存在: {cfg.model_path}（用 ASR_MODEL_PATH 覆盖）")

    # Leave free VRAM for audio encoder activations; limit non-audio modalities.
    engine = VllmEngine(
        cfg.model_path,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_num_seqs=cfg.max_num_seqs,
        enforce_eager=cfg.enforce_eager,
        limit_mm_per_prompt={"audio": 1, "image": 0, "video": 0},
    )
    request_config = RequestConfig(max_tokens=cfg.max_tokens, temperature=0, stream=False, seed=42)
    _ENGINE = (engine, request_config)
    return _ENGINE


def _resolve_audio(audio: str, cfg: AsrConfig) -> Path | None:
    path = Path(audio).expanduser()
    if path.is_absolute() and path.exists():
        return path
    for base in (Path(cfg.audio_dir), Path.cwd()):
        candidate = (base / audio).resolve()
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def _infer_one(engine, request_config, audio_path: Path, prompt: str) -> str:
    from swift.llm import InferRequest  # noqa: WPS433

    req = InferRequest(messages=[{"role": "user", "content": prompt}], audios=[str(audio_path)])
    resp = engine.infer([req], request_config)
    if resp and resp[0].choices:
        return (resp[0].choices[0].message.content or "").replace("\n", " ").strip()
    return ""


def _infer_chunks(engine, request_config, chunk_paths: list[Path], prompt: str) -> list[str]:
    texts: list[str] = []
    for chunk_path in chunk_paths:
        texts.append(_infer_one(engine, request_config, chunk_path, prompt))
    return texts


def transcribe_audio(
    audio: str,
    *,
    two_pass: bool = True,
    domain: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Pass1 bare ASR → dictionary retrieve → Pass2 hotword ASR (ASR_agent pattern).

    Long audio is split into short chunks (default <=15s) before Qwen-Omni inference.
    """
    audio = (audio or "").strip()
    if not audio:
        return {"ok": False, "error": "需要音频路径 audio"}

    cfg = read_asr_config()
    path = _resolve_audio(audio, cfg)
    if path is None:
        return {"ok": False, "error": f"找不到音频文件: {audio}（绝对路径，或相对 ASR_AUDIO_DIR={cfg.audio_dir}）"}

    try:
        engine, request_config = _engine()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    try:
        ensure_importable()
        import context_asr_twopass as cat  # noqa: WPS433
        from context_retrieval import EntityRetriever  # noqa: WPS433
        from domain_terms import config as dt_config  # noqa: WPS433
    except Exception as exc:
        return {"ok": False, "error": f"无法加载两遍法依赖: {exc}"}

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
            pass1_parts = _infer_chunks(engine, request_config, chunk_paths, cat.PASS1_PROMPT)
        except Exception as exc:
            return {"ok": False, "error": f"Pass1 转写失败: {exc}", "chunks": chunk_meta}

        pass1 = join_transcripts(pass1_parts)
        out: dict[str, Any] = {
            "ok": True,
            "audio": str(path),
            "pass1_text": pass1,
            "pass1_chunks": pass1_parts,
            "two_pass": two_pass,
            "backend": "qwen-omni-vllm",
            "duration_sec": chunk_info.get("duration_sec"),
            "chunked": bool(chunk_info.get("chunked")),
            "chunk_count": int(chunk_info.get("chunk_count") or 1),
            "chunks": chunk_meta,
        }
        if not two_pass:
            out["final_text"] = pass1
            out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
            return out

        k = int(top_k or cfg.default_top_k)
        retriever = EntityRetriever(str(dt_config.DICT_PATH))
        retrieved = retriever.retrieve(pass1, domain_label=domain, k=k)
        pass2_prompt = cat.build_pass2_prompt(retrieved)

        try:
            pass2_parts = _infer_chunks(engine, request_config, chunk_paths, pass2_prompt)
            pass2 = join_transcripts(pass2_parts)
        except Exception as exc:
            out["final_text"] = pass1
            out["retrieved"] = retrieved
            out["correction_error"] = str(exc)
            out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
            return out

        out.update(
            {
                "domain": domain,
                "retrieved": retrieved,
                "final_text": pass2,
                "pass2_chunks": pass2_parts,
                "applied_terms": [term for term in retrieved if term in pass2 and term not in pass1],
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }
        )
        return out
    finally:
        cleanup_chunks(chunk_info)
