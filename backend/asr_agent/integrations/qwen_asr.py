"""Qwen-Omni / Qwen-audio single-pass ASR observation adapter."""
from __future__ import annotations

import os
import json
import re
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

DEFAULT_MODEL_PATH = "/home/ma-user/work/dataset/sjk_data/sjk/model_demo/checkpoint-793-merged"
DEFAULT_AUDIO_DIR = "/home/ma-user/work/dataset/sjk_data/ASR_audio"

_ENGINE: Any = None

_PLAIN_PROMPT = "转写这段中文语音。只输出转写文本，不要输出 JSON、标签或解释。"
_OBS_PROMPT = (
    '转写这段中文语音。严格只输出 JSON：'
    '{"text":"完整转写","uncertain_spans":[{"span":"原片段","candidates":["原片段","同音候选"],"confidence":0.0}]}。'
    "只在确有不确定时给出 uncertain_spans；候选必须包含原片段。"
)
_TAG_PROMPT_TMPL = (
    "下面是一段中文 ASR 转写。请标出可能听错的专有名词、品牌、人名、术语（同音/近音不确定处）。"
    '严格只输出 JSON：{{"uncertain_spans":[{{"span":"原片段","candidates":["原片段","同音候选"],"confidence":0.0}}]}}。'
    "候选必须包含原片段；拿不准才标注；没有不确定则输出 {{\"uncertain_spans\":[]}}。\n"
    "转写：{text}"
)


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


def _is_garbage_transcript(text: str) -> bool:
    """Reject diarization/schema dumps and degenerate loops."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    if cleaned.startswith("{") or cleaned.startswith("["):
        return True
    if "start_time" in cleaned and "end_time" in cleaned and "label" in cleaned:
        return True
    if re.fullmatch(r"(.)\1{5,}", cleaned.replace(" ", "")):
        return True
    # heavy repetition of a short token, e.g. 瑶龙瑶龙瑶龙...
    compact = re.sub(r"\s+", "", cleaned)
    if len(compact) >= 8:
        for width in (2, 3, 4):
            token = compact[:width]
            if token and compact == token * (len(compact) // width) + compact[:(len(compact) % width)]:
                if compact.count(token) >= 4:
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
        if _is_garbage_transcript(text):
            return {"text": "", "uncertainty": {}}
        uncertainty = _parse_uncertain_spans(text, payload.get("uncertain_spans"))
        return {"text": text, "uncertainty": uncertainty}

    text = _strip_code_fence(raw)
    if _is_garbage_transcript(text):
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
    """Keep first-pass uncertainty only.

    Inventing phoneme confusables (苹果→平果) creates false hypotheses that never
    get later evidence. Real closed sets are opened by cross-turn / candidate echo.
    """
    del engine, request_config, chunk_path, text
    return uncertainty if uncertainty else {"confidence": {}, "text_candidates": {}}


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
        engine, request_config = _engine()
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
            raw_parts = _infer_chunks(engine, request_config, chunk_paths, _OBS_PROMPT)
        except Exception as exc:
            return {"ok": False, "error": f"Qwen 转写失败: {exc}", "chunks": chunk_meta}

        observations = [parse_observation(raw) for raw in raw_parts]

        # Fallback: empty/garbage chunks get a plain-text second try.
        for index, observation in enumerate(observations):
            if observation.get("text"):
                continue
            try:
                plain = _infer_one(engine, request_config, chunk_paths[index], _PLAIN_PROMPT)
            except Exception:
                continue
            recovered = parse_observation(plain)
            if recovered.get("text"):
                observations[index] = recovered

        # Enrich empty uncertainties with a tagging pass (keeps text fixed).
        for index, observation in enumerate(observations):
            observation["uncertainty"] = _enrich_uncertainty(
                engine,
                request_config,
                chunk_paths[index],
                str(observation.get("text") or ""),
                dict(observation.get("uncertainty") or {}),
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
            "duration_sec": chunk_info.get("duration_sec"),
            "chunked": bool(chunk_info.get("chunked")),
            "chunk_count": int(chunk_info.get("chunk_count") or 1),
            "chunks": chunk_meta,
        }
        out["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        return out
    finally:
        cleanup_chunks(chunk_info)
