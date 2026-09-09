"""Split long audio into short clips suitable for Qwen-Omni ASR."""
from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ChunkConfig:
    max_sec: float = 15.0
    overlap_sec: float = 0.3
    min_sec: float = 0.4
    top_db: float = 30.0
    target_sr: int = 16000


def read_chunk_config() -> ChunkConfig:
    return ChunkConfig(
        max_sec=float(os.getenv("ASR_CHUNK_MAX_SEC", "15")),
        overlap_sec=float(os.getenv("ASR_CHUNK_OVERLAP_SEC", "0.3")),
        min_sec=float(os.getenv("ASR_CHUNK_MIN_SEC", "0.4")),
        top_db=float(os.getenv("ASR_CHUNK_TOP_DB", "30")),
        target_sr=int(os.getenv("ASR_CHUNK_SR", "16000")),
    )


def _load_mono(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def _merge_intervals(intervals: list[tuple[int, int]], max_len: int, min_len: int) -> list[tuple[int, int]]:
    """Pack speech intervals into windows no longer than max_len samples."""
    if not intervals:
        return []
    packed: list[tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for start, end in intervals[1:]:
        if end - cur_s <= max_len and start - cur_e <= max_len // 8:
            cur_e = end
            continue
        if cur_e - cur_s >= min_len:
            packed.append((cur_s, cur_e))
        cur_s, cur_e = start, end
    if cur_e - cur_s >= min_len:
        packed.append((cur_s, cur_e))
    return packed


def _hard_split(start: int, end: int, max_len: int, overlap: int) -> list[tuple[int, int]]:
    if end - start <= max_len:
        return [(start, end)]
    hop = max(1, max_len - overlap)
    out: list[tuple[int, int]] = []
    s = start
    while s < end:
        e = min(s + max_len, end)
        if e - s > 0:
            out.append((s, e))
        if e >= end:
            break
        s += hop
    return out


def plan_chunks(num_samples: int, sr: int, cfg: ChunkConfig, audio: np.ndarray) -> list[tuple[int, int]]:
    max_len = max(1, int(cfg.max_sec * sr))
    overlap = max(0, int(cfg.overlap_sec * sr))
    min_len = max(1, int(cfg.min_sec * sr))
    if num_samples <= max_len:
        return [(0, num_samples)]

    try:
        import librosa

        intervals = librosa.effects.split(audio, top_db=cfg.top_db)
        speech = [(int(s), int(e)) for s, e in intervals if int(e) - int(s) >= min_len]
    except Exception:
        speech = []

    if not speech:
        return _hard_split(0, num_samples, max_len, overlap)

    packed = _merge_intervals(speech, max_len=max_len, min_len=min_len)
    chunks: list[tuple[int, int]] = []
    for start, end in packed:
        chunks.extend(_hard_split(start, end, max_len, overlap))
    if not chunks:
        return _hard_split(0, num_samples, max_len, overlap)
    return chunks


def split_audio_file(path: Path, cfg: ChunkConfig | None = None, work_dir: Path | None = None) -> dict[str, Any]:
    """Return chunk wav paths. Single-element list when audio is already short."""
    cfg = cfg or read_chunk_config()
    audio, sr = _load_mono(path, cfg.target_sr)
    duration = float(len(audio) / sr) if sr else 0.0
    spans = plan_chunks(len(audio), sr, cfg, audio)

    if len(spans) == 1 and spans[0] == (0, len(audio)):
        return {
            "duration_sec": round(duration, 3),
            "chunked": False,
            "chunk_count": 1,
            "chunks": [
                {
                    "index": 0,
                    "path": str(path),
                    "start_sec": 0.0,
                    "end_sec": round(duration, 3),
                    "temporary": False,
                }
            ],
            "work_dir": None,
        }

    out_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="retrace_chunks_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    chunks = []
    stem = path.stem
    for idx, (start, end) in enumerate(spans):
        clip = audio[start:end]
        chunk_path = out_dir / f"{stem}_{uuid.uuid4().hex[:8]}_{idx:03d}.wav"
        sf.write(str(chunk_path), clip, sr, subtype="PCM_16")
        chunks.append(
            {
                "index": idx,
                "path": str(chunk_path),
                "start_sec": round(start / sr, 3),
                "end_sec": round(end / sr, 3),
                "temporary": True,
            }
        )
    return {
        "duration_sec": round(duration, 3),
        "chunked": True,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "work_dir": str(out_dir),
        "max_sec": cfg.max_sec,
    }


def join_transcripts(parts: list[str]) -> str:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    if not cleaned:
        return ""
    # Chinese ASR chunks usually concatenate without spaces.
    return "".join(cleaned)


def cleanup_chunks(chunk_info: dict[str, Any]) -> None:
    work_dir = chunk_info.get("work_dir")
    for item in chunk_info.get("chunks") or []:
        if not item.get("temporary"):
            continue
        try:
            Path(item["path"]).unlink(missing_ok=True)
        except Exception:
            pass
    if work_dir:
        try:
            Path(work_dir).rmdir()
        except Exception:
            pass
