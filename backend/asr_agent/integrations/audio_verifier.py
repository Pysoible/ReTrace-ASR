"""Closed-set historical-audio verification for ReTrace hypotheses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def verify_candidates(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    candidates: list[str],
    *,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score only supplied candidates; malformed or open-set output is rejected."""
    unique = list(dict.fromkeys(item.strip() for item in candidates if item and item.strip()))
    if len(unique) < 2 or end_sec <= start_sec:
        return {"ok": False, "error": "need at least two candidates and a valid audio window"}
    if not Path(audio_path).exists():
        return {"ok": False, "error": f"audio is unavailable: {audio_path}"}
    try:
        payload = (runner or _qwen_runner)(audio_path=audio_path, start_sec=start_sec, end_sec=end_sec, candidates=unique)
    except Exception as exc:
        return {"ok": False, "error": f"audio verification failed: {exc}"}
    scores = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(scores, dict) or set(scores) != set(unique):
        return {"ok": False, "error": "verifier must score exactly the supplied candidates"}
    try:
        values = {candidate: max(0.0, float(scores[candidate])) for candidate in unique}
    except (TypeError, ValueError):
        return {"ok": False, "error": "verifier scores must be numeric"}
    total = sum(values.values())
    if total <= 0:
        return {"ok": False, "error": "verifier scores must have positive mass"}
    return {"ok": True, "scores": {candidate: value / total for candidate, value in values.items()}, "audio_path": audio_path, "start_sec": start_sec, "end_sec": end_sec}


def _qwen_runner(*, audio_path: str, start_sec: float, end_sec: float, candidates: list[str]) -> dict[str, Any]:
    """Ask Qwen-Omni to compare a closed candidate set over a cropped audio window."""
    import soundfile as sf
    from tempfile import NamedTemporaryFile

    from asr_agent.integrations.qwen_asr import _engine, _infer_one

    samples, sample_rate = sf.read(audio_path, always_2d=False)
    start, end = max(0, int(start_sec * sample_rate)), min(len(samples), int(end_sec * sample_rate))
    if end <= start:
        raise ValueError("audio window is empty")
    with NamedTemporaryFile(suffix=".wav") as clipped:
        sf.write(clipped.name, samples[start:end], sample_rate)
        engine, request_config = _engine()
        prompt = (
            "Listen only to this audio and compare the supplied transcript candidates. "
            "Return strict JSON {\"scores\":{candidate:number,...}} with every and only supplied candidate. "
            f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
        )
        raw = _infer_one(engine, request_config, Path(clipped.name), prompt)
    return json.loads(raw)
