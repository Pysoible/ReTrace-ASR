"""Closed-set historical-audio verification for ReTrace hypotheses."""
from __future__ import annotations

import json
import re
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
    if isinstance(payload, dict) and not isinstance(scores, dict):
        selected = payload.get("selected", payload.get("choice"))
        if isinstance(selected, int) and 0 <= selected < len(unique):
            scores = {candidate: (1.0 if index == selected else 0.0) for index, candidate in enumerate(unique)}
        elif isinstance(selected, str) and selected.strip() in unique:
            selected = selected.strip()
            scores = {candidate: (1.0 if candidate == selected else 0.0) for candidate in unique}
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


def retranscribe_window(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    *,
    runner: Callable[..., dict[str, Any]] | None = None,
    domain_hints: list[str] | None = None,
    recover_coverage: bool = False,
) -> dict[str, Any]:
    """Open-vocabulary re-ASR for a historical audio window (degenerate-turn recovery).

    ``domain_hints`` are remembered domain terms (proper nouns) that are likely
    to occur in this conversation. Passing them as hotwords helps the ASR model
    prefer the correct proper noun over a mis-heard homophone — this is standard
    contextual biasing. It is optional: re-ASR still works with no hints.
    """
    if end_sec <= start_sec:
        return {"ok": False, "error": "need a valid audio window"}
    if not Path(audio_path).exists():
        return {"ok": False, "error": f"audio is unavailable: {audio_path}"}
    try:
        payload = (runner or _qwen_retranscribe_runner)(
            audio_path=audio_path,
            start_sec=start_sec,
            end_sec=end_sec,
            domain_hints=domain_hints,
            recover_coverage=recover_coverage,
        )
    except Exception as exc:
        return {"ok": False, "error": f"audio retranscription failed: {exc}"}
    text = str((payload or {}).get("text") or "").strip()
    if not text:
        return {"ok": False, "error": "empty retranscription"}
    return {
        "ok": True,
        "text": text,
        "audio_path": audio_path,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def _crop_audio(audio_path: str, start_sec: float, end_sec: float):
    import soundfile as sf
    from tempfile import NamedTemporaryFile

    samples, sample_rate = sf.read(audio_path, always_2d=False)
    start, end = max(0, int(start_sec * sample_rate)), min(len(samples), int(end_sec * sample_rate))
    if end <= start:
        raise ValueError("audio window is empty")
    clipped = NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(clipped.name, samples[start:end], sample_rate)
    clipped.close()
    return Path(clipped.name)


def _qwen_runner(*, audio_path: str, start_sec: float, end_sec: float, candidates: list[str]) -> dict[str, Any]:
    """Ask Qwen-Omni to compare a closed candidate set over a cropped audio window."""
    from asr_agent.integrations.qwen_asr import _infer_one_audio

    clipped = _crop_audio(audio_path, start_sec, end_sec)
    try:
        prompt = (
            "Listen only to this audio and compare the supplied transcript candidates. "
            "The special candidate [DELETE] means that the focused span is not spoken and should be deleted; "
            "score it only when the audio supports omission of that span. "
            "Return strict JSON with either {\"scores\":{candidate:number,...}} or {\"choice\":index}. "
            "For choice, index candidates from 0 and select exactly one supplied candidate. "
            "Do not invent candidates. "
            f"Candidates: {json.dumps(candidates, ensure_ascii=False)}"
        )
        raw = _infer_one_audio(clipped, prompt)
    finally:
        clipped.unlink(missing_ok=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        text = raw.strip()
        if text in candidates:
            return {"selected": text}
        choice_match = re.search(r"(?:choice|候选|选项)\s*[:：]?\s*([0-9]+)", text, re.IGNORECASE)
        if choice_match:
            return {"choice": int(choice_match.group(1))}
        matching = [candidate for candidate in candidates if candidate in text]
        if len(matching) == 1:
            return {"selected": matching[0]}
        raise


def _qwen_retranscribe_runner(
    *,
    audio_path: str,
    start_sec: float,
    end_sec: float,
    domain_hints: list[str] | None = None,
    recover_coverage: bool = False,
) -> dict[str, Any]:
    """Open re-ASR over a cropped window — used when the first-pass transcript is degenerate.

    ``domain_hints`` act as hotwords: remembered terms likely in this conversation.
    They bias the ASR model toward the correct proper noun without constraining it
    to a closed set.
    """
    from asr_agent.integrations.qwen_asr import _PLAIN_PROMPT, _infer_one_audio

    hints = [str(item).strip() for item in (domain_hints or []) if str(item).strip()]
    prompt = _PLAIN_PROMPT
    if hints:
        prompt = (
            "转写这段中文语音。只输出转写文本，不要输出 JSON、标签或解释。"
            "本对话中可能出现以下专有名词，请优先识别为正确的名称："
            f"{'、'.join(hints[:20])}。"
        )
    # A low transcript-to-speech coverage ratio is an omission failure, not a
    # local character ambiguity. Decode short consecutive sub-windows so one
    # early EOS cannot discard the rest of a 15-second conversational chunk.
    windows = [(start_sec, end_sec)]
    if recover_coverage and end_sec - start_sec > 5.5:
        windows = []
        cursor = start_sec
        while cursor < end_sec:
            next_cursor = min(end_sec, cursor + 5.0)
            windows.append((cursor, next_cursor))
            cursor = next_cursor
    parts: list[str] = []
    for window_start, window_end in windows:
        clipped = _crop_audio(audio_path, window_start, window_end)
        try:
            text = _infer_one_audio(clipped, prompt).strip()
        finally:
            clipped.unlink(missing_ok=True)
        if text:
            parts.append(text)
    return {"text": "".join(parts), "segmented": len(windows) > 1}
