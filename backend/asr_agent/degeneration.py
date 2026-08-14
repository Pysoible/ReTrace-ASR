"""Model-agnostic detection of clearly degenerate ASR transcripts."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DegenerationAssessment:
    degenerate: bool
    score: float
    reasons: tuple[str, ...] = ()


def _compact(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff]", "", text or "")


def _periodic_width(text: str) -> int | None:
    for width in range(2, min(6, len(text) // 3) + 1):
        repeats, remainder = divmod(len(text), width)
        token = text[:width]
        if repeats >= 3 and text == token * repeats + token[:remainder]:
            return width
    return None


def assess_transcript(text: str, *, duration_sec: float | None = None) -> DegenerationAssessment:
    """Score obvious loops and malformed output without assuming specific filler words."""
    raw = (text or "").strip()
    content = re.sub(r"^\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*", "", raw)
    compact = _compact(content)
    reasons: list[str] = []
    score = 0.0

    if not compact:
        return DegenerationAssessment(True, 1.0, ("empty",))

    lower = content.lower()
    if (
        content.startswith(("{", "["))
        or all(marker in lower for marker in ("start_time", "end_time", "label"))
    ):
        reasons.append("format_leak")
        score = 1.0

    if len(compact) >= 6:
        dominant_ratio = max(compact.count(char) for char in set(compact)) / len(compact)
        if dominant_ratio >= 0.8:
            reasons.append("repetition")
            score = max(score, 0.95)

    if len(compact) >= 8 and _periodic_width(compact) is not None:
        reasons.append("periodic_loop")
        score = max(score, 0.9)

    if len(compact) >= 12 and len(set(compact)) / len(compact) <= 0.2:
        reasons.append("low_diversity")
        score = max(score, 0.8)

    if duration_sec is not None and duration_sec >= 8.0 and len(compact) <= 1:
        reasons.append("duration_mismatch")
        score = max(score, 0.8)

    return DegenerationAssessment(score >= 0.75, score, tuple(dict.fromkeys(reasons)))


def should_replace_degenerate(
    original: DegenerationAssessment,
    candidate: DegenerationAssessment,
    *,
    minimum_improvement: float = 0.35,
) -> bool:
    return (
        original.degenerate
        and not candidate.degenerate
        and original.score - candidate.score >= minimum_improvement
    )


_REPEAT_LOOP = re.compile(r"([\u4e00-\u9fff]{1,4}[。！？!?；;，,]?)\1{2,}")


def trim_degenerate_tail(text: str) -> str:
    """Strip a degenerate tail while keeping a meaningful prefix.

    A chunk can start with real content and then collapse into a repeated loop
    (e.g. ``十一的。正好是节假日放假的时候。对。对。对。...``). Retranscription
    of the whole window may still come back degenerate, so as a fallback we drop
    everything from the first repeated loop onward and keep the meaningful prefix.
    """
    raw = (text or "").strip()
    if not raw:
        return raw
    prefix_match = re.match(r"^(\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*)", raw)
    prefix = prefix_match.group(1) if prefix_match else ""
    content = raw[len(prefix):]
    if not content.strip():
        return raw
    # Nothing to trim if the text is not degenerate in the first place.
    if not assess_transcript(content).degenerate:
        return raw
    loop = _REPEAT_LOOP.search(content)
    if loop is None:
        return raw
    kept = content[: loop.start()].strip()
    if not kept:
        return raw
    # Make sure what we keep is genuinely non-degenerate.
    if assess_transcript(kept).degenerate:
        return raw
    return f"{prefix}{kept}"
