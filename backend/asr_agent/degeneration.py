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
