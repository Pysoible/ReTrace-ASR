"""Deterministic uncertainty signals used before ReTrace commits a revision."""
from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass


@dataclass
class SuspiciousSpan:
    text: str
    start: int
    end: int
    score: float
    reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_suspicious_spans(
    text: str,
    *,
    confidence: dict[str, float] | None = None,
    nbest: list[str] | None = None,
    memory_values: list[str] | None = None,
    threshold: float = 0.42,
) -> list[SuspiciousSpan]:
    """Locate questionable regions without changing the ASR observation."""
    found: dict[tuple[int, int, str], SuspiciousSpan] = {}

    def add(fragment: str, reason: str, value: float) -> None:
        start = text.find(fragment)
        if not fragment or start < 0:
            return
        key = (start, start + len(fragment), fragment)
        item = found.get(key)
        if item is None:
            item = SuspiciousSpan(fragment, start, start + len(fragment), 0.0, [])
            found[key] = item
        if reason not in item.reasons:
            item.reasons.append(reason)
        item.score = max(item.score, value) + (0.12 if len(item.reasons) > 1 else 0.0)
        item.score = min(1.0, item.score)

    for fragment, value in (confidence or {}).items():
        if float(value) < 0.65:
            add(fragment, "low_confidence", 1.0 - float(value))

    for alternative in nbest or []:
        matcher = difflib.SequenceMatcher(a=text, b=alternative)
        for tag, start, end, _, _ in matcher.get_opcodes():
            if tag != "equal" and start != end:
                disagreement = text[start:end]
                containing_confident_span = next(
                    (
                        fragment
                        for fragment in (confidence or {})
                        if text.find(fragment) <= start < text.find(fragment) + len(fragment)
                    ),
                    disagreement,
                )
                add(containing_confident_span, "nbest_disagreement", 0.65)

    for value in memory_values or []:
        if value and value not in text:
            close = difflib.get_close_matches(value, [text], n=1, cutoff=0.42)
            if close:
                add(text, "memory_conflict", 0.45)

    return sorted(
        (item for item in found.values() if item.score >= threshold),
        key=lambda item: (-item.score, item.start, item.end),
    )
