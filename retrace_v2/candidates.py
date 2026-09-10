"""Closed-set candidates built only from acoustically attested hypotheses."""
from __future__ import annotations

from .schemas import EditCandidate
from .text import align_edits, normalize_text


def candidates_from_hypothesis(
    raw: str,
    hypothesis: str,
    *,
    source: str,
    acoustic_score: float = 0.0,
) -> tuple[EditCandidate, ...]:
    keep = EditCandidate.keep(raw)
    if normalize_text(raw) == normalize_text(hypothesis):
        return (keep,)
    candidate = EditCandidate(
        candidate_id=f"{source}:0",
        raw_text=raw,
        candidate_text=hypothesis,
        edits=align_edits(raw, hypothesis),
        evidence_sources=(source,),
        acoustic_score=acoustic_score,
    )
    return keep, candidate
