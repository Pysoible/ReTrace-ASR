"""Normalized correction candidates shared by discovery and verification."""
from __future__ import annotations

from dataclasses import dataclass, field

from asr_agent.context_judge import FocusProposal


@dataclass
class CorrectionCandidate:
    target_turn_id: str
    span: str
    candidate: str
    source: str
    evidence_turn_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    semantic_confidence: float = 0.0
    audio_required: bool = True
    audio_start_sec: float | None = None
    audio_end_sec: float | None = None

    def key(self) -> tuple[str, str, str]:
        return self.target_turn_id, self.span, self.candidate


def deduplicate_candidates(candidates: list[CorrectionCandidate]) -> list[CorrectionCandidate]:
    merged: dict[tuple[str, str, str], CorrectionCandidate] = {}
    for candidate in candidates:
        key = candidate.key()
        existing = merged.get(key)
        if existing is None:
            merged[key] = CorrectionCandidate(
                target_turn_id=candidate.target_turn_id,
                span=candidate.span,
                candidate=candidate.candidate,
                source=candidate.source,
                evidence_turn_ids=list(dict.fromkeys(candidate.evidence_turn_ids)),
                rationale=candidate.rationale,
                semantic_confidence=candidate.semantic_confidence,
                audio_required=candidate.audio_required,
                audio_start_sec=candidate.audio_start_sec,
                audio_end_sec=candidate.audio_end_sec,
            )
            continue
        sources = list(dict.fromkeys([*existing.source.split("|"), *candidate.source.split("|")]))
        evidence = list(dict.fromkeys([*existing.evidence_turn_ids, *candidate.evidence_turn_ids]))
        existing.source = "|".join(sources)
        existing.evidence_turn_ids = evidence
        existing.rationale = existing.rationale or candidate.rationale
        existing.semantic_confidence = max(existing.semantic_confidence, candidate.semantic_confidence)
        existing.audio_required = existing.audio_required or candidate.audio_required
        existing.audio_start_sec = existing.audio_start_sec if existing.audio_start_sec is not None else candidate.audio_start_sec
        existing.audio_end_sec = existing.audio_end_sec if existing.audio_end_sec is not None else candidate.audio_end_sec
    return list(merged.values())


def candidate_to_focus(candidate: CorrectionCandidate) -> FocusProposal:
    return FocusProposal(
        target_turn_id=candidate.target_turn_id,
        span=candidate.span,
        proposed_text=candidate.candidate,
        alternatives=[candidate.span, candidate.candidate],
        evidence_turn_ids=list(candidate.evidence_turn_ids),
        rationale=candidate.rationale,
        source=candidate.source,
    )
