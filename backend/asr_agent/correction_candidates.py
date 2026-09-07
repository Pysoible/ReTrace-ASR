"""Normalized correction candidates shared by discovery and verification."""
from __future__ import annotations

from dataclasses import dataclass, field

from asr_agent.context_judge import FocusProposal
from asr_agent.models import Session


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
    relationship: str = "MUTUALLY_EXCLUSIVE"
    operation: str = "REPLACE"
    alternatives: list[str] = field(default_factory=list)

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
                relationship=candidate.relationship,
                operation=candidate.operation,
                alternatives=list(dict.fromkeys(candidate.alternatives)),
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
        existing.alternatives = list(dict.fromkeys([*existing.alternatives, *candidate.alternatives]))
    return list(merged.values())


def focus_to_candidate(focus: FocusProposal, session: Session) -> CorrectionCandidate:
    source = focus.source or "legacy"
    if source == "legacy":
        appears_in_history = any(
            turn.turn_id != focus.target_turn_id
            and (focus.proposed_text in turn.raw_text or focus.proposed_text in turn.current_text)
            for turn in session.turns
            if focus.proposed_text
        )
        source = "history_homophone" if appears_in_history else "semantic_open"
    target = next((turn for turn in session.turns if turn.turn_id == focus.target_turn_id), None)
    meta = target.meta if target is not None else {}
    return CorrectionCandidate(
        target_turn_id=focus.target_turn_id,
        span=focus.span,
        candidate=focus.proposed_text,
        source=source,
        evidence_turn_ids=list(focus.evidence_turn_ids),
        rationale=focus.rationale,
        audio_start_sec=meta.get("start_sec"),
        audio_end_sec=meta.get("end_sec"),
        relationship=focus.relationship,
        operation=focus.operation,
        alternatives=list(focus.alternatives),
    )


def candidate_to_focus(candidate: CorrectionCandidate) -> FocusProposal:
    return FocusProposal(
        target_turn_id=candidate.target_turn_id,
        span=candidate.span,
        proposed_text=candidate.candidate,
        alternatives=list(dict.fromkeys([
            candidate.span,
            *([candidate.candidate] if candidate.candidate else []),
            *candidate.alternatives,
        ])),
        evidence_turn_ids=list(candidate.evidence_turn_ids),
        rationale=candidate.rationale,
        source=candidate.source,
        relationship=candidate.relationship,
        operation=candidate.operation,
    )
