"""Structured context judgment without transcript mutation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asr_agent.models import Session, Turn


JUDGMENT_OUTCOMES = {"CONSISTENT", "NOVEL", "CONFLICT", "UNCERTAIN"}
FOCUS_RELATIONSHIPS = {"MUTUALLY_EXCLUSIVE", "COEXIST", "TEMPORAL_CHANGE"}


@dataclass
class BeliefProposal:
    subject: str
    predicate: str
    value: str
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    valid_from: str | None = None
    valid_to: str | None = None
    evidence_turn_ids: list[str] = field(default_factory=list)


@dataclass
class FocusProposal:
    target_turn_id: str
    span: str
    proposed_text: str
    alternatives: list[str]
    evidence_turn_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    relationship: str = "MUTUALLY_EXCLUSIVE"


@dataclass
class ContextJudgment:
    outcome: str
    confidence: float = 0.0
    rationale: str = ""
    focus: list[FocusProposal] = field(default_factory=list)
    beliefs: list[BeliefProposal] = field(default_factory=list)


def normalize_judgment(value: ContextJudgment | dict[str, Any], session: Session) -> ContextJudgment:
    if isinstance(value, dict):
        focus = [item if isinstance(item, FocusProposal) else FocusProposal(**item) for item in value.get("focus", [])]
        beliefs = [
            item if isinstance(item, BeliefProposal) else BeliefProposal(**item)
            for item in value.get("beliefs", [])
        ]
        value = ContextJudgment(
            outcome=str(value.get("outcome", "UNCERTAIN")).upper(),
            confidence=float(value.get("confidence", 0.0)),
            rationale=str(value.get("rationale", "")),
            focus=focus,
            beliefs=beliefs,
        )
    value.outcome = value.outcome.upper()
    if value.outcome not in JUDGMENT_OUTCOMES:
        raise ValueError(f"invalid context outcome: {value.outcome}")
    value.confidence = min(1.0, max(0.0, float(value.confidence)))
    turns = {turn.turn_id: turn for turn in session.turns}
    for item in value.focus:
        target = turns.get(item.target_turn_id)
        if target is None:
            raise ValueError(f"unknown target turn: {item.target_turn_id}")
        if not item.span or (item.span not in target.raw_text and item.span not in target.current_text):
            raise ValueError(f"focus span is not present in target turn: {item.span!r}")
        if not item.proposed_text or item.proposed_text == item.span:
            raise ValueError("proposed_text must be non-empty and different from span")
        item.alternatives = list(dict.fromkeys(str(candidate) for candidate in item.alternatives if str(candidate)))
        if item.span not in item.alternatives or item.proposed_text not in item.alternatives:
            raise ValueError("alternatives must contain both current and proposed text")
        if any(turn_id not in turns for turn_id in item.evidence_turn_ids):
            raise ValueError("focus references an unknown evidence turn")
        item.relationship = item.relationship.upper()
        if item.relationship not in FOCUS_RELATIONSHIPS:
            raise ValueError(f"invalid focus relationship: {item.relationship}")
    for belief in value.beliefs:
        belief.subject = belief.subject.strip()
        belief.predicate = belief.predicate.strip()
        belief.value = belief.value.strip()
        if not belief.subject or not belief.predicate or not belief.value:
            raise ValueError("belief subject, predicate and value are required")
        belief.confidence = min(1.0, max(0.0, float(belief.confidence)))
        belief.aliases = list(dict.fromkeys(item.strip() for item in belief.aliases if item.strip()))
        if any(turn_id not in turns for turn_id in belief.evidence_turn_ids):
            raise ValueError("belief references an unknown evidence turn")
    if value.outcome in {"CONSISTENT", "NOVEL"} and value.focus:
        raise ValueError(f"{value.outcome} judgment cannot propose revisions")
    return value


class ExplicitSignalFallbackJudge:
    """Safe fallback: react to ASR metadata, never nominate text spans itself."""

    def __init__(self, low_confidence: float = 0.55) -> None:
        self.low_confidence = low_confidence

    def __call__(self, *, session: Session, current_turn: Turn, memory: Any) -> ContextJudgment:
        del session, memory
        signals = current_turn.meta.get("asr_signals") or {}
        confidences = signals.get("confidence") or {}
        nbest = [str(item) for item in signals.get("nbest") or [] if str(item).strip()]
        low = any(float(score) < self.low_confidence for score in confidences.values())
        diverse_nbest = len(set(nbest)) > 1
        if low or diverse_nbest:
            return ContextJudgment(
                outcome="UNCERTAIN",
                confidence=0.5,
                rationale="explicit ASR uncertainty requires an external context judge",
            )
        return ContextJudgment(outcome="CONSISTENT", confidence=0.7, rationale="no explicit conflict signal")
