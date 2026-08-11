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


def _as_list(value: Any) -> list[Any]:
    """Model JSON often returns null for empty arrays; coerce before iterating."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    raise ValueError(f"expected list or null, got {type(value).__name__}")


def _focus_from_raw(item: Any) -> FocusProposal:
    if isinstance(item, FocusProposal):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"focus item must be an object, got {type(item).__name__}")
    raw = dict(item)
    raw["alternatives"] = _as_list(raw.get("alternatives"))
    raw["evidence_turn_ids"] = _as_list(raw.get("evidence_turn_ids"))
    raw.setdefault("rationale", "")
    raw.setdefault("relationship", "MUTUALLY_EXCLUSIVE")
    return FocusProposal(**raw)


def _belief_from_raw(item: Any) -> BeliefProposal:
    if isinstance(item, BeliefProposal):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"belief item must be an object, got {type(item).__name__}")
    raw = dict(item)
    raw["aliases"] = _as_list(raw.get("aliases"))
    raw["evidence_turn_ids"] = _as_list(raw.get("evidence_turn_ids"))
    return BeliefProposal(**raw)


def normalize_judgment(value: ContextJudgment | dict[str, Any], session: Session) -> ContextJudgment:
    if isinstance(value, dict):
        # Some gateways return "label" instead of the protocol field "outcome".
        outcome = value.get("outcome", value.get("label", "UNCERTAIN"))
        focus = [_focus_from_raw(item) for item in _as_list(value.get("focus"))]
        beliefs = [_belief_from_raw(item) for item in _as_list(value.get("beliefs"))]
        value = ContextJudgment(
            outcome=str(outcome or "UNCERTAIN").upper(),
            confidence=float(value.get("confidence") or 0.0),
            rationale=str(value.get("rationale") or ""),
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
        item.alternatives = list(dict.fromkeys(str(candidate) for candidate in _as_list(item.alternatives) if str(candidate)))
        if item.span not in item.alternatives or item.proposed_text not in item.alternatives:
            raise ValueError("alternatives must contain both current and proposed text")
        item.evidence_turn_ids = [str(turn_id) for turn_id in _as_list(item.evidence_turn_ids)]
        if any(turn_id not in turns for turn_id in item.evidence_turn_ids):
            raise ValueError("focus references an unknown evidence turn")
        item.relationship = (item.relationship or "MUTUALLY_EXCLUSIVE").upper()
        if item.relationship not in FOCUS_RELATIONSHIPS:
            raise ValueError(f"invalid focus relationship: {item.relationship}")
    for belief in value.beliefs:
        belief.subject = belief.subject.strip()
        belief.predicate = belief.predicate.strip()
        belief.value = belief.value.strip()
        if not belief.subject or not belief.predicate or not belief.value:
            raise ValueError("belief subject, predicate and value are required")
        belief.confidence = min(1.0, max(0.0, float(belief.confidence)))
        belief.aliases = list(dict.fromkeys(item.strip() for item in _as_list(belief.aliases) if item and str(item).strip()))
        belief.evidence_turn_ids = [str(turn_id) for turn_id in _as_list(belief.evidence_turn_ids)]
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
