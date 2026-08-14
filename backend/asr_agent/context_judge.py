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
    """Model JSON often returns null for empty arrays; coerce before iterating.

    Models sometimes return a single object (a dict) where the protocol wants a
    list — e.g. ``"focus": {...}`` or ``"beliefs": {...}``. Treat that as a
    one-element list instead of failing the whole judgment.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    raise ValueError(f"expected list, null or object, got {type(value).__name__}")


def _focus_from_raw(item: Any) -> FocusProposal:
    """Build a FocusProposal from model JSON, tolerating unknown extra fields.

    Models sometimes add fields not in the protocol (e.g. ``closed_set`` from a
    literal reading of "closed-set alternatives"). We only pick the fields we
    know and map common aliases instead of failing the whole judgment.
    """
    if isinstance(item, FocusProposal):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"focus item must be an object, got {type(item).__name__}")
    alternatives = _as_list(item.get("alternatives"))
    if not alternatives:
        alternatives = _as_list(item.get("closed_set"))
    if not alternatives:
        alternatives = _as_list(item.get("closed_set_alternatives"))
    return FocusProposal(
        target_turn_id=str(item.get("target_turn_id") or ""),
        span=str(item.get("span") or ""),
        proposed_text=str(item.get("proposed_text") or ""),
        alternatives=[str(value) for value in alternatives if str(value)],
        evidence_turn_ids=[str(turn_id) for turn_id in _as_list(item.get("evidence_turn_ids")) if str(turn_id)],
        rationale=str(item.get("rationale") or ""),
        relationship=str(item.get("relationship") or "MUTUALLY_EXCLUSIVE"),
    )


def _belief_from_raw(item: Any) -> BeliefProposal:
    """Build a BeliefProposal from model JSON, tolerating unknown extra fields."""
    if isinstance(item, BeliefProposal):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"belief item must be an object, got {type(item).__name__}")
    return BeliefProposal(
        subject=str(item.get("subject") or ""),
        predicate=str(item.get("predicate") or ""),
        value=str(item.get("value") or ""),
        aliases=[str(alias) for alias in _as_list(item.get("aliases")) if str(alias)],
        confidence=float(item.get("confidence") or 0.0),
        valid_from=str(item["valid_from"]) if item.get("valid_from") else None,
        valid_to=str(item["valid_to"]) if item.get("valid_to") else None,
        evidence_turn_ids=[str(turn_id) for turn_id in _as_list(item.get("evidence_turn_ids")) if str(turn_id)],
    )


def normalize_judgment(
    value: ContextJudgment | dict[str, Any],
    session: Session,
    *,
    strict: bool = False,
) -> ContextJudgment:
    """Validate/normalize a judgment.

    ``strict=True`` (used for programmatically-built judgments, e.g. in tests)
    raises on the first malformed focus/belief. Model JSON (dict input) is
    tolerated item-by-item: a single bad focus/belief is skipped so it cannot
    degrade an otherwise-useful judgment to UNCERTAIN.
    """
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
        # Model JSON is inherently noisy: tolerate per-item issues.
        strict = False
    else:
        # Programmatically-constructed judgments (tests, fallback) are strict.
        strict = True
    value.outcome = value.outcome.upper()
    if value.outcome not in JUDGMENT_OUTCOMES:
        raise ValueError(f"invalid context outcome: {value.outcome}")
    value.confidence = min(1.0, max(0.0, float(value.confidence)))
    turns = {turn.turn_id: turn for turn in session.turns}

    def _validate_focus(item: FocusProposal) -> None:
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
        item.evidence_turn_ids = [str(turn_id) for turn_id in _as_list(item.evidence_turn_ids) if str(turn_id)]
        if any(turn_id not in turns for turn_id in item.evidence_turn_ids):
            raise ValueError("focus references an unknown evidence turn")
        item.relationship = (item.relationship or "MUTUALLY_EXCLUSIVE").upper()
        if item.relationship not in FOCUS_RELATIONSHIPS:
            raise ValueError(f"invalid focus relationship: {item.relationship}")

    def _validate_belief(belief: BeliefProposal) -> None:
        belief.subject = belief.subject.strip()
        belief.predicate = belief.predicate.strip()
        belief.value = belief.value.strip()
        if not belief.subject or not belief.predicate or not belief.value:
            raise ValueError("belief subject, predicate and value are required")
        belief.confidence = min(1.0, max(0.0, float(belief.confidence)))
        belief.aliases = list(dict.fromkeys(item.strip() for item in _as_list(belief.aliases) if item and str(item).strip()))
        belief.evidence_turn_ids = [str(turn_id) for turn_id in _as_list(belief.evidence_turn_ids) if str(turn_id)]
        if any(turn_id not in turns for turn_id in belief.evidence_turn_ids):
            raise ValueError("belief references an unknown evidence turn")

    if strict:
        for item in value.focus:
            _validate_focus(item)
        for belief in value.beliefs:
            _validate_belief(belief)
    else:
        # Model JSON: skip individually-malformed items, but only fail the whole
        # judgment when nothing survives (so a lone bad span still errors loudly
        # rather than silently producing an empty judgment).
        valid_focus: list[FocusProposal] = []
        for item in value.focus:
            try:
                _validate_focus(item)
                valid_focus.append(item)
            except ValueError:
                continue
        if value.focus and not valid_focus:
            raise ValueError("all focus items are malformed")
        value.focus = valid_focus

        valid_beliefs: list[BeliefProposal] = []
        for belief in value.beliefs:
            try:
                _validate_belief(belief)
                valid_beliefs.append(belief)
            except ValueError:
                continue
        if value.beliefs and not valid_beliefs:
            raise ValueError("all beliefs are malformed")
        value.beliefs = valid_beliefs

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
        # A second independent ASR (paraformer) disagreed with the first pass —
        # an acoustic uncertainty signal that is independent of the LLM.
        disagreement = (current_turn.meta.get("uncertainty") or {}).get("acoustic_disagreement")
        if low or diverse_nbest or disagreement:
            return ContextJudgment(
                outcome="UNCERTAIN",
                confidence=0.5,
                rationale="explicit ASR uncertainty requires an external context judge",
            )
        return ContextJudgment(outcome="CONSISTENT", confidence=0.7, rationale="no explicit conflict signal")
