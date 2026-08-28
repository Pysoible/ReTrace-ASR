"""Structured context judgment without transcript mutation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from asr_agent.models import Session, Turn


JUDGMENT_OUTCOMES = {"CONSISTENT", "NOVEL", "CONFLICT", "UNCERTAIN"}
FOCUS_RELATIONSHIPS = {"MUTUALLY_EXCLUSIVE", "COEXIST", "TEMPORAL_CHANGE"}


def text_script_profile(text: str) -> frozenset[str]:
    """Return the writing systems present in text relevant to language locking."""
    profile: set[str] = set()
    for char in text or "":
        if "\u4e00" <= char <= "\u9fff":
            profile.add("cjk")
        elif char.isascii() and char.isalpha():
            profile.add("latin")
    return frozenset(profile)


def language_compatible(source: str, replacement: str) -> bool:
    """Prevent a local correction from changing the transcript language."""
    source_profile = text_script_profile(source)
    replacement_profile = text_script_profile(replacement)
    return bool(replacement_profile) and replacement_profile <= source_profile


def candidates_language_compatible(source: str, candidates: list[str]) -> bool:
    """Require every replacement candidate to use only scripts in the source."""
    return all(language_compatible(source, candidate) for candidate in candidates if candidate.strip())


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
    operation: str = "REPLACE"


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
    if isinstance(value, str):
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
    if not alternatives:
        alternatives = _as_list(item.get("candidates"))
    proposed = (
        item.get("proposed_text") or item.get("replacement_text") or item.get("replacement")
        or item.get("proposed") or item.get("candidate") or item.get("corrected_text")
    )
    span = item.get("span") or item.get("source_span") or item.get("current_text") or item.get("source_text")
    target_turn_id = (
        item.get("target_turn_id") or item.get("target_turn") or item.get("target_id")
        or item.get("turn_id")
    )
    evidence_turn_ids = item.get("evidence_turn_ids")
    if evidence_turn_ids is None:
        evidence_turn_ids = item.get("evidence_turn_id")
    if evidence_turn_ids is None:
        evidence_turn_ids = item.get("evidence") or item.get("supporting_turn_ids")
    relationship = item.get("relationship") or item.get("relation") or "MUTUALLY_EXCLUSIVE"
    operation = str(item.get("operation") or item.get("action") or "REPLACE").upper()
    if operation == "DELETE":
        proposed = ""
    return FocusProposal(
        target_turn_id=str(target_turn_id or ""),
        span=str(span or ""),
        proposed_text=str(proposed or ""),
        alternatives=[str(value) for value in alternatives if str(value)],
        evidence_turn_ids=[str(turn_id) for turn_id in _as_list(evidence_turn_ids) if str(turn_id)],
        rationale=str(item.get("rationale") or ""),
        relationship=str(relationship),
        operation=operation,
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

    def _repair_focus(item: FocusProposal) -> FocusProposal:
        """Repair a nearly valid model focus without inventing a candidate.

        The judge often returns a valid span/proposed pair but omits the
        alternatives or evidence array. Recover only those fields that are
        mechanically grounded in the current session; otherwise normal
        validation still rejects the item.
        """
        if not item.target_turn_id:
            item.target_turn_id = next(reversed(turns), "")
        target = turns.get(item.target_turn_id)
        if target and item.span and item.proposed_text and not item.alternatives:
            item.alternatives = [item.span, item.proposed_text]
        if target:
            valid_evidence = [turn_id for turn_id in item.evidence_turn_ids if turn_id in turns]
            if valid_evidence:
                item.evidence_turn_ids = valid_evidence
            else:
                item.evidence_turn_ids = [item.target_turn_id]
        return item

    def _validate_focus(item: FocusProposal) -> None:
        target = turns.get(item.target_turn_id)
        if target is None:
            raise ValueError(f"unknown target turn: {item.target_turn_id}")
        if not item.span or (item.span not in target.raw_text and item.span not in target.current_text):
            raise ValueError(f"focus span is not present in target turn: {item.span!r}")
        item.operation = (item.operation or "REPLACE").upper()
        if item.operation not in {"REPLACE", "DELETE"}:
            raise ValueError(f"invalid focus operation: {item.operation}")
        if item.operation == "REPLACE" and (not item.proposed_text or item.proposed_text == item.span):
            raise ValueError("proposed_text must be non-empty and different from span")
        if item.operation == "DELETE":
            item.proposed_text = ""
        item.alternatives = list(dict.fromkeys(str(candidate) for candidate in _as_list(item.alternatives) if str(candidate)))
        if item.operation == "REPLACE" and not candidates_language_compatible(item.span, item.alternatives):
            raise ValueError("replacement candidates must preserve the transcript language")
        if item.operation == "REPLACE" and (item.span not in item.alternatives or item.proposed_text not in item.alternatives):
            raise ValueError("alternatives must contain both current and proposed text")
        if item.operation == "DELETE" and item.span not in item.alternatives:
            item.alternatives.append(item.span)
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
                item = _repair_focus(item)
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
        # Preserve an explicitly grounded focus even when the model labels the
        # overall turn as consistent. The focus is still sent through the
        # normal audio evidence gate; dropping it loses a useful correction.
        if not strict:
            value.outcome = "UNCERTAIN"
        else:
            raise ValueError(f"{value.outcome} judgment cannot propose revisions")
    return value


class ExplicitSignalFallbackJudge:
    """Safe fallback: use only explicit ASR alternatives, never invent text."""

    def __init__(self, low_confidence: float = 0.55) -> None:
        self.low_confidence = low_confidence

    def __call__(self, *, session: Session, current_turn: Turn, memory: Any) -> ContextJudgment:
        del session, memory
        signals = current_turn.meta.get("asr_signals") or {}
        confidences = signals.get("confidence") or {}
        candidates_by_span = signals.get("text_candidates") or {}
        nbest = [str(item) for item in signals.get("nbest") or [] if str(item).strip()]
        low = any(float(score) < self.low_confidence for score in confidences.values())
        diverse_nbest = len(set(nbest)) > 1
        # A second independent ASR (paraformer) disagreed with the first pass —
        # an acoustic uncertainty signal that is independent of the LLM.
        uncertainty = current_turn.meta.get("uncertainty") or {}
        disagreement = uncertainty.get("acoustic_disagreement")
        low_conf_chars = uncertainty.get("low_conf_chars")
        focus: list[FocusProposal] = []
        for span, alternatives in candidates_by_span.items():
            span = str(span).strip()
            if not span or span not in current_turn.current_text:
                continue
            values = [str(item).strip() for item in alternatives or [] if str(item).strip()]
            values = list(dict.fromkeys([span, *values]))
            proposed = next((item for item in values[1:] if item != span), "")
            if not proposed or len(values) < 2:
                continue
            score = min((float(confidences.get(span, 0.0)) or 0.0), 1.0)
            if score >= self.low_confidence and not (disagreement or low_conf_chars):
                continue
            focus.append(
                FocusProposal(
                    target_turn_id=current_turn.turn_id,
                    span=span,
                    proposed_text=proposed,
                    alternatives=values,
                    evidence_turn_ids=[current_turn.turn_id],
                    rationale="explicit ASR candidate requires targeted audio verification",
                )
            )
        # Important: a transcript without any explicit ASR uncertainty is still not
        # proof of correctness. The project explicitly treats missing judge/audio
        # evidence as a safe *defer* state, not as a confident accept. Returning
        # CONSISTENT here silently makes the system look more capable than it is.
        if low or diverse_nbest or disagreement or low_conf_chars or focus:
            return ContextJudgment(
                outcome="UNCERTAIN",
                confidence=0.5,
                rationale="explicit ASR uncertainty requires an external context judge",
                focus=focus,
            )
        return ContextJudgment(
            outcome="UNCERTAIN",
            confidence=0.4,
            rationale="no explicit evidence or judge signal available; safe default is defer rather than false confidence",
            focus=[],
        )
