"""Versioned domain models for ReTrace sessions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


@dataclass
class Turn:
    turn_id: str
    raw_text: str
    current_text: str
    source: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Turn":
        return cls(
            turn_id=str(value["turn_id"]),
            raw_text=str(value["raw_text"]),
            current_text=str(value["current_text"]),
            source=str(value.get("source", "text")),
            meta=dict(value.get("meta") or {}),
        )


@dataclass
class EvidenceRef:
    turn_id: str
    kind: str
    value: str
    score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceRef":
        score = value.get("score")
        return cls(
            turn_id=str(value["turn_id"]),
            kind=str(value["kind"]),
            value=str(value["value"]),
            score=float(score) if score is not None else None,
        )


@dataclass
class WorkingHypothesis:
    hypothesis_id: str
    target_turn_ids: list[str]
    current_interpretation: str
    proposed_interpretation: str
    alternatives: list[str]
    relationship: str = "MUTUALLY_EXCLUSIVE"
    supporting_evidence: list[EvidenceRef] = field(default_factory=list)
    contradicting_evidence: list[EvidenceRef] = field(default_factory=list)
    audio_windows: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    status: str = "active"
    created_version: int = 0
    last_evaluated_version: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkingHypothesis":
        return cls(
            hypothesis_id=str(value["hypothesis_id"]),
            target_turn_ids=list(value.get("target_turn_ids", [])),
            current_interpretation=str(value["current_interpretation"]),
            proposed_interpretation=str(value["proposed_interpretation"]),
            alternatives=list(value.get("alternatives", [])),
            relationship=str(value.get("relationship", "MUTUALLY_EXCLUSIVE")),
            supporting_evidence=[EvidenceRef.from_dict(item) for item in value.get("supporting_evidence", [])],
            contradicting_evidence=[EvidenceRef.from_dict(item) for item in value.get("contradicting_evidence", [])],
            audio_windows=list(value.get("audio_windows", [])),
            score=float(value.get("score", 0.0)),
            status=str(value.get("status", "active")),
            created_version=int(value.get("created_version", 0)),
            last_evaluated_version=int(value.get("last_evaluated_version", 0)),
        )


@dataclass
class MemoryBelief:
    belief_id: str
    subject: str
    predicate: str
    value: str
    aliases: list[str] = field(default_factory=list)
    confidence: float = 0.0
    status: str = "provisional"
    valid_from: str | None = None
    valid_to: str | None = None
    source_turn_ids: list[str] = field(default_factory=list)
    source_session_ids: list[str] = field(default_factory=list)
    evidence_kinds: list[str] = field(default_factory=list)
    created_version: int = 0
    updated_version: int = 0
    supersedes: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryBelief":
        return cls(
            belief_id=str(value["belief_id"]),
            subject=str(value["subject"]),
            predicate=str(value["predicate"]),
            value=str(value["value"]),
            aliases=list(value.get("aliases", [])),
            confidence=float(value.get("confidence", 0.0)),
            status=str(value.get("status", "provisional")),
            valid_from=value.get("valid_from"),
            valid_to=value.get("valid_to"),
            source_turn_ids=list(value.get("source_turn_ids", [])),
            source_session_ids=list(value.get("source_session_ids", [])),
            evidence_kinds=list(value.get("evidence_kinds", [])),
            created_version=int(value.get("created_version", 0)),
            updated_version=int(value.get("updated_version", 0)),
            supersedes=value.get("supersedes"),
        )


@dataclass
class RevisionEvent:
    event_id: str
    action: str
    target_turn_id: str
    source_turn_id: str
    span: str
    before_text: str
    after_text: str
    entity_id: str | None
    score: float
    evidence: list[str]
    resolver: str
    active: bool = True
    rationale: str = ""
    reverted_event_id: str | None = None
    reason: str = ""
    replacement: str = ""
    session_version: int = 0
    supersedes_event_id: str | None = None
    event_kind: str = "revision"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RevisionEvent":
        return cls(
            event_id=str(value["event_id"]),
            action=str(value["action"]),
            target_turn_id=str(value["target_turn_id"]),
            source_turn_id=str(value["source_turn_id"]),
            span=str(value["span"]),
            before_text=str(value["before_text"]),
            after_text=str(value["after_text"]),
            entity_id=value.get("entity_id"),
            score=float(value["score"]),
            evidence=list(value.get("evidence", [])),
            resolver=str(value["resolver"]),
            active=_strict_bool(value.get("active", True), field_name="active"),
            rationale=str(value.get("rationale", "")),
            reverted_event_id=value.get("reverted_event_id"),
            reason=str(value.get("reason", "")),
            replacement=str(value.get("replacement", "")),
            session_version=int(value.get("session_version", 0)),
            supersedes_event_id=value.get("supersedes_event_id"),
            event_kind=str(value.get("event_kind", "revision")),
        )


@dataclass
class DecisionState:
    accepted_facts: list[dict[str, Any]] = field(default_factory=list)
    pending_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    status_counts: dict[str, int] = field(
        default_factory=lambda: {"active": 0, "pending": 0, "resolved": 0, "rejected": 0, "closed": 0}
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "DecisionState":
        payload = value or {}
        status_counts = {"active": 0, "pending": 0, "resolved": 0, "rejected": 0, "closed": 0}
        status_counts.update(payload.get("status_counts") or {})
        return cls(
            accepted_facts=list(payload.get("accepted_facts") or []),
            pending_hypotheses=list(payload.get("pending_hypotheses") or []),
            status_counts=status_counts,
        )


@dataclass
class Session:
    session_id: str
    turns: list[Turn] = field(default_factory=list)
    revision_events: list[RevisionEvent] = field(default_factory=list)
    version: int = 0
    memory_scope: str = "default"
    analysis_status: str = "idle"
    working_beliefs: dict[str, MemoryBelief] = field(default_factory=dict)
    open_hypotheses: dict[str, WorkingHypothesis] = field(default_factory=dict)
    dependency_index: dict[str, list[str]] = field(default_factory=dict)
    decision_state: DecisionState = field(default_factory=DecisionState)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Session":
        return cls(
            session_id=str(raw["session_id"]),
            version=int(raw.get("version", 0)),
            memory_scope=str(raw.get("memory_scope", "default")),
            analysis_status=str(raw.get("analysis_status", "idle")),
            turns=[Turn.from_dict(value) for value in raw.get("turns", [])],
            working_beliefs={key: MemoryBelief.from_dict(value) for key, value in raw.get("working_beliefs", {}).items()},
            open_hypotheses={key: WorkingHypothesis.from_dict(value) for key, value in raw.get("open_hypotheses", {}).items()},
            dependency_index={key: list(value) for key, value in raw.get("dependency_index", {}).items()},
            revision_events=[RevisionEvent.from_dict(value) for value in raw.get("revision_events", [])],
            decision_state=DecisionState.from_dict(raw.get("decision_state")),
        )
