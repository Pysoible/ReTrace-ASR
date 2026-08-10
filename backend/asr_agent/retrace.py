"""Agent-first orchestration for realtime, revisable ASR sessions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from asr_agent.context_judge import (
    ContextJudgment,
    normalize_judgment,
)
from asr_agent.ledger import RevisionLedger
from asr_agent.memory import LongTermMemoryRepository, MemoryConsolidator, MemoryRetriever
from asr_agent.models import (
    EntityProfile,
    EvidenceRef,
    Hypothesis,
    MemoryBelief,
    RevisionEvent,
    Session,
    Turn,
    WorkingHypothesis,
)
from asr_agent.resolver import EvidenceResolver
from asr_agent.storage import SessionRepository, VersionConflict


ContextJudge = Callable[..., ContextJudgment | dict[str, Any]]


class ReTraceService:
    """Store immutable ASR observations and revise projections from verified evidence."""

    def __init__(
        self,
        storage_dir: Path,
        *,
        context_judge: ContextJudge | None = None,
        audio_verifier: Callable[..., dict[str, Any]] | None = None,
        resolver: EvidenceResolver | None = None,
        memory_dir: Path | None = None,
        **legacy_adapters: Any,
    ) -> None:
        # Unknown legacy adapters are intentionally ignored: they cannot re-enable
        # the removed heuristic nomination pipeline.
        del legacy_adapters
        root = Path(storage_dir)
        self.repository = SessionRepository(root)
        self.ledger = RevisionLedger()
        self.long_term_memory = LongTermMemoryRepository(memory_dir or root / ".memory")
        self.memory_retriever = MemoryRetriever(self.long_term_memory)
        self.memory_consolidator = MemoryConsolidator(self.long_term_memory)
        if context_judge is None:
            from asr_agent.integrations.deepseek import judge_context

            context_judge = judge_context
        self.context_judge = context_judge
        self.resolver = resolver or EvidenceResolver(audio_verifier=audio_verifier)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self.repository.load(session_id).as_dict()

    def reset_session(self, session_id: str, *, memory_scope: str = "default") -> dict[str, Any]:
        return self.repository.create(Session(session_id, memory_scope=memory_scope)).as_dict()

    def upsert_entities(self, session_id: str, profiles: list[EntityProfile]) -> dict[str, Any]:
        def mutate(session: Session) -> None:
            for profile in profiles:
                session.entities[profile.entity_id] = profile

        return self.repository.update(session_id, mutate).as_dict()

    def observe_turn(
        self,
        session_id: str,
        turn_id: str,
        text: str,
        *,
        confidence: dict[str, float] | None = None,
        text_candidates: dict[str, list[str]] | None = None,
        entity_candidate_ids: dict[str, list[str]] | None = None,
        nbest: list[str] | None = None,
        source: str = "text",
        meta: dict[str, Any] | None = None,
        memory_scope: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        raw_text = str(text)
        if not raw_text.strip():
            raise ValueError("turn text cannot be empty")

        def mutate(session: Session) -> None:
            if any(turn.turn_id == turn_id for turn in session.turns):
                raise ValueError(f"duplicate turn_id: {turn_id}")
            if memory_scope:
                session.memory_scope = memory_scope
            turn_meta = dict(meta or {})
            turn_meta["asr_signals"] = {
                "confidence": dict(confidence or {}),
                "text_candidates": dict(text_candidates or {}),
                "entity_candidate_ids": dict(entity_candidate_ids or {}),
                "nbest": list(nbest or []),
            }
            session.turns.append(Turn(turn_id, raw_text, raw_text, source=source, meta=turn_meta))
            session.analysis_status = "queued"

        session = self.repository.update(session_id, mutate)
        return {"status": "queued", "observed_version": session.version, "session": session.as_dict()}

    def analyze_turn(self, session_id: str, turn_id: str, *, max_retries: int = 3) -> dict[str, Any]:
        for attempt in range(max_retries):
            snapshot = self.repository.load(session_id)
            trigger = next((turn for turn in snapshot.turns if turn.turn_id == turn_id), None)
            if trigger is None:
                raise ValueError(f"unknown turn_id: {turn_id}")
            snapshot.analysis_status = "analyzing"
            memory = self.memory_retriever.retrieve(snapshot, trigger)
            try:
                raw_judgment = self.context_judge(session=snapshot, current_turn=trigger, memory=memory)
                judgment = normalize_judgment(raw_judgment, snapshot)
            except Exception as exc:
                judgment = ContextJudgment("UNCERTAIN", rationale=f"context judge unavailable: {exc}")

            events: list[RevisionEvent] = []
            deferred = judgment.outcome == "UNCERTAIN" and not judgment.focus
            for focus in judgment.focus:
                hypothesis_id = f"hyp-{uuid4().hex}"
                hypothesis = WorkingHypothesis(
                    hypothesis_id=hypothesis_id,
                    target_turn_ids=[focus.target_turn_id],
                    current_interpretation=focus.span,
                    proposed_interpretation=focus.proposed_text,
                    alternatives=focus.alternatives,
                    supporting_evidence=[EvidenceRef(item, "context", focus.proposed_text) for item in focus.evidence_turn_ids],
                    created_version=snapshot.version,
                    last_evaluated_version=snapshot.version,
                )
                resolution = self.resolver.resolve(
                    snapshot,
                    trigger,
                    focus,
                    context_confidence=judgment.confidence,
                    memory_support=self._memory_support(memory.long_term_beliefs, focus.proposed_text),
                )
                hypothesis.score = resolution.score
                if resolution.action in {"REVISE_CURRENT", "REVISE_HISTORY"}:
                    hypothesis.status = "resolved"
                    target = next(turn for turn in snapshot.turns if turn.turn_id == resolution.target_turn_id)
                    reversed_event = next(
                        (
                            event
                            for event in reversed(self.ledger.active_events(snapshot))
                            if event.target_turn_id == target.turn_id
                            and event.span == resolution.replacement
                            and event.replacement == resolution.span
                        ),
                        None,
                    )
                    if reversed_event is not None:
                        events.append(
                            self.ledger.rollback(
                                snapshot,
                                reversed_event.event_id,
                                source_turn_id=trigger.turn_id,
                                reason=resolution.rationale or "newer verified evidence reversed the prior revision",
                                event_version=snapshot.version + 1,
                            )
                        )
                    else:
                        after_text = target.current_text.replace(resolution.span, resolution.replacement, 1)
                        events.append(RevisionEvent(
                            event_id=f"revision-{uuid4().hex}",
                            action=resolution.action,
                            target_turn_id=target.turn_id,
                            source_turn_id=trigger.turn_id,
                            span=resolution.span,
                            before_text=target.current_text,
                            after_text=after_text,
                            entity_id=None,
                            score=resolution.score,
                            evidence=resolution.evidence,
                            resolver="agent-context-audio",
                            rationale=resolution.rationale,
                            replacement=resolution.replacement,
                        ))
                    self._record_verified_belief(snapshot, focus, trigger)
                elif resolution.action == "KEEP_OLD":
                    hypothesis.status = "rejected"
                else:
                    deferred = True
                snapshot.open_hypotheses[hypothesis_id] = hypothesis
                snapshot.dependency_index.setdefault(focus.target_turn_id, [])
                for evidence_turn_id in focus.evidence_turn_ids:
                    if evidence_turn_id not in snapshot.dependency_index[focus.target_turn_id]:
                        snapshot.dependency_index[focus.target_turn_id].append(evidence_turn_id)

            self.ledger.append_many(snapshot, events, event_version=snapshot.version + 1)
            snapshot.analysis_status = "deferred" if deferred else "idle"
            try:
                committed = self.repository.commit(snapshot, expected_version=snapshot.version)
            except VersionConflict:
                if attempt + 1 == max_retries:
                    raise
                continue
            self.memory_consolidator.consolidate(committed.memory_scope, list(committed.working_beliefs.values()))
            return {
                "status": committed.analysis_status,
                "revisions": [event.as_dict() for event in events],
                "judgment": {
                    "outcome": judgment.outcome,
                    "confidence": judgment.confidence,
                    "rationale": judgment.rationale,
                },
                "session": committed.as_dict(),
            }
        raise VersionConflict("analysis could not commit after retries")

    def process_turn(self, session_id: str, turn_id: str, text: str, **kwargs: Any) -> dict[str, Any]:
        self.observe_turn(session_id, turn_id, text, **kwargs)
        return self.analyze_turn(session_id, turn_id)

    @staticmethod
    def _memory_support(beliefs: list[MemoryBelief], proposed_text: str) -> float:
        scores = [item.confidence for item in beliefs if proposed_text == item.value or proposed_text in item.aliases]
        return max(scores, default=0.0)

    @staticmethod
    def _record_verified_belief(session: Session, focus: Any, source_turn: Turn) -> None:
        belief_id = f"belief-{uuid4().hex}"
        session.working_beliefs[belief_id] = MemoryBelief(
            belief_id=belief_id,
            subject=f"turn:{focus.target_turn_id}:canonical_span",
            predicate="canonical_text",
            value=focus.proposed_text,
            aliases=[focus.span],
            confidence=0.9,
            status="provisional",
            source_turn_ids=list(dict.fromkeys([focus.target_turn_id, source_turn.turn_id])),
            source_session_ids=[session.session_id],
            evidence_kinds=["context", "audio_verified"],
        )


__all__ = [
    "EntityProfile",
    "EvidenceRef",
    "Hypothesis",
    "MemoryBelief",
    "ReTraceService",
    "RevisionEvent",
    "Session",
    "Turn",
    "WorkingHypothesis",
]
