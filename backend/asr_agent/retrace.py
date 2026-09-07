"""Agent-first orchestration for realtime, revisable ASR sessions."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from asr_agent.context_judge import (
    BeliefProposal,
    ContextJudgment,
    FocusProposal,
    language_compatible,
    text_script_profile,
    normalize_judgment,
)
from asr_agent.correction_candidates import (
    CorrectionCandidate,
    candidate_to_focus,
    deduplicate_candidates,
    focus_to_candidate,
)
from asr_agent.degeneration import (
    DegenerationAssessment,
    assess_transcript,
    assess_repeated_tail,
    should_replace_degenerate,
    trim_degenerate_tail,
    trim_repeated_tail,
)
from asr_agent.ledger import RevisionLedger
from asr_agent.memory import LongTermMemoryRepository, MemoryConsolidator, MemoryRetriever
from asr_agent.models import (
    DecisionState,
    EvidenceRef,
    MemoryBelief,
    RevisionEvent,
    Session,
    Turn,
    WorkingHypothesis,
)
from asr_agent.resolver import EvidenceResolver
from asr_agent.storage import SessionRepository, VersionConflict


ContextJudge = Callable[..., ContextJudgment | dict[str, Any]]
AudioRetranscriber = Callable[..., dict[str, Any]]


class ReTraceService:
    """Store immutable ASR observations and revise projections from verified evidence."""

    def __init__(
        self,
        storage_dir: Path,
        *,
        context_judge: ContextJudge | None = None,
        audio_verifier: Callable[..., dict[str, Any]] | None = None,
        audio_retranscriber: AudioRetranscriber | None = None,
        resolver: EvidenceResolver | None = None,
        memory_dir: Path | None = None,
    ) -> None:
        root = Path(storage_dir)
        self.repository = SessionRepository(root)
        self.ledger = RevisionLedger()
        self.long_term_memory = LongTermMemoryRepository(memory_dir or root / ".memory")
        self.memory_retriever = MemoryRetriever(self.long_term_memory)
        if context_judge is None:
            from asr_agent.integrations.deepseek import judge_context

            context_judge = judge_context
        self.context_judge = context_judge
        self.resolver = resolver or EvidenceResolver(audio_verifier=audio_verifier)
        if audio_retranscriber is None:
            from asr_agent.integrations.audio_verifier import retranscribe_window

            audio_retranscriber = retranscribe_window
        self.audio_retranscriber = audio_retranscriber
        self.memory_consolidator = MemoryConsolidator(
            self.long_term_memory,
            confidence_threshold=self.resolver.policy.thresholds.long_memory,
        )

    def get_session(self, session_id: str) -> dict[str, Any]:
        session = self.repository.load(session_id)
        self.ledger.replay(session)
        session.decision_state = self._decision_state_for_session(session)
        result = session.as_dict()
        result["observability"] = self._observability(session)
        return result

    def summarize_session(self, session_id: str) -> dict[str, Any]:
        session = self.repository.load(session_id)
        self.ledger.replay(session)
        session.decision_state = self._decision_state_for_session(session)
        return self._session_summary(session)

    def reset_session(self, session_id: str, *, memory_scope: str = "default") -> dict[str, Any]:
        return self.repository.create(Session(session_id, memory_scope=memory_scope)).as_dict()

    def observe_turn(
        self,
        session_id: str,
        turn_id: str,
        text: str,
        *,
        confidence: dict[str, float] | None = None,
        text_candidates: dict[str, list[str]] | None = None,
        nbest: list[str] | None = None,
        source: str = "text",
        meta: dict[str, Any] | None = None,
        memory_scope: str | None = None,
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
                "nbest": list(nbest or []),
            }
            session.turns.append(Turn(turn_id, raw_text, raw_text, source=source, meta=turn_meta))
            session.analysis_status = "queued"

        session = self.repository.update(session_id, mutate)
        return {"status": "queued", "observed_version": session.version, "session": session.as_dict()}

    def analyze_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        observed_version: int | None = None,
        session_complete: bool = False,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        for attempt in range(max_retries):
            snapshot = self.repository.load(session_id)
            self.ledger.replay(snapshot)
            trigger = next((turn for turn in snapshot.turns if turn.turn_id == turn_id), None)
            if trigger is None:
                raise ValueError(f"unknown turn_id: {turn_id}")
            if session_complete:
                trigger.meta["session_complete"] = True
            if observed_version is not None and observed_version > snapshot.version:
                raise VersionConflict(f"observed version {observed_version} is newer than session {snapshot.version}")
            if observed_version is not None and trigger.meta.get("analyzed_observed_version") == observed_version:
                return {
                    "status": snapshot.analysis_status,
                    "decisions": [],
                    "revisions": [],
                    "judgment": dict(trigger.meta.get("context_judgment") or {}),
                    "revalidated": snapshot.version != observed_version,
                    "session": snapshot.as_dict(),
                }
            snapshot.analysis_status = "analyzing"
            recovery_event, degeneration = self._recover_degenerate_turn(
                snapshot,
                trigger,
                observed_version=observed_version,
            )
            tail_degenerate = assess_repeated_tail(trigger.raw_text, min_tail_chars=24)
            if recovery_event is None and tail_degenerate.degenerate:
                trimmed_tail = trim_repeated_tail(trigger.raw_text, min_tail_chars=24)
                if trimmed_tail != trigger.raw_text:
                    after_text = trimmed_tail
                    meta = trigger.meta
                    meta["degeneration"] = {
                        **(meta.get("degeneration") or {}),
                        "detected": True,
                        "score": tail_degenerate.score,
                        "reasons": list(dict.fromkeys([*(meta.get("degeneration") or {}).get("reasons", []), "repeated_tail"])),
                        "recovered": True,
                        "trimmed": True,
                        "replacement": after_text,
                    }
                    recovery_event = RevisionEvent(
                        event_id=self._event_id(snapshot.session_id, trigger.turn_id, observed_version, "REVISE_CURRENT", "repeated-tail"),
                        action="REVISE_CURRENT",
                        target_turn_id=trigger.turn_id,
                        source_turn_id=trigger.turn_id,
                        span=trigger.raw_text,
                        before_text=trigger.raw_text,
                        after_text=after_text,
                        entity_id=None,
                        score=tail_degenerate.score,
                        evidence=[f"audio:{trigger.meta.get('audio_path')}:{trigger.meta.get('start_sec')}-{trigger.meta.get('end_sec')}", "degeneration:repeated_tail"],
                        resolver="audio-degeneration-recovery",
                        rationale="repeated tail removed after open re-transcription failed to recover the turn",
                        replacement=after_text,
                    )
            if recovery_event is None and tail_degenerate.degenerate:
                degeneration = DegenerationAssessment(
                    True,
                    max(degeneration.score, tail_degenerate.score),
                    tuple(dict.fromkeys([*degeneration.reasons, *tail_degenerate.reasons])),
                )
            if recovery_event is not None:
                trigger.current_text = recovery_event.after_text
            memory = self.memory_retriever.retrieve(snapshot, trigger)
            if degeneration.degenerate and recovery_event is None:
                judgment = ContextJudgment(
                    "UNCERTAIN",
                    rationale="first-pass transcript is degenerate and audio re-transcription did not recover it",
                )
            else:
                try:
                    raw_judgment = self.context_judge(session=snapshot, current_turn=trigger, memory=memory)
                    judgment = normalize_judgment(raw_judgment, snapshot)
                except Exception as exc:
                    judgment = ContextJudgment("UNCERTAIN", rationale=f"context judge unavailable: {exc}")

            candidate_pool = [focus_to_candidate(focus, snapshot) for focus in judgment.focus]
            candidate_pool.extend(
                focus_to_candidate(focus, snapshot)
                for focus in self._acoustic_focus(trigger)
            )
            trigger_uncertainty = trigger.meta.get("uncertainty") or {}
            has_acoustic_disagreement = bool(trigger_uncertainty.get("acoustic_disagreement"))
            if not judgment.focus and (
                judgment.outcome == "CONFLICT"
                or (judgment.outcome == "UNCERTAIN" and has_acoustic_disagreement)
            ):
                try:
                    from asr_agent.integrations.deepseek import _homophone_candidates

                    candidates = _homophone_candidates(snapshot, trigger)
                    by_span: dict[str, list[dict[str, str]]] = {}
                    for candidate in candidates:
                        by_span.setdefault(candidate["span"], []).append(candidate)
                    for span, matches in by_span.items():
                        matches = [
                            match for match in matches
                            if not any(particle in match["candidate"] for particle in ("啊", "呀", "吧", "呢", "啦"))
                        ]
                        if len(matches) != 1 or span not in (trigger.current_text or trigger.raw_text):
                            continue
                        match = matches[0]
                        candidate_pool.append(CorrectionCandidate(
                            target_turn_id=trigger.turn_id,
                            span=span,
                            candidate=match["candidate"],
                            source="history_homophone",
                            evidence_turn_ids=list(match.get("evidence_turn_ids") or []),
                            rationale="context conflict exposes a same-pronunciation history candidate",
                            semantic_confidence=judgment.confidence,
                            audio_start_sec=(trigger.meta or {}).get("start_sec"),
                            audio_end_sec=(trigger.meta or {}).get("end_sec"),
                        ))
                except Exception:
                    pass

            judgment.focus = [candidate_to_focus(item) for item in deduplicate_candidates(candidate_pool)]
            if judgment.focus and judgment.outcome not in {"CONFLICT", "UNCERTAIN"}:
                judgment.outcome = "UNCERTAIN"
                judgment.confidence = max(judgment.confidence, 0.5)

            self._apply_beliefs(snapshot, judgment.beliefs)
            events: list[RevisionEvent] = [event for event in (recovery_event,) if event is not None]
            # Open-vocabulary relisten fallback: when the judge is uncertain (or
            # conflicted) but could not point at a specific span, go back to the
            # audio and re-transcribe the window. This catches garbled proper
            # nouns even when the correct word was never mentioned in memory.
            # Char-level acoustic confidence (low_conf_chars) is an independent
            # acoustic doubt signal: even a CONSISTENT judgment is overridden so
            # the agent re-listens to a window the acoustic model itself flagged
            # as unreliable — audio evidence drives both *whether* to doubt and
            # *whether* to adopt the relisten, instead of the LLM judge alone.
            trigger_uncertainty = trigger.meta.get("uncertainty") or {}
            has_acoustic_doubt = bool(trigger_uncertainty.get("low_conf_chars"))
            has_coverage_risk = bool((trigger_uncertainty.get("coverage") or {}).get("truncated"))
            if (
                not recovery_event
                and not judgment.focus
                and (judgment.outcome in {"UNCERTAIN", "CONFLICT"} or has_acoustic_doubt or has_coverage_risk)
            ):
                relisten_result = self._relisten_uncertain_window(
                    snapshot,
                    trigger,
                    observed_version=observed_version,
                    memory=memory,
                )
                if isinstance(relisten_result, RevisionEvent):
                    events.append(relisten_result)
                    trigger.current_text = relisten_result.after_text
                elif relisten_result:
                    candidate_pool.extend(relisten_result)
                    judgment.focus = [
                        candidate_to_focus(item)
                        for item in deduplicate_candidates(candidate_pool)
                    ]
            audit_events: list[RevisionEvent] = []
            candidate_audits: list[RevisionEvent] = []
            deferred = judgment.outcome == "UNCERTAIN" and not judgment.focus
            if not judgment.focus:
                action = {"NOVEL": "ACCEPT_NEW", "CONSISTENT": "KEEP_OLD"}.get(judgment.outcome, "DEFER")
                audit_events.append(
                    self._decision_event(
                        snapshot,
                        trigger,
                        action=action,
                        score=judgment.confidence,
                        rationale=judgment.rationale,
                        observed_version=observed_version,
                    )
                )
            for focus_index, focus in enumerate(judgment.focus):
                hypothesis_id = f"hyp-{uuid4().hex}"
                hypothesis = WorkingHypothesis(
                    hypothesis_id=hypothesis_id,
                    target_turn_ids=[focus.target_turn_id],
                    current_interpretation=focus.span,
                    proposed_interpretation=focus.proposed_text,
                    alternatives=focus.alternatives,
                    relationship=focus.relationship,
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
                candidate_id = self._event_id(
                    session_id,
                    focus.target_turn_id,
                    observed_version,
                    "CANDIDATE",
                    f"{focus_index}:{focus.span}:{focus.proposed_text}",
                )
                candidate_stage = (
                    "committed"
                    if resolution.action in {"REVISE_CURRENT", "REVISE_HISTORY"}
                    else "rejected"
                    if resolution.action == "KEEP_OLD"
                    else "deferred"
                )
                candidate_evidence = [
                    f"candidate_source:{focus.source}",
                    *[f"evidence_turn:{item}" for item in focus.evidence_turn_ids],
                    f"verifier_attempted:{str(resolution.verifier_attempted).lower()}",
                ]
                if resolution.verifier_succeeded:
                    candidate_evidence.append(
                        f"audio_verified:{resolution.audio_confidence:.3f}/{resolution.audio_margin:.3f}"
                    )
                candidate_audits.append(RevisionEvent(
                    event_id=candidate_id,
                    action=resolution.action,
                    target_turn_id=resolution.target_turn_id,
                    source_turn_id=trigger.turn_id,
                    span=resolution.span,
                    before_text=next(turn.current_text for turn in snapshot.turns if turn.turn_id == resolution.target_turn_id),
                    after_text=next(turn.current_text for turn in snapshot.turns if turn.turn_id == resolution.target_turn_id),
                    entity_id=None,
                    score=resolution.score,
                    evidence=candidate_evidence,
                    resolver="candidate-pipeline",
                    rationale=resolution.rationale,
                    replacement=focus.proposed_text,
                    event_kind="candidate_audit",
                    candidate_id=candidate_id,
                    candidate_stage=candidate_stage,
                ))
                hypothesis.score = resolution.score
                if resolution.action in {"REVISE_CURRENT", "REVISE_HISTORY"}:
                    hypothesis.status = "resolved"
                    target = next(turn for turn in snapshot.turns if turn.turn_id == resolution.target_turn_id)
                    reversed_event = next(
                        (
                            event
                            for event in reversed(self.ledger.active_events(snapshot))
                            if event.event_kind == "revision"
                            and event.target_turn_id == target.turn_id
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
                            event_id=self._event_id(session_id, turn_id, observed_version, resolution.action, focus_index),
                            action=resolution.action,
                            target_turn_id=target.turn_id,
                            source_turn_id=trigger.turn_id,
                            span=resolution.span,
                            before_text=target.current_text,
                            after_text=after_text,
                            entity_id=None,
                            score=resolution.score,
                            evidence=[
                                f"candidate_source:{focus.source}",
                                *[f"evidence_turn:{turn_id}" for turn_id in focus.evidence_turn_ids],
                                *resolution.evidence,
                            ],
                            resolver="agent-context-audio",
                            rationale=resolution.rationale,
                            replacement=resolution.replacement,
                        ))
                    self._record_verified_belief(snapshot, focus, trigger)
                elif resolution.action == "KEEP_OLD":
                    hypothesis.status = "rejected"
                    events.append(
                        self._decision_event(
                            snapshot,
                            trigger,
                            action="KEEP_OLD",
                            score=resolution.score,
                            rationale=resolution.rationale,
                            target_turn_id=resolution.target_turn_id,
                            span=resolution.span,
                            observed_version=observed_version,
                            discriminator=str(focus_index),
                        )
                    )
                elif resolution.action in {"COEXIST", "ACCEPT_NEW"}:
                    hypothesis.status = "resolved"
                    events.append(
                        self._decision_event(
                            snapshot,
                            trigger,
                            action=resolution.action,
                            score=resolution.score,
                            rationale=resolution.rationale,
                            target_turn_id=resolution.target_turn_id,
                            span=resolution.span,
                            evidence=resolution.evidence,
                            observed_version=observed_version,
                            discriminator=str(focus_index),
                        )
                    )
                else:
                    deferred = True
                    events.append(
                        self._decision_event(
                            snapshot,
                            trigger,
                            action="DEFER",
                            score=resolution.score,
                            rationale=resolution.rationale,
                            target_turn_id=resolution.target_turn_id,
                            span=resolution.span,
                            observed_version=observed_version,
                            discriminator=str(focus_index),
                        )
                    )
                snapshot.open_hypotheses[hypothesis_id] = hypothesis
                snapshot.dependency_index.setdefault(focus.target_turn_id, [])
                for evidence_turn_id in focus.evidence_turn_ids:
                    if evidence_turn_id not in snapshot.dependency_index[focus.target_turn_id]:
                        snapshot.dependency_index[focus.target_turn_id].append(evidence_turn_id)

            self.ledger.append_many(snapshot, [*audit_events, *candidate_audits, *events], event_version=snapshot.version + 1)
            snapshot.analysis_status = "deferred" if deferred else "idle"
            trigger.meta["analyzed_observed_version"] = observed_version
            trigger.meta["analyzed_session_version"] = snapshot.version + 1
            trigger.meta["analysis_revalidated"] = observed_version is not None and snapshot.version != observed_version
            trigger.meta["context_judgment"] = {
                "outcome": judgment.outcome,
                "confidence": judgment.confidence,
                "rationale": judgment.rationale,
            }
            snapshot.decision_state = self._decision_state_for_session(snapshot)
            try:
                committed = self.repository.commit(snapshot, expected_version=snapshot.version)
            except VersionConflict:
                if attempt + 1 == max_retries:
                    raise
                continue
            committed.decision_state = self._decision_state_for_session(committed)
            self.memory_consolidator.consolidate(committed.memory_scope, list(committed.working_beliefs.values()))
            return {
                "status": committed.analysis_status,
                "decisions": [event.as_dict() for event in [*audit_events, *candidate_audits, *events]],
                "revisions": [
                    event.as_dict()
                    for event in events
                    if event.action in {"REVISE_CURRENT", "REVISE_HISTORY", "ROLLBACK"}
                ],
                "judgment": {
                    "outcome": judgment.outcome,
                    "confidence": judgment.confidence,
                    "rationale": judgment.rationale,
                },
                "revalidated": observed_version is not None and committed.version - 1 != observed_version,
                "session": committed.as_dict(),
            }
        raise VersionConflict("analysis could not commit after retries")

    @staticmethod
    def _acoustic_focus(turn: Turn) -> list[FocusProposal]:
        disagreement = (turn.meta.get("uncertainty") or {}).get("acoustic_disagreement") or []
        focus: list[FocusProposal] = []
        for item in disagreement:
            span = str(item.get("span_a") or "").strip()
            proposed = str(item.get("span_b") or "").strip()
            if not span or not proposed or span == proposed or span not in turn.current_text:
                continue
            focus.append(FocusProposal(
                target_turn_id=turn.turn_id,
                span=span,
                proposed_text=proposed,
                alternatives=[span, proposed],
                evidence_turn_ids=[turn.turn_id],
                rationale="independent ASR transcripts disagree on this span",
                source="acoustic_diff",
            ))
            if len(focus) >= 3:
                break
        return focus

    @staticmethod
    def _context_homophone_focus(session: Session, turn: Turn) -> list[FocusProposal]:
        """Find repeated in-session Chinese homophones as verification candidates.

        This is candidate discovery only: the replacement is still decided by
        the resolver's local closed-set audio verification. Requiring a repeated
        historical phrase prevents arbitrary dictionary-style rewrites.
        """
        try:
            from pypinyin import lazy_pinyin
        except ImportError:
            return []
        current = re.sub(r"^\[\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?\]\s*", "", turn.current_text or turn.raw_text)
        current_spans = {
            current[index:index + width]
            for width in range(2, 5)
            for index in range(len(current) - width + 1)
            if all("\u4e00" <= char <= "\u9fff" for char in current[index:index + width])
        }
        occurrences: dict[str, list[str]] = {}
        for historical in session.turns:
            if historical.turn_id == turn.turn_id:
                continue
            text = re.sub(r"^\[\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?\]\s*", "", historical.current_text or historical.raw_text)
            for width in range(2, 5):
                for index in range(len(text) - width + 1):
                    candidate = text[index:index + width]
                    if all("\u4e00" <= char <= "\u9fff" for char in candidate):
                        occurrences.setdefault(candidate, []).append(historical.turn_id)
        focus: list[FocusProposal] = []
        for span in sorted(current_spans, key=lambda value: (len(value), value)):
            span_pinyin = lazy_pinyin(span)
            candidates = [
                (candidate, ids)
                for candidate, ids in occurrences.items()
                if len(ids) >= 2
                and candidate != span
                and lazy_pinyin(candidate) == span_pinyin
            ]
            if not candidates:
                continue
            candidate, evidence_ids = max(candidates, key=lambda item: (len(item[1]), len(item[0])))
            focus.append(FocusProposal(
                target_turn_id=turn.turn_id,
                span=span,
                proposed_text=candidate,
                alternatives=[span, candidate],
                evidence_turn_ids=list(dict.fromkeys(evidence_ids)),
                rationale="repeated session term is a same-pronunciation candidate; verify against local audio",
            ))
            if len(focus) >= 3:
                break
        return focus

    def _recover_degenerate_turn(
        self,
        session: Session,
        turn: Turn,
        *,
        observed_version: int | None,
    ) -> tuple[RevisionEvent | None, DegenerationAssessment]:
        meta = turn.meta
        start_sec, end_sec = meta.get("start_sec"), meta.get("end_sec")
        duration = None
        if start_sec is not None and end_sec is not None:
            duration = max(0.0, float(end_sec) - float(start_sec))
        assessment = assess_transcript(turn.raw_text, duration_sec=duration)
        meta["degeneration"] = {
            "detected": assessment.degenerate,
            "score": assessment.score,
            "reasons": list(assessment.reasons),
            "recovered": False,
        }
        if not assessment.degenerate:
            return None, assessment
        if set(assessment.reasons) == {"duration_mismatch"}:
            meta["degeneration"]["error"] = "duration mismatch requires independent coverage evidence"
            return None, assessment

        audio_path = meta.get("audio_path")
        if not audio_path or start_sec is None or end_sec is None:
            meta["degeneration"]["error"] = "historical audio unavailable"
            return None, assessment
        try:
            retranscribe_kwargs = {
                "audio_path": str(audio_path),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
            if duration is not None and duration > 8.0:
                retranscribe_kwargs["recover_coverage"] = True
            result = self.audio_retranscriber(**retranscribe_kwargs)
        except Exception as exc:
            meta["degeneration"]["error"] = f"audio re-transcription failed: {exc}"
            return None, assessment

        payload = result if isinstance(result, dict) else {}
        candidate = str(payload.get("text") or "").strip() if payload.get("ok") else ""
        if candidate and not language_compatible(turn.raw_text, candidate):
            meta["degeneration"]["error"] = "re-transcription changed the transcript language"
            return None, assessment
        session_text = "".join(item.raw_text for item in session.turns)
        session_profile = text_script_profile(session_text)
        candidate_profile = text_script_profile(candidate)
        if "latin" in session_profile and "cjk" not in session_profile and "cjk" in candidate_profile:
            meta["degeneration"]["error"] = "re-transcription introduced Chinese into an English session"
            return None, assessment
        if set(assessment.reasons) == {"low_diversity"}:
            meta["degeneration"]["error"] = "low_diversity alone is insufficient for whole-turn recovery"
            return None, assessment
        candidate_assessment = assess_transcript(candidate, duration_sec=duration)
        meta["degeneration"]["candidate_score"] = candidate_assessment.score
        meta["degeneration"]["candidate_reasons"] = list(candidate_assessment.reasons)
        if not candidate or not should_replace_degenerate(assessment, candidate_assessment):
            # Fallback: the whole window may contain a meaningful prefix followed by a
            # degenerate loop (e.g. real speech then a repeated "对。"). Trim the tail.
            # trim_degenerate_tail already preserves the [start-end] timestamp prefix.
            trimmed = trim_degenerate_tail(turn.raw_text)
            trimmed_assessment = assess_transcript(trimmed, duration_sec=duration)
            meta["degeneration"]["trimmed"] = bool(trimmed != turn.raw_text)
            if trimmed != turn.raw_text and not trimmed_assessment.degenerate:
                after_text = trimmed
                meta["degeneration"]["recovered"] = True
                meta["degeneration"]["replacement"] = trimmed
                reasons = ",".join(assessment.reasons) or "degenerate transcript"
                return RevisionEvent(
                    event_id=self._event_id(
                        session.session_id,
                        turn.turn_id,
                        observed_version,
                        "REVISE_CURRENT",
                        "degeneration-trim",
                    ),
                    action="REVISE_CURRENT",
                    target_turn_id=turn.turn_id,
                    source_turn_id=turn.turn_id,
                    span=turn.raw_text,
                    before_text=turn.raw_text,
                    after_text=after_text,
                    entity_id=None,
                    score=max(0.0, min(1.0, assessment.score - trimmed_assessment.score)),
                    evidence=[f"audio:{audio_path}:{start_sec}-{end_sec}", f"degeneration:{reasons}"],
                    resolver="audio-degeneration-recovery",
                    rationale=f"degenerate tail trimmed after retranscription stayed degenerate ({reasons})",
                    replacement=after_text,
                ), assessment
            meta["degeneration"]["error"] = str(payload.get("error") or "re-transcription did not improve quality")
            return None, assessment

        prefix_match = re.match(r"^(\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*)", turn.raw_text)
        after_text = f"{prefix_match.group(1) if prefix_match else ''}{candidate}"
        meta["degeneration"]["recovered"] = True
        meta["degeneration"]["replacement"] = candidate
        reasons = ",".join(assessment.reasons) or "degenerate transcript"
        return RevisionEvent(
            event_id=self._event_id(
                session.session_id,
                turn.turn_id,
                observed_version,
                "REVISE_CURRENT",
                "degeneration",
            ),
            action="REVISE_CURRENT",
            target_turn_id=turn.turn_id,
            source_turn_id=turn.turn_id,
            span=turn.raw_text,
            before_text=turn.raw_text,
            after_text=after_text,
            entity_id=None,
            score=max(0.0, min(1.0, assessment.score - candidate_assessment.score)),
            evidence=[f"audio:{audio_path}:{start_sec}-{end_sec}", f"degeneration:{reasons}"],
            resolver="audio-degeneration-recovery",
            rationale=f"open re-transcription recovered first-pass degeneration ({reasons})",
            replacement=after_text,
        ), assessment

    def _relisten_uncertain_window(
        self,
        session: Session,
        turn: Turn,
        *,
        observed_version: int | None,
        memory: Any | None = None,
    ) -> RevisionEvent | list[CorrectionCandidate] | None:
        """Open-vocabulary relisten when the judge is uncertain but gave no focus.

        If the model flagged the turn as UNCERTAIN/CONFLICT without pinpointing a
        specific span (e.g. a garbled proper noun when the correct entity was
        never mentioned), we still go back to the audio and re-transcribe the
        window. If the relisten clearly differs from the first pass, we revise —
        this does not depend on the correct word being remembered anywhere.

        When memory is available, its remembered domain terms are passed as
        hotwords (domain_hints) so the ASR model is biased toward the correct
        proper noun (contextual biasing). Optional: relisten works without hints.
        """
        meta = turn.meta
        start_sec, end_sec = meta.get("start_sec"), meta.get("end_sec")
        audio_path = meta.get("audio_path")
        if not audio_path or start_sec is None or end_sec is None:
            return None

        def _strip(ts: str) -> str:
            return re.sub(r"^\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*", "", ts).strip()

        raw_body = _strip(turn.raw_text)
        # Char-level acoustic confidence already gave us the second ASR's own
        # transcript (paraformer_text) with per-character confidence. Use a
        # *high-confidence* char-level correction as the "second opinion": it
        # only rewrites characters the second ASR was itself sure about, which is
        # far safer than wholesale adopting paraformer's whole transcript.
        uncertainty = meta.get("uncertainty") or {}
        paraformer_text = str(uncertainty.get("paraformer_text") or "").strip()
        char_confs = uncertainty.get("paraformer_char_confs") or []
        coverage = uncertainty.get("coverage") or {}
        coverage_risk = bool(coverage.get("truncated"))
        hints: list[str] | None = None
        is_high_conf = False
        # Omission is not a char-level ambiguity: use a segmented re-ASR to
        # recover speech after the first model stopped early. Only non-omission
        # windows may use paraformer's minimal high-confidence substitutions.
        if paraformer_text and char_confs and not coverage_risk:
            from asr_agent.integrations import acoustic

            candidate = acoustic.high_conf_correction(raw_body, paraformer_text, char_confs)
            is_high_conf = True
        elif paraformer_text and not coverage_risk:
            candidate = paraformer_text
        else:
            if memory is not None:
                try:
                    from asr_agent.integrations.deepseek import _domain_entities

                    hints = _domain_entities(memory) or None
                except Exception:
                    hints = None
            try:
                result = self.audio_retranscriber(
                    audio_path=str(audio_path),
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    domain_hints=hints,
                    recover_coverage=coverage_risk,
                )
            except Exception as exc:
                meta["relisten_uncertain"] = {"error": f"audio re-transcription failed: {exc}"}
                return None
            payload = result if isinstance(result, dict) else {}
            candidate = str(payload.get("text") or "").strip() if payload.get("ok") else ""
        if not candidate:
            meta["relisten_uncertain"] = {"error": "empty relisten"}
            return None

        cand_body = _strip(candidate)
        if not raw_body or not cand_body:
            return None
        if not language_compatible(raw_body, cand_body):
            meta["relisten_uncertain"] = {
                "relistened": True,
                "changed": False,
                "rejected": "relisten changed the transcript language",
            }
            return None
        # Keep only meaningful characters for a diff decision.
        keep = lambda s: "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", s))
        raw_keep = keep(raw_body)
        cand_keep = keep(cand_body)
        if raw_keep == cand_keep:
            meta["relisten_uncertain"] = {"relistened": True, "changed": False}
            return None
        import difflib

        ratio = difflib.SequenceMatcher(None, raw_keep, cand_keep).ratio()
        # A high-confidence char-level correction changes only a few characters
        # by design, so its similarity ratio is naturally near 1.0 — do NOT reject
        # it for being "too similar". The whole point of the correction is that a
        # small, acoustically-confident edit is applied. For other relistens, a
        # ratio > 0.9 means the relisten is essentially the same transcript.
        if not is_high_conf and ratio > 0.9:
            meta["relisten_uncertain"] = {"relistened": True, "changed": False, "ratio": round(ratio, 3)}
            return None
        # A relisten that merely differs from the first pass is NOT automatically
        # better — it is the same ASR model and can be led astray (e.g. a stale
        # domain hotword like "CS:GO" biasing a LoL clip toward "C S go"). The
        # judge already had its say and produced no focus, so an unrelated
        # remembered term passed as a hotword is misleading. Adopt the relisten
        # only when it surfaced a remembered domain entity verbatim, or when it
        # clearly recovers a degenerate first pass.
        raw_assessment = assess_transcript(raw_body)
        cand_assessment = assess_transcript(cand_body)
        if not coverage_risk and not raw_assessment.degenerate:
            local_candidates: list[CorrectionCandidate] = []
            matcher = difflib.SequenceMatcher(None, raw_body, cand_body, autojunk=False)
            for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
                if tag != "replace":
                    continue
                span = raw_body[left_start:left_end].strip()
                replacement = cand_body[right_start:right_end].strip()
                if (
                    not span
                    or not replacement
                    or len(span) > 12
                    or len(replacement) > 12
                    or not language_compatible(span, replacement)
                ):
                    continue
                local_candidates.append(CorrectionCandidate(
                    target_turn_id=turn.turn_id,
                    span=span,
                    candidate=replacement,
                    source="relisten_open",
                    evidence_turn_ids=[turn.turn_id],
                    rationale="bounded audio relisten differs at this local span",
                    semantic_confidence=0.0,
                    audio_start_sec=float(start_sec),
                    audio_end_sec=float(end_sec),
                ))
            meta["relisten_uncertain"] = {
                "relistened": True,
                "changed": False,
                "ratio": round(ratio, 3),
                "candidate_count": len(local_candidates),
            }
            return local_candidates or None
        paraformer_keep = keep(paraformer_text)
        coverage_supported = bool(
            coverage_risk
            and paraformer_keep
            and len(paraformer_keep) > len(raw_keep)
            and difflib.SequenceMatcher(None, cand_keep, paraformer_keep).ratio() >= 0.7
        )
        hint_matches = [h for h in (hints or []) if h and h in cand_body]
        # Exception: when the second ASR itself flagged low-confidence characters
        # (char-level acoustic confidence), we have independent acoustic evidence
        # that this exact window is unreliable. The agent may then adopt the
        # relisten even without a remembered entity, provided the candidate is
        # not itself degenerate. This is the audio-driven correction path.
        low_conf_chars = (turn.meta.get("uncertainty") or {}).get("low_conf_chars")
        has_acoustic_doubt = bool(low_conf_chars)
        recoverable_degeneration = raw_assessment.degenerate and any(
            reason in {"duration_mismatch", "repeated_tail", "truncated"}
            for reason in raw_assessment.reasons
        )
        adopt = (
            (recoverable_degeneration and not cand_assessment.degenerate)
            or (
                coverage_risk
                and len(cand_keep) > len(raw_keep)
                and not cand_assessment.degenerate
            )
        )
        if not adopt:
            meta["relisten_uncertain"] = {
                "relistened": True,
                "changed": False,
                "ratio": round(ratio, 3),
                "rejected": (
                    "normal turn requires targeted focus for revision"
                    if not raw_assessment.degenerate and not coverage_risk
                    else "relisten differs but recovers no supported content"
                ),
            }
            return None
        # A normal, non-degenerate turn must not be replaced wholesale by an
        # open re-ASR result. Such relistens are useful evidence, but a second
        # decoding of a 30-second window can introduce more substitutions than
        # it fixes. Targeted focus resolutions are the only path allowed to
        # change a healthy turn; whole-window adoption is reserved for genuine
        # truncation/degeneration recovery.
        if not raw_assessment.degenerate and not coverage_risk:
            meta["relisten_uncertain"] = {
                "relistened": True,
                "changed": False,
                "ratio": round(ratio, 3),
                "rejected": "normal turn requires targeted focus for revision",
            }
            return None
        # candidate may itself carry a [start-end] prefix; reuse the original
        # timestamp and avoid duplicating it.
        prefix_match = re.match(r"^(\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*)", turn.raw_text)
        original_prefix = prefix_match.group(1) if prefix_match else ""
        cand_no_ts = _strip(candidate)
        after_text = f"{original_prefix}{cand_no_ts}"
        meta["relisten_uncertain"] = {
            "relistened": True,
            "changed": True,
            "ratio": round(ratio, 3),
            "replacement": cand_no_ts,
            "acoustic_doubt": has_acoustic_doubt,
            "low_conf_chars": low_conf_chars,
            "coverage_risk": coverage_risk,
            "coverage": coverage,
        }
        return RevisionEvent(
            event_id=self._event_id(
                session.session_id,
                turn.turn_id,
                observed_version,
                "REVISE_CURRENT",
                "uncertain-relisten",
            ),
            action="REVISE_CURRENT",
            target_turn_id=turn.turn_id,
            source_turn_id=turn.turn_id,
            span=turn.raw_text,
            before_text=turn.raw_text,
            after_text=after_text,
            entity_id=None,
            score=max(0.0, min(1.0, 1.0 - ratio)),
            evidence=[
                "candidate_source:relisten_open",
                f"audio:{audio_path}:{start_sec}-{end_sec}",
                "uncertainty:relisten",
                *(["acoustic:coverage-risk"] if coverage_risk else []),
                *(["acoustic:low-conf-chars"] if has_acoustic_doubt else []),
            ],
            resolver="audio-uncertainty-relisten",
            rationale="open relisten of an uncertain window recovered a different transcript",
            replacement=cand_no_ts,
        )

    def _session_summary(self, session: Session) -> dict[str, Any]:
        live_hypotheses = {
            key: hypothesis
            for key, hypothesis in session.open_hypotheses.items()
            if getattr(hypothesis, "status", "active") in {"active", "pending"}
        }
        status_counts = {"active": 0, "pending": 0, "resolved": 0, "rejected": 0, "closed": 0}
        for hypothesis in session.open_hypotheses.values():
            status = getattr(hypothesis, "status", "active")
            if status in status_counts:
                status_counts[status] += 1
        session.decision_state = self._decision_state_for_session(session)
        summary = {
            "session_id": session.session_id,
            "memory_scope": session.memory_scope,
            "analysis_status": session.analysis_status,
            "turn_count": len(session.turns),
            "revision_count": len(session.revision_events),
            "open_hypotheses": len(live_hypotheses),
            "active_hypotheses": len(live_hypotheses),
            "hypothesis_status_counts": status_counts,
            "latest_outcome": "UNKNOWN",
            "latest_confidence": 0.0,
            "latest_turn_id": None,
        }

        analyzed = next((turn for turn in reversed(session.turns) if turn.meta.get("context_judgment")), None)
        if analyzed is not None:
            judgment = dict(analyzed.meta.get("context_judgment") or {})
            summary["latest_outcome"] = str(judgment.get("outcome", "UNKNOWN"))
            summary["latest_confidence"] = float(judgment.get("confidence", 0.0))
            summary["latest_turn_id"] = analyzed.turn_id
        return summary

    def _observability(self, session: Session) -> dict[str, Any]:
        if not session.turns:
            return {
                "summary": self._session_summary(session),
                "latest_analysis": None,
                "decision_state": {
                    "accepted_facts": [],
                    "pending_hypotheses": [],
                },
                "short_term": {
                    "recent_turns": [],
                    "dependent_turns": [],
                    "working_beliefs": [],
                    "open_hypotheses": [],
                },
                "long_term": {"beliefs": []},
            }

        latest = session.turns[-1]
        packet = self.memory_retriever.retrieve(session, latest)
        # LONG-TERM inspector shows the full stable belief set, not only the
        # subset relevant to the latest turn — users expect to see what has been
        # consolidated across the whole session.
        try:
            stable_long_term = [
                item for item in self.long_term_memory.load(session.memory_scope) if item.status == "stable"
            ]
        except (OSError, ValueError, json.JSONDecodeError):
            stable_long_term = []
        analyzed = next(
            (turn for turn in reversed(session.turns) if turn.meta.get("context_judgment")),
            None,
        )
        latest_analysis = None
        if analyzed is not None:
            judgment = dict(analyzed.meta.get("context_judgment") or {})
            latest_analysis = {
                "turn_id": analyzed.turn_id,
                "outcome": judgment.get("outcome", "UNCERTAIN"),
                "confidence": float(judgment.get("confidence", 0.0)),
                "rationale": str(judgment.get("rationale", "")),
                "observed_version": analyzed.meta.get("analyzed_observed_version"),
                "analyzed_version": analyzed.meta.get("analyzed_session_version"),
                "revalidated": bool(analyzed.meta.get("analysis_revalidated", False)),
            }
        accepted_facts = [
            {
                "subject": belief.subject,
                "predicate": belief.predicate,
                "value": belief.value,
                "confidence": belief.confidence,
                "source_session_ids": belief.source_session_ids,
            }
            for belief in packet.working_beliefs
            if belief.status != "superseded"
        ]
        session.decision_state = self._decision_state_for_session(session)
        return {
            "summary": self._session_summary(session),
            "latest_analysis": latest_analysis,
            "decision_state": {
                "accepted_facts": session.decision_state.accepted_facts,
                "pending_hypotheses": session.decision_state.pending_hypotheses,
                "status_counts": session.decision_state.status_counts,
            },
            "short_term": {
                "recent_turns": [turn.as_dict() for turn in packet.recent_turns],
                "dependent_turns": [turn.as_dict() for turn in packet.dependent_turns],
                "working_beliefs": [belief.as_dict() for belief in packet.working_beliefs],
                "open_hypotheses": [hypothesis.as_dict() for hypothesis in session.open_hypotheses.values()],
            },
            "long_term": {
                "beliefs": [belief.as_dict() for belief in stable_long_term],
            },
        }

    def process_turn(
        self,
        session_id: str,
        turn_id: str,
        text: str,
        *,
        confidence: dict[str, float] | None = None,
        text_candidates: dict[str, list[str]] | None = None,
        nbest: list[str] | None = None,
        source: str = "text",
        meta: dict[str, Any] | None = None,
        memory_scope: str | None = None,
    ) -> dict[str, Any]:
        observed = self.observe_turn(
            session_id,
            turn_id,
            text,
            confidence=confidence,
            text_candidates=text_candidates,
            nbest=nbest,
            source=source,
            meta=meta,
            memory_scope=memory_scope,
        )
        fast_meta = {**(meta or {}), "confidence": confidence or {}, "text_candidates": text_candidates or {}}
        if source == "moss" and self._can_fast_keep_turn(fast_meta):
            committed = self.repository.update(
                session_id,
                lambda session: self._mark_fast_kept_turn(session, turn_id),
            )
            return {
                "status": committed.analysis_status,
                "decisions": [],
                "revisions": [],
                "judgment": {"outcome": "CONSISTENT", "confidence": 1.0, "rationale": "fast normal-turn path"},
                "revalidated": False,
                "session": committed.as_dict(),
            }
        return self.analyze_turn(session_id, turn_id, observed_version=observed["observed_version"])

    @staticmethod
    def _can_fast_keep_turn(meta: dict[str, Any] | None) -> bool:
        if os.getenv("ASR_FAST_NORMAL_TURNS", "0").strip().lower() in {"0", "false", "no", "off"}:
            return False
        uncertainty = (meta or {}).get("uncertainty") or {}
        if (meta or {}).get("session_complete"):
            return False
        if (meta or {}).get("text_candidates") or (meta or {}).get("confidence"):
            return False
        if uncertainty.get("acoustic_disagreement") or uncertainty.get("low_conf_chars"):
            return False
        if uncertainty.get("acoustic_error") or (uncertainty.get("quality") or {}).get("suspect"):
            return False
        if (uncertainty.get("coverage") or {}).get("truncated"):
            return False
        return not bool((meta or {}).get("degeneration", {}).get("detected"))

    @staticmethod
    def _mark_fast_kept_turn(session: Session, turn_id: str) -> None:
        session.analysis_status = "idle"
        turn = next(turn for turn in session.turns if turn.turn_id == turn_id)
        turn.meta["analyzed_observed_version"] = session.version
        turn.meta["analyzed_session_version"] = session.version + 1
        turn.meta["analysis_revalidated"] = False
        turn.meta["context_judgment"] = {
            "outcome": "CONSISTENT",
            "confidence": 1.0,
            "rationale": "fast normal-turn path",
        }

    def _decision_state_for_session(self, session: Session) -> DecisionState:
        accepted_facts = []
        for belief in session.working_beliefs.values():
            if belief.status != "superseded":
                accepted_facts.append(
                    {
                        "subject": belief.subject,
                        "predicate": belief.predicate,
                        "value": belief.value,
                        "confidence": belief.confidence,
                        "source_session_ids": belief.source_session_ids,
                    }
                )
        pending_hypotheses = []
        status_counts = {"active": 0, "pending": 0, "resolved": 0, "rejected": 0, "closed": 0}
        for hypothesis in session.open_hypotheses.values():
            status = getattr(hypothesis, "status", "active")
            if status in status_counts:
                status_counts[status] += 1
            if status in {"active", "pending"}:
                pending_hypotheses.append(hypothesis.as_dict())
        return DecisionState(
            accepted_facts=accepted_facts,
            pending_hypotheses=pending_hypotheses,
            status_counts=status_counts,
        )

    def _apply_beliefs(self, session: Session, proposals: list[BeliefProposal]) -> None:
        by_key = {
            (item.subject, item.predicate, item.value): item
            for item in session.working_beliefs.values()
            if item.status != "superseded"
        }
        for proposal in proposals:
            key = (proposal.subject, proposal.predicate, proposal.value)
            belief = by_key.get(key)
            if belief is None:
                belief = MemoryBelief(
                    belief_id=f"belief-{uuid4().hex}",
                    subject=proposal.subject,
                    predicate=proposal.predicate,
                    value=proposal.value,
                    aliases=list(proposal.aliases),
                    confidence=proposal.confidence,
                    valid_from=proposal.valid_from,
                    valid_to=proposal.valid_to,
                    source_turn_ids=list(proposal.evidence_turn_ids),
                    source_session_ids=[session.session_id],
                    evidence_kinds=["context"],
                    created_version=session.version + 1,
                    updated_version=session.version + 1,
                )
                session.working_beliefs[belief.belief_id] = belief
                by_key[key] = belief
            else:
                belief.confidence = max(belief.confidence, proposal.confidence)
                belief.aliases = list(dict.fromkeys([*belief.aliases, *proposal.aliases]))
                belief.source_turn_ids = list(dict.fromkeys([*belief.source_turn_ids, *proposal.evidence_turn_ids]))
                belief.updated_version = session.version + 1
            session.dependency_index[belief.belief_id] = list(belief.source_turn_ids)

    @staticmethod
    def _event_id(
        session_id: str,
        turn_id: str,
        observed_version: int | None,
        action: str,
        discriminator: str | int = "",
    ) -> str:
        key = f"{session_id}:{turn_id}:{observed_version}:{action}:{discriminator}"
        return f"decision-{uuid5(NAMESPACE_URL, key).hex}"

    def _decision_event(
        self,
        session: Session,
        source_turn: Turn,
        *,
        action: str,
        score: float,
        rationale: str,
        target_turn_id: str | None = None,
        span: str = "",
        evidence: list[str] | None = None,
        observed_version: int | None = None,
        discriminator: str = "",
    ) -> RevisionEvent:
        target_id = target_turn_id or source_turn.turn_id
        target = next(turn for turn in session.turns if turn.turn_id == target_id)
        event_kind = "audit" if not span and action in {"KEEP_OLD", "DEFER", "COEXIST", "ACCEPT_NEW"} else "revision"
        return RevisionEvent(
            event_id=self._event_id(session.session_id, source_turn.turn_id, observed_version, action, discriminator),
            action=action,
            target_turn_id=target_id,
            source_turn_id=source_turn.turn_id,
            span=span,
            before_text=target.current_text,
            after_text=target.current_text,
            entity_id=None,
            score=score,
            evidence=list(evidence or []),
            resolver="context-judge",
            rationale=rationale,
            event_kind=event_kind,
        )

    @staticmethod
    def _memory_support(beliefs: list[MemoryBelief], proposed_text: str) -> float:
        scores = [item.confidence for item in beliefs if proposed_text == item.value or proposed_text in item.aliases]
        return max(scores, default=0.0)

    @staticmethod
    def _record_verified_belief(session: Session, focus: Any, source_turn: Turn) -> None:
        belief_id = f"belief-{uuid4().hex}"
        session.working_beliefs[belief_id] = MemoryBelief(
            belief_id=belief_id,
            subject=f"turn:{focus.target_turn_id}:canonical_entity",
            predicate="canonical_entity",
            value=focus.proposed_text,
            aliases=[focus.span],
            confidence=0.9,
            status="provisional",
            source_turn_ids=list(dict.fromkeys([focus.target_turn_id, source_turn.turn_id])),
            source_session_ids=[session.session_id],
            evidence_kinds=["context", "audio_verified"],
            created_version=session.version + 1,
            updated_version=session.version + 1,
        )


__all__ = [
    "EvidenceRef",
    "MemoryBelief",
    "ReTraceService",
    "RevisionEvent",
    "Session",
    "Turn",
    "WorkingHypothesis",
]
