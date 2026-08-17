"""Agent-first orchestration for realtime, revisable ASR sessions."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid4, uuid5

from asr_agent.context_judge import (
    BeliefProposal,
    ContextJudgment,
    normalize_judgment,
)
from asr_agent.degeneration import (
    DegenerationAssessment,
    assess_transcript,
    should_replace_degenerate,
    trim_degenerate_tail,
)
from asr_agent.ledger import RevisionLedger
from asr_agent.memory import LongTermMemoryRepository, MemoryConsolidator, MemoryRetriever
from asr_agent.models import (
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
        result = session.as_dict()
        result["observability"] = self._observability(session)
        return result

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
        max_retries: int = 3,
    ) -> dict[str, Any]:
        for attempt in range(max_retries):
            snapshot = self.repository.load(session_id)
            self.ledger.replay(snapshot)
            trigger = next((turn for turn in snapshot.turns if turn.turn_id == turn_id), None)
            if trigger is None:
                raise ValueError(f"unknown turn_id: {turn_id}")
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

            self._apply_beliefs(snapshot, judgment.beliefs)
            events: list[RevisionEvent] = [recovery_event] if recovery_event is not None else []
            # Open-vocabulary relisten fallback: when the judge is uncertain (or
            # conflicted) but could not point at a specific span, go back to the
            # audio and re-transcribe the window. This catches garbled proper
            # nouns even when the correct word was never mentioned in memory.
            # Char-level acoustic confidence (low_conf_chars) is an independent
            # acoustic doubt signal: even a CONSISTENT judgment is overridden so
            # the agent re-listens to a window the acoustic model itself flagged
            # as unreliable — audio evidence drives both *whether* to doubt and
            # *whether* to adopt the relisten, instead of the LLM judge alone.
            has_acoustic_doubt = bool((trigger.meta.get("uncertainty") or {}).get("low_conf_chars"))
            if (
                not recovery_event
                and not judgment.focus
                and (judgment.outcome in {"UNCERTAIN", "CONFLICT"} or has_acoustic_doubt)
            ):
                relisten_event = self._relisten_uncertain_window(
                    snapshot,
                    trigger,
                    observed_version=observed_version,
                    memory=memory,
                )
                if relisten_event is not None:
                    events.append(relisten_event)
                    trigger.current_text = relisten_event.after_text
            audit_events: list[RevisionEvent] = []
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
                            event_id=self._event_id(session_id, turn_id, observed_version, resolution.action, focus_index),
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

            self.ledger.append_many(snapshot, [*audit_events, *events], event_version=snapshot.version + 1)
            snapshot.analysis_status = "deferred" if deferred else "idle"
            trigger.meta["analyzed_observed_version"] = observed_version
            trigger.meta["analyzed_session_version"] = snapshot.version + 1
            trigger.meta["analysis_revalidated"] = observed_version is not None and snapshot.version != observed_version
            trigger.meta["context_judgment"] = {
                "outcome": judgment.outcome,
                "confidence": judgment.confidence,
                "rationale": judgment.rationale,
            }
            try:
                committed = self.repository.commit(snapshot, expected_version=snapshot.version)
            except VersionConflict:
                if attempt + 1 == max_retries:
                    raise
                continue
            self.memory_consolidator.consolidate(committed.memory_scope, list(committed.working_beliefs.values()))
            return {
                "status": committed.analysis_status,
                "decisions": [event.as_dict() for event in [*audit_events, *events]],
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

        audio_path = meta.get("audio_path")
        if not audio_path or start_sec is None or end_sec is None:
            meta["degeneration"]["error"] = "historical audio unavailable"
            return None, assessment
        try:
            result = self.audio_retranscriber(
                audio_path=str(audio_path),
                start_sec=float(start_sec),
                end_sec=float(end_sec),
            )
        except Exception as exc:
            meta["degeneration"]["error"] = f"audio re-transcription failed: {exc}"
            return None, assessment

        payload = result if isinstance(result, dict) else {}
        candidate = str(payload.get("text") or "").strip() if payload.get("ok") else ""
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
    ) -> RevisionEvent | None:
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
        hints: list[str] | None = None
        is_high_conf = False
        if paraformer_text and char_confs:
            from asr_agent.integrations import acoustic

            candidate = acoustic.high_conf_correction(raw_body, paraformer_text, char_confs)
            is_high_conf = True
        elif paraformer_text:
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
        hint_matches = [h for h in (hints or []) if h and h in cand_body]
        # Exception: when the second ASR itself flagged low-confidence characters
        # (char-level acoustic confidence), we have independent acoustic evidence
        # that this exact window is unreliable. The agent may then adopt the
        # relisten even without a remembered entity, provided the candidate is
        # not itself degenerate. This is the audio-driven correction path.
        low_conf_chars = (turn.meta.get("uncertainty") or {}).get("low_conf_chars")
        has_acoustic_doubt = bool(low_conf_chars)
        adopt = bool(hint_matches) or (
            raw_assessment.degenerate and not cand_assessment.degenerate
        ) or ((has_acoustic_doubt or is_high_conf) and not cand_assessment.degenerate)
        if not adopt:
            meta["relisten_uncertain"] = {
                "relistened": True,
                "changed": False,
                "ratio": round(ratio, 3),
                "rejected": "relisten differs but recovers no remembered entity nor a degenerate pass",
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
            evidence=[f"audio:{audio_path}:{start_sec}-{end_sec}", "uncertainty:relisten", *(["acoustic:low-conf-chars"] if has_acoustic_doubt else [])],
            resolver="audio-uncertainty-relisten",
            rationale="open relisten of an uncertain window recovered a different transcript",
            replacement=cand_no_ts,
        )

    def _observability(self, session: Session) -> dict[str, Any]:
        if not session.turns:
            return {
                "latest_analysis": None,
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
        return {
            "latest_analysis": latest_analysis,
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
        return self.analyze_turn(session_id, turn_id, observed_version=observed["observed_version"])

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
            subject=f"turn:{focus.target_turn_id}:canonical_span",
            predicate="canonical_text",
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
