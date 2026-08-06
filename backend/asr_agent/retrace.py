"""Evidence-grounded retrospective revision for conversational ASR."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from asr_agent.uncertainty import detect_suspicious_spans


@dataclass
class EntityProfile:
    entity_id: str
    name: str
    aliases: list[str] = field(default_factory=list)
    attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EntityProfile":
        return cls(str(value["entity_id"]), str(value["name"]), list(value.get("aliases", [])), dict(value.get("attributes", {})))


@dataclass
class Hypothesis:
    span: str
    text_candidates: list[str]
    entity_candidate_ids: list[str]
    entity_id: str | None = None
    action: str = "DEFER"
    candidates: list[dict[str, Any]] = field(default_factory=list)
    risk: str = "medium"
    decision: str = "WAIT"
    decision_rationale: list[str] = field(default_factory=list)
    evidence_packet: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    turn_id: str
    raw_text: str
    current_text: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    source: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)


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


@dataclass
class Session:
    session_id: str
    entities: dict[str, EntityProfile] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    verified_memory: dict[str, dict[str, str]] = field(default_factory=dict)
    quarantine_memory: dict[str, dict[str, str]] = field(default_factory=dict)
    revision_events: list[RevisionEvent] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Session":
        session = cls(session_id=raw["session_id"])
        session.entities = {key: EntityProfile.from_dict(value) for key, value in raw.get("entities", {}).items()}
        session.turns = [
            Turn(
                turn_id=item["turn_id"],
                raw_text=item["raw_text"],
                current_text=item["current_text"],
                hypotheses=[Hypothesis(**hypothesis) for hypothesis in item.get("hypotheses", [])],
                source=item.get("source", "text"),
                meta=dict(item.get("meta") or {}),
            )
            for item in raw.get("turns", [])
        ]
        session.verified_memory = raw.get("verified_memory", {})
        session.quarantine_memory = raw.get("quarantine_memory", {})
        session.revision_events = [RevisionEvent(**item) for item in raw.get("revision_events", [])]
        return session


class ReTraceService:
    def __init__(self, storage_dir: Path, evidence_scorer: Callable[..., dict[str, Any]] | None = None, reflector: Callable[..., list[dict[str, Any]]] | None = None) -> None:
        self.storage_dir = storage_dir
        self.evidence_scorer = evidence_scorer
        self.reflector = reflector

    def upsert_entities(self, session_id: str, profiles: list[EntityProfile]) -> dict[str, Any]:
        session = self._load(session_id)
        session.entities.update({profile.entity_id: profile for profile in profiles})
        self._save(session)
        return session.as_dict()

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._load(session_id).as_dict()

    def undo_revision(self, session_id: str, event_id: str, *, reason: str = "") -> dict[str, Any]:
        session = self._load(session_id)
        event = next((item for item in session.revision_events if item.event_id == event_id), None)
        if event is None or event.action not in {"REVISE_TEXT", "REVISE_ENTITY"}:
            raise ValueError(f"unknown revision event: {event_id}")
        if not event.active:
            raise ValueError(f"revision event already inactive: {event_id}")
        turn = next((item for item in session.turns if item.turn_id == event.target_turn_id), None)
        if turn is None:
            raise ValueError(f"revision target is missing: {event.target_turn_id}")
        before = turn.current_text
        event.active = False
        for hypothesis in turn.hypotheses:
            if hypothesis.span == event.span:
                hypothesis.action = "DEFER"
                hypothesis.entity_id = None
        self._replay(session)
        undo = RevisionEvent(
            event_id=uuid.uuid4().hex,
            action="UNDO_REVISION",
            target_turn_id=event.target_turn_id,
            source_turn_id=event.source_turn_id,
            span=event.span,
            before_text=before,
            after_text=turn.current_text,
            entity_id=event.entity_id,
            score=1.0,
            evidence=["manual undo"],
            resolver="controller",
            reverted_event_id=event.event_id,
            reason=reason,
        )
        session.revision_events.append(undo)
        self._save(session)
        return {"session": session.as_dict(), "event": asdict(undo)}

    def confirm_hypothesis(self, session_id: str, turn_id: str, span: str, candidate: str, *, reason: str = "") -> dict[str, Any]:
        """Commit a known candidate only after an operator explicitly confirms it."""
        session = self._load(session_id)
        turn = next((item for item in session.turns if item.turn_id == turn_id), None)
        if turn is None:
            raise ValueError(f"unknown turn: {turn_id}")
        hypothesis = next((item for item in turn.hypotheses if item.span == span), None)
        if hypothesis is None or candidate not in hypothesis.text_candidates:
            raise ValueError("candidate is not available for this hypothesis")
        if candidate == span:
            hypothesis.decision = "WAIT"
            hypothesis.decision_rationale = ["operator_kept_original"]
            event = RevisionEvent(
                event_id=uuid.uuid4().hex,
                action="KEEP_ORIGINAL",
                target_turn_id=turn_id,
                source_turn_id=turn_id,
                span=span,
                before_text=turn.current_text,
                after_text=turn.current_text,
                entity_id=None,
                score=1.0,
                evidence=["operator confirmation"],
                resolver="user-confirmed",
                rationale=reason,
            )
            session.revision_events.append(event)
            self._save(session)
            return {"session": session.as_dict(), "event": asdict(event)}
        before = turn.current_text
        if span not in before:
            raise ValueError("hypothesis span is not present in the current subtitle")
        turn.current_text = before.replace(span, candidate, 1)
        hypothesis.action = "REVISE_TEXT"
        hypothesis.decision = "COMMIT"
        hypothesis.decision_rationale = ["operator_confirmed"]
        event = RevisionEvent(
            event_id=uuid.uuid4().hex,
            action="REVISE_TEXT",
            target_turn_id=turn_id,
            source_turn_id=turn_id,
            span=span,
            before_text=before,
            after_text=turn.current_text,
            entity_id=None,
            score=1.0,
            evidence=["operator confirmation"],
            resolver="user-confirmed",
            rationale=reason,
            replacement=candidate,
        )
        session.revision_events.append(event)
        self._save(session)
        return {"session": session.as_dict(), "event": asdict(event)}

    @staticmethod
    def _replay(session: Session) -> None:
        """Derive visible text and memory only from immutable raw turns and active events."""
        for turn in session.turns:
            turn.current_text = turn.raw_text
        session.verified_memory = {}
        session.quarantine_memory = {}
        turns = {turn.turn_id: turn for turn in session.turns}
        for event in session.revision_events:
            if not event.active or event.action not in {"REVISE_TEXT", "REVISE_ENTITY"}:
                continue
            turn = turns.get(event.target_turn_id)
            if not turn:
                continue
            if event.replacement and event.span in turn.current_text:
                turn.current_text = turn.current_text.replace(event.span, event.replacement, 1)
            else:
                turn.current_text = event.after_text
            if event.entity_id and event.entity_id in session.entities:
                profile = session.entities[event.entity_id]
                session.verified_memory[event.entity_id] = {"name": profile.name, "source_turn_id": event.source_turn_id}
        for turn in session.turns:
            for hypothesis in turn.hypotheses:
                if hypothesis.action == "DEFER":
                    for entity_id in hypothesis.entity_candidate_ids:
                        profile = session.entities.get(entity_id)
                        if profile:
                            session.quarantine_memory[entity_id] = {"name": profile.name, "status": "candidate"}

    def reset_session(self, session_id: str) -> dict[str, Any]:
        """Replace an existing session with an empty one (one long audio = one session)."""
        session = Session(session_id=session_id)
        self._save(session)
        return session.as_dict()

    def process_turn(
        self,
        session_id: str,
        turn_id: str,
        text: str,
        *,
        confidence: dict[str, float] | None = None,
        text_candidates: dict[str, list[str]] | None = None,
        entity_candidate_ids: dict[str, list[str]] | None = None,
        use_llm: bool = False,
        source: str = "text",
        meta: dict[str, Any] | None = None,
        risk: str = "medium",
        nbest: list[str] | None = None,
    ) -> dict[str, Any]:
        session = self._load(session_id)
        if any(turn.turn_id == turn_id for turn in session.turns):
            raise ValueError(f"duplicate turn_id: {turn_id}")

        llm_meta: dict[str, Any] = dict(meta or {})
        working_text = text

        confidence, text_candidates, entity_candidate_ids = confidence or {}, text_candidates or {}, entity_candidate_ids or {}
        for span, candidate, entity_id in self._recall_entity_fallbacks(session, working_text):
            confidence.setdefault(span, 0.6)
            text_candidates.setdefault(span, [span, candidate])
            entity_candidate_ids.setdefault(span, []).append(entity_id)
        detected_spans = detect_suspicious_spans(working_text, confidence=confidence, nbest=nbest)
        for detected in detected_spans:
            text_candidates.setdefault(detected.text, [detected.text])
        spans = set(text_candidates) | set(entity_candidate_ids)
        hypotheses = []
        for span in spans:
            if span not in working_text or confidence.get(span, 1.0) >= 0.65:
                continue
            candidates = text_candidates.get(span, [span])
            recalled = [profile.entity_id for profile in session.entities.values() if any(value in {profile.name, *profile.aliases} for value in candidates)]
            unique_candidates = list(dict.fromkeys(candidates))
            probability = 1.0 / len(unique_candidates)
            decision = "ASK_USER" if risk == "high" else "WAIT"
            hypotheses.append(Hypothesis(
                span=span,
                text_candidates=unique_candidates,
                entity_candidate_ids=list(dict.fromkeys([*entity_candidate_ids.get(span, []), *recalled])),
                candidates=[{"text": candidate, "score": probability, "supporting_evidence": [], "contradicting_evidence": []} for candidate in unique_candidates],
                risk=risk,
                decision=decision,
                decision_rationale=["high_risk_requires_confirmation"] if decision == "ASK_USER" else ["competing_candidates"],
                evidence_packet={
                    "asr_uncertainty": {"confidence": confidence.get(span), "nbest": nbest or []},
                    "suspicious_span": next((item.as_dict() for item in detected_spans if item.text == span), None),
                    "later_raw_evidence": [],
                },
            ))
        for hypothesis in hypotheses:
            for entity_id in hypothesis.entity_candidate_ids:
                profile = session.entities.get(entity_id)
                if profile:
                    session.quarantine_memory[entity_id] = {"name": profile.name, "status": "candidate"}
        session.turns.append(
            Turn(
                turn_id=turn_id,
                raw_text=text,
                current_text=working_text,
                hypotheses=hypotheses,
                source=source,
                meta=llm_meta,
            )
        )
        source_index = len(session.turns) - 1
        revisions = self._reflect(session, source_index) + self._reassess(session, source_index=source_index, use_llm=use_llm)
        self._save(session)
        return {
            "session": session.as_dict(),
            "revisions": revisions,
            "actions": [item.action for item in hypotheses],
            "integrations": {"source": source},
        }

    def _reflect(self, session: Session, source_index: int) -> list[dict[str, Any]]:
        if source_index == 0:
            return []
        reflector = self.reflector
        if reflector is None:
            from asr_agent.integrations.deepseek import reflect_timeline

            reflector = reflect_timeline
        proposals = reflector(
            prior_turns=[{"turn_id": turn.turn_id, "raw_text": turn.raw_text} for turn in session.turns[:source_index]],
            new_turn={"turn_id": session.turns[source_index].turn_id, "raw_text": session.turns[source_index].raw_text},
        )
        indexes = {turn.turn_id: index for index, turn in enumerate(session.turns)}
        committed: list[dict[str, Any]] = []
        for proposal in proposals or []:
            target_id = str(proposal.get("target_turn_id") or "")
            target_index = indexes.get(target_id, source_index)
            before = str(proposal.get("before_text") or "")
            after = str(proposal.get("after_text") or "")
            score = float(proposal.get("score") or 0.0)
            target = session.turns[target_index] if target_index < source_index else None
            if target is None or not before or not after or before == after or before not in target.current_text or score < 0.7:
                continue
            hypothesis = next((item for item in target.hypotheses if item.span == before), None)
            if hypothesis and hypothesis.risk == "high":
                hypothesis.decision = "ASK_USER"
                hypothesis.decision_rationale = ["high_risk_requires_confirmation"]
                continue
            evidence: list[str] = []
            valid = True
            for item in proposal.get("evidence") or []:
                if not isinstance(item, dict):
                    valid = False
                    break
                evidence_id, quote = str(item.get("turn_id") or ""), str(item.get("quote") or "")
                evidence_index = indexes.get(evidence_id, -1)
                if not quote or evidence_index <= target_index or evidence_index > source_index or quote not in session.turns[evidence_index].raw_text:
                    valid = False
                    break
                evidence.append(f"{evidence_id}:{quote}")
            if not valid or not evidence:
                continue
            before_display = target.current_text
            target.current_text = target.current_text.replace(before, after, 1)
            event = RevisionEvent(
                event_id=uuid.uuid4().hex, action="REVISE_TEXT", target_turn_id=target.turn_id,
                source_turn_id=session.turns[source_index].turn_id, span=before, before_text=before_display,
                after_text=target.current_text, entity_id=None, score=round(score, 3), evidence=evidence,
                resolver="deepseek-reflect", rationale=str(proposal.get("rationale") or ""),
                replacement=after,
            )
            session.revision_events.append(event)
            committed.append(asdict(event))
        return committed

    @staticmethod
    def _recall_entity_fallbacks(session: Session, text: str) -> list[tuple[str, str, str]]:
        """Recall near-matching known entities as quarantined candidates, never as facts."""
        matches: list[tuple[str, str, str]] = []
        for profile in session.entities.values():
            for candidate in [profile.name, *profile.aliases]:
                if len(candidate) < 2:
                    continue
                for start in range(len(text) - len(candidate) + 1):
                    observed = text[start : start + len(candidate)]
                    suffix = 0
                    for left, right in zip(reversed(observed), reversed(candidate)):
                        if left != right:
                            break
                        suffix += 1
                    if observed != candidate and suffix >= len(candidate) - 1:
                        matches.append((observed, candidate, profile.entity_id))
        return matches

    def _reassess(self, session: Session, source_index: int, *, use_llm: bool = False) -> list[dict[str, Any]]:
        revisions: list[dict[str, Any]] = []
        for target_index, turn in enumerate(session.turns[:source_index]):
            evidence_text = " ".join(item.raw_text for item in session.turns[target_index + 1 : source_index + 1])
            for hypothesis in turn.hypotheses:
                if hypothesis.action != "DEFER":
                    continue
                rule_hit = self._rule_resolve(session, turn, hypothesis, evidence_text, source_index)
                if rule_hit:
                    revisions.append(rule_hit)
                elif use_llm:
                    llm_hit = self._llm_resolve(session, turn, hypothesis, evidence_text, source_index)
                    if llm_hit:
                        revisions.append(llm_hit)
        return revisions

    def _rule_resolve(
        self,
        session: Session,
        turn: Turn,
        hypothesis: Hypothesis,
        evidence_text: str,
        source_index: int,
    ) -> dict[str, Any] | None:
        scores: list[tuple[float, EntityProfile, list[str]]] = []
        for entity_id in hypothesis.entity_candidate_ids:
            profile = session.entities.get(entity_id)
            if not profile:
                continue
            matches = [value for value in profile.attributes.values() if value and value in evidence_text]
            if matches:
                scores.append((len(matches) / max(1, len(profile.attributes)), profile, matches))
        scores.sort(key=lambda item: item[0], reverse=True)
        if not scores or (len(scores) > 1 and scores[0][0] == scores[1][0]) or scores[0][0] < 0.5:
            return None
        score, profile, evidence = scores[0]
        return self._commit_revision(
            session,
            turn,
            hypothesis,
            source_index,
            profile=profile,
            candidate=next(
                (item for item in hypothesis.text_candidates if item == profile.name or item in profile.aliases),
                hypothesis.span,
            ),
            score=score,
            evidence=evidence,
            resolver="rule",
        )

    def _llm_resolve(self, session: Session, turn: Turn, hypothesis: Hypothesis, evidence_text: str, source_index: int) -> dict[str, Any] | None:
        scorer = self.evidence_scorer
        if scorer is None:
            from asr_agent.integrations.deepseek import score_evidence

            scorer = score_evidence
        candidates = list(dict.fromkeys(hypothesis.text_candidates or [hypothesis.span]))
        entities = [session.entities[item] for item in hypothesis.entity_candidate_ids if item in session.entities]
        result = scorer(
            span=hypothesis.span,
            text_candidates=candidates,
            entity_candidates=[asdict(item) for item in entities],
            prior_text=turn.current_text,
            evidence_text=evidence_text,
        )
        action = str(result.get("action") or "DEFER").upper()
        candidate = str(result.get("candidate") or hypothesis.span)
        entity_id = str(result.get("entity_id") or "")
        if action not in {"KEEP", "REVISE_TEXT", "REVISE_ENTITY", "DEFER", "CLARIFY"}:
            return None
        if action in {"KEEP", "DEFER", "CLARIFY"}:
            # A non-revision decision is provisional: a later turn may contain counterevidence.
            return None
        if candidate not in candidates or (entity_id and entity_id not in hypothesis.entity_candidate_ids):
            return None
        profile = session.entities.get(entity_id) if entity_id else None
        if action == "REVISE_ENTITY" and profile is None:
            return None
        target_index = next((index for index, item in enumerate(session.turns) if item.turn_id == turn.turn_id), -1)
        evidence = self._validate_later_evidence(session, target_index, source_index, result.get("evidence"))
        if not evidence:
            return None
        return self._commit_revision(
            session, turn, hypothesis, source_index, profile=profile, candidate=candidate,
            score=float(result.get("score") or 0.0), evidence=evidence,
            resolver="deepseek", forced_action=action, rationale=str(result.get("rationale") or ""),
        )

    @staticmethod
    def _validate_later_evidence(session: Session, target_index: int, source_index: int, items: Any) -> list[str]:
        indexes = {turn.turn_id: index for index, turn in enumerate(session.turns)}
        verified: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                return []
            turn_id, quote = str(item.get("turn_id") or ""), str(item.get("quote") or "")
            evidence_index = indexes.get(turn_id, -1)
            if not quote or evidence_index <= target_index or evidence_index > source_index or quote not in session.turns[evidence_index].raw_text:
                return []
            verified.append(f"{turn_id}:{quote}")
        return verified

    def _commit_revision(
        self,
        session: Session,
        turn: Turn,
        hypothesis: Hypothesis,
        source_index: int,
        *,
        profile: EntityProfile | None,
        candidate: str,
        score: float,
        evidence: list[str],
        resolver: str,
        forced_action: str | None = None,
        rationale: str = "",
    ) -> dict[str, Any] | None:
        if hypothesis.risk == "high" and resolver != "user-confirmed":
            hypothesis.decision = "ASK_USER"
            hypothesis.decision_rationale = ["high_risk_requires_confirmation"]
            return None
        before = turn.current_text
        action = forced_action or ("REVISE_TEXT" if candidate != hypothesis.span else "REVISE_ENTITY")
        if action == "REVISE_TEXT" and candidate != hypothesis.span and hypothesis.span in turn.current_text:
            turn.current_text = turn.current_text.replace(hypothesis.span, candidate, 1)
        hypothesis.action = action
        hypothesis.decision = "COMMIT"
        hypothesis.decision_rationale = ["verified_later_raw_evidence"]
        if profile:
            hypothesis.entity_id = profile.entity_id
            session.verified_memory[profile.entity_id] = {"name": profile.name, "source_turn_id": session.turns[source_index].turn_id}
            session.quarantine_memory.pop(profile.entity_id, None)
        event = RevisionEvent(
            event_id=uuid.uuid4().hex,
            action=action,
            target_turn_id=turn.turn_id,
            source_turn_id=session.turns[source_index].turn_id,
            span=hypothesis.span,
            before_text=before,
            after_text=turn.current_text,
            entity_id=profile.entity_id if profile else None,
            score=round(score, 3),
            evidence=evidence,
            resolver=resolver,
            rationale=rationale,
            replacement=candidate if action == "REVISE_TEXT" else "",
        )
        session.revision_events.append(event)
        return asdict(event)

    def _path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _load(self, session_id: str) -> Session:
        path = self._path(session_id)
        return Session.from_dict(json.loads(path.read_text())) if path.exists() else Session(session_id)

    def _save(self, session: Session) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._path(session.session_id).write_text(json.dumps(session.as_dict(), ensure_ascii=False, indent=2))
