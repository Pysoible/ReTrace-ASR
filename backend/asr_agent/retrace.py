"""Evidence-grounded retrospective revision for conversational ASR."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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


@dataclass
class Turn:
    turn_id: str
    raw_text: str
    current_text: str
    hypotheses: list[Hypothesis] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    entities: dict[str, EntityProfile] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    verified_memory: dict[str, dict[str, str]] = field(default_factory=dict)
    quarantine_memory: dict[str, dict[str, str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Session":
        session = cls(session_id=raw["session_id"])
        session.entities = {key: EntityProfile.from_dict(value) for key, value in raw.get("entities", {}).items()}
        session.turns = [Turn(turn_id=item["turn_id"], raw_text=item["raw_text"], current_text=item["current_text"], hypotheses=[Hypothesis(**hypothesis) for hypothesis in item.get("hypotheses", [])]) for item in raw.get("turns", [])]
        session.verified_memory = raw.get("verified_memory", {})
        session.quarantine_memory = raw.get("quarantine_memory", {})
        return session


class ReTraceService:
    def __init__(self, storage_dir: Path) -> None:
        self.storage_dir = storage_dir

    def upsert_entities(self, session_id: str, profiles: list[EntityProfile]) -> dict[str, Any]:
        session = self._load(session_id)
        session.entities.update({profile.entity_id: profile for profile in profiles})
        self._save(session)
        return session.as_dict()

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._load(session_id).as_dict()

    def process_turn(self, session_id: str, turn_id: str, text: str, *, confidence: dict[str, float] | None = None, text_candidates: dict[str, list[str]] | None = None, entity_candidate_ids: dict[str, list[str]] | None = None) -> dict[str, Any]:
        session = self._load(session_id)
        if any(turn.turn_id == turn_id for turn in session.turns):
            raise ValueError(f"duplicate turn_id: {turn_id}")
        confidence, text_candidates, entity_candidate_ids = confidence or {}, text_candidates or {}, entity_candidate_ids or {}
        spans = set(text_candidates) | set(entity_candidate_ids)
        hypotheses = [
            Hypothesis(
                span=span,
                text_candidates=text_candidates.get(span, [span]),
                entity_candidate_ids=entity_candidate_ids.get(span, []),
            )
            for span in spans
            if span in text and confidence.get(span, 1.0) < 0.65
        ]
        for hypothesis in hypotheses:
            for entity_id in hypothesis.entity_candidate_ids:
                profile = session.entities.get(entity_id)
                if profile:
                    session.quarantine_memory[entity_id] = {"name": profile.name, "status": "candidate"}
        session.turns.append(Turn(turn_id=turn_id, raw_text=text, current_text=text, hypotheses=hypotheses))
        revisions = self._reassess(session, source_index=len(session.turns) - 1)
        self._save(session)
        return {"session": session.as_dict(), "revisions": revisions, "actions": [item.action for item in hypotheses]}

    def _reassess(self, session: Session, source_index: int) -> list[dict[str, Any]]:
        evidence_text = session.turns[source_index].current_text
        revisions: list[dict[str, Any]] = []
        for turn in session.turns[:source_index]:
            for hypothesis in turn.hypotheses:
                if hypothesis.action != "DEFER":
                    continue
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
                    continue
                score, profile, evidence = scores[0]
                before = turn.current_text
                candidate = next((item for item in hypothesis.text_candidates if item == profile.name or item in profile.aliases), hypothesis.span)
                action = "REVISE_TEXT" if candidate != hypothesis.span else "REVISE_ENTITY"
                if action == "REVISE_TEXT":
                    turn.current_text = turn.current_text.replace(hypothesis.span, candidate, 1)
                hypothesis.entity_id, hypothesis.action = profile.entity_id, action
                session.verified_memory[profile.entity_id] = {"name": profile.name, "source_turn_id": session.turns[source_index].turn_id}
                session.quarantine_memory.pop(profile.entity_id, None)
                revisions.append({"action": action, "target_turn_id": turn.turn_id, "source_turn_id": session.turns[source_index].turn_id, "span": hypothesis.span, "before_text": before, "after_text": turn.current_text, "entity_id": profile.entity_id, "score": round(score, 3), "evidence": evidence})
        return revisions

    def _path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    def _load(self, session_id: str) -> Session:
        path = self._path(session_id)
        return Session.from_dict(json.loads(path.read_text())) if path.exists() else Session(session_id)

    def _save(self, session: Session) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._path(session.session_id).write_text(json.dumps(session.as_dict(), ensure_ascii=False, indent=2))
