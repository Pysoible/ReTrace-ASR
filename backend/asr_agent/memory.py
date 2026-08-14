"""Short-term retrieval and durable long-term memory."""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from typing import Callable

from asr_agent.models import MemoryBelief, Session, Turn, WorkingHypothesis


_MEMORY_LOCKS: dict[tuple[Path, str], RLock] = {}
_MEMORY_LOCKS_GUARD = Lock()


@dataclass
class MemoryPacket:
    recent_turns: list[Turn] = field(default_factory=list)
    dependent_turns: list[Turn] = field(default_factory=list)
    working_beliefs: list[MemoryBelief] = field(default_factory=list)
    open_hypotheses: list[WorkingHypothesis] = field(default_factory=list)
    long_term_beliefs: list[MemoryBelief] = field(default_factory=list)


class LongTermMemoryRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, scope: str) -> Path:
        if not scope or not re.fullmatch(r"[A-Za-z0-9_.-]+", scope) or ".." in scope:
            raise ValueError("memory scope must be a safe identifier")
        return self.root / f"{scope}.json"

    def _lock_for(self, scope: str) -> RLock:
        key = (self.root.resolve(), scope)
        with _MEMORY_LOCKS_GUARD:
            return _MEMORY_LOCKS.setdefault(key, RLock())

    def load(self, scope: str) -> list[MemoryBelief]:
        with self._lock_for(scope):
            return self._load_unlocked(scope)

    def _load_unlocked(self, scope: str) -> list[MemoryBelief]:
        path = self._path(scope)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("long-term memory must be a JSON list")
        return [MemoryBelief.from_dict(item) for item in data]

    def save(self, scope: str, beliefs: list[MemoryBelief]) -> None:
        with self._lock_for(scope):
            self._save_unlocked(scope, beliefs)

    def _save_unlocked(self, scope: str, beliefs: list[MemoryBelief]) -> None:
        path = self._path(scope)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.root, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump([item.as_dict() for item in beliefs], handle, ensure_ascii=False, indent=2)
            temporary.replace(path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()

    def update(self, scope: str, mutate: Callable[[list[MemoryBelief]], None]) -> list[MemoryBelief]:
        with self._lock_for(scope):
            beliefs = self._load_unlocked(scope)
            mutate(beliefs)
            self._save_unlocked(scope, beliefs)
            return [MemoryBelief.from_dict(item.as_dict()) for item in beliefs]


class MemoryRetriever:
    def __init__(self, repository: LongTermMemoryRepository, recent_limit: int = 8, long_term_limit: int = 24) -> None:
        self.repository = repository
        self.recent_limit = recent_limit
        self.long_term_limit = long_term_limit

    def retrieve(self, session: Session, current_turn: Turn) -> MemoryPacket:
        related_ids = set(session.dependency_index.get(current_turn.turn_id, []))
        related_ids.update(
            target_id
            for target_id, evidence_ids in session.dependency_index.items()
            if current_turn.turn_id in evidence_ids
        )
        recent_ids = {turn.turn_id for turn in session.turns[-self.recent_limit :]}
        dependent = [turn for turn in session.turns if turn.turn_id in related_ids and turn.turn_id not in recent_ids]
        try:
            stable = [item for item in self.repository.load(session.memory_scope) if item.status == "stable"]
            durable = self._relevant_long_term(stable, session, current_turn)
        except (OSError, ValueError, json.JSONDecodeError):
            durable = []
        return MemoryPacket(
            recent_turns=session.turns[-self.recent_limit :],
            dependent_turns=dependent,
            working_beliefs=list(session.working_beliefs.values()),
            open_hypotheses=[item for item in session.open_hypotheses.values() if item.status == "active"],
            long_term_beliefs=durable[: self.long_term_limit],
        )

    @staticmethod
    def _relevant_long_term(
        beliefs: list[MemoryBelief],
        session: Session,
        current_turn: Turn,
    ) -> list[MemoryBelief]:
        text = current_turn.raw_text
        working_keys = {(item.subject, item.predicate) for item in session.working_beliefs.values()}
        hypothesis_terms = {
            term
            for item in session.open_hypotheses.values()
            if item.status == "active"
            for term in [item.current_interpretation, item.proposed_interpretation, *item.alternatives]
        }

        def score(item: MemoryBelief) -> tuple[float, float]:
            terms = [item.value, *item.aliases]
            lexical = 1.0 if any(term and (term in text or term in hypothesis_terms) for term in terms) else 0.0
            structural = 0.65 if (item.subject, item.predicate) in working_keys else 0.0
            return max(lexical, structural), item.confidence

        scored = [(score(item), item) for item in beliefs]
        return [item for relevance, item in sorted(scored, key=lambda pair: pair[0], reverse=True) if relevance[0] > 0]


class MemoryConsolidator:
    def __init__(self, repository: LongTermMemoryRepository, confidence_threshold: float = 0.85) -> None:
        self.repository = repository
        self.confidence_threshold = confidence_threshold

    def consolidate(self, scope: str, candidates: list[MemoryBelief]) -> list[MemoryBelief]:
        promoted: list[MemoryBelief] = []

        def merge(stored: list[MemoryBelief]) -> None:
            by_id = {item.belief_id: item for item in stored}
            by_value = {
                (item.subject, item.predicate, item.value): item
                for item in stored
                if item.status != "superseded"
            }
            for candidate in candidates:
                key = (candidate.subject, candidate.predicate, candidate.value)
                merged = by_value.get(key)
                if merged is None:
                    merged = MemoryBelief.from_dict(candidate.as_dict())
                    by_id[merged.belief_id] = merged
                    by_value[key] = merged
                else:
                    merged.aliases = list(dict.fromkeys([*merged.aliases, *candidate.aliases]))
                    merged.source_turn_ids = list(dict.fromkeys([*merged.source_turn_ids, *candidate.source_turn_ids]))
                    merged.source_session_ids = list(dict.fromkeys([*merged.source_session_ids, *candidate.source_session_ids]))
                    merged.evidence_kinds = list(dict.fromkeys([*merged.evidence_kinds, *candidate.evidence_kinds]))
                    merged.confidence = max(merged.confidence, candidate.confidence)
                    merged.updated_version = max(merged.updated_version, candidate.updated_version)
                # A belief is independently supported when it is corroborated across
                # sessions OR across distinct turns within the same long-audio session.
                # (A single 20-minute audio is one session, so cross-session-only would
                # never promote anything — multi-turn agreement is equally strong.)
                independently_supported = (
                    len(set(merged.source_session_ids)) >= 2
                    or len(set(merged.source_turn_ids)) >= 2
                )
                audio_verified = "audio_verified" in merged.evidence_kinds
                if merged.confidence < self.confidence_threshold or not (independently_supported or audio_verified):
                    merged.status = "provisional"
                    continue
                merged.status = "stable"
                for old in by_id.values():
                    if old.belief_id != merged.belief_id and old.status != "superseded" and (old.subject, old.predicate) == (merged.subject, merged.predicate) and old.value != merged.value:
                        old.status = "superseded"
                        merged.supersedes = old.belief_id
                if all(item.belief_id != merged.belief_id for item in promoted):
                    promoted.append(merged)
            stored[:] = list(by_id.values())

        try:
            self.repository.update(scope, merge)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        return promoted
