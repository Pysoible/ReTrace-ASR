"""Short-term retrieval and durable long-term memory."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from typing import Any, Callable

from asr_agent.models import MemoryBelief, Session, Turn, WorkingHypothesis


_MEMORY_LOCKS: dict[tuple[Path, str], RLock] = {}
_MEMORY_LOCKS_GUARD = Lock()


def _memory_terms(text: str) -> list[str]:
    """Tokenize Chinese/Latin text for retrieval without a segmentation model."""
    normalized = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text or "")).lower()
    if not normalized:
        return []
    # Character unigrams retain short names; bigrams provide useful semantic-ish
    # locality for Chinese terms without requiring a brittle word segmenter.
    return list(normalized) + [normalized[i : i + 2] for i in range(len(normalized) - 1)]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    dot = sum(value * right.get(term, 0) for term, value in left.items())
    if not dot:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


class _OptionalDenseEmbedder:
    """Optional local Transformer encoder for dense semantic retrieval.

    Set ``ASR_MEMORY_EMBEDDING_MODEL`` to a local HuggingFace checkpoint. The
    memory system keeps working when it is unset or unavailable: hybrid sparse
    retrieval is deterministic and dependency-free, while a deployed embedding
    model upgrades ranking with sentence-level semantics.
    """

    def __init__(self) -> None:
        self.model_path = os.getenv("ASR_MEMORY_EMBEDDING_MODEL", "").strip()
        self._model = None
        self._tokenizer = None
        self._failed = False

    def encode(self, texts: list[str]) -> list[list[float]] | None:
        if not self.model_path or self._failed:
            return None
        try:
            if self._model is None or self._tokenizer is None:
                import torch
                from transformers import AutoModel, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True).eval()
            import torch

            batch = self._tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            with torch.no_grad():
                hidden = self._model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
                vectors = torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().tolist()
            return [[float(value) for value in vector] for vector in vectors]
        except Exception:
            self._failed = True
            return None


class HybridMemoryIndex:
    """Hybrid lexical + semantic retrieval over durable memory beliefs.

    BM25 is high precision for names and aliases, char n-gram cosine recalls
    partial Chinese overlap, and the optional dense encoder adds meaning-level
    similarity. Entity aliases and audio provenance are re-ranked as structured
    evidence rather than being injected blindly into the LLM prompt.
    """

    def __init__(self, beliefs: list[MemoryBelief], dense_embedder: _OptionalDenseEmbedder) -> None:
        self.beliefs = beliefs
        self.documents = [self._document(item) for item in beliefs]
        self.tokens = [Counter(_memory_terms(text)) for text in self.documents]
        self.lengths = [sum(counts.values()) for counts in self.tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        self.doc_frequency = Counter(term for counts in self.tokens for term in counts)
        vectors = dense_embedder.encode(self.documents)
        self.dense_vectors = vectors if vectors and len(vectors) == len(beliefs) else None
        self.dense_embedder = dense_embedder

    @staticmethod
    def _document(belief: MemoryBelief) -> str:
        return " ".join([belief.subject, belief.predicate, belief.value, *belief.aliases])

    def search(self, query: str, *, limit: int) -> list[MemoryBelief]:
        if not self.beliefs or not query.strip():
            return []
        query_terms = _memory_terms(query)
        if not query_terms:
            return []
        query_counts = Counter(query_terms)
        dense_query = self.dense_embedder.encode([query]) if self.dense_vectors is not None else None
        n_docs = len(self.beliefs)
        scored: list[tuple[float, float, MemoryBelief]] = []
        for index, belief in enumerate(self.beliefs):
            bm25 = 0.0
            for term, qtf in query_counts.items():
                tf = self.tokens[index].get(term, 0)
                if not tf:
                    continue
                df = self.doc_frequency[term]
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denominator = tf + 1.2 * (1.0 - 0.75 + 0.75 * self.lengths[index] / self.avg_length)
                bm25 += qtf * idf * tf * 2.2 / denominator
            lexical = _cosine(query_counts, self.tokens[index])
            aliases = [belief.value, *belief.aliases]
            exact = 1.0 if any(term and term in query for term in aliases) else 0.0
            dense = 0.0
            if dense_query and self.dense_vectors:
                dense = sum(a * b for a, b in zip(dense_query[0], self.dense_vectors[index]))
            # Provenance improves trust only after lexical/semantic evidence
            # establishes relevance; it must never retrieve an unrelated belief
            # merely because that belief was audio-verified in the past.
            matched = bm25 > 0.0 or lexical > 0.0 or dense > 0.0 or exact > 0.0
            provenance = 0.15 if matched and "audio_verified" in belief.evidence_kinds else 0.0
            # Normalize BM25 relative to query length, then combine evidence.
            bm25_norm = bm25 / max(1.0, len(query_counts))
            score = 0.45 * bm25_norm + 0.25 * lexical + 0.20 * dense + 0.50 * exact + provenance
            scored.append((score, belief.confidence, belief))
        ranked = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)
        return [belief for score, _, belief in ranked[:limit] if score > 0.03]


@dataclass
class MemoryPacket:
    recent_turns: list[Turn] = field(default_factory=list)
    dependent_turns: list[Turn] = field(default_factory=list)
    working_beliefs: list[MemoryBelief] = field(default_factory=list)
    open_hypotheses: list[WorkingHypothesis] = field(default_factory=list)
    long_term_beliefs: list[MemoryBelief] = field(default_factory=list)
    # Agent-confirmed entity spellings, separate from generic factual beliefs.
    # These are only created after context + targeted-audio verification.
    canonical_entities: list[MemoryBelief] = field(default_factory=list)


class LongTermMemoryRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._status_lock = RLock()
        self._last_status: dict[str, dict[str, Any]] = {}

    def _record(self, scope: str, operation: str, error: Exception | None = None) -> None:
        with self._status_lock:
            self._last_status[scope] = {
                "last_operation": operation,
                "error": None if error is None else {
                    "kind": type(error).__name__,
                    "message": str(error),
                },
            }

    def _path(self, scope: str) -> Path:
        if not scope or not re.fullmatch(r"[A-Za-z0-9_.-]+", scope) or ".." in scope:
            raise ValueError("memory scope must be a safe identifier")
        return self.root / f"{scope}.json"

    def _lock_for(self, scope: str) -> RLock:
        key = (self.root.resolve(), scope)
        with _MEMORY_LOCKS_GUARD:
            return _MEMORY_LOCKS.setdefault(key, RLock())

    def load(self, scope: str) -> list[MemoryBelief]:
        try:
            with self._lock_for(scope):
                beliefs = self._load_unlocked(scope)
        except Exception as exc:
            self._record(scope, "load", exc)
            raise
        self._record(scope, "load")
        return beliefs

    def _load_unlocked(self, scope: str) -> list[MemoryBelief]:
        path = self._path(scope)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("long-term memory must be a JSON list")
        return [MemoryBelief.from_dict(item) for item in data]

    def save(self, scope: str, beliefs: list[MemoryBelief]) -> None:
        try:
            with self._lock_for(scope):
                self._save_unlocked(scope, beliefs)
        except Exception as exc:
            self._record(scope, "save", exc)
            raise
        self._record(scope, "save")

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

    def update(
        self,
        scope: str,
        mutate: Callable[[list[MemoryBelief]], None],
        *,
        operation: str = "update",
    ) -> list[MemoryBelief]:
        try:
            with self._lock_for(scope):
                beliefs = self._load_unlocked(scope)
                mutate(beliefs)
                self._save_unlocked(scope, beliefs)
                result = [MemoryBelief.from_dict(item.as_dict()) for item in beliefs]
        except Exception as exc:
            self._record(scope, operation, exc)
            raise
        self._record(scope, operation)
        return result

    def status(self, scope: str) -> dict[str, Any]:
        with self._status_lock:
            recorded = dict(self._last_status.get(scope) or {})
        try:
            with self._lock_for(scope):
                beliefs = self._load_unlocked(scope)
        except Exception as exc:
            beliefs = []
            if not recorded.get("error"):
                recorded = {
                    "last_operation": "status",
                    "error": {"kind": type(exc).__name__, "message": str(exc)},
                }
        counts = Counter(item.status for item in beliefs)
        return {
            "scope": scope,
            "provisional": counts.get("provisional", 0),
            "stable": counts.get("stable", 0),
            "superseded": counts.get("superseded", 0),
            "last_operation": recorded.get("last_operation"),
            "error": recorded.get("error"),
        }


class MemoryRetriever:
    def __init__(self, repository: LongTermMemoryRepository, recent_limit: int | None = None, long_term_limit: int = 24) -> None:
        self.repository = repository
        self.recent_limit = recent_limit if recent_limit is not None else int(os.getenv("ASR_MEMORY_RECENT_LIMIT", "16"))
        self.long_term_limit = long_term_limit
        self._dense_embedder = _OptionalDenseEmbedder()
        self._index_cache_key: tuple[tuple[str, int, str, float], ...] | None = None
        self._index: HybridMemoryIndex | None = None

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
        canonical_entities = self._canonical_entities(session, durable)
        return MemoryPacket(
            recent_turns=session.turns[-self.recent_limit :],
            dependent_turns=dependent,
            working_beliefs=list(session.working_beliefs.values()),
            open_hypotheses=[item for item in session.open_hypotheses.values() if item.status == "active"],
            long_term_beliefs=durable[: self.long_term_limit],
            canonical_entities=canonical_entities[: self.long_term_limit],
        )

    @staticmethod
    def _canonical_entities(session: Session, durable: list[MemoryBelief]) -> list[MemoryBelief]:
        """Return verified, canonical entity spellings available to the agent.

        Generic ``canonical_text`` beliefs can be ordinary phrase repairs, so
        they must never become entity memory. ``canonical_entity`` beliefs are
        created only by the resolver after context and targeted audio agree.
        """
        candidates = [*session.working_beliefs.values(), *durable]
        entities = [
            item
            for item in candidates
            if item.status != "superseded"
            and item.predicate == "canonical_entity"
            and "audio_verified" in item.evidence_kinds
            and len(item.value.strip()) >= 2
        ]
        return sorted(entities, key=lambda item: item.confidence, reverse=True)

    def _relevant_long_term(
        self,
        beliefs: list[MemoryBelief],
        session: Session,
        current_turn: Turn,
    ) -> list[MemoryBelief]:
        """Retrieve durable memory by hybrid semantic and lexical evidence.

        The query includes the current observation plus active hypotheses. This
        preserves the old dependency signal while replacing exact substring
        matching with ranked recall over belief values, aliases and predicates.
        """
        cache_key = tuple(sorted(
            (item.belief_id, item.updated_version, item.status, item.confidence)
            for item in beliefs
        ))
        if self._index is None or self._index_cache_key != cache_key:
            self._index = HybridMemoryIndex(beliefs, self._dense_embedder)
            self._index_cache_key = cache_key
        hypothesis_terms = [
            term
            for item in session.open_hypotheses.values()
            if item.status == "active"
            for term in [item.current_interpretation, item.proposed_interpretation, *item.alternatives]
            if term
        ]
        working_summary = " ".join(
            f"{item.subject} {item.predicate} {item.value}"
            for item in session.working_beliefs.values()
            if item.status != "superseded"
        )
        query = " ".join([current_turn.current_text or current_turn.raw_text, *hypothesis_terms, working_summary])
        return self._index.search(query, limit=self.long_term_limit)


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
            self.repository.update(scope, merge, operation="consolidate")
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        return promoted
