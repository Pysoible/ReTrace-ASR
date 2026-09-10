"""Immutable data contracts for ReTrace V2."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class ModelRole(str, Enum):
    BASELINE_ASR = "baseline_asr"
    EVIDENCE_ASR = "evidence_asr"
    SEMANTIC_RANKER = "semantic_ranker"
    SEPARATION_BACKEND = "separation_backend"


@dataclass(frozen=True)
class RawSegment:
    segment_id: str
    start_sec: float
    end_sec: float
    speaker: str
    text: str
    model_id: str

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError("segment end must be after start")
        if not self.model_id.strip():
            raise ValueError("model_id is required")


@dataclass(frozen=True)
class CharacterEdit:
    kind: str
    start: int
    end: int
    replacement: str

    def __post_init__(self) -> None:
        if self.kind not in {"SUB", "INS", "DEL"}:
            raise ValueError(f"invalid edit kind: {self.kind}")
        if self.start < 0 or self.end < self.start:
            raise ValueError("invalid edit span")


@dataclass(frozen=True)
class EditCandidate:
    candidate_id: str
    raw_text: str
    candidate_text: str
    edits: tuple[CharacterEdit, ...]
    evidence_sources: tuple[str, ...]
    acoustic_score: float = 0.0
    semantic_score: float = 0.0

    @classmethod
    def keep(cls, raw_text: str) -> "EditCandidate":
        return cls("keep", raw_text, raw_text, (), ())


@dataclass(frozen=True)
class EvidenceHypothesis:
    hypothesis_id: str
    model_id: str
    view: str
    text: str
    start_sec: float
    end_sec: float
    acoustic_score: float
    speaker: str | None = None


@dataclass(frozen=True)
class SpeakerTurn:
    recording_id: str
    speaker: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class SuspiciousRegion:
    region_id: str
    segment_id: str
    start_sec: float
    end_sec: float
    raw_text: str
    raw_acoustic_score: float
    overlap_probability: float
    triggers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.end_sec <= self.start_sec:
            raise ValueError("region end must be after start")
        if not 0.0 <= self.overlap_probability <= 1.0:
            raise ValueError("overlap_probability must be in [0, 1]")


@dataclass(frozen=True)
class AgentAction:
    kind: str
    reason: str
    tool_cost: int = 0


@dataclass(frozen=True)
class AgentResult:
    region_id: str
    raw_text: str
    final_text: str
    decision: str
    actions: tuple[AgentAction, ...]
    candidates: tuple[EditCandidate, ...]
    selected_candidate_id: str | None = None


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    models: Mapping[ModelRole, str]
    rttm_source: str

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        models: Mapping[ModelRole, str],
        rttm_source: str,
    ) -> "RunManifest":
        baseline = models.get(ModelRole.BASELINE_ASR, "").strip()
        evidence = models.get(ModelRole.EVIDENCE_ASR, "").strip()
        if not baseline or not evidence:
            raise ValueError("baseline and evidence ASR identities are required")
        if baseline.casefold() == evidence.casefold():
            raise ValueError("baseline and evidence ASR must be independent")
        for identity in models.values():
            folded = identity.casefold()
            if "moss" in folded and "omni" in folded:
                raise ValueError("each model identity must name one model family")
        if rttm_source not in {"none", "predicted", "oracle"}:
            raise ValueError("invalid rttm_source")
        return cls(
            run_id=run_id,
            models=MappingProxyType(dict(models)),
            rttm_source=rttm_source,
        )
