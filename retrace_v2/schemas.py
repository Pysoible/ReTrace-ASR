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
