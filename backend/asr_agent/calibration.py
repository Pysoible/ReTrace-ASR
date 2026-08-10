"""Calibratable evidence policy for automatic revisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp


@dataclass(frozen=True)
class EvidenceFeatures:
    context_confidence: float = 0.0
    audio_confidence: float = 0.0
    audio_margin: float = 0.0
    memory_support: float = 0.0
    independent_sources: int = 0


@dataclass(frozen=True)
class PolicyThresholds:
    suspect: float = 0.45
    relisten: float = 0.55
    revise: float = 0.78
    long_memory: float = 0.85
    audio_margin: float = 0.15


BOOTSTRAP_THRESHOLDS = PolicyThresholds()


@dataclass(frozen=True)
class CalibrationPoint:
    probability: float
    should_revise: bool


@dataclass(frozen=True)
class SelectedThreshold:
    threshold: float
    precision: float
    recall: float
    f_beta: float
    overcorrection: float
    source: str = "development"


def select_revision_threshold(
    points: list[CalibrationPoint],
    *,
    beta: float = 2.0,
    max_overcorrection: float = 0.10,
) -> SelectedThreshold:
    """Select a recall-weighted threshold on development data only."""
    if not points:
        raise ValueError("development calibration points are required")
    if beta <= 0 or not 0 <= max_overcorrection <= 1:
        raise ValueError("invalid threshold selection parameters")
    positives = sum(point.should_revise for point in points)
    candidates = sorted({min(1.0, max(0.0, point.probability)) for point in points}, reverse=True)
    feasible: list[SelectedThreshold] = []
    for threshold in candidates:
        predicted = [point for point in points if point.probability >= threshold]
        true_positive = sum(point.should_revise for point in predicted)
        false_positive = len(predicted) - true_positive
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / positives if positives else 0.0
        overcorrection = false_positive / len(predicted) if predicted else 0.0
        denominator = beta * beta * precision + recall
        score = (1 + beta * beta) * precision * recall / denominator if denominator else 0.0
        if overcorrection <= max_overcorrection:
            feasible.append(SelectedThreshold(threshold, precision, recall, score, overcorrection))
    if not feasible:
        raise ValueError("no threshold satisfies the overcorrection constraint")
    return max(feasible, key=lambda item: (item.f_beta, item.recall, item.threshold))


@dataclass
class LinearLogitCalibrator:
    intercept: float = -2.0
    context_weight: float = 1.2
    audio_weight: float = 2.5
    margin_weight: float = 1.5
    memory_weight: float = 0.5
    sources_weight: float = 0.2

    def predict(self, features: EvidenceFeatures) -> float:
        z = (
            self.intercept
            + self.context_weight * features.context_confidence
            + self.audio_weight * features.audio_confidence
            + self.margin_weight * features.audio_margin
            + self.memory_weight * features.memory_support
            + self.sources_weight * min(3, features.independent_sources)
        )
        return 1.0 / (1.0 + exp(-z))

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> "LinearLogitCalibrator":
        return cls(**{key: float(score) for key, score in value.items()})


class DecisionPolicy:
    def __init__(
        self,
        thresholds: PolicyThresholds = BOOTSTRAP_THRESHOLDS,
        calibrator: LinearLogitCalibrator | None = None,
    ) -> None:
        self.thresholds = thresholds
        self.calibrator = calibrator or LinearLogitCalibrator()

    def decide(self, features: EvidenceFeatures, *, has_audio: bool) -> str:
        if not has_audio:
            return "DEFER"
        if features.audio_confidence < self.thresholds.relisten:
            return "DEFER"
        if features.audio_margin < self.thresholds.audio_margin:
            return "DEFER"
        if self.calibrator.predict(features) < self.thresholds.revise:
            return "DEFER"
        return "REVISE"
