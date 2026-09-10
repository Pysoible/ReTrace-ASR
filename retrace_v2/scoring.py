"""Reference-aware scoring that runs only after V2 inference."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from Levenshtein import editops

from .text import normalize_text


@dataclass(frozen=True)
class ErrorMetric:
    reference_chars: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def edits(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def cer(self) -> float:
        if self.reference_chars:
            return self.edits / self.reference_chars
        return 1.0 if self.insertions else 0.0


@dataclass(frozen=True)
class ScoreReport:
    raw: ErrorMetric
    final: ErrorMetric
    oracle_candidate: ErrorMetric
    committed_revisions: int
    improving_revisions: int
    harmful_revisions: int
    neutral_revisions: int

    @property
    def revision_precision(self) -> float:
        if not self.committed_revisions:
            return 0.0
        return self.improving_revisions / self.committed_revisions


def _counts(reference: str, hypothesis: str) -> Counter[str]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    operations = Counter(operation[0] for operation in editops(ref, hyp))
    return Counter(
        reference_chars=len(ref),
        substitutions=operations["replace"],
        deletions=operations["delete"],
        insertions=operations["insert"],
    )


def _distance(reference: str, hypothesis: str) -> int:
    counts = _counts(reference, hypothesis)
    return counts["substitutions"] + counts["deletions"] + counts["insertions"]


def _metric(total: Counter[str]) -> ErrorMetric:
    return ErrorMetric(
        reference_chars=total["reference_chars"],
        substitutions=total["substitutions"],
        deletions=total["deletions"],
        insertions=total["insertions"],
    )


def score_rows(rows: Iterable[Mapping[str, object]]) -> ScoreReport:
    raw_total: Counter[str] = Counter()
    final_total: Counter[str] = Counter()
    oracle_total: Counter[str] = Counter()
    committed = improving = harmful = neutral = 0

    for row in rows:
        reference = str(row.get("reference_text") or "")
        raw = str(row.get("raw_text") or "")
        final = str(row.get("final_text") or raw)
        raw_total.update(_counts(reference, raw))
        final_total.update(_counts(reference, final))

        values = row.get("candidates") or [raw]
        candidates = [str(value) for value in values]  # type: ignore[union-attr]
        if raw not in candidates:
            candidates.insert(0, raw)
        oracle = min(candidates, key=lambda value: _distance(reference, value))
        oracle_total.update(_counts(reference, oracle))

        if normalize_text(final) != normalize_text(raw):
            committed += 1
            raw_edits = _distance(reference, raw)
            final_edits = _distance(reference, final)
            if final_edits < raw_edits:
                improving += 1
            elif final_edits > raw_edits:
                harmful += 1
            else:
                neutral += 1

    return ScoreReport(
        raw=_metric(raw_total),
        final=_metric(final_total),
        oracle_candidate=_metric(oracle_total),
        committed_revisions=committed,
        improving_revisions=improving,
        harmful_revisions=harmful,
        neutral_revisions=neutral,
    )
