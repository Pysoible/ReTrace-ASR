"""Bounded observe-act-update Agent for selective acoustic recovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .candidates import candidates_from_hypothesis
from .schemas import (
    AgentAction,
    AgentResult,
    EditCandidate,
    EvidenceHypothesis,
    SuspiciousRegion,
)
from .tools import AgentTools, ToolFailure


@dataclass(frozen=True)
class ThresholdPolicy:
    commit_margin: float = 0.25
    overlap_threshold: float = 0.6
    max_changed_chars: int = 24
    max_tool_calls: int = 3


class ReTraceAgent:
    def __init__(self, policy: ThresholdPolicy | None = None) -> None:
        self.policy = policy or ThresholdPolicy()

    def run(self, region: SuspiciousRegion, tools: AgentTools) -> AgentResult:
        actions: list[AgentAction] = []
        hypotheses: list[EvidenceHypothesis] = []
        tool_calls = 0
        try:
            actions.append(AgentAction("RELISTEN", "initial_acoustic_check", 1))
            hypotheses.extend(tools.relisten(region))
            tool_calls += 1
        except ToolFailure:
            return self._keep(region, actions, (), "tool_failure")

        actions.append(AgentAction("COMPARE_EVIDENCE", "original_view_ready"))
        candidates = self._candidates(region, hypotheses)
        selected = self._select(region, candidates)
        if selected is not None:
            actions.append(AgentAction("COMMIT", "acoustic_margin_passed"))
            return self._commit(region, actions, candidates, selected)

        if (
            region.overlap_probability >= self.policy.overlap_threshold
            and tool_calls + 2 <= self.policy.max_tool_calls
        ):
            try:
                actions.append(AgentAction("SEPARATE", "unresolved_overlap", 1))
                channels = tools.separate(region)
                tool_calls += 1
                actions.append(
                    AgentAction("RELISTEN_SEPARATED", "separated_channels_ready", 1)
                )
                hypotheses.extend(tools.relisten_separated(region, channels))
                tool_calls += 1
            except ToolFailure:
                return self._keep(region, actions, candidates, "tool_failure")
            actions.append(AgentAction("COMPARE_EVIDENCE", "separated_view_ready"))
            candidates = self._candidates(region, hypotheses)
            selected = self._select(region, candidates)
            if selected is not None:
                actions.append(AgentAction("COMMIT", "separated_acoustic_margin_passed"))
                return self._commit(region, actions, candidates, selected)

        return self._keep(region, actions, candidates, "insufficient_evidence")

    def _candidates(
        self,
        region: SuspiciousRegion,
        hypotheses: Iterable[EvidenceHypothesis],
    ) -> tuple[EditCandidate, ...]:
        by_text: dict[str, EditCandidate] = {}
        for hypothesis in hypotheses:
            if hypothesis.view.casefold() == "semantic":
                continue
            generated = candidates_from_hypothesis(
                region.raw_text,
                hypothesis.text,
                source=f"{hypothesis.model_id}:{hypothesis.view}",
                acoustic_score=hypothesis.acoustic_score,
            )
            for candidate in generated[1:]:
                current = by_text.get(candidate.candidate_text)
                if current is None or candidate.acoustic_score > current.acoustic_score:
                    by_text[candidate.candidate_text] = candidate
        return (EditCandidate.keep(region.raw_text), *by_text.values())

    def _select(
        self,
        region: SuspiciousRegion,
        candidates: tuple[EditCandidate, ...],
    ) -> EditCandidate | None:
        eligible = []
        for candidate in candidates:
            if not candidate.edits or not candidate.evidence_sources:
                continue
            changed = sum(
                max(edit.end - edit.start, len(edit.replacement))
                for edit in candidate.edits
            )
            if changed > self.policy.max_changed_chars:
                continue
            if candidate.acoustic_score - region.raw_acoustic_score < self.policy.commit_margin:
                continue
            eligible.append(candidate)
        return max(eligible, key=lambda item: item.acoustic_score, default=None)

    @staticmethod
    def _commit(
        region: SuspiciousRegion,
        actions: list[AgentAction],
        candidates: tuple[EditCandidate, ...],
        selected: EditCandidate,
    ) -> AgentResult:
        return AgentResult(
            region_id=region.region_id,
            raw_text=region.raw_text,
            final_text=selected.candidate_text,
            decision="COMMIT",
            actions=tuple(actions),
            candidates=candidates,
            selected_candidate_id=selected.candidate_id,
        )

    @staticmethod
    def _keep(
        region: SuspiciousRegion,
        actions: list[AgentAction],
        candidates: tuple[EditCandidate, ...],
        reason: str,
    ) -> AgentResult:
        actions.append(AgentAction("KEEP", reason))
        return AgentResult(
            region_id=region.region_id,
            raw_text=region.raw_text,
            final_text=region.raw_text,
            decision="KEEP",
            actions=tuple(actions),
            candidates=candidates or (EditCandidate.keep(region.raw_text),),
        )
