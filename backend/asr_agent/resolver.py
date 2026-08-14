"""Resolve a Context Judge focus with targeted historical audio."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from asr_agent.calibration import DecisionPolicy, EvidenceFeatures
from asr_agent.context_judge import FocusProposal
from asr_agent.integrations.audio_verifier import verify_candidates
from asr_agent.models import Session, Turn


@dataclass
class Resolution:
    action: str
    target_turn_id: str
    span: str
    replacement: str = ""
    score: float = 0.0
    evidence: list[str] = field(default_factory=list)
    rationale: str = ""
    audio_verified: bool = False


class EvidenceResolver:
    def __init__(
        self,
        audio_verifier: Callable[..., dict[str, Any]] | None = None,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self.audio_verifier = audio_verifier or verify_candidates
        self.policy = policy or DecisionPolicy()

    def resolve(
        self,
        session: Session,
        source_turn: Turn,
        focus: FocusProposal,
        *,
        context_confidence: float,
        memory_support: float = 0.0,
    ) -> Resolution:
        target = next((turn for turn in session.turns if turn.turn_id == focus.target_turn_id), None)
        if target is None or (focus.span not in target.raw_text and focus.span not in target.current_text):
            return Resolution("DEFER", focus.target_turn_id, focus.span, rationale="invalid target span")
        # Guard against a class of false positives where BOTH the span and the
        # proposed replacement already appear verbatim in the original raw text
        # (e.g. "时候" → "狮子狗" in "狮子狗刚出的时候…"). Here the span is a
        # normal word and the "correct" name already occurs elsewhere in the same
        # turn, so ASR already recognized it; a closed-set verifier listening to
        # the whole window is misled by that other occurrence. Refusing to revise
        # is safe (we only skip a revision, never invent one).
        if (
            focus.span in target.raw_text
            and focus.proposed_text in target.raw_text
            and focus.proposed_text != focus.span
        ):
            return Resolution(
                "DEFER",
                target.turn_id,
                focus.span,
                rationale=(
                    f"proposed replacement {focus.proposed_text!r} already appears elsewhere in the "
                    "same turn; refusing a low-confidence name swap"
                ),
            )
        if focus.relationship == "COEXIST":
            return Resolution(
                "COEXIST",
                target.turn_id,
                focus.span,
                score=context_confidence,
                evidence=[f"context:{turn_id}" for turn_id in focus.evidence_turn_ids],
                rationale=focus.rationale or "the interpretations can refer to different entities",
            )
        if focus.relationship == "TEMPORAL_CHANGE":
            return Resolution(
                "ACCEPT_NEW",
                target.turn_id,
                focus.span,
                score=context_confidence,
                evidence=[f"context:{turn_id}" for turn_id in focus.evidence_turn_ids],
                rationale=focus.rationale or "the fact changed over time",
            )
        meta = target.meta or {}
        audio_path = meta.get("audio_path")
        start_sec, end_sec = meta.get("start_sec"), meta.get("end_sec")
        if not audio_path or start_sec is None or end_sec is None:
            return Resolution("DEFER", target.turn_id, focus.span, rationale="historical audio unavailable")
        try:
            audio = self.audio_verifier(
                audio_path=str(audio_path),
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                candidates=focus.alternatives,
            )
        except Exception as exc:
            return Resolution("DEFER", target.turn_id, focus.span, rationale=f"audio verifier failed: {exc}")
        scores = audio.get("scores") if isinstance(audio, dict) and audio.get("ok") else None
        if not isinstance(scores, dict) or set(scores) != set(focus.alternatives):
            return Resolution("DEFER", target.turn_id, focus.span, rationale="invalid closed-set audio result")
        normalized = {key: max(0.0, float(value)) for key, value in scores.items()}
        ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        if not ordered or ordered[0][0] != focus.proposed_text:
            return Resolution("KEEP_OLD", target.turn_id, focus.span, score=ordered[0][1] if ordered else 0.0)
        top = ordered[0][1]
        margin = top - (ordered[1][1] if len(ordered) > 1 else 0.0)
        # Second acoustic gate (direction 2): a classical ASR (paraformer) votes
        # on the same focused window. If the two acoustic models disagree — the
        # LLM favors the proposed word but the classical ASR favors the original
        # span — we conservatively DEFER rather than authorize a risky revision.
        acoustic_vote: float | None = None
        try:
            from asr_agent.integrations import acoustic  # local import to avoid cycles

            vote = acoustic.acoustic_verify(
                audio_path=str(audio_path),
                start_sec=float(start_sec),
                end_sec=float(end_sec),
                candidates=focus.alternatives,
            )
            if isinstance(vote, dict) and vote.get("ok") and isinstance(vote.get("scores"), dict):
                vote_scores = vote["scores"]
                vote_winner = max(vote_scores, key=lambda key: vote_scores[key])
                if vote_winner == focus.span:
                    return Resolution(
                        "DEFER",
                        target.turn_id,
                        focus.span,
                        score=top,
                        rationale="second acoustic ASR (paraformer) disagrees with the proposed correction",
                    )
                acoustic_vote = float(vote_scores.get(focus.proposed_text, 0.0))
        except Exception:
            acoustic_vote = None
        features = EvidenceFeatures(
            context_confidence=context_confidence,
            audio_confidence=top,
            audio_margin=margin,
            memory_support=memory_support,
            independent_sources=len(set(focus.evidence_turn_ids)),
        )
        if self.policy.decide(features, has_audio=True) != "REVISE":
            return Resolution("DEFER", target.turn_id, focus.span, score=top, rationale="evidence below bootstrap policy")
        action = "REVISE_CURRENT" if target.turn_id == source_turn.turn_id else "REVISE_HISTORY"
        evidence = [f"context:{turn_id}" for turn_id in focus.evidence_turn_ids]
        evidence.append(f"audio:{audio_path}:{start_sec}-{end_sec}")
        if acoustic_vote is not None:
            evidence.append(f"acoustic_vote:{focus.proposed_text}")
        return Resolution(
            action,
            target.turn_id,
            focus.span,
            replacement=focus.proposed_text,
            score=self.policy.calibrator.predict(features),
            evidence=evidence,
            rationale=focus.rationale or "context and targeted audio agree",
            audio_verified=True,
        )
