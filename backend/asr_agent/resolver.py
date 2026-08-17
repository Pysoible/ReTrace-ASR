"""Resolve a Context Judge focus with targeted historical audio."""
from __future__ import annotations

import re
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

    @staticmethod
    def _acoustic_support(target: Turn, span: str) -> float:
        """1.0 when ``span`` falls inside an acoustic-disagreement region.

        The disagreement regions are recorded on the turn during the first pass
        (see ``acoustic.detect_asr_disagreement``) and mark where a second,
        independent ASR disagreed with the first pass — error-dense regions.
        A revision whose span lands in such a region is *acoustically supported*
        (the audio is genuinely ambiguous there), so it becomes one of the
        evidence features rather than a hard gate.
        """
        disagreement = (target.meta.get("uncertainty") or {}).get("acoustic_disagreement") or []
        if not disagreement or not span:
            return 0.0
        # The disagreement offsets are relative to the timestamp-free transcript
        # (the same text passed to detect_asr_disagreement), so strip the leading
        # [start-end] timestamp from raw_text before locating the span.
        text = re.sub(r"^\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*", "", target.raw_text or "")
        pos = text.find(span)
        if pos < 0:
            return 0.0
        span_end = pos + len(span)
        for item in disagreement:
            start = int(item.get("offset_a") or 0)
            end = start + len(str(item.get("span_a") or ""))
            if start < span_end and pos < end:  # any character-level overlap
                return 1.0
        return 0.0

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
        # Acoustic support: the focused span falls inside a region where two
        # independent ASRs disagreed (error-dense). This is a *feature* fed into
        # the calibration policy — it raises revision confidence when the audio
        # is genuinely ambiguous at the span, and lowers it otherwise. It is not
        # a hard gate: the closed-set verifier already did the authoritative
        # acoustic check above.
        acoustic_support = self._acoustic_support(target, focus.span)
        features = EvidenceFeatures(
            context_confidence=context_confidence,
            audio_confidence=top,
            audio_margin=margin,
            memory_support=memory_support,
            independent_sources=len(set(focus.evidence_turn_ids)),
            acoustic_support=acoustic_support,
        )
        if self.policy.decide(features, has_audio=True) != "REVISE":
            return Resolution("DEFER", target.turn_id, focus.span, score=top, rationale="evidence below bootstrap policy")
        action = "REVISE_CURRENT" if target.turn_id == source_turn.turn_id else "REVISE_HISTORY"
        evidence = [f"context:{turn_id}" for turn_id in focus.evidence_turn_ids]
        evidence.append(f"audio:{audio_path}:{start_sec}-{end_sec}")
        if acoustic_support:
            evidence.append(f"acoustic_disagreement:{focus.span}")
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
