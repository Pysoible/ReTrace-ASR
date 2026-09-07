"""Resolve a Context Judge focus with targeted historical audio."""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from asr_agent.calibration import DecisionPolicy, EvidenceFeatures
from asr_agent.context_judge import FocusProposal, candidates_language_compatible, language_compatible
from asr_agent.integrations.audio_verifier import retranscribe_window, verify_candidates
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
    audio_confidence: float = 0.0
    audio_margin: float = 0.0


class EvidenceResolver:
    DELETE_CANDIDATE = "[DELETE]"

    def __init__(
        self,
        audio_verifier: Callable[..., dict[str, Any]] | None = None,
        policy: DecisionPolicy | None = None,
    ) -> None:
        self.audio_verifier = audio_verifier or verify_candidates
        self.policy = policy or DecisionPolicy()

    @staticmethod
    def _strict_revision_enabled() -> bool:
        return os.getenv("ASR_STRICT_REVISION", "1").strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _is_near_variant(candidate_a: str, candidate_b: str) -> bool:
        """Allow a tighter gate for obvious near-character name mistakes.

        These are cases like "卡兹克" vs "卡兹个": the forms are very close, the
        semantic intent is clear, and the risk is mostly an ASR mis-heard final
        character rather than a completely different token. The legacy ratio-only
        check was too strict for short proper nouns because a single-character swap
        can still fall under the sequence similarity threshold when the names share
        the same prefix and are otherwise identical.
        """
        a = (candidate_a or "").strip()
        b = (candidate_b or "").strip()
        if not a or not b or a == b:
            return False
        if abs(len(a) - len(b)) > 1:
            return False

        # Prefer a precise edit-distance-style check for proper names: a single
        # character error or a one-character insertion/deletion is the common ASR
        # confusion pattern we want to preserve. This is more reliable than a
        # single global similarity ratio for short strings.
        if len(a) == len(b):
            mismatch = sum(1 for left, right in zip(a, b) if left != right)
            if mismatch <= 1:
                return True

        shorter, longer = sorted((a, b), key=len)
        for index in range(len(shorter)):
            if shorter[:index] == longer[:index] and shorter[index:] == longer[index + 1 :]:
                return True

        # Single-character substitutions with a long shared prefix/suffix are also
        # near variants even if the string differs slightly more than a raw Levenshtein
        # distance would suggest.
        prefix = 0
        while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
            prefix += 1
        suffix = 0
        while suffix < min(len(a) - prefix, len(b) - prefix) and a[-1 - suffix] == b[-1 - suffix]:
            suffix += 1
        if prefix + suffix >= min(len(a), len(b)) - 1:
            return True

        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        return ratio >= 0.7 and sum(1 for left, right in zip(a, b) if left != right) <= 2

    @staticmethod
    def _is_safe_local_replacement(span: str, replacement: str, operation: str = "REPLACE") -> bool:
        span = (span or "").strip()
        replacement = (replacement or "").strip()
        if not span:
            return False
        if operation == "DELETE":
            return len(span) <= 12 and not re.fullmatch(r"[^\w\u4e00-\u9fff]+", span)
        if not replacement:
            return False
        if re.fullmatch(r"[^\w\u4e00-\u9fff]+", span) or re.fullmatch(r"[^\w\u4e00-\u9fff]+", replacement):
            return False
        if abs(len(span) - len(replacement)) > 1:
            return False
        return difflib.SequenceMatcher(None, span, replacement).ratio() >= 0.5

    @staticmethod
    def _is_same_pronunciation(left: str, right: str) -> bool:
        try:
            from pypinyin import lazy_pinyin
        except ImportError:
            return False
        left, right = left.strip(), right.strip()
        return bool(left and right and lazy_pinyin(left) == lazy_pinyin(right))

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
            if str(item.get("span_a") or "").strip() == span and item.get("offset_a") is None:
                return 1.0
            start = int(item.get("offset_a") or 0)
            end = start + len(str(item.get("span_a") or ""))
            if start < span_end and pos < end:  # any character-level overlap
                return 1.0
        return 0.0

    @staticmethod
    def _focus_audio_window(target: Turn, span: str, start_sec: float, end_sec: float) -> tuple[float, float]:
        """Narrow long turn windows around the focused transcript span."""
        duration = end_sec - start_sec
        if duration <= 6.0:
            return start_sec, end_sec
        text = re.sub(r"^\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*", "", target.raw_text or "")
        position = text.find(span)
        if position < 0 or not text:
            return start_sec, end_sec
        midpoint = (position + len(span) / 2) / len(text)
        center = start_sec + midpoint * duration
        width = min(max(3.0, duration * 0.22), 7.0, duration)
        return max(start_sec, center - width / 2), min(end_sec, center + width / 2)

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
        operation = (focus.operation or "REPLACE").upper()
        if operation == "REPLACE" and not language_compatible(focus.span, focus.proposed_text):
            return Resolution(
                "DEFER",
                target.turn_id,
                focus.span,
                rationale="replacement changes the language/script of the focused transcript",
            )
        # Guard against a class of false positives where BOTH the span and the
        # proposed replacement already appear verbatim in the original raw text
        # (e.g. "时候" → "狮子狗" in "狮子狗刚出的时候…"). Here the span is a
        # normal word and the "correct" name already occurs elsewhere in the same
        # turn, so ASR already recognized it; a closed-set verifier listening to
        # the whole window is misled by that other occurrence. Refusing to revise
        # is safe (we only skip a revision, never invent one).
        if operation == "REPLACE" and (
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
                evidence=[f"context:{evidence_turn_id}" for evidence_turn_id in focus.evidence_turn_ids],
                rationale=focus.rationale or "the interpretations can refer to different entities",
            )
        if focus.relationship == "TEMPORAL_CHANGE":
            return Resolution(
                "ACCEPT_NEW",
                target.turn_id,
                focus.span,
                score=context_confidence,
                evidence=[f"context:{evidence_turn_id}" for evidence_turn_id in focus.evidence_turn_ids],
                rationale=focus.rationale or "the fact changed over time",
            )
        acoustic_supported = self._acoustic_support(target, focus.span) > 0.0
        homophone_candidate = operation == "REPLACE" and self._is_same_pronunciation(focus.span, focus.proposed_text)
        if (
            not self._is_safe_local_replacement(focus.span, focus.proposed_text, operation)
            and not acoustic_supported
            and not homophone_candidate
        ):
            return Resolution(
                "DEFER",
                target.turn_id,
                focus.span,
                rationale="replacement is not a plausible local edit of the focused span",
            )
        meta = target.meta or {}
        audio_path = meta.get("audio_path")
        start_sec, end_sec = meta.get("start_sec"), meta.get("end_sec")
        if not audio_path or start_sec is None or end_sec is None:
            return Resolution("DEFER", target.turn_id, focus.span, rationale="historical audio unavailable")
        verify_start, verify_end = self._focus_audio_window(
            target,
            focus.span,
            float(start_sec),
            float(end_sec),
        )
        if operation == "REPLACE" and not candidates_language_compatible(focus.span, list(focus.alternatives)):
            return Resolution(
                "DEFER",
                target.turn_id,
                focus.span,
                rationale="replacement candidates contain another language/script",
            )
        candidates = list(focus.alternatives)
        if operation == "DELETE":
            candidates = [focus.span, self.DELETE_CANDIDATE]
        elif self.DELETE_CANDIDATE not in candidates:
            candidates.append(self.DELETE_CANDIDATE)
        try:
            audio = self.audio_verifier(
                audio_path=str(audio_path),
                start_sec=verify_start,
                end_sec=verify_end,
                candidates=candidates,
            )
        except Exception as exc:
            return Resolution("DEFER", target.turn_id, focus.span, rationale=f"audio verifier failed: {exc}")
        scores = audio.get("scores") if isinstance(audio, dict) and audio.get("ok") else None
        if operation == "REPLACE" and (not isinstance(scores, dict) or set(scores) != set(candidates)):
            delete_candidates = [focus.span, self.DELETE_CANDIDATE]
            try:
                delete_audio = self.audio_verifier(
                    audio_path=str(audio_path),
                    start_sec=verify_start,
                    end_sec=verify_end,
                    candidates=delete_candidates,
                )
            except Exception:
                delete_audio = None
            delete_scores = delete_audio.get("scores") if isinstance(delete_audio, dict) and delete_audio.get("ok") else None
            if isinstance(delete_scores, dict) and set(delete_scores) == set(delete_candidates):
                candidates = delete_candidates
                scores = delete_scores
            else:
                candidates.remove(self.DELETE_CANDIDATE)
                try:
                    audio = self.audio_verifier(
                        audio_path=str(audio_path),
                        start_sec=verify_start,
                        end_sec=verify_end,
                        candidates=candidates,
                    )
                except Exception as exc:
                    return Resolution("DEFER", target.turn_id, focus.span, rationale=f"audio verifier failed: {exc}")
                scores = audio.get("scores") if isinstance(audio, dict) and audio.get("ok") else None
        if not isinstance(scores, dict) or set(scores) != set(candidates):
            return Resolution("DEFER", target.turn_id, focus.span, rationale="invalid closed-set audio result")
        normalized = {key: max(0.0, float(value)) for key, value in scores.items()}
        ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
        top = ordered[0][1] if ordered else 0.0
        proposed_key = self.DELETE_CANDIDATE if operation == "DELETE" else focus.proposed_text
        proposed_score = normalized.get(proposed_key, -1.0)
        if not ordered:
            return Resolution("KEEP_OLD", target.turn_id, focus.span, score=0.0)
        effective_operation = "DELETE" if ordered[0][0] == self.DELETE_CANDIDATE else operation
        if effective_operation == "REPLACE" and focus.proposed_text not in normalized:
            return Resolution("KEEP_OLD", target.turn_id, focus.span, score=top, rationale="replacement candidate was not acoustically verified")
        near_variant = effective_operation == "REPLACE" and self._is_near_variant(focus.span, focus.proposed_text)

        # For a near-variant proper noun, the decisive question is whether the
        # proposed spelling is still the best candidate and whether the evidence is
        # strong enough to override the historical raw text. We should not reject it
        # just because the general-purpose audio_margin gate is designed for more
        # divergent candidates.
        if ordered[0][0] != proposed_key and effective_operation != "DELETE":
            if not near_variant:
                return Resolution("KEEP_OLD", target.turn_id, focus.span, score=top)
            if proposed_score < 0.0:
                return Resolution("KEEP_OLD", target.turn_id, focus.span, score=top)
            if top - proposed_score > 0.25:
                return Resolution("KEEP_OLD", target.turn_id, focus.span, score=top)

        margin = top - (ordered[1][1] if len(ordered) > 1 else 0.0)
        strict_revision = self._strict_revision_enabled()
        if strict_revision and near_variant:
            effective_margin_threshold = max(0.10, self.policy.thresholds.audio_margin)
            effective_revise_threshold = max(0.75, self.policy.thresholds.revise)
        else:
            effective_margin_threshold = 0.0 if near_variant else self.policy.thresholds.audio_margin
            effective_revise_threshold = 0.40 if near_variant else self.policy.thresholds.revise
        # Acoustic support: the focused span falls inside a region where two
        # independent ASRs disagreed (error-dense). This is a *feature* fed into
        # the calibration policy — it raises revision confidence when the audio
        # is genuinely ambiguous at the span, and lowers it otherwise. It is not
        # a hard gate: the closed-set verifier already did the authoritative
        # acoustic check above.
        acoustic_support = self._acoustic_support(target, focus.span)
        # In high-precision mode, a normal-word revision may still proceed when
        # targeted verification is exceptionally decisive. Acoustic disagreement
        # is strong supporting evidence, not a mandatory prerequisite: the long
        # RAMC fast path can disable the expensive second ASR.
        features = EvidenceFeatures(
            context_confidence=context_confidence,
            audio_confidence=top,
            audio_margin=margin,
            memory_support=memory_support,
            independent_sources=len(set(focus.evidence_turn_ids)),
            acoustic_support=acoustic_support,
        )
        if strict_revision and not near_variant and (
            context_confidence < 0.85
            or top < 0.85
            or margin < 0.30
            or (not acoustic_support and self.policy.calibrator.predict(features) < 0.75)
        ):
            return Resolution(
                "DEFER",
                target.turn_id,
                focus.span,
                score=top,
                rationale="strict revision mode requires decisive local audio and context evidence",
            )
        if top < self.policy.thresholds.relisten:
            return Resolution("DEFER", target.turn_id, focus.span, score=top, rationale="audio confidence below relisten threshold")
        if margin < effective_margin_threshold and not near_variant:
            return Resolution("DEFER", target.turn_id, focus.span, score=top, rationale="audio margin too small for a safe revision")
        if self.policy.calibrator.predict(features) < effective_revise_threshold:
            return Resolution("DEFER", target.turn_id, focus.span, score=top, rationale="evidence below adjusted revision policy")
        action = "REVISE_CURRENT" if target.turn_id == source_turn.turn_id else "REVISE_HISTORY"
        evidence = [f"context:{evidence_turn_id}" for evidence_turn_id in focus.evidence_turn_ids]
        evidence.append(f"audio:{audio_path}:{start_sec}-{end_sec}")
        if acoustic_support:
            evidence.append(f"acoustic_disagreement:{focus.span}")
        if homophone_candidate:
            evidence.append(f"context_homophone:{focus.span}->{focus.proposed_text}")
        return Resolution(
            action,
            target.turn_id,
            focus.span,
            replacement="" if effective_operation == "DELETE" else focus.proposed_text,
            score=self.policy.calibrator.predict(features),
            evidence=evidence,
            rationale=focus.rationale or "context and targeted audio agree",
            audio_verified=True,
            audio_confidence=top,
            audio_margin=margin,
        )
