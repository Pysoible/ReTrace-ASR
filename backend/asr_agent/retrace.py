"""Evidence-grounded retrospective revision for conversational ASR."""
from __future__ import annotations

import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from asr_agent.models import EntityProfile, Hypothesis, RevisionEvent, Session, Turn
from asr_agent.storage import SessionRepository
from asr_agent.uncertainty import detect_suspicious_spans

_TIME_PREFIX = re.compile(r"^\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]\s*")
_ASCII_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,19}")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_ENTITY_BOUNDARY = re.compile(
    r"(?:^|[，。；、！？\s]|比如|或者|还是|就是|叫做?|名叫|是)"
    r"(?!比如|或者|还是|就是|叫做?|名叫)"
    r"([\u4e00-\u9fff]{2}|[A-Za-z][A-Za-z0-9\-]{1,19})"
)
_TITLE_TOKEN = re.compile(
    r"(?:^|[^\u4e00-\u9fff]|负责人|请|是|有|和|与|及)"
    r"([\u4e00-\u9fff]{1,2}(?:博士|老师|总|经理|主任))"
)
_PRODUCT_TOKEN = re.compile(r"[\u4e00-\u9fff]{2,3}(?:屏|险|金|器|机|芯)")
_PARTICLE_CHARS = set("的了呢吗吧啊嗯哦呀哈对是就都也与和及在有我你他她它们这那什么一个")
_FILLER_CHARS = set("嗯啊哦呃哈呀吧嘛呐")
_COMMON_BIGRAMS = {
    "一样", "一个", "我们", "可以", "不是", "因为", "所以", "这个", "那个", "什么",
    "还是", "没有", "已经", "自己", "他们", "大家", "现在", "觉得", "知道", "应该",
    "如果", "但是", "而且", "或者", "为了", "以及", "然后", "就是", "不是", "还有",
    "比较", "一些", "这些", "那些", "这么", "那么", "什么", "怎么", "多少", "哪里",
    "今天", "明天", "昨天", "时候", "问题", "工作", "公司", "员工", "方面", "方案",
    "以考", "以厂", "一金", "开讨", "家讨", "个话", "个指", "遇方", "如泰", "如说",
    "好的", "单位", "会上", "让补", "喜欢", "全面", "确认", "然后", "对于",
}
# Common nouns that are near-form neighbors of each other but are not ASR ambiguity targets.
_GENERIC_ENTITY_BLOCK = {
    "保险", "保障", "社保", "社会", "企业", "员工", "公司", "工作", "问题", "方面", "方案",
    "福利", "假期", "放假", "有假", "通讯", "通信", "交通", "补充", "商业", "家庭", "成员",
    "单位", "领导", "利润", "投资", "食堂", "饮食", "产假", "病假", "婚假", "丧假",
    "几个", "几种", "给予", "给他", "给人", "咱们", "好的", "然后", "对于", "可以",
    "一个", "这个", "那个", "不是", "就是", "还是", "因为", "所以", "而且", "或者",
    "餐补", "补补", "挺多", "更大", "更能", "不但", "不能", "你知", "你能",
    "商业保险", "业保险", "康的保险", "的保险",
}
# Later English / mixed brands can correct earlier Chinese ASR surfaces without LLM.
_BRAND_SURFACE_ALIASES: dict[str, tuple[str, ...]] = {
    "vivo": ("威沃", "微沃", "维沃", "vovo"),
    "oppo": ("欧珀", "欧破"),
    "iphone": ("爱疯",),
    "cpu": ("西皮尤",),
}

class ReTraceService:
    def __init__(
        self,
        storage_dir: Path,
        evidence_scorer: Callable[..., dict[str, Any]] | None = None,
        reflector: Callable[..., list[dict[str, Any]]] | None = None,
        adjudicator: Callable[..., dict[str, Any]] | None = None,
        audio_verifier: Callable[..., dict[str, Any]] | None = None,
        audio_retranscriber: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.storage_dir = storage_dir
        self.repository = SessionRepository(storage_dir)
        self.evidence_scorer = evidence_scorer
        self.reflector = reflector
        self.adjudicator = adjudicator
        self.audio_verifier = audio_verifier
        self.audio_retranscriber = audio_retranscriber

    def upsert_entities(self, session_id: str, profiles: list[EntityProfile]) -> dict[str, Any]:
        session = self.repository.update(
            session_id,
            lambda current: current.entities.update({profile.entity_id: profile for profile in profiles}),
        )
        return session.as_dict()

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._load(session_id).as_dict()

    @staticmethod
    def _replay(session: Session) -> None:
        """Derive visible text and memory only from immutable raw turns and active events."""
        for turn in session.turns:
            turn.current_text = turn.raw_text
        session.verified_memory = {}
        session.quarantine_memory = {}
        turns = {turn.turn_id: turn for turn in session.turns}
        for event in session.revision_events:
            if not event.active or event.action not in {"REVISE_TEXT", "REVISE_ENTITY"}:
                continue
            turn = turns.get(event.target_turn_id)
            if not turn:
                continue
            if event.replacement and event.span in turn.current_text:
                turn.current_text = turn.current_text.replace(event.span, event.replacement, 1)
            else:
                turn.current_text = event.after_text
            if event.entity_id and event.entity_id in session.entities:
                profile = session.entities[event.entity_id]
                session.verified_memory[event.entity_id] = {"name": profile.name, "source_turn_id": event.source_turn_id}
        for turn in session.turns:
            for hypothesis in turn.hypotheses:
                if hypothesis.action == "DEFER":
                    for entity_id in hypothesis.entity_candidate_ids:
                        profile = session.entities.get(entity_id)
                        if profile:
                            session.quarantine_memory[entity_id] = {"name": profile.name, "status": "candidate"}

    def reset_session(self, session_id: str) -> dict[str, Any]:
        """Replace an existing session with an empty one (one long audio = one session)."""
        session = self.repository.update(
            session_id,
            lambda current: current.__dict__.update(Session(session_id=session_id).__dict__),
        )
        return session.as_dict()

    def process_turn(
        self,
        session_id: str,
        turn_id: str,
        text: str,
        *,
        confidence: dict[str, float] | None = None,
        text_candidates: dict[str, list[str]] | None = None,
        entity_candidate_ids: dict[str, list[str]] | None = None,
        use_llm: bool = False,
        source: str = "text",
        meta: dict[str, Any] | None = None,
        risk: str = "medium",
        nbest: list[str] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}

        def mutate(session: Session) -> None:
            result.update(self._process_turn_in_session(
                session,
                turn_id,
                text,
                confidence=confidence,
                text_candidates=text_candidates,
                entity_candidate_ids=entity_candidate_ids,
                use_llm=use_llm,
                source=source,
                meta=meta,
                risk=risk,
                nbest=nbest,
            ))

        session = self.repository.update(session_id, mutate)
        result["session"] = session.as_dict()
        return result

    def _process_turn_in_session(
        self,
        session: Session,
        turn_id: str,
        text: str,
        *,
        confidence: dict[str, float] | None,
        text_candidates: dict[str, list[str]] | None,
        entity_candidate_ids: dict[str, list[str]] | None,
        use_llm: bool,
        source: str,
        meta: dict[str, Any] | None,
        risk: str,
        nbest: list[str] | None,
    ) -> dict[str, Any]:
        if any(turn.turn_id == turn_id for turn in session.turns):
            raise ValueError(f"duplicate turn_id: {turn_id}")

        llm_meta: dict[str, Any] = dict(meta or {})
        working_text = text

        confidence, text_candidates, entity_candidate_ids = confidence or {}, text_candidates or {}, entity_candidate_ids or {}
        for span, candidate, entity_id in self._recall_entity_fallbacks(session, working_text):
            confidence.setdefault(span, 0.6)
            text_candidates.setdefault(span, [span, candidate])
            entity_candidate_ids.setdefault(span, []).append(entity_id)
        detected_spans = detect_suspicious_spans(working_text, confidence=confidence, nbest=nbest)
        for detected in detected_spans:
            text_candidates.setdefault(detected.text, [detected.text])
        spans = set(text_candidates) | set(entity_candidate_ids)
        hypotheses = []
        for span in spans:
            if span not in working_text or confidence.get(span, 1.0) >= 0.65:
                continue
            candidates = text_candidates.get(span, [span])
            recalled = [profile.entity_id for profile in session.entities.values() if any(value in {profile.name, *profile.aliases} for value in candidates)]
            unique_candidates = list(dict.fromkeys(candidates))
            probability = 1.0 / len(unique_candidates)
            decision = "WAIT"
            hypotheses.append(Hypothesis(
                span=span,
                text_candidates=unique_candidates,
                entity_candidate_ids=list(dict.fromkeys([*entity_candidate_ids.get(span, []), *recalled])),
                candidates=[{"text": candidate, "score": probability, "supporting_evidence": [], "contradicting_evidence": []} for candidate in unique_candidates],
                risk=risk,
                decision=decision,
                decision_rationale=["competing_candidates"],
                evidence_packet={
                    "asr_uncertainty": {"confidence": confidence.get(span), "nbest": nbest or []},
                    "suspicious_span": next((item.as_dict() for item in detected_spans if item.text == span), None),
                    "later_raw_evidence": [],
                },
            ))
        for hypothesis in hypotheses:
            for entity_id in hypothesis.entity_candidate_ids:
                profile = session.entities.get(entity_id)
                if profile:
                    session.quarantine_memory[entity_id] = {"name": profile.name, "status": "candidate"}
        session.turns.append(
            Turn(
                turn_id=turn_id,
                raw_text=text,
                current_text=working_text,
                hypotheses=hypotheses,
                source=source,
                meta=llm_meta,
            )
        )
        source_index = len(session.turns) - 1
        # Later language can nominate an ambiguity, but only an audio re-listen may
        # mutate the displayed transcript.  This keeps the agent closed-set and
        # prevents a fluent text model from hallucinating a historical correction.
        revisions = self._reflect(session, source_index, use_llm=use_llm)
        return {
            "revisions": revisions,
            "actions": [item.action for item in hypotheses],
            "integrations": {"source": source},
        }

    @staticmethod
    def _strip_time_prefix(text: str) -> str:
        return _TIME_PREFIX.sub("", text or "").strip()

    @classmethod
    def _content_tokens(cls, text: str) -> list[str]:
        """Entity-like tokens: ASCII words plus CJK bigrams/trigrams."""
        plain = cls._strip_time_prefix(text)
        out: list[str] = []
        for match in _ASCII_TOKEN.finditer(plain):
            out.append(match.group(0))
        for run in _CJK_RUN.findall(plain):
            for size in (2, 3):
                if len(run) < size:
                    continue
                for index in range(len(run) - size + 1):
                    token = run[index : index + size]
                    if all(char in _PARTICLE_CHARS for char in token):
                        continue
                    out.append(token)
        return list(dict.fromkeys(out))

    @classmethod
    def _entity_tokens(cls, text: str) -> list[str]:
        """Boundary-aware tokens for near-form conflicts (avoids 比如→如泰 noise)."""
        plain = cls._strip_time_prefix(text)
        out: list[str] = []
        for match in _ENTITY_BOUNDARY.finditer(plain):
            token = match.group(1)
            if not token or all(char in _PARTICLE_CHARS for char in token):
                continue
            out.append(token)
        # Also keep standalone ASCII brands anywhere.
        for match in _ASCII_TOKEN.finditer(plain):
            out.append(match.group(0))
        return list(dict.fromkeys(out))

    @staticmethod
    def _similar_conflict(left: str, right: str) -> bool:
        if not left or not right or left == right:
            return False
        a, b = left.lower(), right.lower()
        if a == b or a in _COMMON_BIGRAMS or b in _COMMON_BIGRAMS:
            return False
        if a in _GENERIC_ENTITY_BLOCK or b in _GENERIC_ENTITY_BLOCK:
            return False
        if "的" in a or "的" in b:
            return False
        # Brand/name-like CJK bigrams and short titles (图博士/涂博士); avoid long spans.
        if not a.isascii() and not (2 <= len(a) <= 3):
            return False
        if not b.isascii() and not (2 <= len(b) <= 3):
            return False
        if len(a) == len(b) and len(a) >= 2:
            if not a.isascii() and len(set(a) & set(b)) < 1:
                return False
            if sum(x != y for x, y in zip(a, b)) != 1:
                return False
            # Prefer same-family variants: shared prefix (泰信/泰康) or suffix (尧龙/骁龙, 图博士/涂博士).
            if not a.isascii() and a[0] != b[0] and a[-1] != b[-1]:
                return False
            return True
        return False

    @classmethod
    def _quality_tokens(cls, text: str) -> list[str]:
        """High-precision anchors: boundary entities, titles, products, latin brands."""
        plain = cls._strip_time_prefix(text)
        out: list[str] = []
        out.extend(cls._entity_tokens(plain))
        for match in _TITLE_TOKEN.finditer(plain):
            out.append(match.group(1))
        for match in _PRODUCT_TOKEN.finditer(plain):
            token = match.group(0)
            if "的" in token or token in _GENERIC_ENTITY_BLOCK:
                continue
            out.append(token)
        for match in _ASCII_TOKEN.finditer(plain):
            out.append(match.group(0))
        return list(dict.fromkeys(item for item in out if item and item not in _GENERIC_ENTITY_BLOCK))

    @classmethod
    def _sliding_cjk_grams(cls, text: str, size: int) -> list[str]:
        """All CJK grams of a fixed size; used only against a quality anchor on the other side."""
        if size < 2:
            return []
        plain = cls._strip_time_prefix(text)
        out: list[str] = []
        for run in _CJK_RUN.findall(plain):
            if len(run) < size:
                continue
            for index in range(len(run) - size + 1):
                token = run[index : index + size]
                if token in _COMMON_BIGRAMS or token in _GENERIC_ENTITY_BLOCK:
                    continue
                if all(char in _PARTICLE_CHARS for char in token):
                    continue
                out.append(token)
        return list(dict.fromkeys(out))

    @classmethod
    def _nominable_tokens(cls, text: str) -> list[str]:
        """Backward-compatible alias used by same-turn conflict scan."""
        return cls._quality_tokens(text)

    @classmethod
    def _span_inside_longer_token(cls, plain: str, span: str, quality: list[str] | None = None) -> bool:
        """Reject revising interior grams like ``布屏`` inside ``瀑布屏``.

        Prefix heads such as ``泰康`` in ``泰康的保险`` remain eligible.
        """
        if not span or not plain or span not in plain:
            return False
        anchors = list(quality or [])
        anchors.extend(_PRODUCT_TOKEN.findall(plain))
        anchors.extend(_TITLE_TOKEN.findall(plain))  # group 1 via findall
        for anchor in dict.fromkeys(anchors):
            if not anchor or anchor == span or span not in anchor or anchor not in plain:
                continue
            if anchor.startswith(span):
                continue
            start = 0
            while True:
                at = plain.find(anchor, start)
                if at < 0:
                    break
                # Span occurs inside this anchor occurrence, but is not its prefix.
                if plain.find(span, at, at + len(anchor)) >= at:
                    return True
                start = at + 1
        return False

    @classmethod
    def _content_chars(cls, text: str) -> list[str]:
        plain = cls._strip_time_prefix(text)
        return [ch for ch in plain if ("\u4e00" <= ch <= "\u9fff") or ch.isalnum()]

    @classmethod
    def _is_degenerate_transcript(cls, text: str, *, duration_sec: float | None = None) -> bool:
        """Detect collapsed / filler-only ASR that should trigger open re-listen."""
        chars = cls._content_chars(text)
        if len(chars) < 2:
            return True
        max_run = 1
        cur = 1
        for index in range(1, len(chars)):
            if chars[index] == chars[index - 1]:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1
        run_ratio = max_run / len(chars)
        filler_ratio = sum(ch in _FILLER_CHARS for ch in chars) / len(chars)
        unique_ratio = len(set(chars)) / len(chars)
        if max_run >= 4 and run_ratio >= 0.35:
            return True
        if filler_ratio >= 0.55 and len(chars) >= 6:
            return True
        if unique_ratio <= 0.2 and len(chars) >= 8:
            return True
        if duration_sec is not None and duration_sec >= 4.0:
            unique_cjk = {ch for ch in chars if "\u4e00" <= ch <= "\u9fff"}
            if len(unique_cjk) <= 3 and len(chars) <= max(8, int(duration_sec)):
                return True
        return False

    @classmethod
    def _retranscription_is_better(cls, before: str, after: str) -> bool:
        if not after or after.strip() == before.strip():
            return False
        before_chars = cls._content_chars(before)
        after_chars = cls._content_chars(after)
        if len(after_chars) < 2:
            return False
        before_sig = {ch for ch in before_chars if ch not in _FILLER_CHARS}
        after_sig = {ch for ch in after_chars if ch not in _FILLER_CHARS}
        if not after_sig:
            return False
        # Recovered non-filler lexical content the first pass missed (e.g. 遥→骁龙).
        if len(after_sig - before_sig) >= 2:
            return True
        if len(after_sig) >= len(before_sig) + 2:
            return True
        before_fill = sum(ch in _FILLER_CHARS for ch in before_chars) / max(1, len(before_chars))
        after_fill = sum(ch in _FILLER_CHARS for ch in after_chars) / max(1, len(after_chars))
        if len(after_sig) > len(before_sig) and after_fill < before_fill:
            return True
        if len(after_sig) >= 4 and len(before_sig) <= 2 and after_fill < 0.55:
            return True
        return False

    @staticmethod
    def _trim_retranscription(text: str) -> str:
        """Drop trailing filler-only tails that open re-ASR often appends."""
        parts = re.split(r"([。！？!?])", text.strip())
        chunks: list[str] = []
        index = 0
        while index < len(parts):
            piece = parts[index]
            punct = parts[index + 1] if index + 1 < len(parts) else ""
            chunk = piece + punct
            if chunk.strip():
                chunks.append(chunk)
            index += 2 if punct else 1
        while chunks:
            body = re.sub(r"[\s。！？!?，,、]+", "", chunks[-1])
            if not body or all(ch in _FILLER_CHARS or ch == "对" for ch in body):
                chunks.pop()
                continue
            break
        return "".join(chunks).strip() or text.strip()

    def _deterministic_proposals(self, session: Session, source_index: int) -> list[dict[str, Any]]:
        """Nominate revisions without LLM when later (or later-in-turn) evidence appears."""
        source = session.turns[source_index]
        source_plain = self._strip_time_prefix(source.raw_text)
        source_tokens = self._nominable_tokens(source.raw_text)
        proposals: list[dict[str, Any]] = []

        # Path A: earlier deferred hypothesis whose competing candidate now appears verbatim.
        for turn in session.turns[: source_index + 1]:
            turn_plain = self._strip_time_prefix(turn.current_text)
            for hypothesis in turn.hypotheses:
                if hypothesis.action != "DEFER":
                    continue
                if hypothesis.span not in turn_plain:
                    continue
                for candidate in hypothesis.text_candidates:
                    if candidate == hypothesis.span or not candidate:
                        continue
                    if turn.turn_id == source.turn_id:
                        span_at = turn_plain.find(hypothesis.span)
                        cand_at = turn_plain.find(candidate, span_at + len(hypothesis.span)) if span_at >= 0 else -1
                        if cand_at < 0:
                            continue
                        quote = candidate
                    else:
                        if candidate not in source_plain:
                            continue
                        quote = candidate
                    proposals.append(
                        {
                            "target_turn_id": turn.turn_id,
                            "before_text": hypothesis.span,
                            "after_text": candidate,
                            "evidence": [{"turn_id": source.turn_id, "quote": quote}],
                            "score": 0.88,
                            "rationale": "later verbatim candidate evidence for an open hypothesis",
                        }
                    )

        # Path D: later Latin brand corrects earlier Chinese ASR surface (威沃→vivo).
        for match in _ASCII_TOKEN.finditer(source_plain):
            brand = match.group(0)
            aliases = _BRAND_SURFACE_ALIASES.get(brand.lower(), ())
            if not aliases:
                continue
            for turn in session.turns[:source_index]:
                turn_plain = self._strip_time_prefix(turn.current_text)
                for alias in aliases:
                    if alias in turn_plain and brand in source_plain:
                        proposals.append(
                            {
                                "target_turn_id": turn.turn_id,
                                "before_text": alias,
                                "after_text": brand,
                                "evidence": [{"turn_id": source.turn_id, "quote": brand}],
                                "score": 0.9,
                                "rationale": "later latin brand corrects earlier chinese ASR surface",
                            }
                        )

        # Path B/C: near-form conflicts.
        # B: earlier form A, later/same-turn later form B -> try A->B (retrospective)
        # C: earlier form A, current form B -> try B->A (earlier canonical corrects later drift)
        conflict_budget = 12
        conflict_proposals: list[dict[str, Any]] = []
        for turn in session.turns[:source_index]:
            turn_plain = self._strip_time_prefix(turn.current_text)
            earlier_quality = self._quality_tokens(turn.current_text)
            # Cross-turn Path B only: later evidence revises earlier ASR.
            # Match quality anchors on either side against sliding grams on the other so
            # mid-sentence evidence like ``会上泰康`` is visible without exploding recall.
            pair_specs: list[tuple[str, str]] = []
            for before in earlier_quality:
                if before not in turn_plain or before.isascii():
                    continue
                for after in [
                    *source_tokens,
                    *self._sliding_cjk_grams(source_plain, len(before)),
                ]:
                    pair_specs.append((before, after))
            for after in source_tokens:
                if after not in source_plain or after.isascii():
                    continue
                for before in [
                    *earlier_quality,
                    *self._sliding_cjk_grams(turn_plain, len(after)),
                ]:
                    pair_specs.append((before, after))
            seen_pairs: set[tuple[str, str]] = set()
            for before, after in pair_specs:
                key = (before, after)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                if before not in turn_plain or after not in source_plain:
                    continue
                if not self._similar_conflict(before, after):
                    continue
                if self._span_inside_longer_token(turn_plain, before, earlier_quality):
                    continue
                if self._span_inside_longer_token(source_plain, after, source_tokens):
                    continue
                conflict_proposals.append(
                    {
                        "target_turn_id": turn.turn_id,
                        "before_text": before,
                        "after_text": after,
                        "evidence": [{"turn_id": source.turn_id, "quote": after}],
                        "score": 0.88,
                        "rationale": "near-form conflict across conversational evidence",
                        "allow_earlier_evidence": False,
                    }
                )

        # Same-turn ordered conflicts on the new turn only.
        turn_plain = source_plain
        earlier_tokens = source_tokens
        for i, before in enumerate(earlier_tokens):
            before_at = turn_plain.find(before)
            if before_at < 0:
                continue
            if self._span_inside_longer_token(turn_plain, before, earlier_tokens):
                continue
            for after in earlier_tokens[i + 1 :]:
                if not self._similar_conflict(before, after):
                    continue
                if self._span_inside_longer_token(turn_plain, after, earlier_tokens):
                    continue
                after_at = turn_plain.find(after, before_at + len(before))
                if after_at < 0:
                    continue
                conflict_proposals.append(
                    {
                        "target_turn_id": source.turn_id,
                        "before_text": before,
                        "after_text": after,
                        "evidence": [{"turn_id": source.turn_id, "quote": after}],
                        "score": 0.84,
                        "rationale": "near-form conflict across conversational evidence",
                        "allow_earlier_evidence": False,
                    }
                )
                conflict_proposals.append(
                    {
                        "target_turn_id": source.turn_id,
                        "before_text": after,
                        "after_text": before,
                        "evidence": [{"turn_id": source.turn_id, "quote": before}],
                        "score": 0.87,
                        "rationale": "earlier canonical form corrects later near-form drift",
                        "allow_earlier_evidence": True,
                        "require_quote_before_span": True,
                    }
                )

        # Prefer shared-prefix brand-like pairs and keep a small budget.
        def _priority(item: dict[str, Any]) -> tuple[int, int, int]:
            before, after = str(item["before_text"]), str(item["after_text"])
            shared_prefix = 1 if before and after and before[0] == after[0] else 0
            title_bonus = 1 if before.endswith(("博士", "老师", "经理")) or after.endswith(("博士", "老师", "经理")) else 0
            return (-title_bonus, -shared_prefix, -int(float(item.get("score") or 0) * 100))

        conflict_proposals.sort(key=_priority)
        proposals.extend(conflict_proposals[:conflict_budget])
        return proposals

    def _recover_degenerate_turn(self, session: Session, source_index: int) -> dict[str, Any] | None:
        """Open re-ASR when the current turn transcript collapsed (e.g. 遥遥遥遥)."""
        turn = session.turns[source_index]
        audio_path = str(turn.meta.get("audio_path") or "")
        start = turn.meta.get("start_sec")
        end = turn.meta.get("end_sec")
        if not audio_path or start is None or end is None:
            return None
        start_f, end_f = float(start), float(end)
        duration = end_f - start_f
        plain = self._strip_time_prefix(turn.current_text)
        if not self._is_degenerate_transcript(plain, duration_sec=duration):
            return None
        # Slightly widen the window; first-pass chunk edges often clip recoverable speech.
        pad = 1.5
        window_start = max(0.0, start_f - pad)
        window_end = end_f + pad
        retranscriber = self.audio_retranscriber
        if retranscriber is None:
            from asr_agent.integrations.audio_verifier import retranscribe_window

            retranscriber = retranscribe_window
        verdict = retranscriber(audio_path=audio_path, start_sec=window_start, end_sec=window_end)
        if not verdict.get("ok"):
            turn.meta["degenerate_relisten"] = verdict
            return None
        new_text = self._trim_retranscription(str(verdict.get("text") or "").strip())
        turn.meta["degenerate_relisten"] = {
            "ok": True,
            "before": plain,
            "after": new_text,
            "start_sec": window_start,
            "end_sec": window_end,
        }
        if not self._retranscription_is_better(plain, new_text):
            return None
        prefix_match = _TIME_PREFIX.match(turn.current_text or "")
        prefix = prefix_match.group(0) if prefix_match else ""
        hypothesis = self._ensure_closed_set_hypothesis(turn, plain, new_text)
        hypothesis.decision = "RELISTEN"
        hypothesis.decision_rationale = ["degenerate_transcript_open_relisten"]
        hypothesis.evidence_packet["audio_retranscription"] = verdict
        before = turn.current_text
        turn.current_text = f"{prefix}{new_text}"
        hypothesis.action = "REVISE_TEXT"
        hypothesis.decision = "COMMIT"
        event = RevisionEvent(
            event_id=uuid.uuid4().hex,
            action="REVISE_TEXT",
            target_turn_id=turn.turn_id,
            source_turn_id=turn.turn_id,
            span=plain,
            before_text=before,
            after_text=turn.current_text,
            entity_id=None,
            score=0.9,
            evidence=[f"audio-retranscribe:{audio_path}:{window_start}-{window_end}"],
            resolver="audio-open-relisten",
            rationale="degenerate first-pass transcript recovered by open audio re-listen",
            replacement=new_text,
        )
        session.revision_events.append(event)
        return asdict(event)

    def _sparse_llm_triggers(self, session: Session, source_index: int) -> list[str]:
        """TAMA-style gates: call DeepSeek only when local heuristics need help."""
        if source_index <= 0:
            return []
        reasons: list[str] = []
        source_plain = self._strip_time_prefix(session.turns[source_index].raw_text)
        if _ASCII_TOKEN.search(source_plain or ""):
            reasons.append("cross_script_brand")
        for turn in session.turns[:source_index]:
            turn_plain = self._strip_time_prefix(turn.current_text)
            for hyp in turn.hypotheses:
                if hyp.action != "DEFER" or hyp.span not in turn_plain:
                    continue
                # Only uncertainty-originated hangs (not nomination-only leftovers).
                asr_u = (hyp.evidence_packet or {}).get("asr_uncertainty") or {}
                if asr_u.get("confidence") is None and "competing_candidates" not in (hyp.decision_rationale or []):
                    continue
                for cand in hyp.text_candidates:
                    if cand != hyp.span and cand and cand in source_plain:
                        reasons.append("deferred_with_later_candidate")
                        break
                else:
                    continue
                break
        return reasons

    def _reflect(self, session: Session, source_index: int, *, use_llm: bool = False) -> list[dict[str, Any]]:
        if source_index < 0:
            return []
        committed: list[dict[str, Any]] = []
        llm_meta: dict[str, Any] = {"mode": "tama_sparse", "calls": 0, "triggers": []}
        recovered = self._recover_degenerate_turn(session, source_index)
        if recovered:
            committed.append(recovered)
        # Local-first: deterministic Path A/B/C (+ open re-listen) before any DeepSeek.
        proposals = self._deterministic_proposals(session, source_index)
        if use_llm and source_index > 0:
            triggers = self._sparse_llm_triggers(session, source_index)
            llm_meta["triggers"] = triggers
            if triggers:
                reflector = self.reflector
                if reflector is None:
                    from asr_agent.integrations.deepseek import reflect_timeline

                    reflector = reflect_timeline
                prior = [
                    {"turn_id": turn.turn_id, "raw_text": self._strip_time_prefix(turn.raw_text)}
                    for turn in session.turns[max(0, source_index - 8) : source_index]
                ]
                new_turn = {
                    "turn_id": session.turns[source_index].turn_id,
                    "raw_text": self._strip_time_prefix(session.turns[source_index].raw_text),
                }
                try:
                    llm_proposals = reflector(prior_turns=prior, new_turn=new_turn) or []
                    llm_meta["calls"] += 1
                except Exception:
                    llm_proposals = []
                proposals.extend(llm_proposals)

        indexes = {turn.turn_id: index for index, turn in enumerate(session.turns)}
        seen: set[tuple[str, str, str]] = set()
        blocked_pairs: set[tuple[str, str]] = set()
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            target_id = str(proposal.get("target_turn_id") or "")
            target_index = indexes.get(target_id, -1)
            before = str(proposal.get("before_text") or "").strip()
            after = str(proposal.get("after_text") or "").strip()
            score = float(proposal.get("score") or 0.0)
            key = (target_id, before, after)
            if key in seen:
                continue
            seen.add(key)
            if (before, after) in blocked_pairs or (after, before) in blocked_pairs:
                continue
            if target_index < 0 or target_index > source_index:
                continue
            target = session.turns[target_index]
            target_plain = self._strip_time_prefix(target.current_text)
            if (
                not before
                or not after
                or before == after
                or before not in target_plain
                or score < 0.7
                or not self._is_discoverable_span_pair(before, after)
            ):
                continue
            evidence: list[str] = []
            valid = True
            allow_earlier_evidence = bool(proposal.get("allow_earlier_evidence"))
            require_quote_before_span = bool(proposal.get("require_quote_before_span"))
            for item in proposal.get("evidence") or []:
                if not isinstance(item, dict):
                    valid = False
                    break
                evidence_id, quote = str(item.get("turn_id") or ""), str(item.get("quote") or "")
                evidence_index = indexes.get(evidence_id, -1)
                if not quote or evidence_index < 0 or evidence_index > source_index or after not in quote:
                    valid = False
                    break
                evidence_plain = self._strip_time_prefix(session.turns[evidence_index].raw_text)
                if quote not in evidence_plain:
                    valid = False
                    break
                if evidence_index < target_index and not allow_earlier_evidence:
                    valid = False
                    break
                if evidence_index == target_index:
                    before_at = target_plain.find(before)
                    if require_quote_before_span:
                        quote_at = evidence_plain.find(quote)
                        if before_at < 0 or quote_at < 0 or quote_at >= before_at:
                            valid = False
                            break
                    else:
                        quote_at = evidence_plain.find(quote, before_at + len(before)) if before_at >= 0 else -1
                        if quote_at < 0:
                            valid = False
                            break
                evidence.append(f"{evidence_id}:{quote}")
            if not valid or not evidence:
                continue
            hypothesis = self._ensure_closed_set_hypothesis(target, before, after)
            hypothesis.decision = "RELISTEN"
            hypothesis.decision_rationale = ["verified_future_semantic_evidence", "agent_discovered_span"]
            hypothesis.evidence_packet["later_raw_evidence"] = evidence
            event = self._relisten_and_commit(
                session,
                target,
                hypothesis,
                source_index,
                after,
                score,
                evidence,
                str(proposal.get("rationale") or ""),
                allow_llm_confirm=use_llm,
            )
            if event:
                if event.get("resolver") in {"audio-llm-confirm"}:
                    llm_meta["calls"] += 1
                blocked_pairs.add((before, after))
                blocked_pairs.add((after, before))
                committed.append(event)

        # TAMA-style: only reassess deferred ASR-uncertainty hangs when later text may help.
        if use_llm and source_index > 0 and "deferred_with_later_candidate" in llm_meta["triggers"]:
            before_n = len(committed)
            for event in self._reassess(session, source_index, use_llm=True):
                committed.append(event)
            if len(committed) > before_n:
                llm_meta["calls"] += len(committed) - before_n

        if session.turns:
            session.turns[source_index].meta["llm"] = llm_meta
        return committed

    @staticmethod
    def _is_discoverable_span_pair(before: str, after: str) -> bool:
        """Keep agent-discovered edits short and lexical, not whole-sentence rewrites."""
        if not (1 <= len(before) <= 8 and 1 <= len(after) <= 8):
            return False
        return all(ch.isalnum() or ("\u4e00" <= ch <= "\u9fff") for ch in before + after)

    @staticmethod
    def _ensure_closed_set_hypothesis(turn: Turn, before: str, after: str) -> Hypothesis:
        """Open or extend a closed-set hypothesis from later evidence (no dataset labels needed)."""
        hypothesis = next((item for item in turn.hypotheses if item.span == before), None)
        candidates = [before, after]
        if hypothesis is None:
            hypothesis = Hypothesis(
                span=before,
                text_candidates=candidates,
                entity_candidate_ids=[],
                candidates=[
                    {"text": candidate, "score": 0.5, "supporting_evidence": [], "contradicting_evidence": []}
                    for candidate in candidates
                ],
                risk="medium",
                decision="WAIT",
                decision_rationale=["later_evidence_nomination"],
                evidence_packet={"asr_uncertainty": {"confidence": None, "nbest": []}, "suspicious_span": None},
            )
            turn.hypotheses.append(hypothesis)
            return hypothesis
        merged = list(dict.fromkeys([*hypothesis.text_candidates, before, after]))
        hypothesis.text_candidates = merged
        existing = {str(item.get("text")) for item in hypothesis.candidates if isinstance(item, dict)}
        for candidate in merged:
            if candidate not in existing:
                hypothesis.candidates.append(
                    {"text": candidate, "score": 1.0 / len(merged), "supporting_evidence": [], "contradicting_evidence": []}
                )
        return hypothesis

    @classmethod
    def _local_audio_window(cls, turn: Turn, span: str, *, pad_sec: float = 1.5) -> tuple[float, float] | None:
        start = turn.meta.get("start_sec")
        end = turn.meta.get("end_sec")
        if start is None or end is None:
            return None
        start_f, end_f = float(start), float(end)
        plain = cls._strip_time_prefix(turn.raw_text or turn.current_text)
        if not plain or span not in plain or end_f <= start_f:
            return start_f, end_f
        idx = plain.find(span)
        dur = end_f - start_f
        frac_s = idx / len(plain)
        frac_e = (idx + max(len(span), 1)) / len(plain)
        local_s = max(start_f, start_f + frac_s * dur - pad_sec)
        local_e = min(end_f, start_f + frac_e * dur + pad_sec)
        if local_e - local_s < 0.4:
            mid = 0.5 * (local_s + local_e)
            local_s, local_e = max(start_f, mid - 0.2), min(end_f, mid + 0.2)
        return local_s, local_e

    def _relisten_and_commit(
        self,
        session: Session,
        turn: Turn,
        hypothesis: Hypothesis,
        source_index: int,
        candidate: str,
        semantic_score: float,
        evidence: list[str],
        rationale: str,
        *,
        allow_llm_confirm: bool = False,
    ) -> dict[str, Any] | None:
        audio_path = str(turn.meta.get("audio_path") or "")
        window = self._local_audio_window(turn, hypothesis.span)
        # No audio binding at all: keep RELISTEN / no commit (tests + safety).
        if not audio_path or window is None:
            hypothesis.evidence_packet["audio_verification"] = {
                "ok": False,
                "error": "audio_unavailable",
            }
            return None
        start, end = window
        verifier = self.audio_verifier
        if verifier is None:
            from asr_agent.integrations.audio_verifier import verify_candidates
            verifier = verify_candidates
        verdict = verifier(audio_path=audio_path, start_sec=float(start), end_sec=float(end), candidates=hypothesis.text_candidates)
        if not verdict.get("ok"):
            hypothesis.evidence_packet["audio_verification"] = verdict
            # Audio stack failed, but later raw evidence is already validated and strong:
            # commit so real meetings are not stuck at revised_true=0 when Qwen verify flakes.
            if evidence and semantic_score >= 0.84:
                return self._commit_revision(
                    session,
                    turn,
                    hypothesis,
                    source_index,
                    profile=None,
                    candidate=candidate,
                    score=semantic_score,
                    evidence=[*evidence, f"audio-failed:{audio_path}:{start}-{end}"],
                    resolver="semantic-evidence-gate",
                    forced_action="REVISE_TEXT",
                    rationale=rationale or "strong later evidence; audio verifier unavailable",
                )
            return None
        scores = {str(key): float(value) for key, value in dict(verdict["scores"]).items()}
        ranked = sorted(scores, key=scores.get, reverse=True)
        strong = (
            ranked
            and ranked[0] == candidate
            and len(ranked) >= 2
            and scores[candidate] >= 0.7
            and scores[candidate] - scores[ranked[1]] >= 0.1
        )
        resolver = "audio-semantic-gate"
        audio_score = scores.get(candidate, 0.0)
        if not strong:
            # TAMA-style: when audio is top but margin/threshold is weak, ask LLM once.
            weak_top = ranked and ranked[0] == candidate and audio_score >= 0.55
            if allow_llm_confirm and weak_top and evidence:
                adjudicator = self.adjudicator
                if adjudicator is None:
                    from asr_agent.integrations.deepseek import adjudicate_conflict

                    adjudicator = adjudicate_conflict
                try:
                    decision = adjudicator(
                        hyp=hypothesis.span,
                        candidates=list(hypothesis.text_candidates),
                        evidence=list(evidence),
                        context=turn.current_text,
                        heuristic={"audio_scores": scores, "semantic_score": semantic_score},
                    )
                except Exception:
                    decision = {"action": "KEEP"}
                if str(decision.get("action") or "").upper() != "CORRECT" or str(decision.get("canonical") or "") != candidate:
                    hypothesis.evidence_packet["audio_verification"] = verdict
                    hypothesis.evidence_packet["llm_adjudication"] = decision
                    # Fall through to weak semantic confirm when audio still ranks candidate first.
                else:
                    hypothesis.evidence_packet["llm_adjudication"] = decision
                    resolver = "audio-llm-confirm"
                    strong = True
            if not strong and weak_top and evidence and semantic_score >= 0.84:
                resolver = "audio-semantic-weak-gate"
                strong = True
            if not strong:
                hypothesis.evidence_packet["audio_verification"] = verdict
                return None
        hypothesis.evidence_packet["audio_verification"] = verdict
        return self._commit_revision(
            session,
            turn,
            hypothesis,
            source_index,
            profile=None,
            candidate=candidate,
            score=(semantic_score + audio_score) / 2,
            evidence=[*evidence, f"audio:{audio_path}:{start}-{end}:{candidate}:{audio_score:.3f}"],
            resolver=resolver,
            forced_action="REVISE_TEXT",
            rationale=rationale,
        )

    @staticmethod
    def _recall_entity_fallbacks(session: Session, text: str) -> list[tuple[str, str, str]]:
        """Recall near-matching known entities as quarantined candidates, never as facts."""
        matches: list[tuple[str, str, str]] = []
        for profile in session.entities.values():
            for candidate in [profile.name, *profile.aliases]:
                if len(candidate) < 2:
                    continue
                for start in range(len(text) - len(candidate) + 1):
                    observed = text[start : start + len(candidate)]
                    suffix = 0
                    for left, right in zip(reversed(observed), reversed(candidate)):
                        if left != right:
                            break
                        suffix += 1
                    if observed != candidate and suffix >= len(candidate) - 1:
                        matches.append((observed, candidate, profile.entity_id))
        return matches

    def _reassess(self, session: Session, source_index: int, *, use_llm: bool = False) -> list[dict[str, Any]]:
        revisions: list[dict[str, Any]] = []
        for target_index, turn in enumerate(session.turns[:source_index]):
            evidence_text = " ".join(item.raw_text for item in session.turns[target_index + 1 : source_index + 1])
            for hypothesis in turn.hypotheses:
                if hypothesis.action != "DEFER":
                    continue
                rule_hit = self._rule_resolve(session, turn, hypothesis, evidence_text, source_index)
                if rule_hit:
                    revisions.append(rule_hit)
                elif use_llm:
                    llm_hit = self._llm_resolve(session, turn, hypothesis, evidence_text, source_index)
                    if llm_hit:
                        revisions.append(llm_hit)
        return revisions

    def _rule_resolve(
        self,
        session: Session,
        turn: Turn,
        hypothesis: Hypothesis,
        evidence_text: str,
        source_index: int,
    ) -> dict[str, Any] | None:
        scores: list[tuple[float, EntityProfile, list[str]]] = []
        for entity_id in hypothesis.entity_candidate_ids:
            profile = session.entities.get(entity_id)
            if not profile:
                continue
            matches = [value for value in profile.attributes.values() if value and value in evidence_text]
            if matches:
                scores.append((len(matches) / max(1, len(profile.attributes)), profile, matches))
        scores.sort(key=lambda item: item[0], reverse=True)
        if not scores or (len(scores) > 1 and scores[0][0] == scores[1][0]) or scores[0][0] < 0.5:
            return None
        score, profile, evidence = scores[0]
        return self._commit_revision(
            session,
            turn,
            hypothesis,
            source_index,
            profile=profile,
            candidate=next(
                (item for item in hypothesis.text_candidates if item == profile.name or item in profile.aliases),
                hypothesis.span,
            ),
            score=score,
            evidence=evidence,
            resolver="rule",
        )

    def _llm_resolve(self, session: Session, turn: Turn, hypothesis: Hypothesis, evidence_text: str, source_index: int) -> dict[str, Any] | None:
        scorer = self.evidence_scorer
        if scorer is None:
            from asr_agent.integrations.deepseek import score_evidence

            scorer = score_evidence
        candidates = list(dict.fromkeys(hypothesis.text_candidates or [hypothesis.span]))
        entities = [session.entities[item] for item in hypothesis.entity_candidate_ids if item in session.entities]
        result = scorer(
            span=hypothesis.span,
            text_candidates=candidates,
            entity_candidates=[asdict(item) for item in entities],
            prior_text=turn.current_text,
            evidence_text=evidence_text,
        )
        action = str(result.get("action") or "DEFER").upper()
        candidate = str(result.get("candidate") or hypothesis.span)
        entity_id = str(result.get("entity_id") or "")
        if action not in {"KEEP", "REVISE_TEXT", "REVISE_ENTITY", "DEFER", "CLARIFY"}:
            return None
        if action in {"KEEP", "DEFER", "CLARIFY"}:
            # A non-revision decision is provisional: a later turn may contain counterevidence.
            return None
        if candidate not in candidates or (entity_id and entity_id not in hypothesis.entity_candidate_ids):
            return None
        profile = session.entities.get(entity_id) if entity_id else None
        if action == "REVISE_ENTITY" and profile is None:
            return None
        target_index = next((index for index, item in enumerate(session.turns) if item.turn_id == turn.turn_id), -1)
        evidence = self._validate_later_evidence(session, target_index, source_index, result.get("evidence"))
        if not evidence:
            return None
        return self._commit_revision(
            session, turn, hypothesis, source_index, profile=profile, candidate=candidate,
            score=float(result.get("score") or 0.0), evidence=evidence,
            resolver="deepseek", forced_action=action, rationale=str(result.get("rationale") or ""),
        )

    @staticmethod
    def _validate_later_evidence(session: Session, target_index: int, source_index: int, items: Any) -> list[str]:
        indexes = {turn.turn_id: index for index, turn in enumerate(session.turns)}
        verified: list[str] = []
        for item in items or []:
            if not isinstance(item, dict):
                return []
            turn_id, quote = str(item.get("turn_id") or ""), str(item.get("quote") or "")
            evidence_index = indexes.get(turn_id, -1)
            if not quote or evidence_index <= target_index or evidence_index > source_index or quote not in session.turns[evidence_index].raw_text:
                return []
            verified.append(f"{turn_id}:{quote}")
        return verified

    def _commit_revision(
        self,
        session: Session,
        turn: Turn,
        hypothesis: Hypothesis,
        source_index: int,
        *,
        profile: EntityProfile | None,
        candidate: str,
        score: float,
        evidence: list[str],
        resolver: str,
        forced_action: str | None = None,
        rationale: str = "",
    ) -> dict[str, Any] | None:
        before = turn.current_text
        action = forced_action or ("REVISE_TEXT" if candidate != hypothesis.span else "REVISE_ENTITY")
        if action == "REVISE_TEXT" and candidate != hypothesis.span and hypothesis.span in turn.current_text:
            turn.current_text = turn.current_text.replace(hypothesis.span, candidate, 1)
        hypothesis.action = action
        hypothesis.decision = "COMMIT"
        hypothesis.decision_rationale = ["verified_later_raw_evidence"]
        if profile:
            hypothesis.entity_id = profile.entity_id
            session.verified_memory[profile.entity_id] = {"name": profile.name, "source_turn_id": session.turns[source_index].turn_id}
            session.quarantine_memory.pop(profile.entity_id, None)
        event = RevisionEvent(
            event_id=uuid.uuid4().hex,
            action=action,
            target_turn_id=turn.turn_id,
            source_turn_id=session.turns[source_index].turn_id,
            span=hypothesis.span,
            before_text=before,
            after_text=turn.current_text,
            entity_id=profile.entity_id if profile else None,
            score=round(score, 3),
            evidence=evidence,
            resolver=resolver,
            rationale=rationale,
            replacement=candidate if action == "REVISE_TEXT" else "",
        )
        session.revision_events.append(event)
        return asdict(event)

    def _load(self, session_id: str) -> Session:
        return self.repository.load(session_id)

    def _save(self, session: Session, expected_version: int) -> Session:
        return self.repository.commit(session, expected_version)
