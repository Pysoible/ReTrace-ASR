"""DeepSeek adapter for the structured Context Judge protocol."""
from __future__ import annotations

import json
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any

import requests

from asr_agent.context_judge import ContextJudgment, ExplicitSignalFallbackJudge, FocusProposal, normalize_judgment
from asr_agent.correction_candidates import (
    CorrectionCandidate,
    candidate_to_focus,
    deduplicate_candidates,
    focus_to_candidate,
)
from asr_agent.integrations.domainterms import domainterms_root, ensure_importable
from asr_agent.integrations.audio_verifier import retranscribe_window
from asr_agent.memory import MemoryPacket
from asr_agent.models import Session, Turn


_JUDGE_DISABLED_UNTIL = 0.0


_CONTEXT_JUDGE_SYSTEM = (
    "你是实时 ASR Context Judge。先判断最新观察与短期、长期 Memory 的关系，只能输出 "
    "CONSISTENT、NOVEL、CONFLICT、UNCERTAIN。不要预先分词，不要扫描所有二元词，不要发明专有名词。"
    "语言保持是硬约束：ASR 修订只能纠正同一种语言/文字系统中的听写错误，绝对不能翻译、意译或把英文改写成中文，"
    "也不能把中文改写成英文。若 span 与 proposed_text 的语言不同，即使语义等价或音频候选支持，也必须不输出该 focus，"
    "保留原文并判定为 CONSISTENT 或 UNCERTAIN；中英混合原文只允许保持已有语言集合，不得新增另一种语言。"
    "JSON 顶层必须包含 outcome 和 confidence 字段：outcome 为上述四种之一；"
    "若提出修订候选，可使用 focus 或顶层 candidates；candidates 每项必须包含 target_turn_id、span、"
    "candidate、evidence_turn_ids、rationale 和 source。历史中不存在正确写法时 source 必须为 semantic_open。"
    "confidence 为 0-1 的小数，表示你对这个判断的把握程度——"
    "CONSISTENT 时 confidence 通常应 >= 0.7（有明确一致证据则更高），UNCERTAIN 给 0.4-0.6，"
    "CONFLICT/NOVEL 时 confidence 表示证据强度（> 0.6 才建议修订）。不要省略 confidence，不要填 0。"
    "特别注意：如果最新 turn 的内容与上下文话题明显无关、语义突兀（例如一个口语化的闲聊对话中"
    "突然出现一个不成词的音译短语），这通常是 ASR 识别错误的信号，应判定为 UNCERTAIN 或 CONFLICT，"
    "并给出 focus 提出更符合上下文的候选转写，以便系统重听音频窗口核实。"
    "利用 user 消息里的 domain_entities（对话中已经提到过的专有名词/领域词：人名、地名、品牌、"
    "产品名、专业术语等）：如果 current_turn 中有某个词与这些已知实体的读音相近、但在语义上不搭"
    "（例如一个词读起来像某个已知专有名词，但字形是另一个不成词的组合），"
    "则很可能是 ASR 听错了该专有名词。此时应判定为 UNCERTAIN 或 CONFLICT，并在 focus 的 "
    "proposed_text / alternatives 中给出你认为正确的实体名（从 domain_entities 中选最吻合的，"
    "span 用当前文本中该疑似错词的原文）。"
    "即便 domain_entities 里没有现成候选，只要当前对话语境明确属于某个具体领域（例如在讨论某个"
    "游戏的角色、某个地名、某个品牌或人名），且 current_turn 里出现了一个不成词、读起来像该领域"
    "内某个公认专有名词的音译怪词，你可以依据你自己的世界知识提出那个读音相近的正确专有名词作为 "
    "proposed_text，并在 rationale 中说明依据。但前提是：该词必须是该领域内广泛公认的专有名词，"
    "且与 span 读音相近；不要把两个普通的日常词互相替换（普通词的近音差异通常不是 ASR 错误）。"
    "recent_turns 是最近的历史 turn（每个都带 turn_id 与文本）。你不仅要检查 current_turn，"
    "还必须做一次历史可疑片段审计：逐个快速检查最近历史 turn 中语义突兀、疑似漏词、明显不成词、"
    "或与后续事实不一致的短语。即使 current_turn 本身 CONSISTENT，只要历史 turn 存在一个有具体候选"
    "和证据支持的疑点，也必须输出 outcome=UNCERTAIN 或 CONFLICT，并在 focus 中回溯该历史 turn；"
    "不要因为当前 turn 语义通顺就跳过历史审计。"
    "如果没有具体可验证的历史疑点，才输出 CONSISTENT。不要为了凑 focus 把普通通顺词当成错误。"
    "如果一个历史错误只是语义通顺、没有后文证据或领域候选，不要臆造替换；保留为 CONSISTENT，"
    "因为系统的历史修正必须由后文证据或音频支持触发。"
    "如果历史 turn"
    "的文本里出现了一个读音与该实体相近、但字形不同且语义突兀的疑似错词，则那个历史 turn 当年很可能"
    "被 ASR 听错了。此时应输出 outcome=CONFLICT，并在 focus 中：target_turn_id 指向那个历史 turn 的 "
    "id，span 用该历史 turn 文本中原样存在的疑似错词，proposed_text 用你认为正确的实体名，"
    "evidence_turn_ids 填当前 turn 的 id（或其它支持该判断的 turn）。"
    "不要因为这种历史回溯而改动 current_turn 本身的文本；仅在确有把握、且该疑似错词确实读起来像某个"
    "公认专有名词时才做历史回溯，不要把历史 turn 里的普通日常词也当成错误替换。"
    "当 user 消息里的 session_complete=true 时，整条 session 已经完整到达；此时必须把全 session 信息"
    "视为后文证据，并优先审计更早的历史 turn。若历史 turn 有具体的语义冲突、领域实体候选或声学"
    "线索，focus 的 target_turn_id 必须指向更早的 turn，evidence_turn_ids 必须包含当前或更晚的证据 turn；"
    "不要把同一疑点重新指向当前 turn。"
    "user 消息里的 homophone_candidates 只是从历史 ASR 文本提取的同音候选，不代表正确答案，也不按出现次数决定。"
    "你必须先判断 current_turn 是否符合完整上下文；只有当前词语语义突兀、而某个同音候选明显更符合当前主题和句法时，"
    "才输出 CONFLICT 或 UNCERTAIN 以及 focus。候选仍必须交给后续局部音频验证，不能仅凭同音或历史出现直接修改。"
    "对于普通口语短语（例如‘过冬’、‘共都’这类局部词），优先依据当前句子的直接语义和相邻词判断，"
    "不要强行替换成领域专有名词；只有候选在完整句子中更自然且音频验证支持时才提出该候选。"
    "user 消息里的 acoustic_disagreement 是声学层面的独立信号：它列出了第二个独立 ASR 与首遍转写"
    "不一致的片段。这些位置很可能真的听错了（两个声学模型在同一处都拿不准），你应当优先检查这些片段，"
    "但不要盲目相信——仍需结合语义判断它们是否真的读起来像某个已知专有名词，避免把普通词的近音差异"
    "当成 ASR 错误。"
    "user 消息里的 low_conf_chars 是另一个声学层面的独立信号：它列出了第二个独立 ASR 自己在解码时"
    "置信度偏低的具体字符（带 conf 值）。这些字符说明声学模型在该位置上拿不准，是字级声学置信度的"
    "直接体现。你应当把它们视为'此处可能听错'的线索，与语义判断结合使用：如果某个低置信度字与上下文"
    "不搭、或读起来像某个已知专有名词/领域词，则更可能是 ASR 错误，可在 focus 中针对该字所在的最短"
    "词提出候选。反之，如果低置信度字在语义上完全通顺，则不必强行修订。"
    "user 消息里的 overlap_candidates 来自多通道重叠语音经 GSS 分离后、由同一个首遍 ASR 重新解码得到的"
    "短等长替换候选。它们是独立声学视图，不是正确答案。只在 candidate 明显修正 span 的错字或错词、"
    "符合当前句和相邻上下文时，才为原始 target_turn_id 输出最短 focus；不得接受 GSS 中的新增、删除、"
    "语气词堆叠、翻译或整句改写。不确定时保留原文。"
    "user 消息里的 canonical_entities 是 agent 已经通过上下文与定向音频验证确认过的实体规范写法，"
    "每项有 canonical、aliases 与证据 turn。它们用于保证一个 session 内同一实体只有一种写法：当"
    "current_turn 或 recent_turns 中出现 alias 或另一种近音音译，而语境指向同一实体时，必须提出该历史 "
    "turn 的 focus，把 canonical 作为 proposed_text，并把原写法与 canonical 一起放入 alternatives；"
    "不要把 alias 当作已经正确，也不要只在 rationale 中说明。必须让后续定向音频验证决定，不能仅凭 memory 直接改字。"
    "如果不同写法可能确实指向不同实体，或没有明确语境与声学疑点，则保持 CONSISTENT/UNCERTAIN，不要强行统一。"
    "用 beliefs 输出有原文 Turn 证据的结构化新事实：subject、predicate、value、aliases、confidence、"
    "valid_from、valid_to、evidence_turn_ids。"
    "如果怀疑 span 是 ASR 幻觉、人工语音中没有这段内容，应使用 operation=DELETE、proposed_text 为空；"
    "删除必须有明确的局部音频或覆盖证据，不能仅因语义不喜欢该短语就删除。"
    "仅在 CONFLICT 或 UNCERTAIN 且存在具体证据时给出 focus；每个 focus 必须包含 "
    "target_turn_id、目标当前文本中原样存在的最短 span、operation、proposed_text、alternatives（或 closed_set，"
    "必须是包含 span 与 proposed_text 的候选列表，字段名用 alternatives 即可）、"
    "evidence_turn_ids、rationale 和 relationship。relationship 只能是 MUTUALLY_EXCLUSIVE、COEXIST、"
    "TEMPORAL_CHANGE。只输出 JSON 对象。"
    "当 user 消息包含 analysis_window 时，必须逐条审计该窗口内的所有 turn，而不是只看 current_turn；"
    "focus.target_turn_id 必须保持为被修改 turn 的真实 ID，不能统一改成窗口最后一个 turn。"
)


def _sync_model_env() -> None:
    if not os.getenv("JUDGE_MODEL") and os.getenv("DEEPSEEK_MODEL"):
        os.environ["JUDGE_MODEL"] = os.environ["DEEPSEEK_MODEL"]


def _api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("LLM_WS_API_KEY") or ""


def deepseek_status() -> dict[str, Any]:
    root = domainterms_root()
    model = os.environ.get("JUDGE_MODEL") or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    transport = os.environ.get("LLM_TRANSPORT", "http").strip().lower() or "http"
    error: str | None = None
    if transport == "websocket":
        try:
            ensure_importable()
            _sync_model_env()
            from domain_terms import config as dt_config  # noqa: WPS433

            model = getattr(dt_config, "JUDGE_MODEL", model) or model
            transport = getattr(dt_config, "LLM_TRANSPORT", transport) or transport
        except Exception as exc:
            error = str(exc)
    ready = bool(_api_key()) and error is None
    if not _api_key():
        error = "DEEPSEEK_API_KEY / LLM_API_KEY 未配置"
    return {
        "ready": ready,
        "domainterms_root": str(root),
        "domainterms_exists": root.exists(),
        "transport": transport or None,
        "model": model or None,
        "error": error,
    }


def context_judge_identity() -> dict[str, str]:
    status = deepseek_status()
    return {
        "backend": "deepseek-api",
        "model": str(status.get("model") or "unknown"),
        "role": "context_judge",
    }


def select_closed_set_replacement(
    sentence: str,
    candidate: dict[str, Any],
    *,
    minimum_confidence: float = 0.80,
) -> dict[str, Any]:
    """Use DeepSeek only to keep the baseline or choose one acoustic candidate.

    The model sees neither reference text nor an open generation instruction.
    Any response outside the two allowed decisions is converted to KEEP_BASELINE.
    Selection only routes the candidate to acoustic verification; it never
    authorizes a transcript edit by itself.
    """
    candidate_id = str(candidate.get("candidate_id") or "")
    baseline = str(candidate.get("source_text") or "")
    replacement = str(candidate.get("candidate_text") or "")
    allowed = ["KEEP_BASELINE", candidate_id]
    base_result = {
        "decision": "KEEP_BASELINE",
        "selected": False,
        "eligible_for_acoustic_verification": False,
        "confidence": 0.0,
        "rationale": "",
        "candidate_id": candidate_id,
        "judge": context_judge_identity(),
    }
    if not _api_key():
        return {**base_result, "reason": "deepseek_unavailable"}
    if not candidate_id or not baseline or not replacement:
        return {**base_result, "reason": "invalid_candidate"}
    start = int(candidate.get("anchor_start", -1))
    end = int(candidate.get("anchor_end", -1))
    if start < 0 or end <= start or sentence[start:end] != baseline:
        return {**base_result, "reason": "source_offset_mismatch"}
    candidate_sentence = sentence[:start] + replacement + sentence[end:]
    payload = {
        "sentence": sentence,
        "sentence_options": [
            {"decision": "KEEP_BASELINE", "text": sentence},
            {"decision": candidate_id, "text": candidate_sentence},
        ],
        "acoustic_candidate": {
            "candidate_id": candidate_id,
            "baseline_span": baseline,
            "candidate_span": replacement,
            "supporting_view_count": int(candidate.get("support") or 0),
            "supporting_views": list(candidate.get("sources") or []),
        },
        "allowed_decisions": allowed,
        "instruction": (
            "只做闭集语义选择。直接比较 sentence_options 中两条完整句子，重点检查局部句法、"
            "常见搭配、前后因果关系和口语自然度；不要仅因为原句勉强可解释就忽略明显更自然的候选。"
            "只有候选句明显更合理才选择它，否则保留原句。声学视图只是候选来源，不代表答案。"
            "只能返回 JSON：decision 必须严格等于 allowed_decisions 中一个值；"
            "confidence 为 0 到 1；rationale 简短说明语义理由。不得生成新转写或第三个候选。"
            "不确定时必须选择 KEEP_BASELINE。"
        ),
    }
    try:
        result = _chat_json(
            [
                {
                    "role": "system",
                    "content": "你是 ASR 闭集语义门，只能保留原文或选择给定声学候选。",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=int(os.getenv("ASR_CLOSED_SET_JUDGE_MAX_TOKENS", "180")),
        )
    except Exception as exc:  # noqa: BLE001
        return {**base_result, "reason": "judge_error", "error_type": type(exc).__name__}
    if not isinstance(result, dict):
        return {**base_result, "reason": "malformed_response"}
    decision = str(result.get("decision") or "")
    confidence = max(0.0, min(1.0, float(result.get("confidence") or 0.0)))
    rationale = str(result.get("rationale") or "")[:300]
    if decision not in allowed:
        return {
            **base_result,
            "confidence": confidence,
            "rationale": rationale,
            "reason": "out_of_closed_set_response",
        }
    selected = decision == candidate_id and confidence >= minimum_confidence
    return {
        **base_result,
        "decision": candidate_id if selected else "KEEP_BASELINE",
        "selected": selected,
        "eligible_for_acoustic_verification": selected,
        "confidence": confidence,
        "rationale": rationale,
        "reason": "selected" if selected else (
            "below_confidence_threshold" if decision == candidate_id else "kept_baseline"
        ),
        "minimum_confidence": minimum_confidence,
    }


def _chat_json(messages: list[dict[str, str]], *, max_tokens: int = 800) -> Any:
    transport = os.getenv("LLM_TRANSPORT", "http").strip().lower() or "http"
    if transport == "websocket":
        ensure_importable()
        _sync_model_env()
        from domain_terms import llm_client  # noqa: WPS433

        return llm_client.chat_json(messages, max_tokens=max_tokens, temperature=0.0)

    api_key = _api_key()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY / LLM_API_KEY 未配置")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    response = requests.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        timeout=float(os.getenv("DEEPSEEK_TIMEOUT", "120")),
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    return _extract_json(content) if isinstance(content, str) else content


def _domain_entities(memory: MemoryPacket) -> list[str]:
    """Collect proper nouns / domain terms remembered so far.

    Working & long-term beliefs carry the entities the conversation has already
    discussed (proper nouns, domain terms, …). Surfacing these to the judge lets
    it correct ASR mis-hearings by preferring the remembered entity over a
    garbled transcription that sounds similar.
    """
    entities: list[str] = []
    seen: set[str] = set()
    for belief in [*memory.working_beliefs, *memory.long_term_beliefs]:
        predicate = (belief.predicate or "").lower()
        if predicate == "canonical_text":
            continue
        value = (belief.value or "").strip()
        aliases = [str(item).strip() for item in (belief.aliases or []) if str(item).strip()]
        candidates = [value, *aliases]
        for candidate in candidates:
            if not candidate:
                continue
            # Keep short proper nouns (2-6 chars) that are unlikely to be noise.
            if 2 <= len(candidate) <= 6 and candidate not in seen:
                seen.add(candidate)
                entities.append(candidate)
    return entities


def _canonical_entities(memory: MemoryPacket) -> list[dict[str, object]]:
    """Return agent-verified entity identities, including observed aliases.

    Unlike ``domain_entities``, these have an explicit audio-verification
    provenance and must be treated as the preferred spelling when an alias or
    acoustic alternative appears later in the same session.
    """
    return [
        {
            "canonical": belief.value,
            "aliases": list(belief.aliases),
            "confidence": belief.confidence,
            "evidence_turn_ids": list(belief.source_turn_ids),
        }
        for belief in memory.canonical_entities
    ]


def _homophone_candidates(
    session: Session,
    current_turn: Turn,
    limit: int = 64,
    *,
    turn_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Provide same-pronunciation history candidates to the semantic judge."""
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return []
    current = re.sub(r"^\[\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?\]\s*", "", current_turn.current_text or current_turn.raw_text)
    historical_turns = [
        (
            turn.turn_id,
            re.sub(r"^\[\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?\]\s*", "", turn.current_text or turn.raw_text),
        )
        for turn in session.turns
        if turn.turn_id != current_turn.turn_id
        and (turn_ids is None or turn.turn_id in turn_ids)
    ]
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for width in range(2, 5):
        for index in range(len(current) - width + 1):
            span = current[index:index + width]
            if not all("\u4e00" <= char <= "\u9fff" for char in span):
                continue
            pronunciation = tuple(lazy_pinyin(span))
            for turn_id, historical in historical_turns:
                for candidate_index in range(len(historical) - width + 1):
                    candidate = historical[candidate_index:candidate_index + width]
                    if candidate != span and all("\u4e00" <= char <= "\u9fff" for char in candidate) and tuple(lazy_pinyin(candidate)) == pronunciation:
                        key = (span, candidate)
                        item = candidates.setdefault(
                            key,
                            {"span": span, "candidate": candidate, "evidence_turn_ids": []},
                        )
                        if turn_id not in item["evidence_turn_ids"]:
                            item["evidence_turn_ids"].append(turn_id)
    return list(candidates.values())[:limit]


def _window_homophone_candidates(
    session: Session,
    turn_ids: set[str],
    limit: int = 64,
) -> list[dict[str, Any]]:
    """Find spelling conflicts between every turn in one bounded window."""
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        return []
    selected = [
        (
            turn.turn_id,
            re.sub(r"^\[\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?\]\s*", "", turn.current_text or turn.raw_text),
        )
        for turn in session.turns
        if turn.turn_id in turn_ids
    ]
    pronunciation_index: dict[tuple[int, tuple[str, ...]], list[tuple[str, str]]] = {}
    for turn_id, text in selected:
        for width in range(2, 5):
            for start in range(len(text) - width + 1):
                span = text[start : start + width]
                if not all("\u4e00" <= char <= "\u9fff" for char in span):
                    continue
                key = (width, tuple(lazy_pinyin(span)))
                bucket = pronunciation_index.setdefault(key, [])
                if (turn_id, span) not in bucket:
                    bucket.append((turn_id, span))

    candidates: dict[tuple[str, str, str], dict[str, Any]] = {}
    for variants in pronunciation_index.values():
        if len({span for _, span in variants}) < 2:
            continue
        for target_turn_id, span in variants:
            for evidence_turn_id, candidate in variants:
                if target_turn_id == evidence_turn_id or span == candidate:
                    continue
                key = (target_turn_id, span, candidate)
                item = candidates.setdefault(
                    key,
                    {
                        "target_turn_id": target_turn_id,
                        "span": span,
                        "candidate": candidate,
                        "evidence_turn_ids": [],
                    },
                )
                if evidence_turn_id not in item["evidence_turn_ids"]:
                    item["evidence_turn_ids"].append(evidence_turn_id)
                if len(candidates) >= limit:
                    return list(candidates.values())
    return list(candidates.values())


def _resolve_homophone_focus(
    session: Session,
    current_turn: Turn,
    candidates: list[dict[str, str]],
) -> Any:
    """Ask the semantic judge to choose a concrete candidate after a conflict.

    The first judgment is intentionally broad. When it detects a conflict but
    omits focus, this narrow follow-up prevents a valid conflict from becoming a
    no-op while keeping candidate selection semantic rather than frequency-based.
    """
    if not candidates or not _api_key():
        return None
    payload = {
        "current_turn": {
            "turn_id": current_turn.turn_id,
            "text": current_turn.current_text or current_turn.raw_text,
        },
        "homophone_candidates": candidates,
        "instruction": (
            "当前文本与上下文已判定存在冲突。请逐个判断候选是否符合完整上下文和句法。"
            "如果某个候选明显更合理，只能从 homophone_candidates 中选一个，输出合法 JSON："
            "{\"outcome\":\"CONFLICT\",\"confidence\":0-1,\"focus\":[{"
            "\"target_turn_id\":\"当前 turn_id\",\"span\":\"原文 span\","
            "\"proposed_text\":\"候选 candidate\",\"alternatives\":[\"原文 span\",\"候选 candidate\"],"
            "\"evidence_turn_ids\":[\"支持候选的历史 turn_id\"],"
            "\"relationship\":\"MUTUALLY_EXCLUSIVE\",\"rationale\":\"语境理由\"}]}。"
            "如果没有候选明显符合，focus 必须为空。不要使用候选出现次数作为理由。"
        ),
    }
    try:
        result = _chat_json([
            {"role": "system", "content": _CONTEXT_JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], max_tokens=int(os.getenv("ASR_HOMOPHONE_JUDGE_MAX_TOKENS", "500")))
        if isinstance(result, str):
            result = _extract_json(result)
        if isinstance(result, dict):
            _fill_judgment_confidence(result)
            return normalize_judgment(result, session)
    except Exception:
        return None
    return None


def _session_history_digest(session: Session, *, chars_per_turn: int = 120) -> list[dict[str, str]]:
    """Expose the whole session cheaply enough for long-audio auditing."""
    digest: list[dict[str, str]] = []
    for index, turn in enumerate(session.turns):
        text = re.sub(r"^\[[0-9.]+[-,~][0-9.]+\]\s*", "", turn.current_text or turn.raw_text)
        digest.append({
            "turn_id": turn.turn_id,
            "index": str(index),
            "text": text[:chars_per_turn],
        })
    return digest


def judge_context(*, session: Session, current_turn: Turn, memory: MemoryPacket) -> ContextJudgment:
    """Judge context semantically; malformed or unavailable models safely defer.

    ASR mis-hearing correction is left entirely to the LLM judge: it receives
    the remembered domain entities plus a prompt inviting it to use world
    knowledge, so it can do a *semantic* check (does the garbled word plausibly
    sound like a known proper noun in this domain?) rather than a naive pinyin
    match that keeps flagging normal words as hero names.
    """
    if not _api_key():
        return ExplicitSignalFallbackJudge()(session=session, current_turn=current_turn, memory=memory)
    global _JUDGE_DISABLED_UNTIL
    if time.monotonic() < _JUDGE_DISABLED_UNTIL:
        return ExplicitSignalFallbackJudge()(session=session, current_turn=current_turn, memory=memory)
    uncertainty = current_turn.meta.get("uncertainty") or {}
    acoustic_disagreement = uncertainty.get("acoustic_disagreement")
    low_conf_chars = uncertainty.get("low_conf_chars")
    session_complete = bool((current_turn.meta or {}).get("session_complete"))
    if session_complete:
        recent_turns = memory.recent_turns[-4:]
        history_chars = int(os.getenv("ASR_FINAL_AUDIT_DIGEST_CHARS", "180"))
        current_payload = {
            "turn_id": current_turn.turn_id,
            "text": re.sub(r"^\[[0-9.]+[-,~][0-9.]+\]\s*", "", current_turn.current_text or current_turn.raw_text),
        }
    else:
        recent_turns = memory.recent_turns
        history_chars = int(os.getenv("ASR_HISTORY_DIGEST_CHARS", "120"))
        current_payload = current_turn.as_dict()
    window_ids = set(
        (current_turn.meta or {}).get("analysis_window_turn_ids")
        or [current_turn.turn_id]
    )
    windowed_analysis = bool((current_turn.meta or {}).get("analysis_window_turn_ids"))
    if windowed_analysis:
        overlap_candidates = []
        for turn in session.turns:
            if turn.turn_id not in window_ids:
                continue
            candidates = (((turn.meta or {}).get("uncertainty") or {}).get("overlap") or {}).get("substitution_candidates") or []
            overlap_candidates.extend(
                {**item, "target_turn_id": turn.turn_id}
                for item in candidates
                if isinstance(item, dict)
            )
    else:
        overlap_candidates = (uncertainty.get("overlap") or {}).get("substitution_candidates") or []
    analysis_window = [
        {"turn_id": turn.turn_id, "text": turn.current_text or turn.raw_text}
        for turn in session.turns
        if turn.turn_id in window_ids
    ]
    payload = {
        "current_turn": current_payload,
        "analysis_window": analysis_window,
        "recent_turns": [] if windowed_analysis else [
            {"turn_id": turn.turn_id, "text": re.sub(r"^\[[0-9.]+[-,~][0-9.]+\]\s*", "", turn.current_text or turn.raw_text)[:240]}
            for turn in recent_turns
        ],
        "session_history_digest": (
            _session_history_digest(
                session,
                chars_per_turn=int(os.getenv("ASR_FINAL_AUDIT_DIGEST_CHARS", "40")),
            )
            if windowed_analysis and session_complete
            else []
            if windowed_analysis
            else _session_history_digest(session, chars_per_turn=history_chars)
        ),
        "dependent_turns": [] if windowed_analysis or session_complete else [turn.as_dict() for turn in memory.dependent_turns],
        "working_beliefs": [item.as_dict() for item in memory.working_beliefs[-12:]],
        "open_hypotheses": [] if session_complete else [item.as_dict() for item in memory.open_hypotheses],
        "long_term_beliefs": [item.as_dict() for item in memory.long_term_beliefs[-12:]],
        "domain_entities": _domain_entities(memory),
        "canonical_entities": _canonical_entities(memory),
        "homophone_candidates": (
            _window_homophone_candidates(session, window_ids)
            if windowed_analysis
            else _homophone_candidates(session, current_turn)
        ),
        # Acoustic uncertainty: spans where a second independent ASR disagreed
        # with the first pass. These are strong candidates for mis-hearings.
        "acoustic_disagreement": acoustic_disagreement,
        # Char-level acoustic confidence: characters the second ASR itself was
        # unsure about, with their confidence values.
        "low_conf_chars": low_conf_chars,
        # GSS is candidate evidence only.  Insertions/deletions were already
        # removed deterministically; the semantic judge must still validate
        # each remaining short substitution against its context.
        "overlap_candidates": overlap_candidates,
        "session_complete": session_complete,
    }
    messages = [
        {"role": "system", "content": _CONTEXT_JUDGE_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    try:
        result = _chat_json(messages, max_tokens=int(os.getenv("ASR_JUDGE_MAX_TOKENS", "1200")))
        if isinstance(result, str):
            result = _extract_json(result)
        if not isinstance(result, dict):
            raise ValueError("context judge must return a JSON object")
        _fill_judgment_confidence(result)
        judgment = normalize_judgment(result, session)
        normalized_candidates = [focus_to_candidate(focus, session) for focus in judgment.focus]
        open_candidates = result.get("candidates") or result.get("candidate_pool") or []
        for candidate in open_candidates if isinstance(open_candidates, list) else []:
            if not isinstance(candidate, dict):
                continue
            span = str(candidate.get("span") or candidate.get("source_span") or "").strip()
            proposed = str(candidate.get("candidate") or candidate.get("proposed_text") or "").strip()
            operation = str(candidate.get("operation") or "REPLACE").upper()
            target_turn_id = str(candidate.get("target_turn_id") or current_turn.turn_id)
            evidence_turn_ids = [
                str(turn_id)
                for turn_id in (candidate.get("evidence_turn_ids") or [target_turn_id])
                if str(turn_id)
            ]
            if not span or (operation != "DELETE" and not proposed) or not any(
                turn.turn_id == target_turn_id
                and (span in turn.raw_text or span in turn.current_text)
                for turn in session.turns
            ):
                continue
            normalized_candidates.append(CorrectionCandidate(
                target_turn_id=target_turn_id,
                span=span,
                candidate=proposed,
                source=str(candidate.get("source") or "semantic_open"),
                evidence_turn_ids=evidence_turn_ids,
                rationale=str(candidate.get("rationale") or "semantic-open candidate"),
                semantic_confidence=judgment.confidence,
                relationship=str(candidate.get("relationship") or "MUTUALLY_EXCLUSIVE"),
                operation=operation,
                alternatives=[
                    str(item)
                    for item in (candidate.get("alternatives") or [])
                    if str(item)
                ],
            ))
        judgment.focus = [candidate_to_focus(item) for item in deduplicate_candidates(normalized_candidates)]
        if judgment.outcome in {"CONFLICT", "UNCERTAIN"} and not judgment.focus:
            homophone_candidates = payload.get("homophone_candidates") or []
            focused = _resolve_homophone_focus(
                session,
                current_turn,
                homophone_candidates,
            )
            if focused is not None and focused.focus:
                return focused
            if judgment.outcome == "CONFLICT" and len(homophone_candidates) == 1:
                candidate = homophone_candidates[0]
                return ContextJudgment(
                    "CONFLICT",
                    confidence=judgment.confidence,
                    rationale="context conflict with one same-pronunciation candidate; local audio verification required",
                    focus=[FocusProposal(
                        target_turn_id=current_turn.turn_id,
                        span=candidate["span"],
                        proposed_text=candidate["candidate"],
                        alternatives=[candidate["span"], candidate["candidate"]],
                        evidence_turn_ids=[
                            turn_id for turn_id in candidate.get("evidence_turn_ids", [])
                            if any(turn.turn_id == turn_id for turn in session.turns)
                        ] or [current_turn.turn_id],
                        rationale="the only same-pronunciation candidate must be checked against local audio",
                        source="history_homophone",
                    )],
                )
        return judgment
    except Exception as exc:
        # A transport/model outage is not fixed by sending the same large
        # request again. Avoid doubling latency for every chunk in a long file.
        error_text = str(exc)
        if (
            type(exc).__name__ == "LLMUnavailable"
            or "LLM websocket call failed" in error_text
            or "WS error" in error_text
        ):
            _JUDGE_DISABLED_UNTIL = time.monotonic() + float(os.getenv("ASR_JUDGE_FAILURE_COOLDOWN_SEC", "120"))
            return ExplicitSignalFallbackJudge()(session=session, current_turn=current_turn, memory=memory)
        try:
            retry_messages = [
                {"role": "system", "content": _CONTEXT_JUDGE_SYSTEM + f"上一次输出无法通过协议校验：{exc}。请修复该问题：只输出一个合法 JSON 对象，不要 Markdown、解释或前后缀；如果给出 focus，outcome 必须使用 UNCERTAIN 或 CONFLICT，字段必须包含 target_turn_id、span、operation、proposed_text、alternatives、evidence_turn_ids。DELETE 操作的 proposed_text 必须为空。"},
                messages[1],
            ]
            result = _chat_json(retry_messages, max_tokens=int(os.getenv("ASR_JUDGE_MAX_TOKENS", "1200")))
            if isinstance(result, str):
                result = _extract_json(result)
            if not isinstance(result, dict):
                raise ValueError("context judge retry must return a JSON object")
            _fill_judgment_confidence(result)
            return normalize_judgment(result, session)
        except Exception as retry_exc:
            fallback = ExplicitSignalFallbackJudge()(session=session, current_turn=current_turn, memory=memory)
            if fallback.focus:
                fallback.rationale = f"context judge protocol failed; using explicit ASR candidates: {retry_exc}"
                return fallback
            audio_fallback = _audio_diff_fallback(session, current_turn, memory)
            if audio_fallback.focus:
                return audio_fallback
            return ContextJudgment("UNCERTAIN", rationale=f"context judge unavailable: {exc}; retry failed: {retry_exc}")


def _audio_diff_fallback(session: Session, current_turn: Turn, memory: MemoryPacket) -> ContextJudgment:
    """Create grounded candidates from an independent re-ASR after protocol failure."""
    del session
    meta = current_turn.meta or {}
    audio_path = meta.get("audio_path")
    start_sec, end_sec = meta.get("start_sec"), meta.get("end_sec")
    if not audio_path or start_sec is None or end_sec is None:
        return ContextJudgment("UNCERTAIN", rationale="malformed focus and historical audio unavailable")
    try:
        result = retranscribe_window(
            str(audio_path),
            float(start_sec),
            float(end_sec),
            domain_hints=[item.value for item in memory.long_term_beliefs[-12:]],
        )
    except Exception as exc:
        return ContextJudgment("UNCERTAIN", rationale=f"malformed focus audio fallback failed: {exc}")
    if not isinstance(result, dict) or not result.get("ok"):
        return ContextJudgment("UNCERTAIN", rationale="malformed focus audio fallback returned no text")
    raw = re.sub(r"^\[[0-9.]+[-,~][0-9.]+\]\s*", "", current_turn.current_text or current_turn.raw_text)
    independent = str(result.get("text") or "").strip()
    focus: list[Any] = []
    matcher = SequenceMatcher(None, raw, independent, autojunk=False)
    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag != "replace":
            continue
        span = raw[left_start:left_end].strip()
        proposed = independent[right_start:right_end].strip()
        if (
            not span
            or not proposed
            or span == proposed
            or len(span) > 12
            or len(proposed) > 12
            or not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", span)
            or not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", proposed)
        ):
            continue
        focus.append({
            "target_turn_id": current_turn.turn_id,
            "span": span,
            "proposed_text": proposed,
            "alternatives": [span, proposed],
            "evidence_turn_ids": [current_turn.turn_id],
            "rationale": "independent audio re-transcription differs at this span",
        })
    if not focus:
        return ContextJudgment("UNCERTAIN", rationale="malformed focus audio fallback found no bounded replacement")
    return normalize_judgment(
        {"outcome": "UNCERTAIN", "confidence": 0.55, "focus": focus},
        current_turn_session := _session_with_turn(current_turn),
    )


def _session_with_turn(current_turn: Turn) -> Session:
    """Build a minimal validation session for the fallback's local focus."""
    return Session("audio-fallback", turns=[current_turn])


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _fill_judgment_confidence(payload: dict[str, Any]) -> None:
    """Default the top-level judgment confidence when the model omits it.

    The prompt asks for a 0-1 confidence, but models occasionally still drop the
    field (or return 0). Fall back to sensible per-outcome defaults so the
    frontend never shows a blanket 0% for every decision.
    """
    try:
        current = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        current = 0.0
    defaults = {
        "CONSISTENT": 0.8,
        "NOVEL": 0.75,
        "CONFLICT": 0.85,
        "UNCERTAIN": 0.5,
    }
    outcome = str(payload.get("outcome") or payload.get("label") or "").upper()
    if current <= 0.0 and outcome in defaults:
        payload["confidence"] = defaults[outcome]
