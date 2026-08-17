"""DeepSeek adapter for the structured Context Judge protocol."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from asr_agent.context_judge import ContextJudgment, ExplicitSignalFallbackJudge, normalize_judgment
from asr_agent.integrations.domainterms import domainterms_root, ensure_importable
from asr_agent.memory import MemoryPacket
from asr_agent.models import Session, Turn


_CONTEXT_JUDGE_SYSTEM = (
    "你是实时 ASR Context Judge。先判断最新观察与短期、长期 Memory 的关系，只能输出 "
    "CONSISTENT、NOVEL、CONFLICT、UNCERTAIN。不要预先分词，不要扫描所有二元词，不要发明专有名词。"
    "JSON 顶层必须包含 outcome 和 confidence 字段：outcome 为上述四种之一；"
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
    "还要回看 recent_turns：如果 domain_entities（或你的世界知识）里有某个专有名词，而某个历史 turn"
    "的文本里出现了一个读音与该实体相近、但字形不同且语义突兀的疑似错词，则那个历史 turn 当年很可能"
    "被 ASR 听错了。此时应输出 outcome=CONFLICT，并在 focus 中：target_turn_id 指向那个历史 turn 的 "
    "id，span 用该历史 turn 文本中原样存在的疑似错词，proposed_text 用你认为正确的实体名，"
    "evidence_turn_ids 填当前 turn 的 id（或其它支持该判断的 turn）。"
    "不要因为这种历史回溯而改动 current_turn 本身的文本；仅在确有把握、且该疑似错词确实读起来像某个"
    "公认专有名词时才做历史回溯，不要把历史 turn 里的普通日常词也当成错误替换。"
    "user 消息里的 acoustic_disagreement 是声学层面的独立信号：它列出了第二个独立 ASR 与首遍转写"
    "不一致的片段。这些位置很可能真的听错了（两个声学模型在同一处都拿不准），你应当优先检查这些片段，"
    "但不要盲目相信——仍需结合语义判断它们是否真的读起来像某个已知专有名词，避免把普通词的近音差异"
    "当成 ASR 错误。"
    "user 消息里的 low_conf_chars 是另一个声学层面的独立信号：它列出了第二个独立 ASR 自己在解码时"
    "置信度偏低的具体字符（带 conf 值）。这些字符说明声学模型在该位置上拿不准，是字级声学置信度的"
    "直接体现。你应当把它们视为'此处可能听错'的线索，与语义判断结合使用：如果某个低置信度字与上下文"
    "不搭、或读起来像某个已知专有名词/领域词，则更可能是 ASR 错误，可在 focus 中针对该字所在的最短"
    "词提出候选。反之，如果低置信度字在语义上完全通顺，则不必强行修订。"
    "用 beliefs 输出有原文 Turn 证据的结构化新事实：subject、predicate、value、aliases、confidence、"
    "valid_from、valid_to、evidence_turn_ids。"
    "仅在 CONFLICT 或 UNCERTAIN 且存在具体证据时给出 focus；每个 focus 必须包含 "
    "target_turn_id、目标当前文本中原样存在的最短 span、proposed_text、alternatives（或 closed_set，"
    "必须是包含 span 与 proposed_text 的候选列表，字段名用 alternatives 即可）、"
    "evidence_turn_ids、rationale 和 relationship。relationship 只能是 MUTUALLY_EXCLUSIVE、COEXIST、"
    "TEMPORAL_CHANGE。只输出 JSON 对象。"
)


def _sync_model_env() -> None:
    if not os.getenv("JUDGE_MODEL") and os.getenv("DEEPSEEK_MODEL"):
        os.environ["JUDGE_MODEL"] = os.environ["DEEPSEEK_MODEL"]


def _api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("LLM_WS_API_KEY") or ""


def deepseek_status() -> dict[str, Any]:
    root = domainterms_root()
    model = os.environ.get("JUDGE_MODEL") or os.environ.get("DEEPSEEK_MODEL", "")
    transport = os.environ.get("LLM_TRANSPORT", "")
    error: str | None = None
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


def _chat_json(messages: list[dict[str, str]], *, max_tokens: int = 800) -> Any:
    ensure_importable()
    _sync_model_env()
    from domain_terms import llm_client  # noqa: WPS433

    return llm_client.chat_json(messages, max_tokens=max_tokens, temperature=0.0)


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
    uncertainty = current_turn.meta.get("uncertainty") or {}
    acoustic_disagreement = uncertainty.get("acoustic_disagreement")
    low_conf_chars = uncertainty.get("low_conf_chars")
    payload = {
        "current_turn": current_turn.as_dict(),
        "recent_turns": [turn.as_dict() for turn in memory.recent_turns],
        "dependent_turns": [turn.as_dict() for turn in memory.dependent_turns],
        "working_beliefs": [item.as_dict() for item in memory.working_beliefs],
        "open_hypotheses": [item.as_dict() for item in memory.open_hypotheses],
        "long_term_beliefs": [item.as_dict() for item in memory.long_term_beliefs],
        "domain_entities": _domain_entities(memory),
        # Acoustic uncertainty: spans where a second independent ASR disagreed
        # with the first pass. These are strong candidates for mis-hearings.
        "acoustic_disagreement": acoustic_disagreement,
        # Char-level acoustic confidence: characters the second ASR itself was
        # unsure about, with their confidence values.
        "low_conf_chars": low_conf_chars,
    }
    try:
        result = _chat_json(
            [
                {"role": "system", "content": _CONTEXT_JUDGE_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=700,
        )
        if not isinstance(result, dict):
            raise ValueError("context judge must return a JSON object")
        _fill_judgment_confidence(result)
        return normalize_judgment(result, session)
    except Exception as exc:
        return ContextJudgment("UNCERTAIN", rationale=f"context judge unavailable: {exc}")


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
