"""DeepSeek adapter for the structured Context Judge protocol."""
from __future__ import annotations

import json
import os
from typing import Any

from asr_agent.context_judge import ContextJudgment, ExplicitSignalFallbackJudge, normalize_judgment
from asr_agent.integrations.domainterms import domainterms_root, ensure_importable
from asr_agent.memory import MemoryPacket
from asr_agent.models import Session, Turn


_CONTEXT_JUDGE_SYSTEM = (
    "你是实时 ASR Context Judge。先判断最新观察与短期、长期 Memory 的关系，只能输出 "
    "CONSISTENT、NOVEL、CONFLICT、UNCERTAIN。不要预先分词，不要扫描所有二元词，不要发明专有名词。"
    "用 beliefs 输出有原文 Turn 证据的结构化新事实：subject、predicate、value、aliases、confidence、"
    "valid_from、valid_to、evidence_turn_ids。"
    "仅在 CONFLICT 或 UNCERTAIN 且存在具体证据时给出 focus；每个 focus 必须包含 "
    "target_turn_id、目标当前文本中原样存在的最短 span、proposed_text、包含两者的 closed-set alternatives、"
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


def judge_context(*, session: Session, current_turn: Turn, memory: MemoryPacket) -> ContextJudgment:
    """Judge context semantically; malformed or unavailable models safely defer."""
    if not _api_key():
        return ExplicitSignalFallbackJudge()(session=session, current_turn=current_turn, memory=memory)
    payload = {
        "current_turn": current_turn.as_dict(),
        "recent_turns": [turn.as_dict() for turn in memory.recent_turns],
        "dependent_turns": [turn.as_dict() for turn in memory.dependent_turns],
        "working_beliefs": [item.as_dict() for item in memory.working_beliefs],
        "open_hypotheses": [item.as_dict() for item in memory.open_hypotheses],
        "long_term_beliefs": [item.as_dict() for item in memory.long_term_beliefs],
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
        return normalize_judgment(result, session)
    except Exception as exc:
        return ContextJudgment("UNCERTAIN", rationale=f"context judge unavailable: {exc}")
