"""DeepSeek evidence tool via ASR_domainterms.llm_client (HTTP or Huawei WebSocket)."""
from __future__ import annotations

import json
import os
from typing import Any

from asr_agent.integrations.domainterms import domainterms_root, ensure_importable


_REFLECT_SYSTEM = (
    "You are a sparse retrospective ASR adjudicator (TAMA-style). "
    "Local heuristics already ran; you are called only when a trigger fired "
    "(cross-script brand, open deferred span with later evidence). "
    "Propose corrections ONLY when later raw text clearly contradicts an earlier span. "
    "Prefer empty proposals over guesses. "
    "Look for homophones, brand/name/entity drift "
    "(for example earlier 威沃 later vivo, earlier 图博士 later 涂博士). "
    "Return JSON object {proposals:[{target_turn_id,before_text,after_text,"
    "evidence:[{turn_id,quote}],score,rationale}]}. "
    "before_text and after_text MUST be the shortest contested spans only "
    "(2-8 characters), never the full turn sentence. "
    "before_text must appear verbatim inside the target turn. "
    "Every evidence.quote must be verbatim in a later raw turn and must contain after_text. "
    "If unsure, return {\"proposals\":[]}."
)

_SCORE_SYSTEM = (
    "You are an ASR evidence judge. Use only supplied later evidence and candidates. "
    "Return JSON action KEEP|REVISE_TEXT|REVISE_ENTITY|DEFER|CLARIFY, candidate, "
    "entity_id, score, evidence, rationale. Prefer KEEP/DEFER unless evidence is strong."
)

_ADJUDICATE_SYSTEM = (
    "你是会议ASR歧义消解agent（TAMA风格）。本地启发式已给出候选与后文证据；"
    "只在证据充分时 CORRECT，否则 KEEP。不要发明新词；"
    "canonical 必须来自 candidates。"
    '只输出JSON：{"action":"CORRECT|KEEP","canonical":"...|null","reason":"一句话","confidence":0.0-1.0}'
)


def _sync_model_env() -> None:
    """Prefer DEEPSEEK_MODEL when JUDGE_MODEL is unset (domain_terms reads JUDGE_MODEL)."""
    if not os.getenv("JUDGE_MODEL") and os.getenv("DEEPSEEK_MODEL"):
        os.environ["JUDGE_MODEL"] = os.environ["DEEPSEEK_MODEL"]


def deepseek_status() -> dict[str, Any]:
    root = domainterms_root()
    ready = False
    error: str | None = None
    model = os.environ.get("JUDGE_MODEL") or os.environ.get("DEEPSEEK_MODEL", "")
    transport = os.environ.get("LLM_TRANSPORT", "")
    try:
        ensure_importable()
        _sync_model_env()
        from domain_terms import config as dt_config  # noqa: WPS433

        model = getattr(dt_config, "JUDGE_MODEL", model) or model
        transport = getattr(dt_config, "LLM_TRANSPORT", transport) or transport
        key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or os.environ.get("LLM_WS_API_KEY")
        )
        ready = bool(key)
        if not ready:
            error = "DEEPSEEK_API_KEY / LLM_API_KEY 未配置"
    except Exception as exc:
        error = str(exc)
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


def _shortest_edit_span(before: str, after: str) -> tuple[str, str]:
    """Reduce full-sentence edits to the shortest differing span pair."""
    if not before or not after or before == after:
        return before, after
    prefix = 0
    limit = min(len(before), len(after))
    while prefix < limit and before[prefix] == after[prefix]:
        prefix += 1
    suffix = 0
    while suffix < (limit - prefix) and before[-(suffix + 1)] == after[-(suffix + 1)]:
        suffix += 1
    left = before[prefix : len(before) - suffix if suffix else len(before)]
    right = after[prefix : len(after) - suffix if suffix else len(after)]
    if left and right and left != right:
        return left, right
    return before, after


def _normalize_proposals(proposals: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in proposals:
        if not isinstance(item, dict):
            continue
        before = str(item.get("before_text") or "")
        after = str(item.get("after_text") or "")
        before, after = _shortest_edit_span(before, after)
        normalized = dict(item)
        normalized["before_text"] = before
        normalized["after_text"] = after
        out.append(normalized)
    return out


def reflect_timeline(*, prior_turns: list[dict[str, str]], new_turn: dict[str, str]) -> list[dict[str, Any]]:
    """Sparse discovery only — caller must gate on TAMA-style triggers."""
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("LLM_WS_API_KEY")
        or ""
    )
    if not api_key:
        return []
    try:
        result = _chat_json(
            [
                {"role": "system", "content": _REFLECT_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"prior_turns": prior_turns, "new_turn": new_turn, "mode": "sparse_trigger"},
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=400,
        )
    except Exception:
        return []
    if isinstance(result, dict) and isinstance(result.get("proposals"), list):
        return _normalize_proposals(result["proposals"])
    return []


def adjudicate_conflict(
    *,
    hyp: str,
    candidates: list[str],
    evidence: list[str],
    context: str = "",
    heuristic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """TAMA agent_decide-style confirm: CORRECT only among closed candidates + evidence."""
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("LLM_WS_API_KEY")
        or ""
    )
    base = {
        "action": "KEEP",
        "canonical": None,
        "reason": "no_llm",
        "confidence": 0.0,
        "source": "fallback",
    }
    if not api_key or not evidence or not candidates:
        return base
    allowed = {str(item).strip() for item in candidates if str(item).strip()}
    user = (
        f"挂起词 hyp={hyp}\n"
        f"候选={sorted(allowed)}\n"
        f"当时上下文={context[:200]}\n"
        f"后文证据={evidence[:6]}\n"
        f"启发式={heuristic or {}}"
    )
    try:
        data = _chat_json(
            [
                {"role": "system", "content": _ADJUDICATE_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=400,
        )
    except Exception:
        return base
    if not isinstance(data, dict):
        return base
    action = str(data.get("action") or "").upper()
    if action not in {"CORRECT", "KEEP"}:
        return base
    canonical = str(data.get("canonical") or "").strip() or None
    if action == "CORRECT" and (not canonical or canonical not in allowed):
        return base
    return {
        "action": action,
        "canonical": canonical if action == "CORRECT" else None,
        "reason": str(data.get("reason") or "agent_llm"),
        "confidence": float(data.get("confidence") or 0.7),
        "source": "agent_llm",
    }


def score_evidence(
    *,
    span: str,
    text_candidates: list[str],
    entity_candidates: list[dict[str, Any]],
    prior_text: str,
    evidence_text: str,
) -> dict[str, Any]:
    """Return a decision only; the ReTrace controller validates and commits it."""
    api_key = (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("LLM_WS_API_KEY")
        or ""
    )
    if not api_key:
        return {"action": "DEFER"}
    try:
        result = _chat_json(
            [
                {"role": "system", "content": _SCORE_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "span": span,
                            "text_candidates": text_candidates,
                            "entity_candidates": entity_candidates,
                            "prior_text": prior_text,
                            "later_evidence": evidence_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_tokens=400,
        )
    except Exception:
        return {"action": "DEFER"}
    return result if isinstance(result, dict) else {"action": "DEFER"}
