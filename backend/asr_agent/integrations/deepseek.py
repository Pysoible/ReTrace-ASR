"""DeepSeek access via ASR_domainterms.llm_client (same gateway as ASR_agent tools)."""
from __future__ import annotations

import os
from typing import Any

from asr_agent.integrations.domainterms import domainterms_root, ensure_importable

_CORRECT_SYSTEM = (
    "你是一个中文语音识别(ASR)后处理纠错器。输入是一段 ASR 初步转写，里面可能存在同音/近音导致的错别字。"
    "你的任务是识别并修正所有明显的错误，输出修正后的完整文本。\n"
    "规则：\n"
    "1. 修正所有明显的同音/近音错字。\n"
    "2. 优先参考候选术语列表；如果候选里没有合适词，也只能改成与原错误片段发音高度相近的中文词。\n"
    "3. 绝不改变原文语义、语气、标点和正常用词。拿不准的地方保持原样。\n"
    "4. 只输出修正后的完整文本，不要加任何解释、标记或注释。"
)

_REVISE_SYSTEM = (
    "你是对话式 ASR 的证据驱动修订器。给定一个未决假设（先前转写中的不确定片段）以及之后出现的证据句，"
    "判断是否应根据后续证据修订先前文本或绑定实体。\n"
    "只使用后续证据，不要臆造事实。\n"
    "输出严格 JSON 对象，字段：\n"
    '{"action":"KEEP|REVISE_TEXT|REVISE_ENTITY|DEFER|CLARIFY",'
    '"candidate":"修订后的片段或原片段",'
    '"entity_id":"选中的 entity_id 或 null",'
    '"score":0.0到1.0,'
    '"evidence":["证据短语",...],'
    '"rationale":"一句话原因"}'
)


def deepseek_status() -> dict[str, Any]:
    root = domainterms_root()
    ready = False
    error: str | None = None
    model = os.environ.get("JUDGE_MODEL", "")
    transport = os.environ.get("LLM_TRANSPORT", "")
    try:
        ensure_importable()
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


def correct_text_with_deepseek(text: str, terms: list[str] | None = None, domain: str | None = None) -> dict[str, Any]:
    """Text second-pass correction — mirrors ASR_agent asr_tools._llm_correct."""
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "empty text"}
    try:
        ensure_importable()
        from domain_terms import llm_client  # noqa: WPS433
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    terms = terms or []
    term_str = "、".join(terms)
    domain_line = f"该段大致属于「{domain}」领域。\n" if domain else ""
    term_block = (
        f"参考领域术语候选：{term_str}。\n" if terms else ""
    )
    user = (
        "下面是一段语音识别的初步转写结果，其中可能有同音/近音导致的错字。\n"
        f"{domain_line}{term_block}"
        "请修正所有明显错字，只输出修正后的完整文本，不要任何说明。\n\n"
        f"初步转写：{text}"
    )
    try:
        reply = llm_client.chat(
            [
                {"role": "system", "content": _CORRECT_SYSTEM},
                {"role": "user", "content": user},
            ],
            max_tokens=1500,
            temperature=0.0,
        )
        cleaned = (reply or "").strip().strip("`").strip() or text
        return {"ok": True, "original": text, "corrected": cleaned, "changed": cleaned != text}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "original": text, "corrected": text}


def revise_with_deepseek(
    *,
    span: str,
    text_candidates: list[str],
    entity_candidates: list[dict[str, Any]],
    prior_text: str,
    evidence_text: str,
) -> dict[str, Any]:
    """Ask DeepSeek to resolve a deferred hypothesis using later-turn evidence."""
    try:
        ensure_importable()
        from domain_terms import llm_client  # noqa: WPS433
    except Exception as exc:
        return {"ok": False, "error": str(exc), "action": "DEFER"}

    payload = {
        "span": span,
        "text_candidates": text_candidates,
        "entity_candidates": entity_candidates,
        "prior_text": prior_text,
        "evidence_text": evidence_text,
    }
    try:
        result = llm_client.chat_json(
            [
                {"role": "system", "content": _REVISE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        "请根据后续证据决定是否修订先前 ASR 假设，只返回 JSON。\n"
                        f"{payload}"
                    ),
                },
            ],
            max_tokens=800,
            temperature=0.0,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "action": "DEFER"}

    if not isinstance(result, dict):
        return {"ok": False, "error": f"unexpected LLM payload: {result!r}", "action": "DEFER"}

    action = str(result.get("action") or "DEFER").upper()
    if action not in {"KEEP", "REVISE_TEXT", "REVISE_ENTITY", "DEFER", "CLARIFY"}:
        action = "DEFER"
    return {
        "ok": True,
        "action": action,
        "candidate": str(result.get("candidate") or span),
        "entity_id": result.get("entity_id"),
        "score": float(result.get("score") or 0.0),
        "evidence": list(result.get("evidence") or []),
        "rationale": str(result.get("rationale") or ""),
    }
