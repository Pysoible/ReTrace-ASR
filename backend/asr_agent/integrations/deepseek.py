"""Direct DeepSeek tool adapter for constrained ReTrace evidence decisions."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen


def reflect_timeline(*, prior_turns: list[dict[str, str]], new_turn: dict[str, str]) -> list[dict[str, Any]]:
    """Propose only evidence-anchored retrospective interpretations from raw turns."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return []
    prompt = (
        "You are a retrospective conversational ASR agent. A new raw ASR turn arrived. Reconsider earlier raw turns without assuming any user-provided entities. "
        "Return JSON object {proposals:[{target_turn_id,before_text,after_text,evidence:[{turn_id,quote}],score,rationale}]}. "
        "Only propose a correction when every quote is verbatim in a later raw turn and directly supports the reinterpretation. Otherwise return an empty proposals list."
    )
    payload = {"model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), "temperature": 0, "response_format": {"type": "json_object"}, "messages": [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"prior_turns": prior_turns, "new_turn": new_turn}, ensure_ascii=False)},
    ]}
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    request = Request(f"{base}/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(json.loads(response.read().decode())["choices"][0]["message"]["content"])
        return result.get("proposals", []) if isinstance(result, dict) and isinstance(result.get("proposals"), list) else []
    except Exception:
        return []


def score_evidence(*, span: str, text_candidates: list[str], entity_candidates: list[dict[str, Any]], prior_text: str, evidence_text: str) -> dict[str, Any]:
    """Return a decision only; the ReTrace controller validates and commits it."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return {"action": "DEFER"}
    payload = {"model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"), "temperature": 0, "response_format": {"type": "json_object"}, "messages": [
        {"role": "system", "content": "You are an ASR evidence judge. Use only supplied later evidence and candidates. Return JSON action KEEP|REVISE_TEXT|REVISE_ENTITY|DEFER|CLARIFY, candidate, entity_id, score, evidence, rationale."},
        {"role": "user", "content": json.dumps({"span": span, "text_candidates": text_candidates, "entity_candidates": entity_candidates, "prior_text": prior_text, "later_evidence": evidence_text}, ensure_ascii=False)},
    ]}
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    request = Request(f"{base}/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode())
        content = body["choices"][0]["message"]["content"]
        result = json.loads(content)
        return result if isinstance(result, dict) else {"action": "DEFER"}
    except Exception:
        return {"action": "DEFER"}
