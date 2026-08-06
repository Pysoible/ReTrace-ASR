"""Direct DeepSeek tool adapter for constrained ReTrace evidence decisions."""
from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen


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
