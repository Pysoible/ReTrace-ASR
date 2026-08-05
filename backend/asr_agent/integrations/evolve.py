"""Self-evolution loop: ASR → mine terms → ingest dictionary → re-ASR → stop when stable."""
from __future__ import annotations

import os
import re
from typing import Any

from asr_agent.integrations.domainterms import ensure_importable
from asr_agent.integrations.qwen_asr import transcribe_audio

_EXTRACT_SYSTEM = (
    "你是专业术语抽取器。从给定 ASR 转写文本里挑出**中文专业术语/领域词**，目标是"
    "『语音识别可能识别错』的专业词。规则：①每个词 2-8 个汉字；②去掉太常见的通用词；"
    "③偏向生僻、近音易混、行业黑话；④只输出 JSON 数组，如 "
    '["坍落度","报验"]，不要解释。'
)


def _max_iters_default() -> int:
    return max(1, int(os.getenv("ASR_EVOLVE_MAX_ITERS", "3")))


def _max_terms_default() -> int:
    return max(1, int(os.getenv("ASR_EVOLVE_MAX_TERMS", "20")))


def extract_candidates_from_transcript(text: str, max_terms: int | None = None) -> list[str]:
    max_terms = max_terms or _max_terms_default()
    if not (text or "").strip():
        return []
    ensure_importable()
    from domain_terms import llm_client  # noqa: WPS433

    snippet = text.strip()[:6000]
    try:
        data = llm_client.chat_json(
            [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": f"最多抽取 {max_terms} 个候选词。文本：\n{snippet}"},
            ],
            max_tokens=2000,
            temperature=0.2,
        )
    except Exception:
        return _heuristic_candidates(snippet, max_terms)

    out: list[str] = []
    seen: set[str] = set()
    for item in data if isinstance(data, list) else []:
        term = (item if isinstance(item, str) else "").strip()
        if term and 2 <= len(term) <= 8 and term not in seen:
            seen.add(term)
            out.append(term)
        if len(out) >= max_terms:
            break
    return out or _heuristic_candidates(snippet, max_terms)


def _heuristic_candidates(text: str, max_terms: int) -> list[str]:
    # Fallback when LLM unavailable: CJK n-grams that look term-like.
    chars = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    counts: dict[str, int] = {}
    for n in (2, 3, 4):
        for i in range(max(0, len(chars) - n + 1)):
            term = chars[i : i + n]
            counts[term] = counts.get(term, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))
    stop = {"我们", "你们", "他们", "一个", "这个", "那个", "可以", "没有", "因为", "所以", "然后", "今天", "工作", "会议"}
    out = []
    for term, _ in ranked:
        if term in stop:
            continue
        out.append(term)
        if len(out) >= max_terms:
            break
    return out


def ingest_candidates(terms: list[str], *, dry_run: bool = False) -> dict[str, Any]:
    ensure_importable()
    from domain_terms import dictionary, pipeline  # noqa: WPS433

    if not terms:
        return {
            "n_added": 0,
            "n_rejected": 0,
            "n_dup": 0,
            "added": [],
            "rejected": [],
            "duplicates": [],
            "dict_size": len(dictionary.Dictionary().terms),
        }
    summary = pipeline.ingest(terms, dry_run=dry_run)

    def _pair(item: Any) -> list[str]:
        if isinstance(item, dict):
            return [str(item.get("term") or ""), ""]
        if isinstance(item, (list, tuple)):
            term = str(item[0]) if item else ""
            reason = str(item[1]) if len(item) > 1 else ""
            return [term, reason]
        return [str(item), ""]

    return {
        "n_added": int(summary.get("n_added") or 0),
        "n_rejected": int(summary.get("n_rejected") or 0),
        "n_dup": int(summary.get("n_dup") or 0),
        "dict_size": int(summary.get("dict_size") or 0),
        "added": [_pair(x)[0] for x in (summary.get("added") or [])],
        "rejected": [_pair(x) for x in (summary.get("rejected") or [])][:30],
        "duplicates": [_pair(x) for x in (summary.get("duplicates") or [])][:30],
    }


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip())


def evolve_audio(
    audio: str,
    *,
    max_iters: int | None = None,
    two_pass: bool = True,
    domain: str | None = None,
    top_k: int | None = None,
    dry_run_dict: bool = False,
    max_terms: int | None = None,
) -> dict[str, Any]:
    """Iterative self-evolution over one audio file.

    Each iteration:
      1) Qwen two-pass ASR (uses current domain dictionary)
      2) Mine candidate terms from the transcript (DeepSeek)
      3) Judge + ingest into dictionary (ASR_domainterms pipeline)
      4) Stop when no new terms accepted and transcript stabilizes
    """
    max_iters = max(1, int(max_iters or _max_iters_default()))
    history: list[dict[str, Any]] = []
    prev_norm = ""
    final_asr: dict[str, Any] | None = None

    for iteration in range(1, max_iters + 1):
        asr = transcribe_audio(audio, two_pass=two_pass, domain=domain, top_k=top_k)
        if not asr.get("ok"):
            return {"ok": False, "error": asr.get("error") or "ASR failed", "history": history, "iterations": iteration - 1}

        text = str(asr.get("final_text") or asr.get("pass1_text") or "")
        candidates = extract_candidates_from_transcript(text, max_terms=max_terms)
        ingest = ingest_candidates(candidates, dry_run=dry_run_dict)
        norm = _normalize_text(text)
        changed = bool(prev_norm) and norm != prev_norm
        stop = iteration > 1 and ingest["n_added"] == 0 and not changed

        step = {
            "iteration": iteration,
            "duration_sec": asr.get("duration_sec"),
            "chunk_count": asr.get("chunk_count"),
            "applied_terms": asr.get("applied_terms") or [],
            "final_text": text,
            "candidates": candidates,
            "ingest": ingest,
            "text_changed": changed,
            "stop_reason": "stable" if stop else None,
        }
        history.append(step)
        final_asr = asr
        prev_norm = norm or prev_norm

        if stop:
            break

    return {
        "ok": True,
        "iterations": len(history),
        "max_iters": max_iters,
        "history": history,
        "asr": final_asr,
        "evolved": any((step.get("ingest") or {}).get("n_added", 0) > 0 for step in history),
        "final_text": (final_asr or {}).get("final_text") or (final_asr or {}).get("pass1_text") or "",
    }
