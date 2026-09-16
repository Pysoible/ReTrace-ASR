#!/usr/bin/env python3
"""Run ReTrace-ASR on AliMeeting clips and report CER + revision stats."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _norm(text: str) -> str:
    text = text or ""
    text = re.sub(r"\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text, flags=re.UNICODE)
    return text.lower()


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _cer(ref: str, hyp: str) -> dict[str, Any]:
    r, h = _norm(ref), _norm(hyp)
    edits = _edit_distance(r, h)
    n_ref = len(r)
    overlap = sum((Counter(r) & Counter(h)).values())
    precision = overlap / len(h) if h else 0.0
    recall = overlap / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cer": (edits / n_ref) if n_ref else 0.0,
        "n_ref": n_ref,
        "n_hyp": len(h),
        "edits": edits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _join_turns(session: dict[str, Any], field: str) -> str:
    parts = []
    for turn in session.get("turns") or []:
        parts.append(str(turn.get(field) or ""))
    return "".join(parts)


def evaluate_one(
    *,
    base_url: str,
    stem: str,
    audio: Path,
    ref_text: str,
    use_llm: bool,
    session_id: str,
    timeout: float,
) -> dict[str, Any]:
    t0 = time.time()
    try:
        payload = _post_json(
            f"{base_url}/api/sessions/{session_id}/audio",
            {"turn_id": "t000", "audio": str(audio), "use_llm": use_llm},
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"stem": stem, "session_id": session_id, "error": f"HTTP {exc.code}: {detail}"}
    except Exception as exc:  # noqa: BLE001
        return {"stem": stem, "session_id": session_id, "error": str(exc)}

    elapsed = round(time.time() - t0, 1)
    session = payload.get("session") or {}
    bound_id = payload.get("session_id") or session_id
    if not session:
        try:
            session = _get_json(f"{base_url}/api/sessions/{bound_id}").get("session") or {}
        except Exception:
            session = {}

    events = [e for e in (session.get("revision_events") or []) if e.get("active")]
    audit_events = [e for e in events if e.get("event_kind") == "audit" or (not e.get("span") and e.get("resolver") == "context-judge")]
    committed_events = [
        e for e in events
        if e.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY", "ROLLBACK"}
        and e.get("event_kind") != "audit"
    ]
    raw = _join_turns(session, "raw_text")
    cur = _join_turns(session, "current_text")
    asr_text = str(payload.get("final_text") or raw)
    cer_asr = _cer(ref_text, asr_text)
    cer_cur = _cer(ref_text, cur)
    revisions = [
        {
            "turn": e.get("target_turn_id"),
            "resolver": e.get("resolver"),
            "span": e.get("span"),
            "to": e.get("replacement") or e.get("after_text"),
            "rationale": e.get("rationale"),
        }
        for e in committed_events
    ]
    return {
        "stem": stem,
        "session_id": bound_id,
        "use_llm": use_llm,
        "elapsed_sec": elapsed,
        "turns": len(session.get("turns") or []),
        "chunk_count": payload.get("chunk_count") or len(session.get("turns") or []),
        "n_revisions": len(committed_events),
        "decision_events": len(events),
        "audit_events": len(audit_events),
        "committed_revisions": len(committed_events),
        "changed_turns": sum(turn.get("raw_text") != turn.get("current_text") for turn in session.get("turns") or []),
        "cer_asr": cer_asr,
        "cer_current": cer_cur,
        "cer_delta": round(cer_asr["cer"] - cer_cur["cer"], 4),
        "revisions": revisions,
        "raw_preview": _norm(raw)[:180],
        "cur_preview": _norm(cur)[:180],
        "ref_preview": _norm(ref_text)[:180],
        "impact_note": (
            f"{stem} · CER {cer_asr['cer']:.3f}→{cer_cur['cer']:.3f} · "
            f"{len(events)} revisions · {elapsed}s"
        ),
        "result": payload,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--eval-dir",
        default=str(Path(__file__).resolve().parents[1] / "retrace_state" / "alimeeting_eval"),
    )
    parser.add_argument("--stems", nargs="+", default=["R0015_M0135"])
    parser.add_argument("--suffix", default="3min")
    parser.add_argument("--use-llm", action="store_true", default=True)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--tag", default="rerun")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    use_llm = False if args.no_llm else args.use_llm
    eval_dir = Path(args.eval_dir)
    ts = int(time.time())
    summaries: list[dict[str, Any]] = []

    status = _get_json(f"{args.base_url}/api/integrations/status")
    print(json.dumps({"integrations": status}, ensure_ascii=False, indent=2))

    for stem in args.stems:
        audio = eval_dir / f"{stem}_{args.suffix}.wav"
        ref_path = eval_dir / f"{stem}_{args.suffix}.ref.txt"
        if not ref_path.exists():
            ref_path = eval_dir / f"{stem}.ref.txt"
        if not audio.exists():
            summaries.append({"stem": stem, "error": f"missing audio: {audio}"})
            continue
        if not ref_path.exists():
            summaries.append({"stem": stem, "error": f"missing ref: {ref_path}"})
            continue
        ref_text = ref_path.read_text(encoding="utf-8")
        session_id = f"{args.tag}_{stem}_{ts}"
        print(f"\n=== {stem} audio={audio.name} llm={use_llm} session={session_id} ===", flush=True)
        summary = evaluate_one(
            base_url=args.base_url.rstrip("/"),
            stem=stem,
            audio=audio,
            ref_text=ref_text,
            use_llm=use_llm,
            session_id=session_id,
            timeout=args.timeout,
        )
        out_result = eval_dir / f"{stem}_{args.tag}.result.json"
        out_summary = eval_dir / f"{stem}_{args.tag}.summary.json"
        # Keep summary compact on disk; full payload in result.
        compact = {k: v for k, v in summary.items() if k != "result"}
        out_result.write_text(json.dumps(summary.get("result", summary), ensure_ascii=False, indent=2), encoding="utf-8")
        out_summary.write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(compact)
        print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)

    all_path = eval_dir / f"{args.tag}_summary.json"
    all_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {all_path}", flush=True)


if __name__ == "__main__":
    main()
