#!/usr/bin/env python3
"""Run RAMC vs. one-sided baseline comparisons on AliMeeting clips.

The script evaluates each sample through the live ReTrace-ASR service, records the
raw first-pass ASR CER as the one-sided baseline, and records the final replayed
transcript CER as the RAMC result. It saves per-sample metrics and an aggregate
comparison file to disk.

Example:
    python scripts/run_ramc_experiment.py \
        --base-url http://127.0.0.1:8000 \
        --stems R0015_M0135 R0020_M0176 R0004_M0025 \
        --tag ramc_compare
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
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
    return {
        "cer": (edits / n_ref) if n_ref else 0.0,
        "n_ref": n_ref,
        "n_hyp": len(h),
        "edits": edits,
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
    parts: list[str] = []
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
    raw = _join_turns(session, "raw_text")
    cur = _join_turns(session, "current_text")
    asr_text = str(payload.get("final_text") or raw)

    cer_asr = _cer(ref_text, asr_text)
    cer_cur = _cer(ref_text, cur)
    delta = round(cer_asr["cer"] - cer_cur["cer"], 6)
    relative_gain = (delta / cer_asr["cer"]) if cer_asr["cer"] else 0.0

    revisions = [
        {
            "turn": e.get("target_turn_id"),
            "resolver": e.get("resolver"),
            "span": e.get("span"),
            "to": e.get("replacement") or e.get("after_text"),
            "rationale": e.get("rationale"),
        }
        for e in events
    ]

    return {
        "stem": stem,
        "session_id": bound_id,
        "use_llm": use_llm,
        "elapsed_sec": elapsed,
        "turns": len(session.get("turns") or []),
        "chunk_count": payload.get("chunk_count") or len(session.get("turns") or []),
        "n_revisions": len(events),
        "baseline_cer": cer_asr["cer"],
        "ramc_cer": cer_cur["cer"],
        "cer_delta": delta,
        "relative_gain_pct": relative_gain * 100.0,
        "baseline_edits": cer_asr["edits"],
        "ramc_edits": cer_cur["edits"],
        "revisions": revisions,
        "raw_preview": _norm(raw)[:180],
        "current_preview": _norm(cur)[:180],
        "ref_preview": _norm(ref_text)[:180],
        "impact_note": (
            f"{stem} · baseline CER {cer_asr['cer']:.3f} → RAMC {cer_cur['cer']:.3f} "
            f"· {len(events)} revisions · {elapsed}s"
        ),
        "result": payload,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if "error" not in r]
    if not valid:
        return {
            "n_samples": 0,
            "baseline_mean_cer": 0.0,
            "ramc_mean_cer": 0.0,
            "mean_cer_delta": 0.0,
            "mean_relative_gain_pct": 0.0,
            "mean_revisions": 0.0,
        }

    baseline = [float(r["baseline_cer"]) for r in valid]
    ramc = [float(r["ramc_cer"]) for r in valid]
    deltas = [float(r["cer_delta"]) for r in valid]
    rel_gain = [float(r["relative_gain_pct"]) for r in valid]
    revisions = [float(r["n_revisions"]) for r in valid]

    return {
        "n_samples": len(valid),
        "baseline_mean_cer": statistics.mean(baseline),
        "ramc_mean_cer": statistics.mean(ramc),
        "mean_cer_delta": statistics.mean(deltas),
        "mean_relative_gain_pct": statistics.mean(rel_gain),
        "mean_revisions": statistics.mean(revisions),
        "median_baseline_cer": statistics.median(baseline),
        "median_rAMC_cer": statistics.median(ramc),
        "median_cer_delta": statistics.median(deltas),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stem",
        "session_id",
        "use_llm",
        "baseline_cer",
        "ramc_cer",
        "cer_delta",
        "relative_gain_pct",
        "n_revisions",
        "turns",
        "chunk_count",
        "elapsed_sec",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "stem": row.get("stem", ""),
                "session_id": row.get("session_id", ""),
                "use_llm": row.get("use_llm", ""),
                "baseline_cer": row.get("baseline_cer", ""),
                "ramc_cer": row.get("ramc_cer", ""),
                "cer_delta": row.get("cer_delta", ""),
                "relative_gain_pct": row.get("relative_gain_pct", ""),
                "n_revisions": row.get("n_revisions", ""),
                "turns": row.get("turns", ""),
                "chunk_count": row.get("chunk_count", ""),
                "elapsed_sec": row.get("elapsed_sec", ""),
                "error": row.get("error", ""),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--eval-dir",
        default=str(Path(__file__).resolve().parents[1] / "retrace_state" / "alimeeting_eval"),
    )
    parser.add_argument("--stems", nargs="+", default=["R0015_M0135", "R0020_M0176", "R0004_M0025"])
    parser.add_argument("--suffix", default="3min")
    parser.add_argument("--use-llm", action="store_true", default=True)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--tag", default="ramc_compare")
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    use_llm = False if args.no_llm else args.use_llm
    eval_dir = Path(args.eval_dir)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    try:
        status = _get_json(f"{args.base_url.rstrip('/')}/api/integrations/status")
        print(json.dumps({"integrations": status}, ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: unable to query service status: {exc}")

    for stem in args.stems:
        audio = eval_dir / f"{stem}_{args.suffix}.wav"
        ref_path = eval_dir / f"{stem}_{args.suffix}.ref.txt"
        if not ref_path.exists():
            ref_path = eval_dir / f"{stem}.ref.txt"
        if not audio.exists():
            rows.append({"stem": stem, "error": f"missing audio: {audio}"})
            continue
        if not ref_path.exists():
            rows.append({"stem": stem, "error": f"missing ref: {ref_path}"})
            continue

        ref_text = ref_path.read_text(encoding="utf-8")
        session_id = f"{args.tag}_{stem}_{int(time.time())}"
        print(f"\n=== {stem} audio={audio.name} llm={use_llm} session={session_id} ===", flush=True)

        result = evaluate_one(
            base_url=args.base_url.rstrip("/"),
            stem=stem,
            audio=audio,
            ref_text=ref_text,
            use_llm=use_llm,
            session_id=session_id,
            timeout=args.timeout,
        )

        if "error" in result:
            rows.append(result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            continue

        sample_out = eval_dir / f"{stem}_{args.tag}.result.json"
        sample_summary = eval_dir / f"{stem}_{args.tag}.summary.json"
        write_json(sample_out, result.get("result", result))
        write_json(sample_summary, {k: v for k, v in result.items() if k != "result"})

        rows.append({
            "stem": result["stem"],
            "session_id": result["session_id"],
            "use_llm": result["use_llm"],
            "baseline_cer": result["baseline_cer"],
            "ramc_cer": result["ramc_cer"],
            "cer_delta": result["cer_delta"],
            "relative_gain_pct": result["relative_gain_pct"],
            "n_revisions": result["n_revisions"],
            "turns": result["turns"],
            "chunk_count": result["chunk_count"],
            "elapsed_sec": result["elapsed_sec"],
            "error": "",
        })
        summary_rows.append(rows[-1])
        print(json.dumps(rows[-1], ensure_ascii=False, indent=2), flush=True)

    # aggregate all valid rows and write report files
    aggregate = aggregate_rows(rows)
    report = {
        "experiment": "RAMC_vs_one_sided_baseline",
        "tag": args.tag,
        "base_url": args.base_url.rstrip("/"),
        "split": {"stems": args.stems, "eval_dir": str(eval_dir), "suffix": args.suffix},
        "metrics": aggregate,
        "samples": rows,
    }

    compare_path = eval_dir / f"{args.tag}_ramc_vs_oneside_summary.json"
    csv_path = eval_dir / f"{args.tag}_ramc_vs_oneside.csv"
    write_json(compare_path, report)
    write_csv(csv_path, rows)

    print("\n=== RAMC vs one-sided summary ===")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f"Wrote {compare_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
