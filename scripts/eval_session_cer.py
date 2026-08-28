#!/usr/bin/env python3
"""Evaluate a RAMC session: upload audio -> poll until done -> compute CER.

Usage:
    python scripts/eval_session_cer.py --session eval-101 \
        --audio /path/to/audio.wav \
        --gt /path/to/groundtruth.txt \
        [--base-url http://127.0.0.1:8000] [--poll-sec 20]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def _norm(text: str) -> str:
    # Strip every "[start-end] " chunk header (not just the leading one).
    text = re.sub(r"\[\d+\.?\d*[-~]\d+\.?\d*\]", "", text or "")
    return "".join(_CHAR_RE.findall(text)).lower()


def _cer(ref: str, hyp: str) -> float:
    """Character error rate (Levenshtein) over normalized strings."""
    a, b = _norm(ref), _norm(hyp)
    if not a:
        return 0.0 if not b else 1.0
    m, n = len(a), len(b)
    if m == 0:
        return float(n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            sub = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + sub)
            prev = tmp
    return dp[n] / m


def load_gt(path: str) -> str:
    """Concat the text column of a RAMC TXT ground-truth file."""
    parts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cols = line.split()
        if len(cols) < 4:
            continue
        # [start,end] SPK gender,accent text...
        text = line.split("，", 1)[-1] if "，" in line else ""
        # Simpler: text is everything after the accent column.
        # Columns: [start,end], SPK, gender,accent, text
        toks = line.split()
        # join from the 4th token (index 3) onward
        if len(toks) >= 4:
            text = " ".join(toks[3:])
        parts.append(text)
    return "".join(parts)


def load_session_hyp(session: dict) -> str:
    turns = session.get("turns") or []
    parts = []
    for t in turns:
        cur = t.get("current_text") or t.get("raw_text") or ""
        parts.append(cur)
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--poll-sec", type=int, default=20)
    ap.add_argument("--skip-upload", action="store_true", help="session already processed")
    ap.add_argument("--out", help="write hypothesis text to this file")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")

    if not args.skip_upload:
        with open(args.audio, "rb") as fh:
            r = requests.post(
                f"{base}/api/sessions/{args.session}/audio/upload",
                files={"file": (Path(args.audio).name, fh)},
                timeout=120,
            )
        print("upload:", r.status_code, r.text[:200], flush=True)

    # Poll until the transcript is complete (turn count stops growing).
    stable = 0
    last_turns = -1
    deadline = time.time() + 90 * 60
    while time.time() < deadline:
        try:
            r = requests.get(f"{base}/api/sessions/{args.session}", timeout=30)
            data = r.json()["session"]
        except Exception as exc:  # noqa: BLE001
            print("poll err:", exc, flush=True)
            time.sleep(args.poll_sec)
            continue
        n_turns = len(data.get("turns") or [])
        status = data.get("analysis_status", "?")
        print(f"  status={status} turns={n_turns}", flush=True)
        if n_turns and n_turns == last_turns:
            stable += 1
        else:
            stable = 0
        last_turns = n_turns
        if n_turns and stable >= 3:
            print("transcription complete (turn count stable).", flush=True)
            break
        time.sleep(args.poll_sec)

    r = requests.get(f"{base}/api/sessions/{args.session}", timeout=30)
    session = r.json()["session"]

    hyp = load_session_hyp(session)
    ref = load_gt(args.gt)
    cer = _cer(ref, hyp)
    print(f"\n=== CER ===\nref chars: {len(_norm(ref))}\nhyp chars: {len(_norm(hyp))}\nCER: {cer*100:.2f}%")

    revs = session.get("revision_events") or []
    revised = sum(1 for r in revs if (r.get("changed") or r.get("action") == "revise"))
    print(f"turns: {len(session.get('turns') or [])}  revision_events: {len(revs)}  revised: {revised}")

    # Truncation-relisten stats
    relisten = [t.get("meta", {}).get("relisten_uncertain") for t in session.get("turns", [])]
    print(f"relisten_uncertain recorded: {sum(1 for x in relisten if x)}")

    if args.out:
        Path(args.out).write_text(hyp, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
