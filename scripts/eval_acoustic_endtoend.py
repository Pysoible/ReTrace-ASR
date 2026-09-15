#!/usr/bin/env python3
"""End-to-end effect of the acoustic signal: does "correcting" disagreement
spans with the second ASR's transcript actually reduce CER?

This is the decisive experiment separating "the signal is a good error
indicator" (previous eval) from "the signal *improves* the final transcript".

For each sentence we have:
  - first   : Qwen-Omni first pass
  - second  : paraformer second pass (the acoustic signal source)
  - ref     : reference transcript

We build a "corrected" transcript by replacing every disagreement span of
``first`` with the corresponding ``second`` span (the simplest possible
acoustic-driven correction), then compare:

    CER(first, ref)   vs   CER(corrected, ref)

and count, per character, whether the correction *improved* (was wrong, now
right) or *worsened* (was right, now wrong).

Usage:
    python scripts/eval_acoustic_endtoend.py \
        --data-dir /path/to/mix_wav_no_overlap --limit 50
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_ROOT / "backend"))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from eval_acoustic_disagreement import (  # noqa: E402
    _load_env,
    _norm,
    _qwen_transcribe,
    _ref_text,
)

_load_env()


def _cer(hyp: str, ref: str) -> float:
    """Character error rate (Levenshtein / len(ref))."""
    if not ref:
        return 1.0 if hyp else 0.0
    n, m = len(hyp), len(ref)
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dist[i][0] = i
    for j in range(m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + (0 if hyp[i - 1] == ref[j - 1] else 1),
            )
    return dist[n][m] / m


def _apply_acoustic_correction(first: str, second: str) -> str:
    """Replace every disagreement span of ``first`` with the ``second`` version."""
    sm = difflib.SequenceMatcher(None, first, second, autojunk=False)
    parts: list[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(first[i1:i2])
        elif tag == "replace":
            parts.append(second[j1:j2])
        elif tag == "delete":
            parts.append("")  # first has extra chars the second ASR did not hear
        elif tag == "insert":
            parts.append(second[j1:j2])
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pairs = sorted(data_dir.glob("*.wav"))
    if args.limit:
        pairs = pairs[: args.limit]

    from asr_agent.integrations import acoustic

    rows: list[dict[str, Any]] = []
    for index, wav in enumerate(pairs, 1):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        ref = _ref_text(txt)
        first = _qwen_transcribe(args.base_url, wav)
        if not first:
            print(f"[{index}/{len(pairs)}] {wav.name}: Qwen 空转写，跳过")
            continue
        second = _norm(acoustic._transcribe_paraformer(wav))
        if not second:
            print(f"[{index}/{len(pairs)}] {wav.name}: 二遍 ASR 空转写，跳过")
            continue

        baseline_cer = _cer(first, ref)
        corrected = _apply_acoustic_correction(first, second)
        corrected_cer = _cer(corrected, ref)

        # per-character improvement/worsening relative to the reference
        improved = max(0.0, baseline_cer - corrected_cer)
        worsened = max(0.0, corrected_cer - baseline_cer)

        rows.append(
            {
                "baseline_cer": baseline_cer,
                "corrected_cer": corrected_cer,
                "improved": improved,
                "worsened": worsened,
                "n_ref": len(ref),
            }
        )
        print(
            f"[{index}/{len(pairs)}] {wav.name}: CER {baseline_cer:.3f} → {corrected_cer:.3f} "
            f"({'↓改善' if corrected_cer < baseline_cer else '↑恶化' if corrected_cer > baseline_cer else '='})"
        )

    if not rows:
        print("无有效样本")
        return

    n = len(rows)
    total_ref = sum(row["n_ref"] for row in rows)
    avg_base = sum(row["baseline_cer"] * row["n_ref"] for row in rows) / total_ref
    avg_corr = sum(row["corrected_cer"] * row["n_ref"] for row in rows) / total_ref
    n_improved = sum(1 for row in rows if row["corrected_cer"] < row["baseline_cer"])
    n_worsened = sum(1 for row in rows if row["corrected_cer"] > row["baseline_cer"])
    n_tied = n - n_improved - n_worsened

    print("\n================ 端到端汇总 ================")
    print(f"样本数: {n}")
    print(f"加权基线 CER: {avg_base:.4f}")
    print(f"加权修正 CER: {avg_corr:.4f}")
    print(f"相对变化: {(avg_corr - avg_base) / avg_base * 100:+.2f}% (负=改善)")
    print(f"改善句: {n_improved}  恶化句: {n_worsened}  持平句: {n_tied}")


if __name__ == "__main__":
    main()
