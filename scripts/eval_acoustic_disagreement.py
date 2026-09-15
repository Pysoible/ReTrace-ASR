#!/usr/bin/env python3
"""Quantify whether the acoustic-disagreement signal predicts real ASR errors.

Design: for each sentence clip we run two ASR systems (default: Qwen-Omni first
pass + paraformer second pass) and the reference transcript. We then mark, per
character of the first pass, whether it is (a) *wrong* relative to the reference
and (b) in a *disagreement* span relative to the second pass. If disagreement is
a useful acoustic-uncertainty signal, then

    P(wrong | disagree)  >>  P(wrong | agree)

and a large fraction of real errors should fall inside disagreement spans
(recall), while disagreement spans should mostly be real errors (precision).

Usage:
    python scripts/eval_acoustic_disagreement.py \
        --data-dir /path/to/Alimeeting_eval_stamps_new/mix_wav_overlap \
        --limit 50
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(_ROOT / "backend"))


def _load_env() -> None:
    """Load backend/.env so the ASR engine config matches the running server."""
    env_file = _ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env()


def _norm(text: str) -> str:
    text = text or ""
    text = re.sub(r"\[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "", text, flags=re.UNICODE)
    return text.lower()


def _ref_text(txt_path: Path) -> str:
    raw = txt_path.read_text(encoding="utf-8")
    return _norm(raw.replace("<sc>", ""))


def _qwen_transcribe(base_url: str, audio_path: Path) -> str:
    """First-pass Qwen-Omni transcript via the running ReTrace service."""
    import json
    import urllib.request

    payload = json.dumps({"audio": str(audio_path)}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/sessions/eval-acoustic/audio",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ""
    asr = data.get("asr") or {}
    final = str(asr.get("final_text") or "")
    if not final:
        # 从 session turns 拼接 raw_text 作为兜底
        session = data.get("session") or {}
        parts = [str(turn.get("raw_text") or "") for turn in (session.get("turns") or [])]
        final = "".join(parts)
    return _norm(final)


def _char_mask_wrong(first: str, reference: str) -> list[bool]:
    """Per-character mask of ``first`` marking positions that differ from reference."""
    sm = difflib.SequenceMatcher(None, first, reference, autojunk=False)
    mask = [False] * len(first)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag != "equal":
            for index in range(i1, i2):
                mask[index] = True
    return mask


def _char_mask_disagree(first: str, second: str) -> list[bool]:
    """Per-character mask of ``first`` marking positions that differ from second."""
    return _char_mask_wrong(first, second)


def evaluate_pair(
    audio_path: Path,
    ref: str,
    first: str,
    second: str,
) -> dict[str, Any]:
    wrong = _char_mask_wrong(first, ref)
    disagree = _char_mask_disagree(first, second)
    n = len(first)
    tp = sum(1 for i in range(n) if wrong[i] and disagree[i])  # 分歧且错误
    fp = sum(1 for i in range(n) if (not wrong[i]) and disagree[i])  # 分歧但正确
    fn = sum(1 for i in range(n) if wrong[i] and (not disagree[i]))  # 错误但未分歧
    tn = sum(1 for i in range(n) if (not wrong[i]) and (not disagree[i]))  # 正确且未分歧
    n_err = tp + fn
    n_dis = tp + fp
    return {
        "audio": str(audio_path),
        "n_chars": n,
        "n_err": n_err,
        "n_disagree": n_dis,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="wav+txt 配对的句子级目录")
    parser.add_argument("--limit", type=int, default=50, help="最多评测多少句")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="ReTrace 服务地址")
    parser.add_argument("--second-asr", default="paraformer", choices=["paraformer"], help="二遍 ASR")
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
        second = _norm(acoustic._transcribe_paraformer(wav)) if args.second_asr == "paraformer" else ""
        if not second:
            print(f"[{index}/{len(pairs)}] {wav.name}: 二遍 ASR 空转写，跳过")
            continue

        row = evaluate_pair(wav, ref, first, second)
        rows.append(row)
        prec = row["tp"] / row["n_disagree"] if row["n_disagree"] else 0.0
        rec = row["tp"] / row["n_err"] if row["n_err"] else 0.0
        print(f"[{index}/{len(pairs)}] {wav.name}: 错 {row['n_err']} 分歧 {row['n_disagree']} | 精确率 {prec:.2f} 召回率 {rec:.2f}")

    if not rows:
        print("无有效样本")
        return

    total = {key: sum(row[key] for row in rows) for key in ("n_chars", "n_err", "n_disagree", "tp", "fp", "fn", "tn")}
    n_err, n_dis = total["n_err"], total["n_disagree"]
    precision = total["tp"] / n_dis if n_dis else 0.0
    recall = total["tp"] / n_err if n_err else 0.0
    p_wrong_given_disagree = total["tp"] / n_dis if n_dis else 0.0
    p_wrong_given_agree = total["fn"] / (total["fn"] + total["tn"]) if (total["fn"] + total["tn"]) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n================ 汇总 ================")
    print(f"样本数: {len(rows)}")
    print(f"总字符: {total['n_chars']}  错误字符: {n_err}  分歧字符: {n_dis}")
    print(f"精确率 (分歧处真错的占比): {precision:.3f}")
    print(f"召回率 (错误被分歧覆盖): {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"P(错|分歧) = {p_wrong_given_disagree:.3f}")
    print(f"P(错|不分歧) = {p_wrong_given_agree:.3f}")
    print(f"提升倍数 = {p_wrong_given_disagree / p_wrong_given_agree:.2f}x" if p_wrong_given_agree else "提升倍数 = inf（不分歧处零错误）")


if __name__ == "__main__":
    main()
