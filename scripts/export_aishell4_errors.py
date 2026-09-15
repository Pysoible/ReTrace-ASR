#!/usr/bin/env python3
"""Small helpers for rendering AI-SHELL-4 error rows for human review."""
from __future__ import annotations

import re
from typing import Any


def _strip_time(text: str) -> str:
    return re.sub(r"^\[\d+(?:\.\d+)?[-~]\d+(?:\.\d+)?\]\s*", "", text or "")


def _group_for_display(rows: list[dict[str, Any]], *, window_sec: float = 60.0) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for row in rows:
        start = float(row.get("start_sec") or 0.0)
        end = float(row.get("end_sec") or start)
        if grouped and start - grouped[-1]["timestamp"]["start_sec"] <= window_sec:
            current = grouped[-1]
            current["raw"] += _strip_time(str(row.get("raw_text") or ""))
            current["retrace"] += _strip_time(str(row.get("final_text") or ""))
            current["gt"] += str(row.get("ground_truth_text") or "")
            current["timestamp"]["end_sec"] = end
            continue
        grouped.append({
            "raw": _strip_time(str(row.get("raw_text") or "")),
            "retrace": _strip_time(str(row.get("final_text") or "")),
            "gt": str(row.get("ground_truth_text") or ""),
            "timestamp": {"start_sec": start, "end_sec": end},
        })
    return grouped
