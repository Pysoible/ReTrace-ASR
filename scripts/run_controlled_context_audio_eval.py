#!/usr/bin/env python3
"""Controlled context-plus-audio recovery probe on real AISHELL-4 windows."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from uuid import uuid4


CASES = [
    {
        "name": "title_homophone",
        "audio": "L_R003S03C02.flac",
        "start_sec": 41.25,
        "end_sec": 50.64,
        "context": ["总经理刚才介绍方案", "仍由总经理负责"],
        "injected": "行吧我是刚才总经里也说了",
        "expected": "行吧我是刚才总经理也说了",
    },
    {
        "name": "passage_homophone",
        "audio": "L_R003S04C02.flac",
        "start_sec": 90.0,
        "end_sec": 115.0,
        "context": ["消防安全通道需要检查", "保持消防安全通道畅通"],
        "injected": "消防器材要巡查消防安全通到咱都得检查",
        "expected": "消防器材要巡查消防安全通道咱都得检查",
    },
    {
        "name": "clearance_homophone",
        "audio": "S_R004S03C01.flac",
        "start_sec": 544.74,
        "end_sec": 550.44,
        "context": ["这个清仓的菜仍然能吃", "下午会处理清仓的菜"],
        "injected": "他们来看看有没有出仓的清苍的菜",
        "expected": "他们来看看有没有出仓的清仓的菜",
    },
]


def _post(base_url: str, session_id: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{base_url}/api/sessions/{session_id}/turns",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _session(base_url: str, session_id: str) -> dict:
    with urllib.request.urlopen(f"{base_url}/api/sessions/{session_id}", timeout=30) as response:
        return json.load(response)["session"]


def _settle(base_url: str, session_id: str, turn_count: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    latest: dict = {}
    while time.monotonic() < deadline:
        time.sleep(0.25)
        latest = _session(base_url, session_id)
        if len(latest.get("turns", [])) >= turn_count and latest.get("analysis_status") not in {
            "pending", "running", "queued"
        }:
            return latest
    raise TimeoutError(f"session {session_id} did not settle in {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    rows = []
    for case in CASES:
        session_id = f"controlled_{case['name']}_{uuid4().hex[:8]}"
        for index, text in enumerate(case["context"], 1):
            _post(args.base_url, session_id, {"turn_id": f"t{index}", "text": text})
            _settle(args.base_url, session_id, index, args.timeout)
        started = time.monotonic()
        _post(args.base_url, session_id, {
            "turn_id": "t3",
            "text": case["injected"],
            "audio_path": str(args.wav_dir / case["audio"]),
            "start_sec": case["start_sec"],
            "end_sec": case["end_sec"],
        })
        session = _settle(args.base_url, session_id, 3, args.timeout)
        elapsed = time.monotonic() - started
        final = session["turns"][2]["current_text"]
        revisions = [
            event for event in session.get("revision_events", [])
            if event.get("source_turn_id") == "t3"
            and event.get("event_kind") == "revision"
            and event.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY"}
        ]
        rows.append({
            **case,
            "session_id": session_id,
            "final": final,
            "recovered": final == case["expected"],
            "elapsed_sec": round(elapsed, 3),
            "committed": len(revisions),
            "revisions": revisions,
        })

    recovered = sum(row["recovered"] for row in rows)
    committed = sum(row["committed"] for row in rows)
    result = {
        "protocol": "controlled_same_pronunciation_substitution_on_real_aishell4_audio",
        "n": len(rows),
        "recovered": recovered,
        "recovery_rate": recovered / len(rows),
        "committed": committed,
        "revision_precision": recovered / committed if committed else None,
        "harmful_edits": sum(not row["recovered"] and row["committed"] for row in rows),
        "mean_target_turn_sec": sum(row["elapsed_sec"] for row in rows) / len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
