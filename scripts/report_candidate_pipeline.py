#!/usr/bin/env python3
"""Summarize candidate discovery and revision provenance from completed sessions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    events = [event for event in session.get("revision_events") or [] if event.get("active", True)]
    candidate_events = [
        event for event in events
        if any(str(item).startswith("candidate_source:") for item in event.get("evidence") or [])
    ]
    verified = [
        event for event in candidate_events
        if any(str(item).startswith("audio_verified:") for item in event.get("evidence") or [])
    ]
    committed = [
        event for event in candidate_events
        if event.get("action") in {"REVISE_CURRENT", "REVISE_HISTORY", "ROLLBACK"}
    ]
    sources = Counter(
        str(item).split(":", 1)[1]
        for event in candidate_events
        for item in event.get("evidence") or []
        if str(item).startswith("candidate_source:")
    )
    return {
        "session_id": session.get("session_id"),
        "turns": len(session.get("turns") or []),
        "candidate_proposals": len(candidate_events),
        "candidate_to_verifier_rate": len(verified) / len(candidate_events) if candidate_events else 0.0,
        "verified_candidate_rate": len(committed) / len(verified) if verified else 0.0,
        "committed_revision_count": len(committed),
        "candidate_sources": dict(sources),
        "candidate_recall": None,
        "note": "candidate_recall requires offline reference alignment and is not used online",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.session_dir.glob("*.json")):
        try:
            rows.append(summarize_session(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    print(json.dumps({"sessions": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
