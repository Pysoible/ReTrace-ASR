#!/usr/bin/env python3
"""Summarize candidate discovery and revision provenance from completed sessions."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def candidate_pipeline_metrics(session: dict[str, Any]) -> dict[str, Any]:
    events = [event for event in session.get("revision_events") or [] if event.get("active", True)]
    candidate_events = [
        event for event in events
        if event.get("event_kind") == "candidate_audit"
    ]
    verifier_attempts = [
        event for event in candidate_events
        if "verifier_attempted:true" in (event.get("evidence") or [])
    ]
    verified = [event for event in verifier_attempts if any(str(item).startswith("audio_verified:") for item in event.get("evidence") or [])]
    committed = [event for event in candidate_events if event.get("candidate_stage") == "committed"]
    deferred = [event for event in candidate_events if event.get("candidate_stage") == "deferred"]
    rejected = [event for event in candidate_events if event.get("candidate_stage") == "rejected"]
    sources = Counter(
        str(item).split(":", 1)[1]
        for event in candidate_events
        for item in event.get("evidence") or []
        if str(item).startswith("candidate_source:")
    )
    return {
        "candidate_recall": None,
        "candidate_proposals": len(candidate_events),
        "validated_candidates": len(candidate_events),
        "verifier_attempts": len(verifier_attempts),
        "verified_candidates": len(verified),
        "candidate_to_verifier_rate": len(verifier_attempts) / len(candidate_events) if candidate_events else 0.0,
        "verified_candidate_rate": len(verified) / len(verifier_attempts) if verifier_attempts else 0.0,
        "committed_revision_count": len(committed),
        "deferred_candidate_count": len(deferred),
        "rejected_candidate_count": len(rejected),
        "candidate_sources": dict(sources),
        "note": "candidate_recall requires post-inference reference alignment and is intentionally not used online",
    }


def summarize_session(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session.get("session_id"),
        "turns": len(session.get("turns") or []),
        **candidate_pipeline_metrics(session),
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
