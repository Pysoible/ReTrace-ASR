import argparse
import json
from pathlib import Path

from asr_agent.retrace import ReTraceService


def main() -> None:
    parser = argparse.ArgumentParser(prog="retrace-asr")
    parser.add_argument("session_id")
    parser.add_argument("turn_id")
    parser.add_argument("text")
    args = parser.parse_args()
    print(json.dumps(ReTraceService(Path.cwd() / "retrace_state").process_turn(args.session_id, args.turn_id, args.text), ensure_ascii=False, indent=2))
