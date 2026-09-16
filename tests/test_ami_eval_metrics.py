import importlib.util
import json
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_ami_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_ami_eval", _SCRIPT)
assert _SPEC and _SPEC.loader
ami_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ami_eval)


def test_ami_diagnostic_report_marks_overlap_and_official_status(tmp_path):
    result = {
        "session": {
            "turns": [
                {"turn_id": "t1", "raw_text": "hello there", "current_text": "hello there", "meta": {"start_sec": 0.0, "end_sec": 2.0}},
            ],
            "revision_events": [],
        }
    }
    result_path = tmp_path / "result.json"
    stm_path = tmp_path / "sample.stm"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    stm_path.write_text(
        "\n".join(
            [
                "EN2002a 1 A 0.0 1.5 hello there",
                "EN2002a 1 B 0.5 2.0 other speaker",
            ]
        ),
        encoding="utf-8",
    )

    report = ami_eval.evaluate(result_path, stm_path, "time_sorted", None)

    assert report["metric_kind"] == "diagnostic_chunk_aligned"
    assert report["official_scoring"] is False
    assert report["ambiguous_turn_count"] == 1
    assert report["eligible_turn_count"] == 0
    assert report["unscorable_turn_count"] == 1
