import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_inference_has_no_reference_option_and_scorer_is_separate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_retrace_v2.py"),
            "--raw",
            str(ROOT / "tests_v2/fixtures/raw_moss.jsonl"),
            "--evidence",
            str(ROOT / "tests_v2/fixtures/evidence.jsonl"),
            "--output",
            str(run_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr
    help_text = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_retrace_v2.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.casefold()
    assert "reference" not in help_text
    assert not (run_dir / "scoring").exists()
    final_row = json.loads(
        (run_dir / "final_hypothesis.jsonl").read_text(encoding="utf-8")
    )
    assert final_row["segment_id"] == "s1"
    assert final_row["region_id"] == "r1"

    score = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/score_retrace_v2.py"),
            "--run",
            str(run_dir),
            "--reference",
            str(ROOT / "tests_v2/fixtures/reference.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert score.returncode == 0, score.stderr
    metrics = json.loads((run_dir / "scoring/metrics.json").read_text(encoding="utf-8"))
    assert metrics["raw"]["edits"] == 1
    assert metrics["final"]["edits"] == 0
    assert metrics["oracle_candidate"]["edits"] == 0


def test_inference_rejects_mixed_evidence_model_identity(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        (ROOT / "tests_v2/fixtures/evidence.jsonl").read_text(encoding="utf-8")
        + '{"segment_id":"s1","hypothesis_id":"h2","model_id":"OMNI","view":"original","text":"方案","start_sec":0,"end_sec":1,"acoustic_score":0.9}\n',
        encoding="utf-8",
    )

    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_retrace_v2.py"),
            "--raw",
            str(ROOT / "tests_v2/fixtures/raw_moss.jsonl"),
            "--evidence",
            str(evidence),
            "--output",
            str(tmp_path / "run"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert run.returncode != 0
    assert "one evidence model" in run.stderr
