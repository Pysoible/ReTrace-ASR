import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_prepare_cli_exports_only_raw_moss(tmp_path: Path) -> None:
    result_dir = tmp_path / "results/rec1"
    result_dir.mkdir(parents=True)
    wav_dir = tmp_path / "wav"
    wav_dir.mkdir()
    (wav_dir / "rec1.wav").write_bytes(b"audio-placeholder")
    payload = {
        "asr": {"backend": "moss", "model": "MOSS-Transcribe-Diarize"},
        "session": {
            "turns": [
                {
                    "turn_id": "t1",
                    "raw_text": "原文",
                    "current_text": "旧修改",
                    "meta": {"start_sec": 0.0, "end_sec": 1.0},
                }
            ]
        },
    }
    (result_dir / "retrace_result.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    output_dir = tmp_path / "prepared"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_retrace_v2_dev.py"),
            "--results-root",
            str(tmp_path / "results"),
            "--wav-dir",
            str(wav_dir),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    row = json.loads((output_dir / "raw_moss.jsonl").read_text(encoding="utf-8"))
    assert row["text"] == "原文"
    assert "reference" not in completed.stdout.casefold()
    assert not any("reference" in key for key in row)


def test_reference_cli_assigns_textgrid_after_raw_export(tmp_path: Path) -> None:
    raw = tmp_path / "raw_moss.jsonl"
    raw.write_text(
        json.dumps(
            {
                "recording_id": "rec1",
                "segment_id": "rec1:t1",
                "start_sec": 0.0,
                "end_sec": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    textgrid_dir = tmp_path / "TextGrid"
    textgrid_dir.mkdir()
    (textgrid_dir / "rec1.TextGrid").write_text(
        '''item [1]:
        name = "speaker_a"
        intervals [1]:
            xmin = 0.1
            xmax = 0.9
            text = "方案"
        ''',
        encoding="utf-8",
    )
    output = tmp_path / "reference.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_retrace_v2_reference.py"),
            "--raw",
            str(raw),
            "--textgrid-dir",
            str(textgrid_dir),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "segment_id": "rec1:t1",
        "reference_text": "方案",
    }
    assert json.loads(
        (tmp_path / "recording_reference.jsonl").read_text(encoding="utf-8")
    ) == {"recording_id": "rec1", "reference_text": "方案"}
