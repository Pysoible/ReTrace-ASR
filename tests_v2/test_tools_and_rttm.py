from pathlib import Path

import pytest

from retrace_v2.adapters.rttm import overlap_windows, parse_rttm
from retrace_v2.tools import EvidenceASR, Separator


def test_predicted_rttm_yields_overlap(tmp_path: Path) -> None:
    path = tmp_path / "predicted.rttm"
    path.write_text(
        "SPEAKER rec 1 1.0 2.0 <NA> <NA> spk0 <NA> <NA>\n"
        "SPEAKER rec 1 2.0 2.0 <NA> <NA> spk1 <NA> <NA>\n",
        encoding="utf-8",
    )

    turns = parse_rttm(path, expected_recording="rec")

    assert overlap_windows(turns) == [(2.0, 3.0)]


def test_adjacent_turns_do_not_count_as_overlap(tmp_path: Path) -> None:
    path = tmp_path / "predicted.rttm"
    path.write_text(
        "SPEAKER rec 1 0.0 1.0 <NA> <NA> spk0 <NA> <NA>\n"
        "SPEAKER rec 1 1.0 1.0 <NA> <NA> spk1 <NA> <NA>\n",
        encoding="utf-8",
    )

    assert overlap_windows(parse_rttm(path, expected_recording="rec")) == []


def test_rttm_rejects_wrong_recording(tmp_path: Path) -> None:
    path = tmp_path / "predicted.rttm"
    path.write_text(
        "SPEAKER other 1 0 1 <NA> <NA> spk0 <NA> <NA>\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="recording"):
        parse_rttm(path, expected_recording="rec")


def test_tool_contracts_are_runtime_checkable() -> None:
    assert getattr(EvidenceASR, "_is_runtime_protocol", False)
    assert getattr(Separator, "_is_runtime_protocol", False)
