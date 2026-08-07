from pathlib import Path


def test_paper_is_anonymous_and_describes_automatic_audio_verification():
    source = Path("paper/main.tex").read_text(encoding="utf-8")
    assert "\\documentclass[conference]{IEEEtran}" in source
    assert "Selective Retrospective Audio Verification" in source
    assert "human" not in source.lower()
    assert "RELISTEN" in source
    assert "XXX" not in source
