from pathlib import Path


def test_workspace_uses_timeline_and_decision_explainer_without_legacy_terms():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "live subtitles" in source
    assert "decision explainer" in source
    for forbidden in ("dictionary", "domain term", "tama", "tts", "proposal"):
        assert forbidden not in source


def test_workspace_uses_academic_dashboard_labels_and_example_empty_state():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "asr observation" in source
    assert "experimental session" in source
    assert "· revised" in source


def test_workspace_renders_live_subtitles_and_inline_revisions():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "live subtitles" in source
    assert "revision-link" in source
    assert "incorrect" in source
    assert "corrected" in source
