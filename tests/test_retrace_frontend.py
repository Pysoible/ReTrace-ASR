from pathlib import Path


def test_workspace_uses_timeline_and_decision_explainer_without_legacy_terms():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "live subtitles" in source
    assert "decision explainer" in source
    for forbidden in ("dictionary", "domain term", "tama", "tts", "evolve", "two-pass"):
        assert forbidden not in source


def test_workspace_uses_academic_dashboard_labels_and_example_empty_state():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "qwen-omni audio input" in source
    assert "evidence agent" in source
    assert "· revised" in source


def test_workspace_renders_live_subtitles_and_inline_revisions():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "live subtitles" in source
    assert "revision-link" in source
    assert "incorrect" in source
    assert "corrected" in source


def test_workspace_has_audio_input_panel():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    lower = source.lower()
    assert "qwen-omni audio input" in lower
    assert "audio/upload" in lower
    assert 'type="file"' in source


def test_workspace_exposes_reversible_audit_control():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "undo revision" in source
    assert "/undo" in source
    assert "append-only audit" in source
