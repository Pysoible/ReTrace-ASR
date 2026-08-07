from pathlib import Path


def test_workspace_uses_timeline_and_decision_explainer_without_legacy_terms():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "live transcript" in source
    assert "agent note" in source
    for forbidden in ("dictionary", "domain term", "tama", "tts", "evolve", "two-pass"):
        assert forbidden not in source


def test_workspace_uses_academic_dashboard_labels_and_example_empty_state():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "qwen-omni audio input" in source
    assert "research notebook" in source
    assert "revised" in source


def test_workspace_renders_live_subtitles_and_inline_revisions():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "live transcript" in source
    assert "revision-link" in source
    assert "incorrect" in source
    assert "corrected" in source


def test_workspace_has_audio_input_panel():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    lower = source.lower()
    assert "qwen-omni audio input" in lower
    assert "audio/upload" in lower
    assert 'type="file"' in source


def test_workspace_exposes_read_only_audit_without_human_controls():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "append-only audit" in source
    assert "method trace" in source
    assert "undo revision" not in source
    assert "/undo" not in source
    assert "人工复核" not in source
    assert "append-only audit" in source


def test_workspace_uses_research_notebook_structure_for_transcript_and_evidence():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "research notebook" in source
    assert "transcript-paper" in source
    assert "evidence-note" in source
    assert "method-rail" in source


def test_workspace_renders_uncertainty_candidates_and_autonomous_audio_evidence():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    assert "candidate-card" in source
    assert "evidence-packet" in source
    assert "Audio verification" in source
    assert "confirm-candidate" not in source
    assert "/hypotheses/" not in source


def test_workspace_visualizes_the_full_autonomous_decision_path():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    for label in ("OBSERVE", "SEMANTIC TRIGGER", "SELECTIVE RELISTEN", "DUAL-EVIDENCE GATE", "AUTONOMOUS DECISION"):
        assert label in source
