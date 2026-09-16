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
    assert "analysis_status" in source
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


def test_workspace_uses_transcript_and_evidence_structure():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8").lower()
    assert "transcript-paper" in source
    assert "evidence-note" in source
    assert "method-rail" in source


def test_workspace_renders_uncertainty_candidates_and_autonomous_audio_evidence():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    assert "candidate-card" in source
    assert "evidence-packet" in source
    assert "TARGETED RELISTEN" in source
    assert "confirm-candidate" not in source
    assert "/hypotheses/" not in source


def test_workspace_visualizes_the_full_autonomous_decision_path():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    for label in ("OBSERVE", "MEMORY RETRIEVE", "CONTEXT JUDGE", "TARGETED RELISTEN", "EVENT REPLAY"):
        assert label in source


def test_workspace_exposes_dual_timescale_memory_and_belief_provenance():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    for label in ("SHORT-TERM", "LONG-TERM", "WORKING BELIEFS", "STABLE BELIEFS", "Source turns", "Supersedes"):
        assert label in source
    assert "observability" in source


def test_workspace_shows_all_agent_decisions_relationships_and_versions():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    for label in (
        "KEEP_OLD", "ACCEPT_NEW", "COEXIST", "DEFER", "ROLLBACK",
        "MUTUALLY_EXCLUSIVE", "TEMPORAL_CHANGE", "Observed v", "Analyzed v",
    ):
        assert label in source


def test_workspace_removes_stale_demo_and_degenerate_prompt_copy():
    source = Path("frontend/src/main.ts").read_text(encoding="utf-8")
    assert "R0015" not in source
    assert "遥遥遥" not in source
    assert "Hear it twice" not in source


def test_method_trace_step_number_keeps_centering_after_generic_span_rule():
    styles = Path("frontend/src/styles.css").read_text(encoding="utf-8")
    assert ".method-trace .stage-index" in styles
    centered_rule = styles.split(".method-trace .stage-index", 1)[1].split("}", 1)[0]
    assert "display: grid" in centered_rule
    assert "place-items: center" in centered_rule
