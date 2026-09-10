from retrace_v2.candidates import candidates_from_hypothesis
from retrace_v2.text import apply_edits, normalize_text


def test_candidates_cover_substitution() -> None:
    candidates = candidates_from_hypothesis("方按", "方案", source="paraformer")

    assert {edit.kind for edit in candidates[1].edits} == {"SUB"}
    assert apply_edits("方按", candidates[1].edits) == "方案"


def test_candidates_cover_baseline_insertion() -> None:
    candidates = candidates_from_hypothesis("这个个方案", "这个方案", source="paraformer")

    assert "INS" in {edit.kind for edit in candidates[1].edits}
    assert apply_edits("这个个方案", candidates[1].edits) == "这个方案"


def test_candidates_cover_baseline_deletion() -> None:
    candidates = candidates_from_hypothesis("这个案", "这个方案", source="paraformer")

    assert "DEL" in {edit.kind for edit in candidates[1].edits}
    assert apply_edits("这个案", candidates[1].edits) == "这个方案"


def test_raw_candidate_is_always_first() -> None:
    candidates = candidates_from_hypothesis("原文", "候选", source="paraformer")

    assert candidates[0].candidate_text == "原文"
    assert candidates[0].edits == ()


def test_identical_normalized_text_has_no_edit_candidate() -> None:
    candidates = candidates_from_hypothesis("方案。", "方案", source="paraformer")

    assert len(candidates) == 1
    assert normalize_text(candidates[0].candidate_text) == "方案"
