from asr_agent.context_judge import FocusProposal
from asr_agent.correction_candidates import CorrectionCandidate, candidate_to_focus, deduplicate_candidates, focus_to_candidate
from asr_agent.models import Session, Turn


def test_deduplicate_candidates_merges_sources_and_evidence():
    candidates = [
        CorrectionCandidate("t1", "南庄", "男装", "history_homophone", ["t0"]),
        CorrectionCandidate("t1", "南庄", "男装", "acoustic_diff", ["t1"]),
    ]

    result = deduplicate_candidates(candidates)

    assert len(result) == 1
    assert result[0].evidence_turn_ids == ["t0", "t1"]
    assert result[0].source == "history_homophone|acoustic_diff"


def test_focus_without_history_candidate_is_inferred_as_semantic_open():
    turn = Turn("t1", "我负责南庄部门", "我负责南庄部门")
    focus = FocusProposal("t1", "南庄", "男装", ["南庄", "男装"], ["t1"])

    candidate = focus_to_candidate(focus, Session("s", turns=[turn]))

    assert candidate.source == "semantic_open"


def test_focus_with_candidate_in_another_turn_is_inferred_as_history_homophone():
    target = Turn("t2", "我负责南庄部门", "我负责南庄部门")
    focus = FocusProposal("t2", "南庄", "男装", ["南庄", "男装"], ["t1"])
    session = Session("s", turns=[Turn("t1", "男装业务", "男装业务"), target])

    candidate = focus_to_candidate(focus, session)

    assert candidate.source == "history_homophone"


def test_focus_round_trip_preserves_all_closed_set_alternatives():
    turn = Turn("t1", "我负责南庄部门", "我负责南庄部门")
    focus = FocusProposal(
        "t1",
        "南庄",
        "男装",
        ["南庄", "男装", "南章"],
        ["t1"],
    )

    restored = candidate_to_focus(focus_to_candidate(focus, Session("s", turns=[turn])))

    assert restored.alternatives == ["南庄", "男装", "南章"]


def test_deduplicate_candidates_merges_closed_set_alternatives():
    first = CorrectionCandidate(
        "t1", "南庄", "男装", "semantic_open", ["t1"],
        alternatives=["南庄", "男装"],
    )
    second = CorrectionCandidate(
        "t1", "南庄", "男装", "acoustic_diff", ["t2"],
        alternatives=["南庄", "男装", "南章"],
    )

    result = deduplicate_candidates([first, second])

    assert result[0].alternatives == ["南庄", "男装", "南章"]
