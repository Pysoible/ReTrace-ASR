from asr_agent.correction_candidates import CorrectionCandidate, deduplicate_candidates


def test_deduplicate_candidates_merges_sources_and_evidence():
    candidates = [
        CorrectionCandidate("t1", "南庄", "男装", "history_homophone", ["t0"]),
        CorrectionCandidate("t1", "南庄", "男装", "acoustic_diff", ["t1"]),
    ]

    result = deduplicate_candidates(candidates)

    assert len(result) == 1
    assert result[0].evidence_turn_ids == ["t0", "t1"]
    assert result[0].source == "history_homophone|acoustic_diff"
