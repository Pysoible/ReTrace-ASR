from asr_agent.metrics import evaluate_revisions


def test_revision_metrics_measure_precision_overcorrection_and_latency():
    report = evaluate_revisions(
        events=[
            {"action": "REVISE_TEXT", "target_turn_id": "t1", "source_turn_id": "t3", "after_text": "涂博士来了"},
            {"action": "REVISE_TEXT", "target_turn_id": "t2", "source_turn_id": "t4", "after_text": "错误修订"},
        ],
        reference_by_turn={"t1": "涂博士来了", "t2": "正确原文"},
        turn_order=["t1", "t2", "t3", "t4"],
        ambiguous_turn_count=3,
    )

    assert report["revision_precision"] == 0.5
    assert report["overcorrection_rate"] == 0.5
    assert report["revision_coverage"] == 2 / 3
    assert report["mean_resolution_latency_turns"] == 2.0
