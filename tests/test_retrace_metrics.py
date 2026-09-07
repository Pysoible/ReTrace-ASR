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


def test_revision_metrics_support_new_actions_and_automatic_rollback():
    report = evaluate_revisions(
        events=[
            {
                "event_id": "bad",
                "action": "REVISE_HISTORY",
                "target_turn_id": "t1",
                "source_turn_id": "t2",
                "after_text": "错误",
            },
            {
                "event_id": "undo",
                "action": "ROLLBACK",
                "target_turn_id": "t1",
                "source_turn_id": "t3",
                "after_text": "正确",
                "supersedes_event_id": "bad",
            },
            {
                "event_id": "good",
                "action": "REVISE_CURRENT",
                "target_turn_id": "t2",
                "source_turn_id": "t2",
                "after_text": "修正后",
            },
        ],
        reference_by_turn={"t1": "正确", "t2": "修正后"},
        turn_order=["t1", "t2", "t3"],
        ambiguous_turn_count=2,
    )

    assert report["committed_revisions"] == 1
    assert report["automatic_rollbacks"] == 1
    assert report["revision_precision"] == 1.0
    assert report["revision_recall"] == 0.5
    assert report["revision_f2"] > report["revision_recall"]


def test_revision_metrics_ignore_candidate_audit_events():
    report = evaluate_revisions(
        events=[
            {
                "event_id": "candidate-1",
                "event_kind": "candidate_audit",
                "action": "REVISE_CURRENT",
                "target_turn_id": "t1",
                "source_turn_id": "t1",
                "after_text": "修正后",
            },
            {
                "event_id": "revision-1",
                "event_kind": "revision",
                "action": "REVISE_CURRENT",
                "target_turn_id": "t1",
                "source_turn_id": "t1",
                "after_text": "修正后",
            },
        ],
        reference_by_turn={"t1": "修正后"},
        turn_order=["t1"],
        ambiguous_turn_count=1,
    )

    assert report["committed_revisions"] == 1
    assert report["correct_revisions"] == 1


def test_revision_metrics_stratify_historical_and_current_routes():
    report = evaluate_revisions(
        events=[
            {
                "event_id": "history-good",
                "action": "REVISE_HISTORY",
                "target_turn_id": "t1",
                "source_turn_id": "t3",
                "after_text": "历史正确",
                "evidence": ["audio:/tmp/a.wav:0-2"],
            },
            {
                "event_id": "current-bad",
                "action": "REVISE_CURRENT",
                "target_turn_id": "t2",
                "source_turn_id": "t2",
                "after_text": "当前错误",
                "evidence": ["acoustic_disagreement:词"],
            },
            {
                "event_id": "rollback-good",
                "action": "ROLLBACK",
                "target_turn_id": "t4",
                "source_turn_id": "t5",
                "after_text": "回滚正确",
            },
        ],
        reference_by_turn={"t1": "历史正确", "t2": "当前正确", "t4": "回滚正确"},
        turn_order=["t1", "t2", "t3", "t4", "t5"],
        ambiguous_turn_count=2,
    )

    assert report["historical_revision_count"] == 1
    assert report["current_revision_count"] == 1
    assert report["historical_revision_precision"] == 1.0
    assert report["current_revision_precision"] == 0.0
    assert report["historical_resolution_latency_turns"] == 2.0
    assert report["current_overcorrection_rate"] == 1.0
    assert report["rollback_success_rate"] == 1.0
