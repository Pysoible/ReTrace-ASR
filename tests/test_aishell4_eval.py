from scripts.run_aishell4_eval import clean, retrace_miss_analysis
from scripts.export_aishell4_errors import _group_for_display, _strip_time


def test_clean_excludes_chunk_timestamp_from_cer():
    assert clean("[41.8-56.5] 学校<sil>表现") == "学校表现"


def test_error_report_strips_chunk_timestamp_from_transcript_fields():
    assert _strip_time("[41.8-56.5] 学校表现") == "学校表现"


def test_error_report_groups_short_turns_for_human_review():
    rows = [
        {"sample": "x", "start_sec": 0.0, "end_sec": 12.0, "raw_text": "甲", "final_text": "甲", "ground_truth_text": "乙"},
        {"sample": "x", "start_sec": 12.0, "end_sec": 24.0, "raw_text": "丙", "final_text": "丙", "ground_truth_text": "丁"},
    ]

    grouped = _group_for_display(rows, window_sec=60.0)

    assert grouped == [{
        "raw": "甲丙",
        "retrace": "甲丙",
        "gt": "乙丁",
        "timestamp": {"start_sec": 0.0, "end_sec": 24.0},
    }]


def test_miss_analysis_distinguishes_missing_signal_from_acoustic_failure():
    session = {
        "turns": [
            {
                "turn_id": "t1",
                "raw_text": "[0.0-2.0] 学着表现",
                "current_text": "[0.0-2.0] 学着表现",
                "meta": {"start_sec": 0.0, "end_sec": 2.0, "uncertainty": {}},
            },
            {
                "turn_id": "t2",
                "raw_text": "[2.0-4.0] 欢迎",
                "current_text": "[2.0-4.0] 欢迎",
                "meta": {
                    "start_sec": 2.0,
                    "end_sec": 4.0,
                    "uncertainty": {"acoustic_error": "RuntimeError: unavailable"},
                },
            },
        ]
    }
    rows = [
        {"start": 0.0, "end": 2.0, "spk": "A", "text": "学校表现"},
        {"start": 2.0, "end": 4.0, "spk": "A", "text": "欢饮"},
    ]

    analysis = retrace_miss_analysis(session, rows)

    assert analysis["categories"] == {
        "missed_no_online_signal": 1,
        "missed_acoustic_failure": 1,
    }
    assert analysis["purpose"] == "offline_gt_diagnostics_only"


def test_miss_analysis_counts_unreferenced_expansion_as_harmful_revision():
    session = {
        "turns": [{
            "turn_id": "t1",
            "raw_text": "[0.0-12.0] 嗯。",
            "current_text": "[0.0-12.0] 嗯。好，我们来看一下。",
            "meta": {"start_sec": 0.0, "end_sec": 12.0, "uncertainty": {}},
        }]
    }

    analysis = retrace_miss_analysis(session, [])

    assert analysis["categories"] == {"revision_harmed": 1}
    assert analysis["revision_quality"]["harmed"] == 1
    assert analysis["revision_quality"]["harm_rate"] == 1.0
    assert analysis["net_edits_removed"] < 0


def test_miss_analysis_reports_candidate_pipeline_metrics():
    session = {
        "turns": [],
        "revision_events": [{
            "active": True,
            "action": "REVISE_CURRENT",
            "evidence": ["candidate_source:semantic_open", "audio_verified:1.0/0.0"],
        }],
    }

    analysis = retrace_miss_analysis(session, [])

    assert analysis["candidate_pipeline"] == {
        "candidate_recall": None,
        "candidate_proposals": 1,
        "candidate_to_verifier_rate": 1.0,
        "verified_candidate_rate": 1.0,
        "committed_revision_count": 1,
        "note": "candidate_recall requires post-inference reference alignment and is intentionally not used online",
    }
