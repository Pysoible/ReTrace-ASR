from scripts.run_aishell4_eval import (
    clean,
    classify_gt_error,
    evaluate_pair_payloads,
    aggregate_results,
    parse_reference,
    resolve_reference_path,
    retrace_miss_analysis,
)
from scripts.export_aishell4_errors import _group_for_display, _strip_time


def test_clean_excludes_chunk_timestamp_from_cer():
    assert clean("[41.8-56.5] 学校<sil>表现") == "学校表现"


def test_gt_taxonomy_routes_omission_to_audio_actions_without_oracle_revision():
    result = classify_gt_error(
        "看看咱这用的话业主们怎么停车呀",
        "看看咱这用的话",
        assigned_rows=[{"spk": "S01"}],
        later_reference="",
    )

    assert result["primary_error_type"] == "omission_or_undercoverage"
    assert "audio_action_eligible" in result["eligibility"]
    assert result["recommended_actions"][:3] == ["EXPAND_WINDOW", "RESEGMENT", "REDECODE"]


def test_gt_taxonomy_marks_repeated_later_span_as_context_candidate():
    result = classify_gt_error(
        "再去雕琢再去细化",
        "再去交流再去细化",
        assigned_rows=[{"spk": "S01"}],
        later_reference="这个方案还需要雕琢",
    )

    assert "acoustic_substitution" in result["error_types"]
    assert "later_evidence_candidate" in result["error_types"]
    assert result["later_supported_spans"] == ["雕琢"]
    assert "context_action_candidate" in result["eligibility"]


def test_gt_taxonomy_keeps_multi_speaker_assignment_separate_from_text_agent():
    result = classify_gt_error(
        "甲乙丙丁",
        "甲乙",
        assigned_rows=[{"spk": "S01"}, {"spk": "S02"}],
        later_reference="",
    )

    assert result["primary_error_type"] == "segmentation_or_speaker_assignment"
    assert "alignment_only" in result["eligibility"]
    assert "REALIGN_SPEAKER_BOUNDARIES" in result["recommended_actions"]


def test_gt_taxonomy_marks_reference_interval_crossing_asr_turns_as_alignment_only():
    result = classify_gt_error(
        "我这儿男装这儿说两句啊男装销量不理想",
        "男装销量不理想",
        assigned_rows=[{"spk": "S01", "_asr_turn_overlap_count": 2}],
        later_reference="",
    )

    assert result["primary_error_type"] == "segmentation_or_speaker_assignment"
    assert "alignment_only" in result["eligibility"]


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


def test_parse_aishell4_textgrid_reference(tmp_path):
    reference = tmp_path / "sample.TextGrid"
    reference.write_text(
        '''
        item [1]:
            class = "IntervalTier"
            name = "S01"
            intervals [1]:
                xmin = 0
                xmax = 1.2
                text = "大家好"
            intervals [2]:
                xmin = 1.2
                xmax = 2.0
                text = ""
        item [2]:
            class = "IntervalTier"
            name = "S02"
            intervals [1]:
                xmin = 2.1
                xmax = 3.0
                text = "开始开会"
        ''',
        encoding="utf-8",
    )

    text, rows = parse_reference(reference)

    assert text == "大家好开始开会"
    assert rows == [
        {"start": 0.0, "end": 1.2, "spk": "S01", "text": "大家好"},
        {"start": 2.1, "end": 3.0, "spk": "S02", "text": "开始开会"},
    ]


def test_resolve_reference_path_prefers_txt_then_textgrid(tmp_path):
    (tmp_path / "a.TextGrid").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")

    assert resolve_reference_path(tmp_path, "a") == tmp_path / "a.TextGrid"
    assert resolve_reference_path(tmp_path, "b") == tmp_path / "b.txt"


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
        "revision_events": [
            {"active": True, "event_kind": "candidate_audit", "action": "REVISE_CURRENT", "candidate_id": "c1", "candidate_stage": "committed", "evidence": ["candidate_source:semantic_open", "verifier_attempted:true", "audio_verified:1.0/0.0"]},
            {"active": True, "event_kind": "candidate_audit", "action": "DEFER", "candidate_id": "c2", "candidate_stage": "deferred", "evidence": ["candidate_source:semantic_open", "verifier_attempted:true"]},
            {"active": True, "event_kind": "candidate_audit", "action": "DEFER", "candidate_id": "c3", "candidate_stage": "rejected", "evidence": ["candidate_source:history_homophone", "verifier_attempted:false"]},
        ],
    }

    analysis = retrace_miss_analysis(session, [])

    assert analysis["candidate_pipeline"] == {
        "candidate_recall": None,
        "candidate_proposals": 3,
        "validated_candidates": 3,
        "verifier_attempts": 2,
        "verified_candidates": 1,
        "candidate_to_verifier_rate": 2 / 3,
        "verified_candidate_rate": 1 / 2,
        "committed_revision_count": 1,
        "deferred_candidate_count": 1,
        "rejected_candidate_count": 1,
        "note": "candidate_recall requires post-inference reference alignment and is intentionally not used online",
    }


def test_paired_payloads_report_primary_effectiveness_metrics():
    baseline = {
        "first_pass_artifact_id": "artifact-1",
        "transcript": "图博士发言",
        "asr": {"ok": True, "completeness": {"truncated": False}},
        "stage_timings_ms": {"first_pass_asr": 1000.0},
    }
    retrace = {
        "first_pass_artifact_id": "artifact-1",
        "asr": {"ok": True, "completeness": {"truncated": False}},
        "stage_timings_ms": {"context_judge": 20.0},
        "session": {
            "memory_scope": "model--moss",
            "turns": [{
                "turn_id": "t001", "raw_text": "图博士发言", "current_text": "涂博士发言",
                "meta": {"start_sec": 0.0, "end_sec": 1.0},
            }],
            "revision_events": [
                {"active": True, "event_kind": "revision", "action": "REVISE_CURRENT", "target_turn_id": "t001", "before_text": "图博士发言", "after_text": "涂博士发言"},
                {"active": True, "event_kind": "candidate_audit", "candidate_stage": "committed", "evidence": ["verifier_attempted:true", "audio_verified:0.9/0.1"]},
            ],
            "observability": {"memory_status": {"error": None}},
        },
    }
    rows = [{"start": 0.0, "end": 1.0, "spk": "S01", "text": "涂博士发言"}]

    item = evaluate_pair_payloads(
        baseline,
        retrace,
        reference="涂博士发言",
        rows=rows,
        integrations={"deepseek": {"ready": True}},
    )

    assert item["valid"] is True
    assert item["first_pass_artifact_id"] == "artifact-1"
    assert item["raw"] == item["baseline"]
    assert item["effectiveness"]["ecer"] == 1.0
    assert item["effectiveness"]["revision_precision"] == 1.0
    assert item["effectiveness"]["candidate_to_correction_yield"] == 1.0
    assert item["effectiveness"]["error_type_reduction"] == {
        "substitutions": {"net_removed": 1, "relative": 1.0},
        "insertions": {"net_removed": 0, "relative": 0.0},
        "deletions": {"net_removed": 0, "relative": 0.0},
    }
    assert item["retrace"]["committed_revisions"] == 1


def test_paired_payloads_reject_mismatched_incomplete_or_unready_runs():
    baseline = {
        "first_pass_artifact_id": "artifact-a",
        "transcript": "甲",
        "asr": {"ok": True, "completeness": {"truncated": False}},
    }
    retrace = {
        "first_pass_artifact_id": "artifact-b",
        "asr": {"ok": False, "completeness": {"truncated": True}},
        "session": {"turns": [{"turn_id": "t1", "raw_text": "乙", "current_text": "乙", "meta": {}}]},
    }

    item = evaluate_pair_payloads(
        baseline,
        retrace,
        reference="甲",
        rows=[],
        integrations={"deepseek": {"ready": False}},
    )

    assert item["valid"] is False
    assert "first_pass_artifact_mismatch" in item["invalid_reasons"]
    assert "incomplete_first_pass" in item["invalid_reasons"]
    assert "deepseek_not_ready" in item["invalid_reasons"]
    assert "raw_transcript_mismatch" in item["invalid_reasons"]
    assert item["effectiveness"] is None


def test_aggregate_uses_only_valid_pooled_counts():
    valid = {
        "sample": "ok", "valid": True, "invalid_reasons": [],
        "baseline": {"edits": 10, "reference_chars": 20}, "final": {"edits": 6, "reference_chars": 20},
        "retrace": {"revision_quality": {"committed": 4, "improved": 3}, "candidate_funnel": {"validated_candidates": 6}},
        "stage_timings_ms": {"context_judge": 100.0},
        "stage_call_counts": {"context_judge": 2},
    }
    invalid = {
        "sample": "bad", "valid": False, "invalid_reasons": ["deepseek_not_ready"],
        "baseline": {"edits": 100}, "final": {"edits": 0},
    }

    result = aggregate_results([valid, invalid])

    assert result["valid_samples"] == 1
    assert result["invalid_samples"] == 1
    assert result["effectiveness"] == {
        "ecer": 0.4,
        "baseline_cer": 0.5,
        "final_cer": 0.3,
        "cer_absolute_change": -0.2,
        "cer_relative_reduction": 0.4,
        "revision_precision": 0.75,
        "harmful_revision_rate": 0.0,
        "candidate_to_correction_yield": 0.5,
        "error_type_reduction": {
            "substitutions": {"net_removed": 0, "relative": 0.0},
            "insertions": {"net_removed": 0, "relative": 0.0},
            "deletions": {"net_removed": 0, "relative": 0.0},
        },
    }
    assert result["stage_timings_ms"] == {"context_judge": 100.0}
