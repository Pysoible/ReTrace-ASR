import numpy as np
import soundfile as sf

from asr_agent.multiview import (
    apply_closed_set_replacements,
    closed_set_differences,
    circular_delay_and_sum,
    consensus_medoid,
    coverage_recovery_choice,
    detect_acoustic_windows,
    filter_semantic_replacement_candidates,
    strong_multiview_semantic_decision,
)


def test_consensus_medoid_prefers_supported_hypothesis_without_reference():
    hypotheses = {
        "baseline_anchor": "宣传代业",
        "multichannel": "宣传单页",
        "channel_average": "宣传单页",
        "channel_0": "宣传单页",
    }

    assert consensus_medoid(hypotheses) == "channel_0"


def test_circular_delay_and_sum_returns_finite_mono_audio():
    samples = np.zeros((800, 8), dtype=np.float32)
    samples[200, :] = 1.0

    output = circular_delay_and_sum(samples, 16000, 90.0)

    assert output.shape == (800,)
    assert np.isfinite(output).all()


def test_closed_set_differences_include_insertions_and_deletions_but_do_not_commit():
    hypotheses = {
        "baseline_anchor": "今天开会然后结束",
        "view_a": "今天下午开会结束",
        "view_b": "今天下午开会结束",
    }

    candidates = closed_set_differences(hypotheses["baseline_anchor"], hypotheses)

    assert {item["operation"] for item in candidates} == {"insert", "delete"}
    assert all(item["support"] == 2 for item in candidates)
    assert all(item["automatic_commit_allowed"] is False for item in candidates)


def test_semantic_candidate_filter_only_keeps_short_supported_cjk_replacements():
    candidates = [
        {
            "operation": "replace",
            "anchor_start": 2,
            "anchor_end": 4,
            "source_text": "时间",
            "candidate_text": "室内",
            "support": 3,
            "sources": ["beam_270", "channel_2", "channel_5"],
        },
        {
            "operation": "insert",
            "anchor_start": 4,
            "anchor_end": 4,
            "source_text": "",
            "candidate_text": "室内",
            "support": 5,
            "sources": ["a", "b", "c", "d", "e"],
        },
        {
            "operation": "replace",
            "anchor_start": 6,
            "anchor_end": 8,
            "source_text": "天气",
            "candidate_text": "weather",
            "support": 4,
            "sources": ["a", "b", "c", "d"],
        },
    ]

    selected = filter_semantic_replacement_candidates(candidates)

    assert len(selected) == 1
    assert selected[0]["candidate_id"] == "replace:2:4:时间:室内"
    assert selected[0]["automatic_commit_allowed"] is False


def test_apply_closed_set_replacements_uses_offsets_not_first_text_occurrence():
    anchor = "时间安排以后还是时间比较好"
    candidate = {
        "candidate_id": "replace:8:10:时间:室内",
        "operation": "replace",
        "anchor_start": 8,
        "anchor_end": 10,
        "source_text": "时间",
        "candidate_text": "室内",
    }

    result = apply_closed_set_replacements(anchor, [candidate])

    assert result["text"] == "时间安排以后还是室内比较好"
    assert result["applied_candidate_ids"] == [candidate["candidate_id"]]


def test_apply_closed_set_replacements_rejects_stale_or_overlapping_offsets():
    anchor = "还是时间比较好"
    stale = {
        "candidate_id": "stale",
        "operation": "replace",
        "anchor_start": 2,
        "anchor_end": 4,
        "source_text": "天气",
        "candidate_text": "室内",
    }

    result = apply_closed_set_replacements(anchor, [stale])

    assert result["text"] == anchor
    assert result["rejected"] == [{"candidate_id": "stale", "reason": "source_mismatch"}]


def test_strong_multiview_semantic_decision_accepts_diverse_clear_winner():
    winner = {
        "candidate_id": "before",
        "anchor_start": 3,
        "anchor_end": 5,
        "source_text": "这些",
        "candidate_text": "之前",
        "support": 5,
        "sources": ["beam_000", "beam_090", "beam_270", "channel_2", "channel_3"],
    }
    competitor = {
        **winner,
        "candidate_id": "first",
        "candidate_text": "就先",
        "support": 3,
        "sources": ["beam_180", "channel_average", "multichannel"],
    }

    decision = strong_multiview_semantic_decision(
        winner,
        [winner, competitor],
        {"selected": True, "confidence": 0.9},
    )

    assert decision["decision"] == "COMMIT_REPLACEMENT"
    assert decision["support_margin"] == 2
    assert decision["view_families"] == ["beam", "channel"]


def test_strong_multiview_semantic_decision_rejects_close_competitor():
    winner = {
        "candidate_id": "a",
        "anchor_start": 3,
        "anchor_end": 5,
        "source_text": "这些",
        "candidate_text": "之前",
        "support": 4,
        "sources": ["beam_000", "beam_090", "channel_2", "channel_3"],
    }
    competitor = {**winner, "candidate_id": "b", "candidate_text": "就先", "support": 3}

    decision = strong_multiview_semantic_decision(
        winner,
        [winner, competitor],
        {"selected": True, "confidence": 0.95},
    )

    assert decision["decision"] == "KEEP_BASELINE"
    assert "insufficient_support_margin" in decision["reasons"]


def test_detector_finds_speech_not_covered_by_asr(tmp_path):
    sampling_rate = 8000
    silence = np.zeros((sampling_rate, 2), dtype=np.float32)
    time = np.arange(sampling_rate * 2) / sampling_rate
    speech = np.stack(
        [0.2 * np.sin(2 * np.pi * 220 * time), 0.2 * np.sin(2 * np.pi * 337 * time)],
        axis=1,
    ).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])
    path = tmp_path / "meeting.wav"
    sf.write(path, audio, sampling_rate)

    result = detect_acoustic_windows(path, [], max_windows=4)

    assert result["kind"] == "gt_free_multichannel_energy_spatial_audit"
    assert result["uncovered_speech_frame_count"] > 0
    assert any(item["detector"] == "uncovered_speech" for item in result["windows"])


def test_coverage_policy_accepts_supported_addition_and_rejects_shortening():
    accepted = coverage_recovery_choice(
        {
            "baseline_anchor": "活动结束",
            "multichannel": "活动结束大家合影留念",
            "channel_average": "活动结束大家合影留念",
            "channel_0": "活动结束大家合影留念",
        }
    )
    rejected = coverage_recovery_choice(
        {
            "baseline_anchor": "活动结束大家合影留念",
            "multichannel": "活动结束",
            "channel_average": "活动结束",
            "channel_0": "活动结束",
        }
    )

    assert accepted["decision"] == "PROPOSE_COVERAGE_RECOVERY"
    assert accepted["selected_view"] != "baseline_anchor"
    assert accepted["automatic_commit_allowed"] is False
    assert rejected["decision"] == "KEEP_BASELINE"
    assert rejected["selected_view"] == "baseline_anchor"


def test_coverage_policy_rejects_content_already_present_in_neighboring_turn():
    result = coverage_recovery_choice(
        {
            "baseline_anchor": "老师们一个合影",
            "multichannel": "老师们一个惊喜吧合影留念然后合影",
            "channel_average": "老师们一个惊喜吧合影留念然后合影",
            "channel_0": "老师们一个惊喜吧合影留念然后合影",
        },
        neighbor_text="一个惊喜吧合影留念，然后进行互动，合影",
    )

    assert result["decision"] == "KEEP_BASELINE"
    assert result["selected_view"] == "baseline_anchor"
    assert result["insertion_candidates"]
    assert all(not item["accepted"] for item in result["insertion_candidates"])
