from asr_agent.analysis_windows import AnalysisWindowPolicy, partition_turns


def test_partition_turns_closes_on_first_reached_bound():
    turns = [
        {"turn_id": "t1", "text": "甲" * 40, "start_sec": 0.0, "end_sec": 8.0},
        {"turn_id": "t2", "text": "乙" * 40, "start_sec": 8.0, "end_sec": 16.0},
        {"turn_id": "t3", "text": "丙" * 40, "start_sec": 16.0, "end_sec": 31.0},
        {"turn_id": "t4", "text": "丁", "start_sec": 31.0, "end_sec": 32.0},
    ]
    policy = AnalysisWindowPolicy(max_turns=10, max_chars=180, max_audio_sec=30.0)

    groups = partition_turns(turns, policy)

    assert [[item["turn_id"] for item in group] for group in groups] == [
        ["t1", "t2"],
        ["t3", "t4"],
    ]


def test_partition_turns_keeps_single_oversized_turn():
    turns = [
        {"turn_id": "t1", "text": "甲" * 200, "start_sec": 0.0, "end_sec": 40.0},
        {"turn_id": "t2", "text": "乙", "start_sec": 40.0, "end_sec": 41.0},
    ]

    groups = partition_turns(
        turns,
        AnalysisWindowPolicy(max_turns=10, max_chars=180, max_audio_sec=30.0),
    )

    assert [[item["turn_id"] for item in group] for group in groups] == [["t1"], ["t2"]]


def test_default_policy_uses_meeting_scale_windows():
    policy = AnalysisWindowPolicy()

    assert (policy.max_turns, policy.max_chars, policy.max_audio_sec) == (20, 600, 90.0)
