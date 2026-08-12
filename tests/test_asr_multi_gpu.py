from asr_agent.integrations.qwen_asr import parse_gpu_ids, shard_indices


def test_parse_gpu_ids_supports_comma_and_semicolon():
    assert parse_gpu_ids("0") == ["0"]
    assert parse_gpu_ids("0,1") == ["0", "1"]
    assert parse_gpu_ids("0; 1") == ["0", "1"]
    assert parse_gpu_ids("") == ["0"]


def test_shard_indices_round_robin_across_workers():
    assert shard_indices(5, 2) == [[0, 2, 4], [1, 3]]
    assert shard_indices(4, 2) == [[0, 2], [1, 3]]
    assert shard_indices(3, 1) == [[0, 1, 2]]
    assert shard_indices(0, 2) == [[], []]
