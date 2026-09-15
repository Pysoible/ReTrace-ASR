from scripts.preflight_moss_eval import parse_vllm_profile, validate_sample


def test_parse_vllm_profile_reads_required_limits():
    profile = parse_vllm_profile(
        environ={
            "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "1024",
            "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600",
        },
        argv=["vllm", "serve", "/models/MOSS", "--max-num-batched-tokens", "32768"],
    )

    assert profile.max_upload_mb == 1024
    assert profile.max_duration_sec == 3600
    assert profile.max_num_batched_tokens == 32768


def test_sample_validation_identifies_each_insufficient_bound():
    sample = {"size_mb": 273.0, "duration_sec": 2363.0, "estimated_audio_tokens": 29538}

    failures = validate_sample(
        sample,
        parse_vllm_profile(
            environ={"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "100", "VLLM_MAX_AUDIO_DECODE_DURATION_S": "600"},
            argv=["vllm", "serve", "/models/MOSS", "--max-num-batched-tokens", "2048"],
        ),
    )

    assert {item["code"] for item in failures} == {
        "audio_upload_limit_insufficient",
        "audio_duration_limit_insufficient",
        "encoder_capacity_insufficient",
    }


def test_selected_profile_accepts_long_aishell4_sample():
    profile = parse_vllm_profile(
        environ={"VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "1024", "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600"},
        argv=["vllm", "serve", "/models/MOSS", "--max-num-batched-tokens", "32768"],
    )

    assert validate_sample(
        {"size_mb": 273.0, "duration_sec": 2363.0, "estimated_audio_tokens": 29538},
        profile,
    ) == []
