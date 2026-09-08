from asr_agent.model_identity import ModelIdentity, model_memory_scope


def test_unknown_moss_model_is_not_inferred_from_qwen_configuration():
    identity = ModelIdentity.from_asr_result({"backend": "moss-transcribe-diarize"})

    assert identity.backend == "moss-transcribe-diarize"
    assert identity.model == "unknown"
    assert identity.family == "moss"


def test_same_model_shares_memory_scope_but_models_are_isolated():
    moss = ModelIdentity("moss-transcribe-diarize", "MOSS-Audio-7B", "first_pass")
    omni = ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "first_pass")

    assert model_memory_scope(moss, namespace="eval") == model_memory_scope(moss, namespace="eval")
    assert model_memory_scope(moss, namespace="eval") != model_memory_scope(omni, namespace="eval")


def test_roles_do_not_split_memory_for_the_same_concrete_model():
    first_pass = ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "first_pass")
    verifier = ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "targeted_verifier")

    assert model_memory_scope(first_pass, namespace="eval") == model_memory_scope(verifier, namespace="eval")
