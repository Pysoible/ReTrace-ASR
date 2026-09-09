from asr_agent.memory import LongTermMemoryRepository, MemoryConsolidator, MemoryRetriever
from asr_agent.models import MemoryBelief, Session, Turn
from asr_agent.model_identity import ModelIdentity, model_memory_scope


def belief(belief_id: str, value: str, *, sessions: list[str], kinds: list[str]) -> MemoryBelief:
    return MemoryBelief(
        belief_id=belief_id,
        subject="speaker-a",
        predicate="name",
        value=value,
        confidence=0.92,
        source_turn_ids=["t1"],
        source_session_ids=sessions,
        evidence_kinds=kinds,
    )


def test_retriever_combines_recent_short_term_and_relevant_long_term(tmp_path):
    store = LongTermMemoryRepository(tmp_path)
    stable = belief("b1", "涂博士", sessions=["s0"], kinds=["audio_verified"])
    stable.status = "stable"
    store.save("team", [stable])
    session = Session(
        "s1",
        memory_scope="team",
        turns=[Turn(f"t{i}", "涂博士发言" if i == 11 else f"第{i}句", "涂博士发言" if i == 11 else f"第{i}句") for i in range(12)],
    )

    packet = MemoryRetriever(store, recent_limit=4).retrieve(session, session.turns[-1])

    assert [turn.turn_id for turn in packet.recent_turns] == ["t8", "t9", "t10", "t11"]
    assert [item.value for item in packet.long_term_beliefs] == ["涂博士"]


def test_consolidator_requires_independent_support_or_audio(tmp_path):
    store = LongTermMemoryRepository(tmp_path)
    consolidator = MemoryConsolidator(store, confidence_threshold=0.85)
    ordinary = belief("ordinary", "甲", sessions=["s1"], kinds=["context"])
    verified = belief("verified", "乙", sessions=["s1"], kinds=["audio_verified"])

    promoted = consolidator.consolidate("team", [ordinary, verified])

    assert [item.belief_id for item in promoted] == ["verified"]
    stored = {item.belief_id: item for item in store.load("team")}
    assert stored["verified"].status == "stable"
    assert stored["ordinary"].status == "superseded"


def test_long_term_store_failure_degrades_to_empty_memory(tmp_path):
    class BrokenStore(LongTermMemoryRepository):
        def load(self, scope: str):
            raise OSError("disk unavailable")

    session = Session("s", turns=[Turn("t1", "测试", "测试")])
    packet = MemoryRetriever(BrokenStore(tmp_path)).retrieve(session, session.turns[0])
    assert packet.long_term_beliefs == []


def test_repository_status_exposes_consolidation_failure(tmp_path, monkeypatch):
    store = LongTermMemoryRepository(tmp_path)

    def fail_save(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_save_unlocked", fail_save)

    try:
        store.update("model--test", lambda beliefs: None, operation="consolidate")
    except OSError:
        pass

    assert store.status("model--test") == {
        "scope": "model--test",
        "provisional": 0,
        "stable": 0,
        "superseded": 0,
        "last_operation": "consolidate",
        "error": {"kind": "OSError", "message": "disk unavailable"},
    }


def test_model_memory_scope_is_shared_across_audio_but_isolated_by_model():
    qwen = ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "first_pass")
    moss = ModelIdentity("moss-transcribe-diarize", "MOSS-Transcribe-Diarize", "first_pass")

    assert model_memory_scope(qwen, namespace="experiment-a") == model_memory_scope(
        qwen, namespace="experiment-a"
    )
    assert model_memory_scope(qwen, namespace="experiment-a") != model_memory_scope(
        moss, namespace="experiment-a"
    )


def test_consolidator_merges_independent_sessions_before_promotion(tmp_path):
    store = LongTermMemoryRepository(tmp_path)
    consolidator = MemoryConsolidator(store, confidence_threshold=0.85)

    assert consolidator.consolidate("team", [belief("b1", "涂博士", sessions=["s1"], kinds=["context"])]) == []
    promoted = consolidator.consolidate("team", [belief("b2", "涂博士", sessions=["s2"], kinds=["context"])])

    assert len(promoted) == 1
    assert promoted[0].status == "stable"
    assert set(promoted[0].source_session_ids) == {"s1", "s2"}


def test_retriever_includes_dependency_turns_and_filters_unrelated_long_term(tmp_path):
    store = LongTermMemoryRepository(tmp_path)
    relevant = belief("relevant", "涂博士", sessions=["s0"], kinds=["audio_verified"])
    unrelated = belief("unrelated", "完全无关", sessions=["s0"], kinds=["audio_verified"])
    relevant.status = unrelated.status = "stable"
    store.save("team", [unrelated, relevant])
    session = Session(
        "s1",
        memory_scope="team",
        turns=[
            Turn("t1", "负责人可能是图博士", "负责人可能是图博士"),
            Turn("t2", "后来确认负责人是涂博士", "后来确认负责人是涂博士"),
            Turn("t3", "继续开会", "继续开会"),
        ],
        dependency_index={"t1": ["t2"]},
    )

    packet = MemoryRetriever(store, recent_limit=1).retrieve(session, session.turns[1])

    assert [turn.turn_id for turn in packet.dependent_turns] == ["t1"]
    assert [item.belief_id for item in packet.long_term_beliefs] == ["relevant"]


def test_retriever_exposes_audio_verified_canonical_entity_to_agent(tmp_path):
    store = LongTermMemoryRepository(tmp_path)
    canonical = MemoryBelief(
        belief_id="champion",
        subject="turn:t1:canonical_entity",
        predicate="canonical_entity",
        value="卡兹克",
        aliases=["卡斯克"],
        confidence=0.9,
        status="provisional",
        source_turn_ids=["t1"],
        source_session_ids=["s1"],
        evidence_kinds=["context", "audio_verified"],
    )
    session = Session(
        "s1",
        turns=[Turn("t1", "卡兹克", "卡兹克"), Turn("t2", "卡斯克", "卡斯克")],
        working_beliefs={canonical.belief_id: canonical},
    )

    packet = MemoryRetriever(store).retrieve(session, session.turns[-1])

    assert [(item.value, item.aliases) for item in packet.canonical_entities] == [("卡兹克", ["卡斯克"])]


def test_hybrid_retriever_ranks_alias_match_above_unrelated_memory(tmp_path):
    store = LongTermMemoryRepository(tmp_path)
    canonical = belief("champion", "卡兹克", sessions=["s0"], kinds=["audio_verified"])
    canonical.aliases = ["卡斯克"]
    canonical.status = "stable"
    unrelated = belief("other", "和平精英", sessions=["s0"], kinds=["audio_verified"])
    unrelated.status = "stable"
    store.save("team", [unrelated, canonical])
    session = Session(
        "s1",
        memory_scope="team",
        turns=[Turn("t1", "这局卡斯克要怎么出装", "这局卡斯克要怎么出装")],
    )

    packet = MemoryRetriever(store).retrieve(session, session.turns[0])

    assert packet.long_term_beliefs[0].belief_id == "champion"


def test_hybrid_retriever_uses_character_ngram_overlap_without_exact_phrase(tmp_path):
    store = LongTermMemoryRepository(tmp_path)
    relevant = belief("relevant", "网络连接异常", sessions=["s0"], kinds=["audio_verified"])
    relevant.status = "stable"
    unrelated = belief("unrelated", "游戏角色技能", sessions=["s0"], kinds=["audio_verified"])
    unrelated.status = "stable"
    store.save("team", [unrelated, relevant])
    session = Session(
        "s1",
        memory_scope="team",
        turns=[Turn("t1", "网络连接不上怎么办", "网络连接不上怎么办")],
    )

    packet = MemoryRetriever(store).retrieve(session, session.turns[0])

    assert packet.long_term_beliefs[0].belief_id == "relevant"
