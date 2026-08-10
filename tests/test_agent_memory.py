from asr_agent.memory import LongTermMemoryRepository, MemoryConsolidator, MemoryRetriever
from asr_agent.models import MemoryBelief, Session, Turn


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
