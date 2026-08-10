from asr_agent.models import (
    EvidenceRef,
    MemoryBelief,
    RevisionEvent,
    Session,
    WorkingHypothesis,
)


def test_legacy_session_loads_with_versioned_memory_defaults():
    session = Session.from_dict({
        "session_id": "s1",
        "turns": [{"turn_id": "t1", "raw_text": "泰信", "current_text": "泰信"}],
        "verified_memory": {},
        "quarantine_memory": {},
        "revision_events": [],
    })

    assert session.version == 0
    assert session.memory_scope == "default"
    assert session.open_hypotheses == {}
    assert Session.from_dict(session.as_dict()).as_dict() == session.as_dict()


def test_working_hypothesis_keeps_competing_interpretations():
    hypothesis = WorkingHypothesis(
        hypothesis_id="h1",
        target_turn_ids=["t1"],
        current_interpretation="泰信",
        proposed_interpretation="泰康",
        alternatives=["泰信", "泰康"],
    )

    assert hypothesis.current_interpretation == "泰信"
    assert hypothesis.proposed_interpretation == "泰康"
    assert hypothesis.alternatives == ["泰信", "泰康"]
    assert hypothesis.status == "active"


def test_versioned_session_deserializes_nested_models():
    session = Session.from_dict({
        "session_id": "s1",
        "version": 3,
        "working_beliefs": {
            "b1": {
                "belief_id": "b1",
                "subject": "provider",
                "predicate": "name",
                "value": "泰康",
                "aliases": ["泰信"],
                "source_turn_ids": ["t1"],
            },
        },
        "open_hypotheses": {
            "h1": {
                "hypothesis_id": "h1",
                "target_turn_ids": ["t1"],
                "current_interpretation": "泰信",
                "proposed_interpretation": "泰康",
                "alternatives": ["泰信", "泰康"],
                "supporting_evidence": [
                    {"turn_id": "t2", "kind": "text", "value": "泰康", "score": 0.9},
                ],
                "contradicting_evidence": [
                    {"turn_id": "t1", "kind": "asr", "value": "泰信"},
                ],
            },
        },
        "revision_events": [{
            "event_id": "e1",
            "action": "REVISE_TEXT",
            "target_turn_id": "t1",
            "source_turn_id": "t2",
            "span": "泰信",
            "before_text": "泰信",
            "after_text": "泰康",
            "entity_id": None,
            "score": 0.9,
            "evidence": ["t2:泰康"],
            "resolver": "test",
            "session_version": 3,
            "supersedes_event_id": "e0",
        }],
    })

    assert isinstance(session.working_beliefs["b1"], MemoryBelief)
    assert isinstance(session.open_hypotheses["h1"], WorkingHypothesis)
    assert isinstance(session.open_hypotheses["h1"].supporting_evidence[0], EvidenceRef)
    assert isinstance(session.open_hypotheses["h1"].contradicting_evidence[0], EvidenceRef)
    assert isinstance(session.revision_events[0], RevisionEvent)
    assert session.revision_events[0].session_version == 3
    assert session.revision_events[0].supersedes_event_id == "e0"
    assert Session.from_dict(session.as_dict()).as_dict() == session.as_dict()
