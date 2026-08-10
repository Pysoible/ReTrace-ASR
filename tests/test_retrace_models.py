import json

import pytest

from asr_agent.models import (
    EvidenceRef,
    MemoryBelief,
    RevisionEvent,
    Session,
    WorkingHypothesis,
)
from asr_agent.retrace import ReTraceService, Session as ExportedSession


def _revision_event_dict(**overrides):
    value = {
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
    }
    value.update(overrides)
    return value


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
    serialized = json.loads(json.dumps(session.as_dict(), ensure_ascii=False))
    assert Session.from_dict(serialized).as_dict() == session.as_dict()


def test_retrace_exports_current_session_model():
    assert ExportedSession is Session


def test_service_saves_and_reloads_json_session(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn("s1", "t1", "泰康")

    saved_json = json.loads((tmp_path / "s1.json").read_text())
    reloaded = ReTraceService(tmp_path).get_session("s1")

    assert saved_json == reloaded
    assert reloaded["turns"][0]["raw_text"] == "泰康"


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
                "created_version": 2,
                "updated_version": 3,
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
            **_revision_event_dict(),
            "session_version": 3,
            "supersedes_event_id": "e0",
        }],
    })

    assert isinstance(session.working_beliefs["b1"], MemoryBelief)
    assert session.working_beliefs["b1"].created_version == 2
    assert session.working_beliefs["b1"].updated_version == 3
    assert isinstance(session.open_hypotheses["h1"], WorkingHypothesis)
    assert isinstance(session.open_hypotheses["h1"].supporting_evidence[0], EvidenceRef)
    assert isinstance(session.open_hypotheses["h1"].contradicting_evidence[0], EvidenceRef)
    assert isinstance(session.revision_events[0], RevisionEvent)
    assert session.revision_events[0].session_version == 3
    assert session.revision_events[0].supersedes_event_id == "e0"
    serialized = json.loads(json.dumps(session.as_dict(), ensure_ascii=False))
    assert Session.from_dict(serialized).as_dict() == session.as_dict()


def test_evidence_ref_converts_non_null_score_to_float():
    evidence = EvidenceRef.from_dict({
        "turn_id": "t1",
        "kind": "text",
        "value": "泰康",
        "score": "0.9",
    })

    assert evidence.score == 0.9
    assert isinstance(evidence.score, float)


def test_revision_event_rejects_non_boolean_active():
    with pytest.raises(ValueError, match="active"):
        RevisionEvent.from_dict(_revision_event_dict(active="false"))


def test_revision_event_handles_boolean_and_legacy_active_values():
    assert RevisionEvent.from_dict(_revision_event_dict(active=False)).active is False
    assert RevisionEvent.from_dict(_revision_event_dict()).active is True
