import json

import pytest

from asr_agent.ledger import RevisionLedger
from asr_agent.models import RevisionEvent, Session, Turn


def revision_event(
    event_id,
    *,
    action="REVISE_HISTORY",
    target_turn_id="t1",
    source_turn_id="t2",
    span="泰信方案",
    after_text="泰康方案",
    replacement="泰康方案",
    supersedes_event_id=None,
    active=True,
):
    return RevisionEvent(
        event_id=event_id,
        action=action,
        target_turn_id=target_turn_id,
        source_turn_id=source_turn_id,
        span=span,
        before_text=span,
        after_text=after_text,
        entity_id=None,
        score=0.9,
        evidence=[source_turn_id],
        resolver="test",
        active=active,
        replacement=replacement,
        supersedes_event_id=supersedes_event_id,
    )


def test_rollback_restores_raw_text_without_mutating_prior_event():
    ledger = RevisionLedger()
    session = Session("s", turns=[Turn("t1", "泰信方案", "泰信方案")])
    revision = revision_event("e1")

    ledger.append(session, revision, event_version=4)
    rollback = ledger.rollback(
        session,
        "e1",
        source_turn_id="t3",
        reason="new evidence contradicted the revision",
        event_version=5,
    )

    assert session.turns[0].raw_text == "泰信方案"
    assert session.turns[0].current_text == "泰信方案"
    assert revision.active is True
    assert rollback.action == "ROLLBACK"
    assert rollback.supersedes_event_id == "e1"
    assert rollback.resolver == "automatic-rollback"


def test_append_many_assigns_one_event_version_without_changing_session_version():
    ledger = RevisionLedger()
    session = Session(
        "s",
        turns=[
            Turn("t1", "泰信方案", "泰信方案"),
            Turn("t2", "旧地址", "旧地址"),
        ],
        version=12,
    )
    events = [
        revision_event("e1"),
        revision_event(
            "e2",
            target_turn_id="t2",
            span="旧地址",
            after_text="新地址",
            replacement="新地址",
        ),
    ]

    appended = ledger.append_many(session, events, event_version=13)

    assert appended == events
    assert [event.session_version for event in appended] == [13, 13]
    assert session.version == 12


def test_duplicate_event_id_is_idempotent():
    ledger = RevisionLedger()
    session = Session("s", turns=[Turn("t1", "泰信方案", "泰信方案")])
    original = revision_event("e1")

    ledger.append(session, original, event_version=1)
    duplicate_result = ledger.append_many(
        session,
        [revision_event("e1", after_text="错误方案", replacement="错误方案")],
        event_version=2,
    )

    assert duplicate_result == []
    assert session.revision_events == [original]
    assert session.turns[0].current_text == "泰康方案"


def test_superseding_chain_only_applies_final_active_revision():
    ledger = RevisionLedger()
    session = Session("s", turns=[Turn("t1", "泰信方案", "泰信方案")])
    first = revision_event("e1", after_text="泰康方案", replacement="泰康方案")
    second = revision_event(
        "e2",
        after_text="泰安方案",
        replacement="泰安方案",
        supersedes_event_id="e1",
    )
    final = revision_event(
        "e3",
        after_text="泰和方案",
        replacement="泰和方案",
        supersedes_event_id="e2",
    )

    ledger.append_many(session, [first, second, final], event_version=7)

    assert [event.event_id for event in ledger.active_events(session)] == ["e3"]
    assert session.turns[0].current_text == "泰和方案"


def test_json_round_trip_replays_to_the_same_text():
    ledger = RevisionLedger()
    session = Session("s", turns=[Turn("t1", "泰信方案", "泰信方案")])
    ledger.append(session, revision_event("e1"), event_version=3)
    expected = session.turns[0].current_text

    restored = Session.from_dict(json.loads(json.dumps(session.as_dict(), ensure_ascii=False)))
    restored.turns[0].current_text = "corrupted projection"
    ledger.replay(restored)

    assert restored.turns[0].raw_text == "泰信方案"
    assert restored.turns[0].current_text == expected


@pytest.mark.parametrize(
    "target_id,existing_events",
    [
        ("missing", []),
        ("e1", [revision_event("e1", action="ROLLBACK")]),
        (
            "e1",
            [revision_event("e1"), revision_event("e2", action="ROLLBACK", supersedes_event_id="e1")],
        ),
    ],
)
def test_invalid_rollback_is_rejected(target_id, existing_events):
    ledger = RevisionLedger()
    session = Session(
        "s",
        turns=[Turn("t1", "泰信方案", "泰信方案")],
        revision_events=existing_events,
    )

    with pytest.raises(ValueError):
        ledger.rollback(session, target_id, source_turn_id="t3", reason="invalid", event_version=4)
