"""Append-only revision events and deterministic session text projection."""
from __future__ import annotations

from collections.abc import Iterable
from uuid import uuid4

from asr_agent.models import RevisionEvent, Session


REVISION_ACTIONS = {
    "REVISE_CURRENT",
    "REVISE_HISTORY",
    "REVISE_TEXT",
    "REVISE_ENTITY",
}


class RevisionLedger:
    def append_many(
        self,
        session: Session,
        events: Iterable[RevisionEvent],
        event_version: int,
    ) -> list[RevisionEvent]:
        known_ids = {event.event_id for event in session.revision_events}
        appended: list[RevisionEvent] = []
        for event in events:
            if event.event_id in known_ids:
                continue
            event.session_version = event_version
            session.revision_events.append(event)
            appended.append(event)
            known_ids.add(event.event_id)

        self.replay(session)
        return appended

    def append(
        self,
        session: Session,
        event: RevisionEvent,
        event_version: int,
    ) -> RevisionEvent:
        appended = self.append_many(session, [event], event_version=event_version)
        if appended:
            return appended[0]
        return next(existing for existing in session.revision_events if existing.event_id == event.event_id)

    def rollback(
        self,
        session: Session,
        event_id: str,
        source_turn_id: str,
        reason: str,
        event_version: int,
    ) -> RevisionEvent:
        target = next((event for event in session.revision_events if event.event_id == event_id), None)
        superseded_ids = {
            event.supersedes_event_id
            for event in session.revision_events
            if event.active and event.supersedes_event_id is not None
        }
        if (
            target is None
            or not target.active
            or target.action not in REVISION_ACTIONS
            or target.event_id in superseded_ids
        ):
            raise ValueError(f"event {event_id!r} is not an active revision target")

        rollback = RevisionEvent(
            event_id=f"rollback-{uuid4().hex}",
            action="ROLLBACK",
            target_turn_id=target.target_turn_id,
            source_turn_id=source_turn_id,
            span=target.span,
            before_text=target.after_text,
            after_text=target.before_text,
            entity_id=target.entity_id,
            score=target.score,
            evidence=[target.event_id],
            resolver="automatic-rollback",
            reason=reason,
            supersedes_event_id=target.event_id,
        )
        return self.append(session, rollback, event_version=event_version)

    def replay(self, session: Session) -> None:
        turns_by_id = {turn.turn_id: turn for turn in session.turns}
        for turn in session.turns:
            turn.current_text = turn.raw_text

        for event in self.active_events(session):
            if event.event_kind != "revision" or event.action not in REVISION_ACTIONS:
                continue
            try:
                turn = turns_by_id[event.target_turn_id]
            except KeyError as error:
                raise ValueError(f"unknown target turn: {event.target_turn_id}") from error

            if event.span and event.replacement and event.span in turn.current_text:
                turn.current_text = turn.current_text.replace(event.span, event.replacement, 1)
            else:
                turn.current_text = event.after_text

    def active_events(self, session: Session) -> list[RevisionEvent]:
        superseded_ids = {
            event.supersedes_event_id
            for event in session.revision_events
            if event.active and event.supersedes_event_id is not None
        }
        return [
            event
            for event in session.revision_events
            if event.active and event.event_id not in superseded_ids
        ]
