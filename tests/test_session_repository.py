import json
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import pytest

from asr_agent.models import Session, Turn
from asr_agent.retrace import ReTraceService
from asr_agent.storage import SessionRepository, VersionConflict


def test_stale_snapshot_commit_is_rejected_without_overwriting_newer_data(tmp_path):
    repo = SessionRepository(tmp_path)
    repo.create(Session("s"))
    stale = repo.load("s")

    repo.update("s", lambda current: current.turns.append(Turn("t1", "甲", "甲")))
    stale.turns.append(Turn("t2", "乙", "乙"))

    with pytest.raises(VersionConflict, match="expected version 0, got 1"):
        repo.commit(stale, expected_version=0)

    assert [turn.turn_id for turn in repo.load("s").turns] == ["t1"]


def test_update_increments_version_exactly_once(tmp_path):
    repo = SessionRepository(tmp_path)
    repo.create(Session("s", version=4))

    updated = repo.update("s", lambda current: current.turns.append(Turn("t1", "甲", "甲")))

    assert updated.version == 5
    assert repo.load("s").version == 5


def test_update_does_not_write_or_increment_when_mutation_raises(tmp_path):
    repo = SessionRepository(tmp_path)
    repo.create(Session("s"))

    def fail_after_mutation(current):
        current.turns.append(Turn("t1", "甲", "甲"))
        raise RuntimeError("mutation failed")

    with pytest.raises(RuntimeError, match="mutation failed"):
        repo.update("s", fail_after_mutation)

    unchanged = repo.load("s")
    assert unchanged.version == 0
    assert unchanged.turns == []


def test_failed_serialization_preserves_session_and_removes_temporary_file(tmp_path):
    repo = SessionRepository(tmp_path)
    repo.create(Session("s", turns=[Turn("t1", "甲", "甲")], version=4))
    original_json = (tmp_path / "s.json").read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        repo.update("s", lambda current: current.turns[0].meta.update(bad=object()))

    stored = repo.load("s")
    assert stored.version == 4
    assert stored.turns[0].meta == {}
    assert (tmp_path / "s.json").read_text(encoding="utf-8") == original_json
    assert list(tmp_path.glob(".s.*.tmp")) == []


def test_returned_objects_are_detached_from_persisted_data(tmp_path):
    repo = SessionRepository(tmp_path)
    created = repo.create(Session("s"))
    created.turns.append(Turn("created", "甲", "甲"))

    loaded = repo.load("s")
    loaded.turns.append(Turn("loaded", "乙", "乙"))

    updated = repo.update("s", lambda current: current.turns.append(Turn("stored", "丙", "丙")))
    updated.turns.append(Turn("updated", "丁", "丁"))

    assert [turn.turn_id for turn in repo.load("s").turns] == ["stored"]


def test_unicode_json_round_trips_without_ascii_escaping(tmp_path):
    repo = SessionRepository(tmp_path)
    repo.create(Session("unicode", turns=[Turn("t1", "涂博士", "涂博士")]))

    loaded = repo.load("unicode")
    raw_json = (tmp_path / "unicode.json").read_text(encoding="utf-8")

    assert loaded.turns[0].raw_text == "涂博士"
    assert "涂博士" in raw_json
    assert json.loads(raw_json)["turns"][0]["raw_text"] == "涂博士"


@pytest.mark.parametrize("session_id", ["../escape", "a/b", "a\\b", "two..dots"])
def test_invalid_session_ids_are_rejected(tmp_path, session_id):
    repo = SessionRepository(tmp_path)

    with pytest.raises(ValueError):
        repo.load(session_id)
    with pytest.raises(ValueError):
        repo.create(Session(session_id))
    with pytest.raises(ValueError):
        repo.update(session_id, lambda current: None)
    with pytest.raises(ValueError):
        repo.commit(Session(session_id), expected_version=0)


def test_service_writes_increment_version_but_reads_do_not(tmp_path):
    service = ReTraceService(tmp_path)

    first = service.process_turn("s", "t1", "甲")["session"]
    read = service.get_session("s")

    assert first["version"] == 1
    assert read["version"] == 1


def test_duplicate_turn_failure_does_not_increment_version(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn("s", "t1", "甲")

    with pytest.raises(ValueError, match="duplicate turn_id: t1"):
        service.process_turn("s", "t1", "乙")

    assert service.get_session("s")["version"] == 1


def test_concurrent_service_turns_are_not_lost(tmp_path):
    service = ReTraceService(tmp_path)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.process_turn, "s", "t1", "甲"),
            executor.submit(service.process_turn, "s", "t2", "乙"),
        ]
        for future in futures:
            future.result()

    session = service.get_session("s")
    assert {turn["turn_id"] for turn in session["turns"]} == {"t1", "t2"}
    assert session["version"] == 2


def test_repository_instances_share_atomic_updates_for_the_same_root(tmp_path):
    first_repo = SessionRepository(tmp_path)
    second_repo = SessionRepository(tmp_path)
    first_repo.create(Session("s"))

    def append_after_delay(repo, turn):
        def mutate(current):
            sleep(0.05)
            current.turns.append(turn)

        return repo.update("s", mutate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(append_after_delay, first_repo, Turn("t1", "甲", "甲")),
            executor.submit(append_after_delay, second_repo, Turn("t2", "乙", "乙")),
        ]
        for future in futures:
            future.result()

    session = first_repo.load("s")
    assert {turn.turn_id for turn in session.turns} == {"t1", "t2"}
    assert session.version == 2


def test_stale_commit_is_rejected_across_repository_instances(tmp_path):
    first_repo = SessionRepository(tmp_path)
    second_repo = SessionRepository(tmp_path)
    first_repo.create(Session("s"))
    stale = first_repo.load("s")

    second_repo.update("s", lambda current: current.turns.append(Turn("t1", "甲", "甲")))

    with pytest.raises(VersionConflict, match="expected version 0, got 1"):
        first_repo.commit(stale, expected_version=0)
