from threading import Event, Lock

from asr_agent.realtime import RealtimeAnalysisCoordinator


def test_same_session_analysis_runs_in_submission_order():
    calls: list[str] = []
    lock = Lock()

    class Service:
        def analyze_turn(self, session_id: str, turn_id: str):
            with lock:
                calls.append(f"{session_id}:{turn_id}")

    coordinator = RealtimeAnalysisCoordinator(Service(), max_workers=2)
    try:
        coordinator.submit("s", "t1")
        coordinator.submit("s", "t2")
        coordinator.drain("s")
    finally:
        coordinator.close()

    assert calls == ["s:t1", "s:t2"]


def test_different_sessions_can_analyze_concurrently():
    entered = {"a": Event(), "b": Event()}
    release = Event()

    class Service:
        def analyze_turn(self, session_id: str, turn_id: str):
            del turn_id
            entered[session_id].set()
            release.wait(timeout=2)

    coordinator = RealtimeAnalysisCoordinator(Service(), max_workers=2)
    try:
        coordinator.submit("a", "t1")
        coordinator.submit("b", "t1")
        assert entered["a"].wait(timeout=1)
        assert entered["b"].wait(timeout=1)
        release.set()
        coordinator.drain()
    finally:
        coordinator.close()
