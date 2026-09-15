"""Background coordination for realtime session analysis."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any


class RealtimeAnalysisCoordinator:
    """Serialize work per session while allowing sessions to run in parallel."""

    def __init__(self, service: Any, *, max_workers: int = 4) -> None:
        self.service = service
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="retrace-analysis")
        self._tails: dict[str, Future[Any]] = {}
        self._lock = Lock()
        self._closed = False

    def submit(self, session_id: str, turn_id: str, *, observed_version: int | None = None) -> Future[Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("realtime coordinator is closed")
            previous = self._tails.get(session_id)
            future = self.executor.submit(self._run_after, previous, session_id, turn_id, observed_version)
            self._tails[session_id] = future
            return future

    def _run_after(
        self,
        previous: Future[Any] | None,
        session_id: str,
        turn_id: str,
        observed_version: int | None,
    ) -> Any:
        if previous is not None:
            try:
                previous.result()
            except Exception:
                # One malformed turn must not permanently block this session.
                pass
        if observed_version is None:
            return self.service.analyze_turn(session_id, turn_id)
        return self.service.analyze_turn(session_id, turn_id, observed_version=observed_version)

    def drain(self, session_id: str | None = None) -> None:
        with self._lock:
            futures = [self._tails[session_id]] if session_id in self._tails else [] if session_id else list(self._tails.values())
        for future in futures:
            future.result()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.executor.shutdown(wait=True)
