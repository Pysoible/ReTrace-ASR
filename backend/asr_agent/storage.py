"""Atomic, versioned persistence for ReTrace sessions."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from threading import Lock, RLock
from typing import Callable

from asr_agent.models import Session


class VersionConflict(RuntimeError):
    """Raised when an optimistic commit targets a stale session version."""


class SessionRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, RLock] = {}
        self._locks_lock = Lock()

    def load(self, session_id: str) -> Session:
        lock = self._lock_for(session_id)
        with lock:
            return self._copy(self._load_unlocked(session_id))

    def create(self, session: Session) -> Session:
        session_id = self._validate_session_id(session.session_id)
        lock = self._lock_for(session_id)
        with lock:
            stored = self._copy(session)
            self._write_unlocked(stored)
            return self._copy(stored)

    def update(self, session_id: str, mutate: Callable[[Session], object]) -> Session:
        lock = self._lock_for(session_id)
        with lock:
            current = self._load_unlocked(session_id)
            previous_version = current.version
            mutate(current)
            if current.session_id != session_id:
                raise ValueError("mutation cannot change session_id")
            current.version = previous_version + 1
            self._write_unlocked(current)
            return self._copy(current)

    def commit(self, session: Session, expected_version: int) -> Session:
        session_id = self._validate_session_id(session.session_id)
        lock = self._lock_for(session_id)
        with lock:
            current = self._load_unlocked(session_id)
            if current.version != expected_version:
                raise VersionConflict(f"expected version {expected_version}, got {current.version}")
            session.version = expected_version + 1
            stored = self._copy(session)
            self._write_unlocked(stored)
            return self._copy(stored)

    def _lock_for(self, session_id: str) -> RLock:
        session_id = self._validate_session_id(session_id)
        with self._locks_lock:
            return self._locks.setdefault(session_id, RLock())

    def _path(self, session_id: str) -> Path:
        return self.root / f"{self._validate_session_id(session_id)}.json"

    def _load_unlocked(self, session_id: str) -> Session:
        path = self._path(session_id)
        if not path.exists():
            return Session(session_id)
        return Session.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_unlocked(self, session: Session) -> None:
        path = self._path(session.session_id)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{session.session_id}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                json.dump(session.as_dict(), temporary, ensure_ascii=False, indent=2)
                temporary_path = Path(temporary.name)
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _copy(session: Session) -> Session:
        return Session.from_dict(session.as_dict())

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
            raise ValueError("session_id must be a safe file identifier")
        return session_id
