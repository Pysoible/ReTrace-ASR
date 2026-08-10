"""Atomic, versioned persistence for ReTrace sessions."""
from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Callable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - Unix and macOS provide fcntl.
    fcntl = None

from asr_agent.models import Session


_LOCK_REGISTRY: dict[tuple[Path, str], RLock] = {}
_LOCK_REGISTRY_GUARD = Lock()


class VersionConflict(RuntimeError):
    """Raised when an optimistic commit targets a stale session version."""


class SessionRepository:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock_root = self.root / ".locks"
        self._lock_root.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> Session:
        with self.exclusive(session_id):
            return self._copy(self._load_unlocked(session_id))

    def create(self, session: Session) -> Session:
        session_id = self._validate_session_id(session.session_id)
        with self.exclusive(session_id):
            stored = self._copy(session)
            self._write_unlocked(stored)
            return self._copy(stored)

    def update(self, session_id: str, mutate: Callable[[Session], object]) -> Session:
        with self.exclusive(session_id):
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
        with self.exclusive(session_id):
            current = self._load_unlocked(session_id)
            if current.version != expected_version:
                raise VersionConflict(f"expected version {expected_version}, got {current.version}")
            session.version = expected_version + 1
            stored = self._copy(session)
            self._write_unlocked(stored)
            return self._copy(stored)

    @contextmanager
    def exclusive(self, session_id: str) -> Iterator[None]:
        """Hold this session's process and filesystem guards without exposing state."""
        session_id = self._validate_session_id(session_id)
        with self._lock_for(session_id):
            if fcntl is None:
                # Non-Unix platforms retain cross-instance safety within this process.
                yield
                return

            lock_path = self._lock_root / f"{session_id}.lock"
            with lock_path.open("a+b") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _lock_for(self, session_id: str) -> RLock:
        session_id = self._validate_session_id(session_id)
        key = (self.root, session_id)
        with _LOCK_REGISTRY_GUARD:
            return _LOCK_REGISTRY.setdefault(key, RLock())

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
