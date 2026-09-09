"""Load the sibling ASR_domainterms repo the same way ASR_agent does."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ModelArts sibling default (ASR_agent ships a macOS path; override here).
DEFAULT_DOMAINTERMS_ROOT = (
    Path(__file__).resolve().parents[4] / "ASR_domainterms"
)


def domainterms_root() -> Path:
    configured = os.getenv("ASR_DOMAINTERMS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_DOMAINTERMS_ROOT.resolve()


def _load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_path)
        return
    except Exception:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def ensure_importable() -> Path:
    """Put ASR_domainterms on sys.path and load its .env (DeepSeek WS settings)."""
    root = domainterms_root()
    if not root.exists():
        raise RuntimeError(
            f"ASR_domainterms 仓库不存在: {root}。设置 ASR_DOMAINTERMS_ROOT 指向正确路径。"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    _load_env(root / ".env")
    return root
