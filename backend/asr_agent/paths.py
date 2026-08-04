from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    # backend/asr_agent/paths.py -> parents[2] == project root
    return Path(__file__).resolve().parents[2]


def frontend_root() -> Path:
    return project_root() / "frontend"


def default_workspace_path() -> Path:
    configured = os.getenv("ASR_AGENT_WORKSPACE")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".asr_agent" / "workspaces" / "default").resolve()
