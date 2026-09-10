"""Safe subprocess adapter for an external multichannel separator."""
from __future__ import annotations

from pathlib import Path
import subprocess
import time
import uuid

from ..tools import ToolFailure


class CommandSeparator:
    def __init__(
        self,
        command: list[str],
        *,
        backend_id: str,
        output_root: Path,
        timeout_sec: float = 300.0,
    ) -> None:
        if not command:
            raise ValueError("separation command is required")
        self.command = tuple(command)
        self.backend_id = backend_id
        self.output_root = output_root
        self.timeout_sec = timeout_sec

    def separate(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
    ) -> tuple[Path, ...]:
        output_dir = self.output_root / f"{audio_path.stem}-{uuid.uuid4().hex[:12]}"
        output_dir.mkdir(parents=True, exist_ok=False)
        values = {
            "audio": str(audio_path),
            "start": str(start_sec),
            "end": str(end_sec),
            "output": str(output_dir),
        }
        command = [token.format_map(values) for token in self.command]
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ToolFailure(f"{self.backend_id} separation failed: {error}") from error
        elapsed = time.perf_counter() - started
        if completed.returncode:
            detail = completed.stderr.strip()[-500:]
            raise ToolFailure(
                f"{self.backend_id} separation failed after {elapsed:.2f}s: {detail}"
            )
        outputs = tuple(sorted(output_dir.rglob("*.wav")))
        if not outputs:
            raise ToolFailure(f"{self.backend_id} separation produced no wav files")
        return outputs
