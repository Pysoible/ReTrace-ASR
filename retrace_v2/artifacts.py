"""Safe, reference-free inference artifact writer."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Iterable, Mapping


def _forbidden_reference_key(value: object) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).casefold()
            if "reference" in key_text or "ground_truth" in key_text or key_text == "gt":
                return str(key)
            nested = _forbidden_reference_key(child)
            if nested:
                return nested
    elif isinstance(value, (list, tuple)):
        for child in value:
            nested = _forbidden_reference_key(child)
            if nested:
                return nested
    return None


def _json_ready(value: object) -> object:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(_json_ready(key)): _json_ready(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


class ArtifactWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_manifest(self, payload: Mapping[str, object]) -> Path:
        forbidden = _forbidden_reference_key(payload)
        if forbidden:
            raise ValueError(f"reference data is forbidden in inference manifest: {forbidden}")
        return self._write_json("manifest.json", payload)

    def write_jsonl(self, filename: str, rows: Iterable[object]) -> Path:
        if Path(filename).name != filename or not filename.endswith(".jsonl"):
            raise ValueError("artifact filename must be a local .jsonl name")
        target = self.output_dir / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")
        temporary.replace(target)
        return target

    def _write_json(self, filename: str, payload: object) -> Path:
        target = self.output_dir / filename
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target
