"""Text normalization, explicit character alignment, and edit replay."""
from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Iterable

from .schemas import CharacterEdit


def normalize_text(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text or "")).casefold()


def align_edits(raw: str, candidate: str) -> tuple[CharacterEdit, ...]:
    edits: list[CharacterEdit] = []
    for tag, left_start, left_end, right_start, right_end in SequenceMatcher(
        None, raw, candidate, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        kind = {"replace": "SUB", "delete": "INS", "insert": "DEL"}[tag]
        edits.append(
            CharacterEdit(
                kind=kind,
                start=left_start,
                end=left_end,
                replacement=candidate[right_start:right_end],
            )
        )
    return tuple(edits)


def apply_edits(raw: str, edits: Iterable[CharacterEdit]) -> str:
    result = raw
    previous_start = len(raw) + 1
    for edit in sorted(edits, key=lambda item: (item.start, item.end), reverse=True):
        if edit.end > len(raw) or edit.end > previous_start:
            raise ValueError("edit is outside text or overlaps another edit")
        result = result[: edit.start] + edit.replacement + result[edit.end :]
        previous_start = edit.start
    return result
