# Complete MOSS ReTrace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce complete, model-pure MOSS baseline/ReTrace runs with bounded DeepSeek calls, MOSS-local candidate verification, durable first-pass reuse, and auditable AISHELL-4 metrics.

**Architecture:** Keep first-pass completeness and protocol handling in the MOSS adapter; add pure window scheduling and first-pass artifact modules instead of expanding `server.py` further. The selected ASR backend supplies a complete tool bundle so MOSS can only invoke MOSS verification/relisten functions, while the evaluator reuses one immutable complete first pass for baseline and ReTrace.

**Tech Stack:** Python 3.10+, FastAPI/Pydantic, urllib/OpenAI-compatible transcription API, NumPy, SoundFile optional MOSS extra, pypinyin, JSON file repositories, pytest.

---

## Execution constraints

- Work on `feat/open-candidate-fast-retrace-20260907`.
- Preserve unrelated changes and untracked `.superpowers/`; never use reset or checkout to discard them.
- Use `apply_patch` for edits and add only named files for each commit.
- Never print, log, commit, or transmit `DEEPSEEK_API_KEY` outside the configured DeepSeek request.
- Never use AISHELL-4 TextGrid text in online inference.
- Do not claim improvement until baseline and ReTrace score the same complete cached first pass.

### Task 1: Checkpoint the verified model-isolation prerequisite

**Files:**
- Modify: `backend/asr_agent/model_identity.py`
- Modify: `backend/asr_agent/server.py`
- Create: `backend/asr_agent/integrations/asr_backend.py`
- Create: `backend/asr_agent/integrations/moss_asr.py`
- Modify: `scripts/run_aishell4_eval.py`
- Test: `tests/test_agent_memory.py`
- Test: `tests/test_aishell4_eval.py`
- Test: `tests/test_asr_backend.py`
- Test: `tests/test_context_resolution.py`
- Test: `tests/test_model_identity.py`
- Test: `tests/test_moss_asr.py`
- Test: `tests/test_retrace_api.py`
- Test: `tests/test_retrace_service.py`

- [ ] **Step 1: Review the existing prerequisite diff**

Run:

```bash
git diff -- backend/asr_agent/model_identity.py backend/asr_agent/server.py \
  backend/asr_agent/integrations/asr_backend.py backend/asr_agent/integrations/moss_asr.py \
  scripts/run_aishell4_eval.py tests/test_agent_memory.py tests/test_aishell4_eval.py \
  tests/test_asr_backend.py tests/test_context_resolution.py tests/test_model_identity.py \
  tests/test_moss_asr.py tests/test_retrace_api.py tests/test_retrace_service.py
```

Expected: only model identity isolation, model-scoped memory, MOSS routing,
TextGrid/FLAC evaluation support, timing/audit tests, and their tests are present.

- [ ] **Step 2: Run the verified prerequisite suites**

Run:

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_deepseek_context_judge.py tests/test_retrace_api.py \
  tests/test_retrace_service.py tests/test_agent_memory.py \
  tests/test_context_resolution.py tests/test_aishell4_eval.py \
  tests/test_moss_asr.py tests/test_asr_backend.py tests/test_model_identity.py
```

Expected: `116 passed` with no failure. A Starlette/httpx deprecation warning is
accepted because it is unrelated to the experiment.

- [ ] **Step 3: Checkpoint only the prerequisite files**

```bash
git add backend/asr_agent/model_identity.py backend/asr_agent/server.py \
  backend/asr_agent/integrations/asr_backend.py backend/asr_agent/integrations/moss_asr.py \
  scripts/run_aishell4_eval.py tests/test_agent_memory.py tests/test_aishell4_eval.py \
  tests/test_asr_backend.py tests/test_context_resolution.py tests/test_model_identity.py \
  tests/test_moss_asr.py tests/test_retrace_api.py tests/test_retrace_service.py
git commit -m "feat: add model-pure MOSS experiment path"
```

Expected: `.superpowers/` remains untracked and no unrelated file enters the
commit.

### Task 2: Fix MOSS completion length and reject incomplete first passes

**Files:**
- Modify: `backend/asr_agent/integrations/moss_asr.py`
- Test: `tests/test_moss_asr.py`

- [ ] **Step 1: Add failing request and completeness tests**

Add tests that inspect the multipart body and exercise vLLM's duration-shaped
usage response:

```python
def test_moss_request_uses_transcription_completion_parameter(monkeypatch, tmp_path):
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")
    seen = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self):
            return json.dumps({
                "text": "[0.00][S01]完整文本[9.90]",
                "usage": {"type": "duration", "seconds": 10.0},
            }).encode()

    def fake_urlopen(request, timeout):
        seen["body"] = request.data.decode("utf-8", errors="ignore")
        return Response()

    monkeypatch.setenv("MOSS_MAX_COMPLETION_TOKENS", "32768")
    monkeypatch.setattr(moss_asr.urllib.request, "urlopen", fake_urlopen)
    result = moss_asr.transcribe_audio(str(audio))

    assert 'name="max_completion_tokens"' in seen["body"]
    assert 'name="max_new_tokens"' not in seen["body"]
    assert result["duration_sec"] == 10.0
    assert result["completeness"]["coverage_ratio"] == 0.99
    assert result["completeness"]["truncated"] is False


def test_moss_marks_silent_http_200_truncation(monkeypatch, tmp_path):
    payload = {
        "text": "[0.00][S01]只有前半段[37.00][8",
        "usage": {"type": "duration", "seconds": 100.0},
    }
    monkeypatch.setattr(moss_asr, "_post_transcription", lambda _: payload)
    audio = tmp_path / "sample.wav"
    audio.write_bytes(b"audio")

    result = moss_asr.transcribe_audio(str(audio))

    assert result["ok"] is False
    assert result["failure_code"] == "incomplete_first_pass"
    assert result["completeness"]["coverage_ratio"] == 0.37
    assert result["completeness"]["parse_tail"] == "[8"
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_moss_asr.py::test_moss_request_uses_transcription_completion_parameter \
  tests/test_moss_asr.py::test_moss_marks_silent_http_200_truncation
```

Expected: FAIL because the body contains `max_new_tokens` and no completeness
record exists.

- [ ] **Step 3: Implement the protocol and completeness contract**

Replace the request field helper and extend parsing with these concrete
functions:

```python
def _env_max_completion_tokens() -> str:
    return os.getenv(
        "MOSS_MAX_COMPLETION_TOKENS",
        os.getenv("MOSS_MAX_NEW_TOKENS", "32768"),
    )


def _response_duration(payload: dict[str, Any]) -> float | None:
    direct = payload.get("duration") or payload.get("duration_sec")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    value = direct or usage.get("seconds")
    return float(value) if value is not None else None


def transcript_completeness(raw: str, chunks: list[dict[str, Any]], duration: float | None) -> dict[str, Any]:
    covered = max((float(item["end_sec"]) for item in chunks if item.get("end_sec") is not None), default=0.0)
    ratio = min(1.0, covered / duration) if duration and duration > 0 else None
    matches = list(_SEGMENT_RE.finditer(raw or ""))
    tail = (raw[matches[-1].end():] if matches else raw).strip()[:160]
    threshold = float(os.getenv("MOSS_MIN_COVERAGE_RATIO", "0.98"))
    return {
        "covered_until_sec": covered,
        "coverage_ratio": ratio,
        "threshold": threshold,
        "truncated": ratio is not None and ratio < threshold,
        "parse_tail": tail,
    }
```

Send `max_completion_tokens` in `_post_transcription`. In `transcribe_audio`,
set `duration_sec`, `completeness`, `failure_code`, and make `ok` false when
`truncated` is true.

- [ ] **Step 4: Run the MOSS tests and verify GREEN**

Run:

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_moss_asr.py
```

Expected: all MOSS adapter tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/asr_agent/integrations/moss_asr.py tests/test_moss_asr.py
git commit -m "fix: require complete MOSS transcriptions"
```

### Task 3: Schedule bounded contextual analysis windows

**Files:**
- Create: `backend/asr_agent/analysis_windows.py`
- Modify: `backend/asr_agent/integrations/deepseek.py`
- Modify: `backend/asr_agent/server.py`
- Test: `tests/test_analysis_windows.py`
- Test: `tests/test_deepseek_context_judge.py`
- Test: `tests/test_retrace_api.py`

- [ ] **Step 1: Add failing pure window-boundary tests**

```python
from asr_agent.analysis_windows import AnalysisWindowPolicy, partition_turns


def test_partition_turns_closes_on_first_reached_bound():
    turns = [
        {"turn_id": "t1", "text": "甲" * 40, "start_sec": 0.0, "end_sec": 8.0},
        {"turn_id": "t2", "text": "乙" * 40, "start_sec": 8.0, "end_sec": 16.0},
        {"turn_id": "t3", "text": "丙" * 40, "start_sec": 16.0, "end_sec": 31.0},
        {"turn_id": "t4", "text": "丁", "start_sec": 31.0, "end_sec": 32.0},
    ]
    policy = AnalysisWindowPolicy(max_turns=10, max_chars=180, max_audio_sec=30.0)

    assert [[item["turn_id"] for item in group] for group in partition_turns(turns, policy)] == [
        ["t1", "t2"], ["t3", "t4"]
    ]
```

The boundary-causing Turn starts the next window when adding it would exceed a
bound. A single oversized Turn remains a one-Turn window.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_analysis_windows.py
```

Expected: collection fails because `asr_agent.analysis_windows` does not exist.

- [ ] **Step 3: Implement the pure partitioner**

```python
@dataclass(frozen=True)
class AnalysisWindowPolicy:
    max_turns: int = 10
    max_chars: int = 180
    max_audio_sec: float = 30.0


def partition_turns(turns: list[dict[str, Any]], policy: AnalysisWindowPolicy) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for turn in turns:
        proposed = [*current, turn]
        start = float(proposed[0].get("start_sec") or 0.0)
        end = float(proposed[-1].get("end_sec") or start)
        exceeds = current and (
            len(proposed) > policy.max_turns
            or sum(len(str(item.get("text") or "")) for item in proposed) > policy.max_chars
            or end - start > policy.max_audio_sec
        )
        if exceeds:
            groups.append(current)
            current = [turn]
        else:
            current = proposed
    if current:
        groups.append(current)
    return groups
```

- [ ] **Step 4: Add failing DeepSeek window-payload test**

Construct a three-Turn session, mark the trigger Turn with
`analysis_window_turn_ids=["t1", "t2", "t3"]`, intercept `_chat_json`, and
assert its user JSON contains:

```python
assert payload["analysis_window"] == [
    {"turn_id": "t1", "text": "第一句"},
    {"turn_id": "t2", "text": "第二句"},
    {"turn_id": "t3", "text": "第三句"},
]
```

Also assert a returned focus targeting `t1` survives normalization.

- [ ] **Step 5: Run the payload test and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_deepseek_context_judge.py -k analysis_window
```

Expected: FAIL because the payload has no `analysis_window`.

- [ ] **Step 6: Extend the Judge contract**

In `judge_context`, resolve IDs only from the current Session and add:

```python
window_ids = set((current_turn.meta or {}).get("analysis_window_turn_ids") or [current_turn.turn_id])
analysis_window = [
    {"turn_id": turn.turn_id, "text": turn.current_text or turn.raw_text}
    for turn in session.turns
    if turn.turn_id in window_ids
]
payload["analysis_window"] = analysis_window
```

Amend the system instruction so every `analysis_window` Turn is audited and
every focus keeps its concrete target Turn ID. Do not include reference text.
Expose a public `context_judge_identity()` that returns:

```python
def context_judge_identity() -> dict[str, str]:
    status = deepseek_status()
    return {
        "backend": "deepseek-api",
        "model": str(status.get("model") or "unknown"),
        "role": "context_judge",
    }
```

At session construction, persist this under
`pipeline_provenance["context_judge"]` only in ReTrace mode. It is a separate
role and never replaces `pipeline_provenance["first_pass"]`.

- [ ] **Step 7: Add failing API call-count test**

Stub a complete 21-segment MOSS result and an injected Judge that records calls.
POST a ReTrace audio request with window bounds of 10 Turns and assert three
Judge calls, not 21. Assert all 21 immutable Turns remain in the session, the
third call has `session_complete=true`, and provenance contains distinct
`first_pass` MOSS and `context_judge` DeepSeek role entries.

- [ ] **Step 8: Run the API test and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_retrace_api.py -k moss_analysis_windows
```

Expected: FAIL because `_build_session_from_asr` calls `process_turn` 21 times.

- [ ] **Step 9: Observe all Turns, then analyze window triggers**

In `_build_session_from_asr`, first call `service.observe_turn` for each MOSS
segment. Partition lightweight Turn dictionaries with `partition_turns`; for
each group set `analysis_window_turn_ids` on the group's last Turn and call
`service.analyze_turn` once. Pass `session_complete=True` only on the final
group, eliminating the separate duplicate final audit for windowed MOSS input.
Retain the existing per-Turn path for Qwen until its own experiment is migrated.

Read bounds from:

```python
AnalysisWindowPolicy(
    max_turns=int(os.getenv("ASR_JUDGE_WINDOW_MAX_TURNS", "10")),
    max_chars=int(os.getenv("ASR_JUDGE_WINDOW_MAX_CHARS", "180")),
    max_audio_sec=float(os.getenv("ASR_JUDGE_WINDOW_MAX_AUDIO_SEC", "30")),
)
```

- [ ] **Step 10: Verify and commit**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_analysis_windows.py tests/test_deepseek_context_judge.py tests/test_retrace_api.py
git add backend/asr_agent/analysis_windows.py backend/asr_agent/integrations/deepseek.py \
  backend/asr_agent/server.py tests/test_analysis_windows.py \
  tests/test_deepseek_context_judge.py tests/test_retrace_api.py
git commit -m "feat: batch MOSS context analysis windows"
```

Expected: all listed tests pass and no reference data enters the Judge payload.

### Task 4: Add a same-model MOSS focused verifier

**Files:**
- Create: `backend/asr_agent/integrations/moss_audio_tools.py`
- Modify: `backend/asr_agent/integrations/asr_backend.py`
- Modify: `backend/asr_agent/server.py`
- Modify: `pyproject.toml`
- Test: `tests/test_moss_audio_tools.py`
- Test: `tests/test_asr_backend.py`
- Test: `tests/test_retrace_api.py`

- [ ] **Step 1: Add the optional MOSS audio dependency**

Add:

```toml
[project.optional-dependencies]
moss = [
  "soundfile>=0.12.1",
]
```

Server installation uses `uv sync --extra moss`; do not add vLLM to the ReTrace
application environment.

- [ ] **Step 2: Add failing crop-cleanup and scoring tests**

Create a 16 kHz two-channel WAV fixture with SoundFile. Monkeypatch
`moss_asr._post_transcription` to return a timestamped local transcript and
assert:

```python
result = moss_audio_tools.verify_candidates(
    audio_path=str(audio),
    start_sec=0.25,
    end_sec=1.25,
    candidates=["南庄", "男装", "[DELETE]"],
)
assert result["ok"] is True
assert set(result["scores"]) == {"南庄", "男装", "[DELETE]"}
assert max(result["scores"], key=result["scores"].get) == "男装"
assert result["backend"] == "moss-transcribe-diarize"
assert result["model"] == "MOSS-Transcribe-Diarize"
assert not Path(result["temporary_path"]).exists()
```

Add a second test where MOSS returns equally similar evidence and assert
`ok=false`, `failure_code="ambiguous_local_transcript"`. Add a third test where
the crop is silent and only `[DELETE]` may win.

- [ ] **Step 3: Run and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_moss_audio_tools.py
```

Expected: collection fails because the module does not exist.

- [ ] **Step 4: Implement crop, normalization, and closed-set scoring**

Create the module with these concrete helpers and public functions:

```python
from __future__ import annotations

import difflib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from pypinyin import lazy_pinyin

from asr_agent.integrations import moss_asr
from asr_agent.model_identity import MOSS_TRANSCRIBE_DIARIZE_MODEL

BACKEND = "moss-transcribe-diarize"
DELETE_CANDIDATE = "[DELETE]"
TARGET_SAMPLE_RATE = 16_000


def _clean(value: str) -> str:
    return "".join(
        char.lower()
        for char in value
        if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def _crop_to_wav(audio_path: str, start_sec: float, end_sec: float) -> Path:
    if end_sec <= start_sec:
        raise ValueError("audio crop end must be after start")
    with sf.SoundFile(audio_path) as source:
        start_frame = max(0, round(start_sec * source.samplerate))
        end_frame = min(len(source), round(end_sec * source.samplerate))
        source.seek(start_frame)
        frames = source.read(
            max(0, end_frame - start_frame),
            dtype="float32",
            always_2d=True,
        )
        source_rate = source.samplerate
    if not len(frames):
        raise ValueError("audio crop is empty")
    mono = frames.mean(axis=1)
    if source_rate != TARGET_SAMPLE_RATE:
        output_size = max(1, round(len(mono) * TARGET_SAMPLE_RATE / source_rate))
        mono = np.interp(
            np.linspace(0.0, len(mono) - 1, output_size),
            np.arange(len(mono)),
            mono,
        ).astype("float32")
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    path = Path(handle.name)
    sf.write(path, mono, TARGET_SAMPLE_RATE, subtype="PCM_16")
    return path


def retranscribe_window(
    *,
    audio_path: str,
    start_sec: float,
    end_sec: float,
    **_: Any,
) -> dict[str, Any]:
    temporary_path: Path | None = None
    try:
        temporary_path = _crop_to_wav(audio_path, start_sec, end_sec)
        payload = moss_asr._post_transcription(temporary_path)
        parsed = moss_asr.parse_moss_transcript(
            str(payload.get("text") or payload.get("transcript") or "")
        )
        text = str(parsed.get("text") or "").strip()
        return {
            "ok": bool(text),
            "text": text,
            "backend": BACKEND,
            "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
            "temporary_path": str(temporary_path),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"MOSS focused retranscription failed: {exc}",
            "failure_code": "moss_focused_retranscription_failed",
            "backend": BACKEND,
            "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
            "temporary_path": str(temporary_path) if temporary_path else None,
        }
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _best_similarity(candidate: str, transcript: str) -> float:
    candidate_clean = _clean(candidate)
    transcript_clean = _clean(transcript)
    if not candidate_clean or not transcript_clean:
        return 0.0
    widths = range(
        max(1, len(candidate_clean) - 1),
        min(len(transcript_clean), len(candidate_clean) + 1) + 1,
    )
    best = 0.0
    candidate_pinyin = " ".join(lazy_pinyin(candidate_clean))
    for width in widths:
        for offset in range(len(transcript_clean) - width + 1):
            observed = transcript_clean[offset : offset + width]
            char_score = difflib.SequenceMatcher(
                None, candidate_clean, observed, autojunk=False
            ).ratio()
            pinyin_score = difflib.SequenceMatcher(
                None,
                candidate_pinyin,
                " ".join(lazy_pinyin(observed)),
                autojunk=False,
            ).ratio()
            best = max(best, 0.55 * char_score + 0.45 * pinyin_score)
    return best


def verify_candidates(
    *, audio_path: str, start_sec: float, end_sec: float, candidates: list[str]
) -> dict[str, Any]:
    if not candidates or len(set(candidates)) != len(candidates):
        return {
            "ok": False,
            "failure_code": "invalid_candidate_set",
            "backend": BACKEND,
            "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
        }
    observed = retranscribe_window(
        audio_path=audio_path,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    if not observed.get("ok") and observed.get("text") is None:
        return observed
    transcript = str(observed.get("text") or "")
    scores = {
        candidate: (
            1.0 if candidate == DELETE_CANDIDATE and not _clean(transcript)
            else 0.0 if candidate == DELETE_CANDIDATE
            else _best_similarity(candidate, transcript)
        )
        for candidate in candidates
    }
    ordered = sorted(scores.values(), reverse=True)
    top = ordered[0] if ordered else 0.0
    margin = top - (ordered[1] if len(ordered) > 1 else 0.0)
    if top < 0.55 or margin < 0.10:
        return {
            "ok": False,
            "scores": scores,
            "text": transcript,
            "failure_code": "ambiguous_local_transcript",
            "backend": BACKEND,
            "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
            "temporary_path": observed.get("temporary_path"),
        }
    return {
        "ok": True,
        "scores": scores,
        "text": transcript,
        "backend": BACKEND,
        "model": MOSS_TRANSCRIBE_DIARIZE_MODEL,
        "temporary_path": observed.get("temporary_path"),
    }
```

`retranscribe_window` reads only requested frames with `soundfile.SoundFile`,
downmixes by channel mean, resamples to 16 kHz with deterministic NumPy linear
interpolation when needed, writes a `NamedTemporaryFile(suffix=".wav",
delete=False)`, calls `moss_asr._post_transcription`, strips MOSS tags, and
unlinks the temporary file in `finally`.

For every non-delete candidate, score the best same-length local substring with:

```python
score = 0.55 * character_similarity + 0.45 * pinyin_similarity
```

The returned contract sets the delete score to `1.0` only when normalized local
text is empty, otherwise `0.0`. It returns `ok=false` when the top score is below
0.55 or its margin over second place is below 0.10. It always returns scores for
exactly the supplied candidate set when `ok=true`.

- [ ] **Step 5: Add failing backend-bundle routing test**

Define the expected interface in the test:

```python
bundle = asr_backend.selected_backend_bundle()
assert bundle.identity.family == "moss"
assert bundle.verifier_identity.family == "moss"
assert bundle.relistener_identity.family == "moss"
assert bundle.audio_verifier is moss_audio_tools.verify_candidates
assert bundle.audio_retranscriber is moss_audio_tools.retranscribe_window
```

Under Qwen configuration, assert every identity is Qwen and no MOSS callable is
returned.

- [ ] **Step 6: Run the bundle test and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_asr_backend.py -k backend_bundle
```

Expected: FAIL because `selected_backend_bundle` does not exist.

- [ ] **Step 7: Implement and inject the backend bundle**

Add this immutable contract to `asr_backend.py`:

```python
@dataclass(frozen=True)
class ASRBackendBundle:
    identity: ModelIdentity
    audio_verifier: Callable[..., dict[str, Any]] | None
    verifier_identity: ModelIdentity | None
    audio_retranscriber: Callable[..., dict[str, Any]] | None
    relistener_identity: ModelIdentity | None
```

`selected_backend_bundle()` returns matching MOSS functions and identities for
MOSS, and existing Qwen functions and identities for Qwen. Modify `create_app`
to inject all four bundle fields into `ReTraceService`. Remove direct default
production dependence on the Qwen verifier import when MOSS is selected.

- [ ] **Step 8: Verify model-pure resolution and commit**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_moss_audio_tools.py tests/test_asr_backend.py \
  tests/test_context_resolution.py tests/test_retrace_service.py tests/test_retrace_api.py
git add pyproject.toml backend/asr_agent/integrations/moss_audio_tools.py \
  backend/asr_agent/integrations/asr_backend.py backend/asr_agent/server.py \
  tests/test_moss_audio_tools.py tests/test_asr_backend.py tests/test_retrace_api.py
git commit -m "feat: verify MOSS candidates with focused MOSS audio"
```

Expected: MOSS candidates can reach a verifier, while every cross-family test
still fails closed before executing an audio tool.

### Task 5: Cache and reuse an immutable complete first pass

**Files:**
- Create: `backend/asr_agent/asr_artifacts.py`
- Modify: `backend/asr_agent/server.py`
- Test: `tests/test_asr_artifacts.py`
- Test: `tests/test_retrace_api.py`

- [ ] **Step 1: Add failing repository tests**

```python
def test_artifact_key_changes_with_audio_or_inference_contract(tmp_path):
    audio = tmp_path / "a.flac"
    audio.write_bytes(b"same audio")
    first = artifact_key(audio, identity={"backend": "moss-transcribe-diarize", "model": "MOSS-Transcribe-Diarize"}, config={"max_completion_tokens": 32768})
    second = artifact_key(audio, identity={"backend": "moss-transcribe-diarize", "model": "MOSS-Transcribe-Diarize"}, config={"max_completion_tokens": 5120})
    assert first != second


def test_repository_refuses_incomplete_artifact(tmp_path):
    repo = FirstPassArtifactRepository(tmp_path)
    with pytest.raises(ValueError, match="incomplete"):
        repo.save("key", {"ok": False, "completeness": {"truncated": True}})
```

Also save/load a complete artifact and assert its `artifact_id` and payload are
stable.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_asr_artifacts.py
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement checksum-keyed atomic artifacts**

`artifact_key` streams the audio through SHA-256 in 1 MiB blocks, hashes a
canonical JSON object containing audio hash, first-pass backend/model,
completion budget, endpoint model name, and minimum coverage ratio, and returns
the hex digest. `FirstPassArtifactRepository.save` writes JSON to a temporary
sibling and uses `Path.replace`; it rejects `ok=false` or
`completeness.truncated=true`.

- [ ] **Step 4: Add failing baseline/ReTrace reuse test**

Call baseline and ReTrace against the same audio with a stub `transcribe_audio`
counter. Assert the counter is one, both responses have the same
`first_pass_artifact_id`, and ReTrace `session.turns[*].raw_text` concatenates to
the baseline transcript.

- [ ] **Step 5: Run and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_retrace_api.py -k first_pass_artifact
```

Expected: FAIL because the current server transcribes once per request.

- [ ] **Step 6: Integrate the repository**

Construct `FirstPassArtifactRepository(root / "first_pass_artifacts")` in
`create_app`. Before `transcribe_audio`, compute the key from the selected
identity and:

```python
asr = artifact_repository.load(key)
cache_hit = asr is not None
if asr is None:
    asr = transcribe_audio(audio_path)
    if asr.get("ok"):
        artifact_repository.save(key, asr)
```

Return `first_pass_artifact_id` and `first_pass_cache_hit` in both experiment
modes. Never cache an incomplete or failed response.

- [ ] **Step 7: Verify and commit**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q \
  tests/test_asr_artifacts.py tests/test_retrace_api.py
git add backend/asr_agent/asr_artifacts.py backend/asr_agent/server.py \
  tests/test_asr_artifacts.py tests/test_retrace_api.py
git commit -m "feat: reuse complete first-pass ASR artifacts"
```

### Task 6: Make AISHELL-4 evaluation validity-aware

**Files:**
- Modify: `scripts/run_aishell4_eval.py`
- Test: `tests/test_aishell4_eval.py`

- [ ] **Step 1: Add failing validity and paired-mode tests**

Use a fake `requests.post` returning baseline and ReTrace bodies. Assert:

```python
assert item["valid"] is True
assert item["first_pass_artifact_id"] == "artifact-1"
assert item["raw"] == item["baseline"]
assert item["retrace"]["committed_revisions"] == 1
```

Add invalid cases for different artifact IDs, incomplete first pass, and
`integrations.deepseek.ready=false`; each must set `valid=false` with a distinct
`invalid_reasons` entry and must not enter aggregate effectiveness metrics.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_aishell4_eval.py
```

Expected: FAIL because the evaluator currently sends only one ReTrace request.

- [ ] **Step 3: Run paired requests and compute selected metrics**

For each audio, POST baseline first and ReTrace second with the same path. Save
`baseline_result.json` and `retrace_result.json`. Require matching artifact IDs,
complete first pass, DeepSeek readiness, identical normalized raw transcript,
and non-error memory status.

Compute:

```python
ecer = (baseline_edits - final_edits) / baseline_edits if baseline_edits else 0.0
revision_precision = improved / committed if committed else None
candidate_to_correction_yield = improved / validated_candidates if validated_candidates else None
```

Keep CER as a supporting diagnostic. Store per-stage milliseconds, calls,
real-time factor, Judge fallback/retry/cooldown counts, candidate funnel, memory
scope/status, and completeness.

- [ ] **Step 4: Aggregate only valid paired samples**

Add `aggregate.json` with valid/invalid sample counts, pooled edit totals and
ECER, micro revision precision, micro candidate-to-correction yield, total
stage costs, and a list of invalid sample reasons. Never average per-file CER or
ratios when pooled counts are available.

- [ ] **Step 5: Verify and commit**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_aishell4_eval.py
git add scripts/run_aishell4_eval.py tests/test_aishell4_eval.py
git commit -m "feat: evaluate paired complete MOSS runs"
```

### Task 7: Add server preflight and reproducible MOSS launch profile

**Files:**
- Create: `scripts/preflight_moss_eval.py`
- Create: `scripts/start_moss_vllm.sh`
- Modify: `.env.example`
- Test: `tests/test_preflight_moss_eval.py`

- [ ] **Step 1: Add failing process-profile parser tests**

Test pure functions with a fake environment and cmdline:

```python
profile = parse_vllm_profile(
    environ={
        "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "1024",
        "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600",
    },
    argv=["vllm", "serve", "/models/MOSS", "--max-num-batched-tokens", "32768"],
)
assert profile.max_upload_mb == 1024
assert profile.max_duration_sec == 3600
assert profile.max_num_batched_tokens == 32768
```

Add failure tests for a 2,363-second/273-MiB/29,537-token sample against each
insufficient bound, and a success case for the selected profile.

- [ ] **Step 2: Run and verify RED**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_preflight_moss_eval.py
```

Expected: collection fails because the script does not exist.

- [ ] **Step 3: Implement read-only preflight**

The CLI accepts `--vllm-pid`, `--wav-dir`, `--base-url`, and
`--retrace-base-url`. It reads only named `/proc/<pid>/environ` and `cmdline`,
checks `/v1/models`, derives FLAC duration/size with SoundFile, estimates MOSS
audio tokens using the confirmed 12.5 tokens/second ceiling, calls
`/api/integrations/status`, verifies output/temp paths are writable with a
create/unlink probe, and prints JSON with `ready`, `checks`, and `failures`.
Never include secret values.

- [ ] **Step 4: Add the launch script**

`scripts/start_moss_vllm.sh` requires explicit model/log paths and launches:

```bash
env CUDA_VISIBLE_DEVICES="${MOSS_GPU_INDEX:-4}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=1024 \
  VLLM_MAX_AUDIO_DECODE_DURATION_S=3600 \
  vllm serve "$MOSS_MODEL_PATH" \
  --trust-remote-code --host 127.0.0.1 --port 8010 \
  --served-model-name MOSS-Transcribe-Diarize \
  --max-num-batched-tokens 32768
```

The script checks the model directory and refuses to start if port 8010 is
already listening. It does not kill processes automatically.

- [ ] **Step 5: Document application environment**

Add these non-secret keys to `.env.example`:

```dotenv
ASR_BACKEND=moss-transcribe-diarize
MOSS_TRANSCRIBE_URL=http://127.0.0.1:8010/v1/audio/transcriptions
MOSS_MODEL_PATH=/home/panyibo/models/openmoss/MOSS-Transcribe-Diarize
MOSS_MODEL_NAME=MOSS-Transcribe-Diarize
MOSS_MAX_COMPLETION_TOKENS=32768
MOSS_MIN_COVERAGE_RATIO=0.98
ASR_JUDGE_WINDOW_MAX_TURNS=10
ASR_JUDGE_WINDOW_MAX_CHARS=180
ASR_JUDGE_WINDOW_MAX_AUDIO_SEC=30
ASR_EXPERIMENT_NAMESPACE=aishell4-moss-test20
```

Keep the DeepSeek key commented and empty.

- [ ] **Step 6: Verify and commit**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_preflight_moss_eval.py
bash -n scripts/start_moss_vllm.sh
git add scripts/preflight_moss_eval.py scripts/start_moss_vllm.sh \
  tests/test_preflight_moss_eval.py .env.example
git commit -m "feat: preflight MOSS evaluation deployment"
```

### Task 8: Full local verification and 118 acceptance

**Files:**
- No new source files unless a failing acceptance check identifies a root cause.

- [ ] **Step 1: Run the complete local verification suite**

```bash
PYTHONPATH=backend .venv/bin/python -m pytest -q
git diff --check
```

Expected: all tests pass and diff check prints nothing. Do not claim completion
from focused tests alone.

- [ ] **Step 2: Push the feature branch**

```bash
git status --short --branch
git push retrace-public feat/open-candidate-fast-retrace-20260907
```

Expected: remote branch advances without force push.

- [ ] **Step 3: Sync the exact branch to 118**

Because `/home/panyibo/ReTrace-ASR` is currently a plain copied directory, use
`rsync` with an explicit include set or clone into a new versioned directory.
Do not use `--delete` against the existing directory. Verify deployed source by
commit hash recorded in the result directory.

- [ ] **Step 4: Install the app with the MOSS extra**

```bash
/home/panyibo/.local/bin/uv sync --extra moss --python 3.12
```

Expected: `soundfile` imports in the ReTrace application environment. Do not
replace the already working vLLM development build.

- [ ] **Step 5: Configure DeepSeek without exposing the key**

The user runs interactively on 118:

```bash
cd /home/panyibo/ReTrace-ASR
umask 077
read -rsp 'DEEPSEEK_API_KEY: ' key
printf '\nDEEPSEEK_API_KEY=%s\n' "$key" >> .env
unset key
```

Verify only `/api/integrations/status` reports `deepseek.ready=true`; never read
or print the key.

- [ ] **Step 6: Run preflight**

```bash
PYTHONPATH=backend .venv/bin/python scripts/preflight_moss_eval.py \
  --vllm-pid "$(pgrep -f 'vllm serve.*/MOSS-Transcribe-Diarize' | head -1)" \
  --wav-dir /home/panyibo/datasets/retrace_asr/aishell4_test20/test/wav \
  --base-url http://127.0.0.1:8010 \
  --retrace-base-url http://127.0.0.1:8000
```

Expected: `ready=true`, served model is exactly
`MOSS-Transcribe-Diarize`, DeepSeek is ready, and every selected file fits.

- [ ] **Step 7: Run one paired sample**

Place one symlink in an explicit one-sample input directory and run:

```bash
PYTHONPATH=backend .venv/bin/python scripts/run_aishell4_eval.py \
  --wav-dir /home/panyibo/datasets/retrace_asr/aishell4_test1/wav \
  --reference-dir /home/panyibo/datasets/retrace_asr/aishell4_test20/test/TextGrid \
  --output-dir /home/panyibo/results/retrace/aishell4-moss-test1 \
  --base-url http://127.0.0.1:8000
```

Accept only if the first pass is complete, artifact IDs match, DeepSeek is
ready, model identities are pure MOSS, candidate terminal states reconcile,
memory status is healthy, and timings are present.

- [ ] **Step 8: Run three paired samples and identify the bottleneck**

Run the same command on an explicit three-sample directory. Compare total
`first_pass_asr`, `context_judge`, `candidate_build`, `targeted_verifier`,
`open_relisten`, and `ledger_commit` time. Inspect every committed/harmed
revision against offline TextGrid only after inference.

Stop before 20 samples when any of these holds:

- first-pass coverage below 0.98;
- any MOSS Turn names a Qwen/Omni audio tool;
- DeepSeek fallback/cooldown makes the run invalid;
- revision precision is below 0.5 on the audited three-sample pilot;
- one ReTrace layer dominates incremental time without producing candidates or
  improved revisions.

- [ ] **Step 9: Report evidence without overclaiming**

Report the exact commit, server paths, per-sample validity, ECER, revision
precision, candidate-to-correction yield, raw/final diagnostic CER, stage
timings, candidate funnel, memory status, and the next bottleneck. If no
revision improves, state that ReTrace did not improve the pilot and use the
audit funnel to identify whether discovery, verification, or policy blocked it.
