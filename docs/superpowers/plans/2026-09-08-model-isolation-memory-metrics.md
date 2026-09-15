# Model-Isolated ReTrace and Durable Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent MOSS/Omni cross-model execution, persist long-term memory by concrete model across audio sessions, expose component timing and memory failures, and report LECR, revision precision, and ECER.

**Architecture:** Add a small model-identity module used by ASR results, Turn metadata, audio-tool routing, and memory scope derivation. Existing Qwen audio tools become explicitly Qwen-bound; MOSS inputs fail closed unless the service is constructed with MOSS-bound tools. Memory remains file-backed but scopes are derived from backend/model/experiment namespace rather than audio session id, and repository status becomes observable.

**Tech Stack:** Python 3.10+, dataclasses, FastAPI/Pydantic, pytest, JSON session storage, TypeScript/Vite.

---

### Task 1: Define stage identity and model-scoped memory keys

**Files:**
- Create: `backend/asr_agent/model_identity.py`
- Test: `tests/test_model_identity.py`

- [ ] **Step 1: Write failing identity tests**

```python
from asr_agent.model_identity import ModelIdentity, model_memory_scope


def test_unknown_moss_model_is_not_inferred_from_qwen_configuration():
    identity = ModelIdentity.from_asr_result({"backend": "moss-transcribe-diarize"})
    assert identity.backend == "moss-transcribe-diarize"
    assert identity.model == "unknown"
    assert identity.family == "moss"


def test_same_model_shares_memory_scope_but_models_are_isolated():
    moss = ModelIdentity("moss-transcribe-diarize", "MOSS-Audio-7B", "first_pass")
    omni = ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "first_pass")
    assert model_memory_scope(moss, namespace="eval") == model_memory_scope(moss, namespace="eval")
    assert model_memory_scope(moss, namespace="eval") != model_memory_scope(omni, namespace="eval")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_model_identity.py`

Expected: collection fails because `asr_agent.model_identity` does not exist.

- [ ] **Step 3: Implement the identity contract**

Create an immutable `ModelIdentity` with `backend`, `model`, `role`, a normalized
`family` property, `as_dict()`, `from_dict()`, and `from_asr_result()`. Backend
families normalize `moss-*` to `moss`, `qwen-omni-*` to `qwen-omni`, and all
others to a safe lowercase identifier. `model_memory_scope()` must sanitize the
namespace/backend/model and append the first 10 hex characters of a SHA-256 over
the unsanitized identity.

```python
@dataclass(frozen=True)
class ModelIdentity:
    backend: str
    model: str
    role: str

    @property
    def family(self) -> str: ...

    @classmethod
    def from_asr_result(cls, value: dict[str, Any]) -> "ModelIdentity":
        return cls(
            backend=str(value.get("backend") or "unknown"),
            model=str(value.get("model") or "unknown"),
            role="first_pass",
        )
```

- [ ] **Step 4: Run identity tests and verify GREEN**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_model_identity.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/asr_agent/model_identity.py tests/test_model_identity.py
git commit -m "feat: define model-isolated pipeline identity"
```

### Task 2: Emit truthful first-pass provenance

**Files:**
- Modify: `backend/asr_agent/integrations/qwen_asr.py:139-157,919-935`
- Modify: `backend/asr_agent/server.py:302-407`
- Modify: `backend/asr_agent/models.py:214-244`
- Test: `tests/test_retrace_api.py`
- Test: `tests/test_retrace_models.py`

- [ ] **Step 1: Write failing API provenance tests**

Extend the MOSS boundary test so a MOSS result without `model` returns an
explicit unknown MOSS identity and never copies `ASR_MODEL_PATH`:

```python
def test_moss_backend_never_acquires_qwen_model_name(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_MODEL_PATH", "/models/Qwen3_Omni_30B")
    monkeypatch.setattr(server, "transcribe_audio", lambda _: {
        "ok": True,
        "backend": "moss-transcribe-diarize",
        "chunks_text": ["测试文本"],
        "chunks": [{"start_sec": 0.0, "end_sec": 1.0}],
        "uncertainties": [{}],
    })
    with TestClient(create_app(tmp_path)) as client:
        body = client.post("/api/sessions/s/audio", json={"audio": "/tmp/moss.wav"}).json()
    assert body["provenance"]["first_pass"] == {
        "backend": "moss-transcribe-diarize", "model": "unknown", "role": "first_pass"
    }
    assert body["session"]["pipeline_provenance"]["first_pass"]["model"] == "unknown"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_retrace_api.py::test_moss_backend_never_acquires_qwen_model_name`

Expected: response has no structured `provenance`.

- [ ] **Step 3: Add provenance to Qwen and sessions**

Qwen results return `model=Path(cfg.model_path).name`. Add
`pipeline_provenance: dict[str, dict[str, str]]` to `Session` with a default
empty dict and backward-compatible `from_dict`. In `_build_session_from_asr`,
build `ModelIdentity.from_asr_result(asr)`, persist it as `first_pass`, and
return the same map at response top level. Never consult Qwen configuration when
normalizing a MOSS result.

- [ ] **Step 4: Run provenance/model compatibility tests**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_retrace_api.py tests/test_retrace_models.py`

Expected: all tests pass, including old session JSON without provenance.

- [ ] **Step 5: Commit**

```bash
git add backend/asr_agent/integrations/qwen_asr.py backend/asr_agent/server.py backend/asr_agent/models.py tests/test_retrace_api.py tests/test_retrace_models.py
git commit -m "feat: expose truthful pipeline provenance"
```

### Task 3: Enforce same-model audio verification and relisten

**Files:**
- Modify: `backend/asr_agent/resolver.py:19-394`
- Modify: `backend/asr_agent/retrace.py:52-84,716-953`
- Modify: `backend/asr_agent/server.py:225-243`
- Test: `tests/test_context_resolution.py`
- Test: `tests/test_retrace_service.py`

- [ ] **Step 1: Write failing cross-model rejection tests**

```python
def test_moss_turn_cannot_call_qwen_verifier(tmp_path):
    called = False
    def qwen_verify(**_):
        nonlocal called
        called = True
        return {"ok": True, "scores": {"南庄": 0.1, "男装": 0.9}}
    resolver = EvidenceResolver(
        audio_verifier=qwen_verify,
        verifier_identity=ModelIdentity("qwen-omni-vllm", "Qwen3_Omni_30B", "targeted_verifier"),
    )
    target = Turn("t1", "南庄", "南庄", source="moss", meta={
        "audio_path": str(tmp_path / "a.wav"), "start_sec": 0, "end_sec": 1,
        "first_pass_identity": ModelIdentity("moss-transcribe-diarize", "MOSS-Audio-7B", "first_pass").as_dict(),
    })
    focus = FocusProposal(
        "t1", "南庄", "男装", ["南庄", "男装"], ["t1"], source="semantic_open"
    )
    result = resolver.resolve(Session("s", turns=[target]), target, focus, context_confidence=0.99)
    assert result.action == "DEFER"
    assert result.verifier_attempted is False
    assert result.failure_code == "verifier_backend_mismatch"
    assert called is False
```

Add a corresponding relisten test proving `_relisten_uncertain_window` does not
call a Qwen-bound relistener for a MOSS Turn.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_context_resolution.py tests/test_retrace_service.py -k 'cross_model or cannot_call_qwen'`

Expected: constructors do not accept bound identities and the calls are not blocked.

- [ ] **Step 3: Implement fail-closed family checks**

Add `verifier_identity` and `failure_code` to resolver state/results. Before any
audio call, compare the target's first-pass family with the verifier family.
Apply the same rule to the service's `relistener_identity`. Text-only unit-test
Turns without first-pass identity retain current injected-test behavior; audio
API Turns always carry identity. Candidate audits include verifier identity and
failure code.

The production app binds the built-in `verify_candidates` and
`retranscribe_window` explicitly to the configured Qwen identity. No MOSS alias
is registered in this repository.

- [ ] **Step 4: Run resolver/service suites and verify GREEN**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_context_resolution.py tests/test_retrace_service.py`

Expected: all tests pass and MOSS/Qwen calls cannot cross.

- [ ] **Step 5: Commit**

```bash
git add backend/asr_agent/resolver.py backend/asr_agent/retrace.py backend/asr_agent/server.py tests/test_context_resolution.py tests/test_retrace_service.py
git commit -m "fix: forbid cross-model audio evidence"
```

### Task 4: Persist and observe model-scoped long-term memory

**Files:**
- Modify: `backend/asr_agent/server.py:302-535`
- Modify: `backend/asr_agent/memory.py:155-347`
- Modify: `backend/asr_agent/retrace.py:81-92,486-508,995-1040`
- Modify: `.env.example`
- Test: `tests/test_agent_memory.py`
- Test: `tests/test_retrace_api.py`

- [ ] **Step 1: Write failing cross-session memory tests**

Create two audio sessions with the same Qwen identity and assert equal memory
scopes; create a MOSS identity and assert its scope differs. Seed a stable Qwen
belief and prove it is retrievable from the second Qwen session but absent from
the MOSS session.

Add a broken repository fixture and assert session observability contains:

```python
{
    "scope": "model--...",
    "provisional": 0,
    "stable": 0,
    "superseded": 0,
    "last_operation": "consolidate",
    "error": {"kind": "OSError", "message": "disk unavailable"},
}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_agent_memory.py tests/test_retrace_api.py -k 'model_scope or memory_status'`

Expected: audio scopes remain session-specific and no status exists.

- [ ] **Step 3: Derive scopes and expose repository status**

Use `ASR_EXPERIMENT_NAMESPACE` (default `default`) plus
`model_memory_scope(first_pass_identity, namespace=...)` in batch and streaming
audio paths. Add a thread-safe last-operation/error map and `status(scope)` to
`LongTermMemoryRepository`; `load`, `save`, and `update` record success or error
before re-raising. Consolidation stays non-fatal, but ReTrace observability reads
and returns repository status.

Document that changing model identity or experiment namespace intentionally
creates a fresh memory scope.

- [ ] **Step 4: Run memory/API suites and verify GREEN**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_agent_memory.py tests/test_retrace_api.py tests/test_retrace_service.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/asr_agent/server.py backend/asr_agent/memory.py backend/asr_agent/retrace.py .env.example tests/test_agent_memory.py tests/test_retrace_api.py
git commit -m "feat: persist model-scoped long-term memory"
```

### Task 5: Separate baseline and ReTrace execution modes

**Files:**
- Modify: `backend/asr_agent/server.py:47-55,302-535`
- Test: `tests/test_retrace_api.py`

- [ ] **Step 1: Write a failing baseline isolation test**

Construct the app with a service whose Judge/verifier/relistener raise if called.
POST an audio request with `experiment_mode="baseline"` and assert the response
contains the first-pass transcript, empty revisions, no session memory writes,
and provenance containing only `first_pass`.

- [ ] **Step 2: Run the test and verify RED**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_retrace_api.py -k baseline_mode`

Expected: request validation rejects `experiment_mode` or ReTrace is called.

- [ ] **Step 3: Implement explicit modes**

Add `experiment_mode: Literal["baseline", "retrace"] = "retrace"` to JSON
audio requests and an equivalent validated form/query field for uploads. Carry
the selected mode explicitly into the background stream worker. Baseline returns
ASR output without constructing/analyzing a ReTrace session; a baseline stream
may publish first-pass chunks but must never call `reset_session`, Judge,
verifier, relistener, or memory persistence. ReTrace retains the existing session
behavior with model-pure routing and model-scoped memory. Include mode in every
response and saved evaluation artifact.

- [ ] **Step 4: Run API tests and verify GREEN**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_retrace_api.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/asr_agent/server.py tests/test_retrace_api.py
git commit -m "feat: isolate baseline and retrace experiment modes"
```

### Task 6: Record component timing and the three primary metrics

**Files:**
- Modify: `backend/asr_agent/resolver.py`
- Modify: `backend/asr_agent/retrace.py`
- Modify: `backend/asr_agent/server.py`
- Modify: `backend/asr_agent/metrics.py`
- Modify: `scripts/run_ramc_dataset_eval.py`
- Test: `tests/test_retrace_metrics.py`
- Test: `tests/test_retrace_service.py`

- [ ] **Step 1: Write failing metric fixtures**

Define offline gold fixtures containing eligible later-supported error spans and
entity mentions. Assert exact values for:

```python
assert report["lecr"] == 2 / 3
assert report["revision_precision"] == 2 / 3
assert report["entity_consistency_error_rate"] == 1 / 4
```

Include a `candidate_audit` with a revision-shaped action and prove it is not
counted as a committed revision.

- [ ] **Step 2: Run metric tests and verify RED**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_retrace_metrics.py -k primary_task_metrics`

Expected: the primary metric function does not exist.

- [ ] **Step 3: Implement deterministic metric calculation**

Add `evaluate_primary_metrics(events, eligible_errors, entity_mentions)`.
Eligible errors and entity mentions are offline inputs derived from GT; online
candidate output cannot define the denominator. A revision is beneficial only
when its target/replacement matches the corresponding gold correction.

- [ ] **Step 4: Write failing timing test**

Use injected Judge/verifier/relistener functions and assert the session/result
contains non-negative milliseconds and call counts for `context_judge`,
`candidate_build`, `targeted_verifier`, `open_relisten`, `ledger_commit`, and
`end_to_end`. ASR responses additionally retain first-pass elapsed time.

- [ ] **Step 5: Implement timing instrumentation**

Use `time.perf_counter()` at component boundaries. Store per-Turn timing in
`turn.meta["stage_timings_ms"]`, aggregate response timing by summing Turn
values, and record verifier/relistener timings on their candidate audit or Turn.
Do not infer unavailable component time from end-to-end duration.

- [ ] **Step 6: Run focused metrics/timing tests**

Run: `PYTHONPATH=backend .venv/bin/python -m pytest -q tests/test_retrace_metrics.py tests/test_retrace_service.py tests/test_ramc_eval_metrics.py`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add backend/asr_agent/resolver.py backend/asr_agent/retrace.py backend/asr_agent/server.py backend/asr_agent/metrics.py scripts/run_ramc_dataset_eval.py tests/test_retrace_metrics.py tests/test_retrace_service.py
git commit -m "feat: report retrace effectiveness and component cost"
```

### Task 7: Final compatibility and build verification

**Files:**
- Modify only if a verification failure reveals an in-scope defect.

- [ ] **Step 1: Run the complete backend suite offline**

Run:

```bash
ASR_DOMAINTERMS_ROOT=/private/tmp/retrace-asr-offline-domainterms \
PYTHONPATH=backend .venv/bin/python -m pytest -q
```

Expected: all tests pass; only the known Starlette/httpx deprecation warning is acceptable.

- [ ] **Step 2: Verify compilation and whitespace**

```bash
.venv/bin/python -m compileall -q backend scripts tests
git diff --check
```

Expected: exit code 0 and no output.

- [ ] **Step 3: Build the frontend**

Run: `./scripts/build_frontend.sh`

Expected: TypeScript check and Vite build pass.

- [ ] **Step 4: Inspect final provenance/memory behavior**

Run focused tests with `-vv` and confirm outputs demonstrate: no mixed model
identity, no cross-model tool call, same-model memory reuse, cross-model memory
isolation, visible memory errors, and exact primary metric fixtures.

- [ ] **Step 5: Commit any verification-only correction**

If no correction is needed, do not create an empty commit. Otherwise stage only
the in-scope files and commit with `fix: close model isolation verification gap`.
