# Candidate Pipeline Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make open, historical, acoustic, and relisten correction candidates reach one strict audio-verification path with durable stage metrics.

**Architecture:** DeepSeek and deterministic discovery produce `CorrectionCandidate` objects. `ReTraceService` validates and deduplicates them, adapts them to `FocusProposal` only for `EvidenceResolver`, records a candidate audit event for every outcome, and permits transcript mutation only from resolver-approved revisions.

**Tech Stack:** Python 3.11, dataclasses, FastAPI, pytest, uv

---

### Task 1: Restore a trustworthy baseline

**Files:**
- Modify: `tests/test_retrace_fast_path.py`
- Modify: `backend/asr_agent/retrace.py`
- Modify: `backend/asr_agent/server.py`
- Test: `tests/test_retrace_api.py`

- [ ] **Step 1: Write failing tests for correctness-first defaults and status.**

```python
def test_moss_normal_turn_calls_context_judge_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("ASR_FAST_NORMAL_TURNS", raising=False)
    calls = []
    service = ReTraceService(tmp_path, context_judge=lambda **kwargs: calls.append(kwargs) or ContextJudgment("CONSISTENT", 0.9))
    service.process_turn("s", "t1", "正常文本", source="moss")
    assert len(calls) == 1

def test_fast_path_is_explicit_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("ASR_FAST_NORMAL_TURNS", "1")
    service = ReTraceService(tmp_path, context_judge=lambda **_: (_ for _ in ()).throw(AssertionError()))
    assert service.process_turn("s", "t1", "正常文本", source="moss")["status"] == "idle"
```

- [ ] **Step 2: Run the two tests and confirm the default-path assertion fails.**

Run: `PYTHONPATH=backend .venv/bin/pytest -q tests/test_retrace_fast_path.py`

- [ ] **Step 3: Change `_can_fast_keep_turn` so `ASR_FAST_NORMAL_TURNS` defaults to `0`, and extend `/api/integrations/status` with policy flags and homophone availability.**

```python
enabled = os.getenv("ASR_FAST_NORMAL_TURNS", "0").strip().lower() not in {"0", "false", "no", "off"}
```

- [ ] **Step 4: Add a server-boundary test proving a result with `backend=moss-transcribe-diarize` produces MOSS turns.**
- [ ] **Step 5: Run focused tests and commit.**

### Task 2: Normalize every Judge candidate

**Files:**
- Modify: `backend/asr_agent/correction_candidates.py`
- Modify: `backend/asr_agent/integrations/deepseek.py`
- Modify: `backend/asr_agent/context_judge.py`
- Modify: `tests/test_correction_candidates.py`
- Modify: `tests/test_deepseek_context_judge.py`

- [ ] **Step 1: Add failing tests for focus/top-level equivalence and source inference.**

```python
def test_absent_history_focus_is_semantic_open():
    candidates = judgment_candidates(result, session, current_turn)
    assert len(candidates) == 1
    assert candidates[0].target_turn_id == "t1"
    assert candidates[0].span == "南庄"
    assert candidates[0].candidate == "男装"
    assert candidates[0].source == "semantic_open"
    assert candidates[0].evidence_turn_ids == ["t1"]

def test_focus_and_candidate_pool_deduplicate():
    assert len(deduplicate_candidates(candidates)) == 1
```

- [ ] **Step 2: Run tests and confirm failures show the adapter is missing.**
- [ ] **Step 3: Implement `focus_to_candidate`, candidate validation, source inference, and one DeepSeek normalization function used for both JSON shapes.**
- [ ] **Step 4: Update the system prompt with the exact top-level candidate schema and safety statement.**
- [ ] **Step 5: Move the four accidentally nested tests to module scope and verify pytest collects them.**
- [ ] **Step 6: Run focused tests and commit.**

### Task 3: Route candidates through one resolver

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Modify: `backend/asr_agent/resolver.py`
- Modify: `tests/test_context_resolution.py`
- Modify: `tests/test_retrace_service.py`

- [ ] **Step 1: Write a failing end-to-end test for absent-history semantic correction without Paraformer disagreement.**

```python
def test_semantic_open_candidate_commits_with_decisive_local_audio(tmp_path):
    audio = tmp_path / "turn.wav"
    audio.touch()
    result = service.process_turn(
        "s",
        "t1",
        "我负责南庄部门",
        meta={"audio_path": str(audio), "start_sec": 0.0, "end_sec": 2.0},
    )
    assert result["session"]["turns"][0]["current_text"] == "我负责男装部门"
```

- [ ] **Step 2: Confirm the test fails with `semantic-open candidate requires independent acoustic support`.**
- [ ] **Step 3: Remove the semantic-open acoustic hard gate; retain acoustic disagreement as `EvidenceFeatures.acoustic_support`.**
- [ ] **Step 4: Replace `direct_context_event` with history candidates added to the same pool and resolver loop.**
- [ ] **Step 5: Make all committed replacements use `replace(span, replacement, 1)` and add a repeated-span regression test.**
- [ ] **Step 6: Run focused tests and commit.**

### Task 4: Make relisten nomination safe and restore coverage recovery

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Modify: `tests/test_retrace_service.py`
- Modify: `tests/test_context_resolution.py`

- [ ] **Step 1: Preserve the existing failing coverage-risk test and add a failing healthy-turn relisten candidate test.**
- [ ] **Step 2: Confirm coverage recovery fails on the branch baseline.**
- [ ] **Step 3: Restore supported coverage recovery when the bounded relisten is non-degenerate, while retaining independent support when available as evidence.**
- [ ] **Step 4: For healthy turns, convert bounded diff replacements to `CorrectionCandidate(source="relisten_open")` and pass them to the resolver instead of creating a revision directly.**
- [ ] **Step 5: Verify ambiguous/unbounded relisten output never mutates text.**
- [ ] **Step 6: Run focused tests and commit.**

### Task 5: Add candidate-stage audit records and correct metrics

**Files:**
- Modify: `backend/asr_agent/models.py`
- Modify: `backend/asr_agent/retrace.py`
- Modify: `scripts/report_candidate_pipeline.py`
- Modify: `scripts/run_aishell4_eval.py`
- Modify: `tests/test_aishell4_eval.py`
- Modify: `tests/test_retrace_service.py`

- [ ] **Step 1: Add failing tests using candidate audit events for discovered, verifier-attempted, committed, deferred, and rejected candidates.**

```python
assert metrics == {
    "candidate_proposals": 3,
    "validated_candidates": 3,
    "verifier_attempts": 2,
    "verified_candidates": 1,
    "committed_revision_count": 1,
    "candidate_to_verifier_rate": 2 / 3,
    "verified_candidate_rate": 1 / 2,
}
```

- [ ] **Step 2: Confirm old revision-derived metrics fail.**
- [ ] **Step 3: Add `candidate_id`, `candidate_stage`, and structured candidate evidence to audit events without changing immutable raw turns.**
- [ ] **Step 4: Record resolver invocation, audio result, final action, and rejection rationale for every normalized candidate.**
- [ ] **Step 5: Update both reports to consume candidate audit events and keep candidate recall `None` until valid reference alignment exists.**
- [ ] **Step 6: Run focused tests and commit.**

### Task 6: Declare runtime dependencies and validate the complete branch

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `.env.example`
- Modify: `tests/test_deepseek_context_judge.py`

- [ ] **Step 1: Add a dependency/status test that fails when homophone support is unavailable.**
- [ ] **Step 2: Add `pypinyin`, `requests`, and `python-Levenshtein` to project dependencies and regenerate `uv.lock`.**
- [ ] **Step 3: Document `ASR_FAST_NORMAL_TURNS=0` in `.env.example`.**
- [ ] **Step 4: Run `uv sync --group dev`, then the focused suite.**
- [ ] **Step 5: Run the complete pytest suite, Python compilation, and `git diff --check`.**
- [ ] **Step 6: Review the final diff against every design requirement and commit.**

### Task 7: Three-sample evaluation gate

**Files:**
- Modify only after real evaluation: `WEEKLY_PROGRESS.md`

- [ ] **Step 1: Verify the server is running the final commit and exposes expected policy status.**
- [ ] **Step 2: Run at most three representative samples with fast mode disabled.**
- [ ] **Step 3: Inspect candidate audit records and confirm at least one real open-candidate verifier attempt.**
- [ ] **Step 4: Compare raw/final errors and stop if any unexplained harmful revision appears.**
- [ ] **Step 5: Record measured results only; defer the full evaluation until this gate passes.**
