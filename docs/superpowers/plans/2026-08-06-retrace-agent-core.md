# ReTrace Agent Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the restored domain-dictionary pipeline with an evidence-grounded, reversible ReTrace controller for Qwen-Omni conversation ASR.

**Architecture:** Qwen-Omni remains an audio observation adapter: it emits immutable chunks and uncertainty candidates. The ReTrace controller owns hypothesis creation, later-turn evidence assessment, append-only revision events, verified/quarantine memory, and undo. An optional LLM is a constrained evidence scorer; it never writes transcript text directly.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, pytest, TypeScript, Vite.

---

## File structure

- `backend/asr_agent/retrace.py`: session event model, hypothesis extraction, deterministic evidence decisions, revision and undo replay.
- `backend/asr_agent/server.py`: text/audio/undo APIs; Qwen chunks become turns with generated candidates.
- `backend/asr_agent/integrations/qwen_asr.py`: single-pass Qwen adapter and per-chunk uncertainty metadata only.
- `backend/asr_agent/integrations/evidence_llm.py`: optional strict-schema evidence scorer, isolated from transcript writes.
- `frontend/src/main.ts`: revision timeline, evidence explanation, and undo interaction.
- `tests/test_retrace_service.py`, `tests/test_retrace_api.py`, `tests/test_retrace_frontend.py`: behavior and regression coverage.

### Task 1: Event-sourced ReTrace state

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Modify: `tests/test_retrace_service.py`

- [ ] Write failing tests for: a revision event remains in `session.revision_events`; undo appends an `UNDO_REVISION` event and restores the previous displayed text; raw ASR text never changes.
- [ ] Run `uv run --group dev python -m pytest tests/test_retrace_service.py -q` and confirm failures occur because no event stream or undo API exists.
- [ ] Add `RevisionEvent` with event id, action, target/source turn ids, before/after text, candidate/entity, evidence, score, resolver, and optional `reverted_event_id`.
- [ ] Make transcript display derive from active events; persist `revision_events`; implement `undo_revision(session_id, event_id, reason)` as a new event rather than deletion.
- [ ] Re-run the focused tests and commit `feat: persist reversible revision events`.

### Task 2: Remove legacy dictionary and make audio create hypotheses

**Files:**
- Modify: `backend/asr_agent/integrations/qwen_asr.py`
- Delete: `backend/asr_agent/integrations/domainterms.py`
- Delete: `backend/asr_agent/integrations/evolve.py`
- Modify: `backend/asr_agent/server.py`
- Modify: `README.md`, `.env.example`
- Modify: `tests/test_retrace_api.py`

- [ ] Write failing API tests proving `/audio` produces one immutable turn per Qwen chunk with automatic uncertain-span candidates, and that no dictionary/evolve option is accepted.
- [ ] Run `uv run --group dev python -m pytest tests/test_retrace_api.py -q` and confirm the expected failures.
- [ ] Replace Pass1/Pass2 dictionary retrieval with a single-pass Qwen adapter returning text plus lightweight uncertainty candidates; create hypotheses from those candidates in `_build_session_from_asr`.
- [ ] Delete legacy imports, configuration and UI/API fields for domains, dictionaries and evolution. Keep a clear 503 error when Qwen is unavailable.
- [ ] Re-run focused API tests and commit `refactor: remove dictionary evolution from audio flow`.

### Task 3: Constrain optional LLM evidence scoring

**Files:**
- Create: `backend/asr_agent/integrations/evidence_llm.py`
- Modify: `backend/asr_agent/retrace.py`
- Modify: `backend/asr_agent/server.py`
- Modify: `tests/test_retrace_service.py`

- [ ] Write a failing test showing an LLM result outside known text/entity candidates is rejected as `DEFER`, and a valid candidate is persisted with its rationale/evidence.
- [ ] Run `uv run --group dev python -m pytest tests/test_retrace_service.py -q` and confirm failure.
- [ ] Replace direct full-text correction with an optional scorer that can only return `KEEP`, `REVISE_TEXT`, `REVISE_ENTITY`, `DEFER`, or `CLARIFY` over supplied candidates; validate all fields in the controller before committing an event.
- [ ] Re-run focused service tests and commit `feat: constrain llm evidence decisions`.

### Task 4: Expose evidence and undo in the Studio UI

**Files:**
- Modify: `frontend/src/main.ts`
- Modify: `frontend/src/styles.css`
- Modify: `tests/test_retrace_frontend.py`

- [ ] Write failing source-level tests requiring an undo action, active/reverted event state, and no legacy dictionary/domain/evolve labels.
- [ ] Run `uv run --group dev python -m pytest tests/test_retrace_frontend.py -q` and confirm failure.
- [ ] Render each revision from persisted events, link it to later-turn evidence, show original/current text, and add an undo button with optional reason input.
- [ ] Re-run focused tests and `npm --prefix frontend run build`; commit `feat: expose auditable subtitle undo`.

### Task 5: End-to-end verification and documentation

**Files:**
- Modify: `README.md`
- Modify: `tests/test_retrace_api.py`

- [ ] Add an end-to-end test: early ambiguous Turn, later evidence, revision event, GET session recovery, undo, and recovered raw text.
- [ ] Run the full suite with `uv run --group dev python -m pytest -q` and require zero failures.
- [ ] Run `npm --prefix frontend run build` and require exit code 0.
- [ ] Update README architecture and API examples to describe the evidence-only agent loop and undo semantics; commit `docs: describe reversible evidence agent`.
