# Observable Agent Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the realtime memory protocol through a compact, data-driven frontend.

**Architecture:** Add a read-only observability projection to session responses, persist hypothesis relationships and analysis version metadata, then render those structures in a three-column TypeScript workbench. Existing autonomous resolution and append-only replay remain unchanged.

**Tech Stack:** Python 3.11, FastAPI, dataclasses, pytest, TypeScript, Vite, CSS.

---

### Task 1: Observability Contract

**Files:**
- Modify: `tests/test_retrace_service.py`
- Modify: `tests/test_retrace_models.py`
- Modify: `backend/asr_agent/models.py`
- Modify: `backend/asr_agent/retrace.py`

- [ ] Write tests asserting `observability.latest_analysis`, short/long memory separation, relationship persistence, and revalidation metadata.
- [ ] Run focused tests and verify they fail because the contract is absent.
- [ ] Add relationship serialization and the read-only observability projection.
- [ ] Run focused tests and verify they pass.

### Task 2: Research Workbench UI

**Files:**
- Modify: `tests/test_retrace_frontend.py`
- Modify: `frontend/src/main.ts`
- Modify: `frontend/src/styles.css`

- [ ] Write source-contract tests for memory tabs, autonomous decision actions, relationship labels, version state, and removal of stale demo copy.
- [ ] Run the focused test and verify it fails for the missing UI.
- [ ] Refactor the page into transcript, live agent trace, memory inspector, ledger timeline, and compact input composer.
- [ ] Add responsive styling for desktop and mobile.
- [ ] Run frontend tests and the TypeScript production build.

### Task 3: End-to-End Verification

**Files:**
- Modify only if verification reveals a defect.

- [ ] Run the full Python test suite.
- [ ] Build the frontend and restart the local application.
- [ ] Exercise a text turn in the browser and inspect desktop and mobile screenshots.
- [ ] Confirm the worktree contains only intended changes, then commit them.
