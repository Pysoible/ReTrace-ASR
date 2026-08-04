# ReTrace-ASR Clean-Slate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy domain-term/TAMA project with a standalone, explainable retrospective conversational ASR system.

**Architecture:** The backend is a single ReTrace service over versioned transcript state and evidence-grounded entity resolution. The Vite frontend renders the approved timeline and decision-explainer view from its REST API.

**Tech Stack:** Python 3.12, FastAPI, Pydantic, JSON persistence, Vite, TypeScript, CSS, pytest.

---

### Task 1: Specify the new public contract

**Files:**

- Create: `tests/test_retrace_service.py`
- Create: `tests/test_retrace_api.py`
- Create: `tests/test_retrace_frontend.py`

- [ ] Write failing tests for future-only text revision, entity-only revision, deferred ties, quarantine-to-verified promotion, and immutable observations.
- [ ] Write failing FastAPI tests for `POST /api/sessions/{id}/turns`, `GET /api/sessions/{id}`, and entity profile registration.
- [ ] Write static frontend tests asserting timeline, decision-explainer, and absence of legacy terminology.
- [ ] Run the focused test files and verify failure is caused by the absent ReTrace-only contract.

### Task 2: Replace backend and API

**Files:**

- Create: `backend/asr_agent/retrace.py`
- Replace: `backend/asr_agent/server.py`
- Replace: `backend/asr_agent/mcp/server.py`
- Replace: `backend/asr_agent/cli.py`
- Delete: `backend/asr_agent/tools/asr_tools.py`
- Delete: `backend/asr_agent/tools/tama.py`
- Delete: legacy agent, state, realtime, settings, and API modules when no longer imported.

- [ ] Implement state, evidence scoring, action policy, verified/quarantine memory, and JSON persistence.
- [ ] Implement the new REST, Qwen observation adapter, MCP, and CLI contracts without legacy aliases.
- [ ] Run focused backend tests until green.

### Task 3: Replace the frontend

**Files:**

- Replace: `frontend/src/main.ts`
- Replace: `frontend/src/styles.css`
- Replace: `frontend/src/api.ts`
- Delete: legacy frontend tests and add ReTrace workspace tests.

- [ ] Render session timeline, revisions, evidence cards, candidates, and audit log.
- [ ] Add responsive layout and accessible status labels.
- [ ] Build the Vite application and run frontend regression tests.

### Task 4: Remove legacy surface and verify

**Files:**

- Replace: `README.md`
- Modify: `pyproject.toml`
- Delete: legacy tests and unused modules.

- [ ] Remove old dependencies, routes, MCP tools, documentation, and terminology.
- [ ] Run `rg` hygiene checks, the full test suite, backend compilation, and frontend build.
- [ ] Review the diff and commit the finished migration.
