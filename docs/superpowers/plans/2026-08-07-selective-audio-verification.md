# Selective Audio Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a fully automatic `WAIT`/`RELISTEN`/`REVISE` ReTrace controller and an anonymous, compilable ICASSP Overleaf paper project.

**Architecture:** Later raw transcript evidence activates a closed-set audio verification job for a past uncertain span. Only a unique candidate supported by both sources becomes an event-sourced subtitle revision. The LaTeX paper documents this implemented protocol and leaves empirical table cells explicit rather than fabricated.

**Tech Stack:** Python 3.11, FastAPI, pytest, Qwen-Omni adapter, TypeScript/Vite, LaTeX/BibTeX.

---

### Task 1: Closed-set audio verifier

**Files:**
- Create: `backend/asr_agent/integrations/audio_verifier.py`
- Create: `tests/test_audio_verifier.py`

- [ ] Write a failing test where a deterministic verifier returns normalized scores only for supplied candidates.
- [ ] Run `uv run pytest tests/test_audio_verifier.py -q`; confirm import failure.
- [ ] Implement `verify_candidates(audio_path, start_sec, end_sec, candidates)` as a Qwen JSON adapter with injectable runner; reject malformed output, unknown candidates, missing audio and ties.
- [ ] Re-run the focused test; commit `feat: add closed-set audio verifier`.

### Task 2: Automatic ReTrace actions

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Modify: `backend/asr_agent/server.py`
- Modify: `tests/test_retrace_service.py`, `tests/test_retrace_api.py`

- [ ] Write failing tests showing later raw evidence produces `RELISTEN` but no edit until verifier support arrives; dual evidence produces `REVISE`; no public confirmation endpoint remains.
- [ ] Run focused service/API tests and confirm failures.
- [ ] Persist audio reference metadata and add verifier injection. Delete `confirm_hypothesis`, confirmation schema/endpoint and all `ASK_USER` states. Populate candidate acoustic/semantic scores, call verifier only after valid later evidence, and gate commit on threshold/margin.
- [ ] Re-run focused tests and all backend tests; commit `feat: automate retrospective audio verification`.

### Task 3: Autonomous Studio state

**Files:**
- Modify: `frontend/src/main.ts`, `frontend/src/styles.css`
- Modify: `tests/test_retrace_frontend.py`

- [ ] Write failing source assertions that exclude confirmation routes/buttons and include `RELISTEN`, audio evidence and semantic evidence labels.
- [ ] Run the focused test and confirm failure.
- [ ] Render read-only action/evidence cards and remove all confirmation handlers/styles.
- [ ] Run frontend tests and production build; commit `feat: render autonomous audio verification`.

### Task 4: ICASSP Overleaf project

**Files:**
- Create: `paper/main.tex`, `paper/retrace.bib`, `paper/README.md`
- Create: `tests/test_paper_source.py`

- [ ] Write source tests requiring anonymous ICASSP document class, method equations, no human-in-the-loop claim, and no numeric result fabrication.
- [ ] Run test and confirm failure.
- [ ] Write a compact anonymous ICASSP paper: abstract, introduction, method, experiment protocol, limits, conclusion and verified references. Use clearly marked placeholders only in the experimental-results table.
- [ ] Run source test and `latexmk -pdf main.tex` when available; commit `docs: add ICASSP Overleaf manuscript`.

### Task 5: Full verification

- [ ] Run `uv run pytest -q`, `npm --prefix frontend run build`, LaTex verification and `git diff --check`.
- [ ] Inspect README and paper claims against the implementation; commit any corrections.
