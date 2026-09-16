# ReTrace Uncertainty Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TAMA-style uncertainty discovery, competing hypotheses, action policy, evidence packets and auditable confirmations to ReTrace-ASR without weakening raw-evidence revision rules.

**Architecture:** A focused `uncertainty.py` module derives suspicious spans and action decisions from ASR metadata and ReTrace state. `retrace.py` persists that state, uses raw turns exclusively for evidence, and converts only valid `COMMIT` decisions into existing revision events. FastAPI exposes decisions and confirmations; the notebook UI renders them alongside the existing audit trail.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, pytest, TypeScript, Vite.

---

## File structure

- Create: `backend/asr_agent/uncertainty.py` — pure, testable suspicious-span and action-policy types/functions.
- Modify: `backend/asr_agent/retrace.py` — persisted hypothesis state, raw-evidence reassessment, confirmation event and controller wiring.
- Modify: `backend/asr_agent/server.py` — extended request schema and confirmation endpoint.
- Modify: `frontend/src/main.ts` — decision/candidate/evidence rendering and confirmation controls.
- Modify: `frontend/src/styles.css` — compact action and candidate-card styles.
- Modify: `tests/test_retrace_service.py`, `tests/test_retrace_api.py`, `tests/test_retrace_frontend.py` — regressions and integration coverage.

### Task 1: Add pure suspicious-span discovery

**Files:**
- Create: `backend/asr_agent/uncertainty.py`
- Modify: `tests/test_retrace_service.py`

- [ ] Write failing tests for low-confidence and N-best disagreement discovery:

```python
from asr_agent.uncertainty import detect_suspicious_spans

def test_detect_suspicious_span_combines_asr_signals():
    spans = detect_suspicious_spans(
        "请图博士审批合同",
        confidence={"图博士": 0.31},
        nbest=["请涂博士审批合同"],
    )
    assert spans[0].text == "图博士"
    assert {"low_confidence", "nbest_disagreement"} <= set(spans[0].reasons)
```

- [ ] Run `uv run pytest tests/test_retrace_service.py::test_detect_suspicious_span_combines_asr_signals -q`; confirm it fails because `asr_agent.uncertainty` is missing.
- [ ] Implement `SuspiciousSpan` and `detect_suspicious_spans(text, confidence, nbest, memory_values)` with deterministic text-diff, confidence and near-match signals. Return score-sorted spans and never alter input text.
- [ ] Re-run the focused test; confirm it passes.
- [ ] Commit `feat: detect ReTrace suspicious spans`.

### Task 2: Persist candidate competition and conservative actions

**Files:**
- Modify: `backend/asr_agent/uncertainty.py`
- Modify: `backend/asr_agent/retrace.py`
- Modify: `tests/test_retrace_service.py`

- [ ] Write failing tests:

```python
def test_high_entropy_hypothesis_waits_without_mutating_subtitle(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn("s", "t1", "请图博士审批", confidence={"图博士": .2}, text_candidates={"图博士": ["图博士", "涂博士", "屠博士"]})
    state = service.get_session("s")["turns"][0]["hypotheses"][0]
    assert state["decision"] == "WAIT"
    assert state["candidates"]
    assert service.get_session("s")["turns"][0]["current_text"] == "请图博士审批"

def test_high_risk_hypothesis_requests_user_before_commit(tmp_path):
    service = ReTraceService(tmp_path)
    service.process_turn("s", "t1", "请图博士审批", confidence={"图博士": .2}, text_candidates={"图博士": ["图博士", "涂博士"]}, risk="high")
    assert service.get_session("s")["turns"][0]["hypotheses"][0]["decision"] == "ASK_USER"
```

- [ ] Run both focused tests and confirm they fail for absent fields/argument.
- [ ] Add JSON-compatible `CandidateState` to `Hypothesis`, with candidate score, supporting/contradicting raw evidence lists; add `risk`, `decision`, `decision_rationale`, `evidence_packet`. Build candidates from current text candidates and select `WAIT`, `RETRIEVE_MEMORY`, `RELISTEN_AUDIO`, or `ASK_USER`; do not call `_commit_revision` unless action is `COMMIT`.
- [ ] Re-run focused tests and the existing service suite.
- [ ] Commit `feat: persist competing ReTrace hypotheses`.

### Task 3: Enforce raw-only evidence and commit gating

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Modify: `tests/test_retrace_service.py`

- [ ] Write failing regression tests:

```python
def test_reassessment_passes_only_later_raw_text_to_scorer(tmp_path):
    seen = []
    service = ReTraceService(tmp_path, evidence_scorer=lambda **p: seen.append(p["evidence_text"]) or {"action": "DEFER"})
    service.process_turn("s", "t1", "图博士来了", confidence={"图博士": .2}, text_candidates={"图博士": ["图博士", "涂博士"]})
    service.process_turn("s", "t2", "负责人涂博士来了", use_llm=True)
    assert seen == ["负责人涂博士来了"]

def test_verified_later_quote_promotes_candidate_to_commit(tmp_path):
    def reflector(**_):
        return [{"target_turn_id": "t1", "before_text": "图博士", "after_text": "涂博士", "score": .91,
                 "evidence": [{"turn_id": "t2", "quote": "负责人涂博士"}]}]
    service = ReTraceService(tmp_path, reflector=reflector)
    service.process_turn("s", "t1", "图博士来了")
    result = service.process_turn("s", "t2", "负责人涂博士来了")
    assert result["session"]["turns"][0]["current_text"] == "涂博士来了"
    assert result["revisions"][0]["action"] == "REVISE_TEXT"
```

- [ ] Run the raw-evidence test and confirm it fails because the scorer currently receives `current_text` aggregation.
- [ ] Change reassessment/evidence packet assembly to join later `raw_text`; validate every LLM evidence item as a `{turn_id, quote}` reference before promoting candidate evidence or calling `_commit_revision`. Preserve autonomous reflection’s existing validation behavior.
- [ ] Re-run service tests; confirm both old reflection tests and new raw-only tests pass.
- [ ] Commit `fix: gate ReTrace revisions on raw evidence`.

### Task 4: Add auditable user confirmation API

**Files:**
- Modify: `backend/asr_agent/retrace.py`
- Modify: `backend/asr_agent/server.py`
- Modify: `tests/test_retrace_api.py`

- [ ] Write failing API test:

```python
def test_confirmation_commits_known_candidate_as_auditable_event(tmp_path):
    client = TestClient(create_app(tmp_path))
    client.post("/api/sessions/s/turns", json={"turn_id": "t1", "text": "图博士到了", "confidence": {"图博士": .2}, "text_candidates": {"图博士": ["图博士", "涂博士"]}, "risk": "high"})
    response = client.post("/api/sessions/s/hypotheses/t1/confirm", json={"span": "图博士", "candidate": "涂博士", "reason": "operator confirmed"})
    assert response.status_code == 200
    assert response.json()["session"]["turns"][0]["current_text"] == "涂博士到了"
    assert response.json()["event"]["resolver"] == "user-confirmed"
```

- [ ] Run the focused API test and confirm it fails with 404.
- [ ] Add `risk`, `nbest`, `speaker` fields to `TurnRequest`; add `ConfirmHypothesisRequest`; implement `confirm_hypothesis` that only accepts an existing candidate and appends a normal revision event with `resolver="user-confirmed"`. Add the endpoint and return the session/event.
- [ ] Re-run the focused API tests plus existing API suite.
- [ ] Commit `feat: add auditable hypothesis confirmation`.

### Task 5: Render decision state and confirmations in Studio

**Files:**
- Modify: `frontend/src/main.ts`
- Modify: `frontend/src/styles.css`
- Modify: `tests/test_retrace_frontend.py`

- [ ] Write failing source-level tests requiring `candidate-card`, `evidence-packet`, `ASK_USER`, and `/hypotheses/` confirmation request strings.
- [ ] Run `uv run pytest tests/test_retrace_frontend.py -q`; confirm the new assertions fail.
- [ ] Extend frontend session types; render a candidate/action card in Agent Note for selected or first unresolved hypothesis, grouped by ASR, memory and later raw-turn evidence. Add confirm/keep-original controls only when action is `ASK_USER`; call the new API and refresh session state. Add CSS that distinguishes WAIT, RELISTEN_AUDIO, ASK_USER and COMMIT without obscuring the existing timeline.
- [ ] Re-run frontend tests and `npm --prefix frontend run build`.
- [ ] Commit `feat: show ReTrace uncertainty decisions`.

### Task 6: Document and verify the integrated workflow

**Files:**
- Modify: `README.md`
- Modify: `tests/test_retrace_api.py`

- [ ] Write a failing end-to-end test for low-confidence initial text, unresolved high-risk state, user confirmation, session reload and undo restoring raw display text.
- [ ] Run that test and confirm it fails before its required wiring is present.
- [ ] Update README’s method/API sections with suspicious spans, actions, raw-only evidence rule and confirmation endpoint.
- [ ] Run `uv run pytest -q` and `npm --prefix frontend run build`; inspect exit codes and output.
- [ ] Commit `docs: describe ReTrace uncertainty workflow`.
