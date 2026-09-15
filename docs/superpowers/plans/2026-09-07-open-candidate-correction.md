# Open Candidate Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase ASR correction recall beyond history-seen terms while preserving precision through one auditable, locally verified candidate-to-revision path.

**Architecture:** Normalize all correction proposals into a small internal `CorrectionCandidate` record. History, domain, semantic-open, acoustic-diff, and relisten sources feed the same pool; the existing `EvidenceResolver` remains the only replacement gate. Revision events record candidate provenance and complete projected text, so offline evaluation can distinguish recall failures from rejected candidates and harmful commits.

**Tech Stack:** Python 3, dataclasses, existing `ContextJudgment`/`FocusProposal`, `EvidenceResolver`, `RevisionLedger`, pytest, existing Levenshtein-based offline evaluation.

**Spec:** `docs/superpowers/specs/2026-09-07-open-candidate-correction-design.md`

## Global Constraints

- Optimize end-to-end correction quality, not architectural purity.
- Candidate recall may be open-world; revision submission may not be open-world.
- Ground Truth is offline evaluation only and never enters candidate generation, Judge input, resolver input, or online audio correction.
- Historical occurrence is evidence, never a correctness requirement.
- Candidate frequency must not decide correctness.
- Preserve language/script compatibility and immutable raw ASR text.
- Missing audio, invalid verifier output, or ambiguous scores produces `DEFER`/`KEEP_OLD`, never an online rewrite.
- Preserve the current strict audio thresholds for the first slice: proposed score at least `0.85` and margin at least `0.30`.
- Do not change the frontend, ASR checkpoint, speaker routing, or strict verifier thresholds in this plan.

---

### Task 1: Introduce the normalized candidate contract

**Files:**
- Create: `backend/asr_agent/correction_candidates.py`
- Test: `tests/test_correction_candidates.py`

**Interfaces:**
- Produces `CorrectionCandidate` with fields `target_turn_id`, `span`, `candidate`, `source`, `evidence_turn_ids`, `rationale`, `semantic_confidence`, `audio_required`, `audio_start_sec`, and `audio_end_sec`.
- Produces `deduplicate_candidates(candidates) -> list[CorrectionCandidate]`, keyed by `(target_turn_id, span, candidate)` and merging source labels and evidence IDs deterministically.
- Produces `candidate_to_focus(candidate) -> FocusProposal` without changing the existing public `FocusProposal` API.

- [ ] **Step 1: Write failing tests**

```python
from asr_agent.correction_candidates import CorrectionCandidate, deduplicate_candidates


def test_deduplicate_candidates_merges_sources_and_evidence():
    candidates = [
        CorrectionCandidate("t1", "南庄", "男装", "history_homophone", ["t0"]),
        CorrectionCandidate("t1", "南庄", "男装", "acoustic_diff", ["t1"]),
    ]
    result = deduplicate_candidates(candidates)
    assert len(result) == 1
    assert result[0].evidence_turn_ids == ["t0", "t1"]
    assert result[0].source == "history_homophone|acoustic_diff"
```

- [ ] **Step 2: Run `PYTHONPATH=backend pytest -q tests/test_correction_candidates.py -v` and verify it fails because the module does not exist.**
- [ ] **Step 3: Implement the dataclass, deterministic merge rules, validation, and FocusProposal adapter.**
- [ ] **Step 4: Run the focused test, then `PYTHONPATH=backend pytest -q tests/test_correction_candidates.py tests/test_context_resolution.py`.**

### Task 2: Make history and acoustic candidate discovery bounded and provenance-preserving

**Files:**
- Modify: `backend/asr_agent/integrations/deepseek.py:_homophone_candidates`
- Modify: `backend/asr_agent/retrace.py:_acoustic_focus`
- Test: `tests/test_deepseek_context_judge.py`
- Test: `tests/test_context_resolution.py`

**Interfaces:**
- `_homophone_candidates(session, current_turn, limit=64) -> list[CorrectionCandidate]` or an equivalent payload that includes `span`, `candidate`, and `evidence_turn_ids`.
- `_acoustic_focus` must produce candidates with `source="acoustic_diff"` before converting to `FocusProposal`.

- [ ] **Step 1: Keep the existing red tests and add an acoustic provenance test:** an acoustic disagreement candidate must target the current turn and include that turn as evidence.
- [ ] **Step 2: Run the focused tests and confirm the old return shape/path fails the new provenance assertions.**
- [ ] **Step 3: Scan each historical turn independently; never concatenate history across turn boundaries. Merge duplicate candidates while retaining all source turn IDs. Convert acoustic disagreement spans into the same candidate representation.**
- [ ] **Step 4: Run `PYTHONPATH=backend pytest -q tests/test_deepseek_context_judge.py tests/test_context_resolution.py`.**

### Task 3: Add semantic-open candidates and route every source through one verification gate

**Files:**
- Modify: `backend/asr_agent/integrations/deepseek.py:judge_context`
- Modify: `backend/asr_agent/retrace.py:analyze_turn`
- Modify: `backend/asr_agent/resolver.py:EvidenceResolver`
- Test: `tests/test_deepseek_context_judge.py`
- Test: `tests/test_context_resolution.py`
- Test: `tests/test_retrace_service.py`

**Interfaces:**
- Judge payload may return `semantic_open` candidate objects with `span`, `candidate/proposed_text`, `target_turn_id`, `evidence_turn_ids`, and rationale even when the candidate is absent from history.
- ReTrace converts valid candidate objects to `FocusProposal` and sends them through `EvidenceResolver`; no source bypasses the resolver.
- `RevisionEvent.evidence` includes `candidate_source:<source>` and all candidate evidence turn IDs.

- [ ] **Step 1: Add failing test for an absent-history candidate:** current text `我负责南庄部门`, Judge returns semantic-open `男装`, mock verifier scores `男装=1.0`, and the service must revise to `我负责男装部门`. Add a second test with ambiguous scores that must keep `南庄`.**
- [ ] **Step 2: Run only these tests and confirm the semantic-open case fails while the ambiguous case remains safe.**
- [ ] **Step 3: Normalize Judge-provided open candidates, validate target span/script, attach `semantic_open` provenance, and route through the existing closed-set verifier with current thresholds.**
- [ ] **Step 4: Ensure revision events retain complete `after_text`, source labels, evidence IDs, audio scores, and rationale.**
- [ ] **Step 5: Run `PYTHONPATH=backend pytest -q tests/test_deepseek_context_judge.py tests/test_context_resolution.py tests/test_retrace_service.py`.**

### Task 4: Unify bounded relisten output with the candidate pool

**Files:**
- Modify: `backend/asr_agent/retrace.py:_relisten_uncertain_window`
- Modify: `backend/asr_agent/integrations/deepseek.py:_audio_diff_fallback`
- Test: `tests/test_retrace_service.py`
- Test: `tests/test_context_resolution.py`

**Interfaces:**
- Relisten replacements enter as `CorrectionCandidate(source="relisten_open")` or `source="acoustic_diff"` with explicit audio bounds.
- Unsupported or unbounded relisten output remains an audit/defer event and cannot mutate text.

- [ ] **Step 1: Add a failing test where open relisten proposes a bounded replacement absent from history and decisive verifier scores commit it.**
- [ ] **Step 2: Add a failing safety test where relisten returns an unbounded or ambiguous replacement and text remains unchanged.**
- [ ] **Step 3: Route relisten output through the candidate adapter and existing resolver; remove only the duplicate direct submission path that bypasses candidate provenance.**
- [ ] **Step 4: Run the focused ReTrace/context suite and verify both commit and safety cases.**

### Task 5: Add offline candidate-recall and revision-quality reporting

**Files:**
- Modify: `scripts/run_aishell4_eval.py:retrace_miss_analysis`
- Create: `scripts/report_candidate_pipeline.py`
- Test: `tests/test_aishell4_eval.py`

**Interfaces:**
- Offline report consumes completed session JSON and reference text only after inference.
- Report fields: `candidate_recall`, `candidate_to_verifier_rate`, `verified_candidate_rate`, `committed_revision_count`, `corrected_turn_count`, `harmed_turn_count`, `neutral_turn_count`, `raw_cer`, `final_cer`, and `net_edits_removed`.

- [ ] **Step 1: Add metric tests using synthetic session events with candidate source and verifier evidence.**
- [ ] **Step 2: Run the metric tests and verify they fail because the new counters are absent.**
- [ ] **Step 3: Parse candidate and revision evidence without reading reference text until post-inference scoring; compute the new counters.**
- [ ] **Step 4: Run `PYTHONPATH=backend pytest -q tests/test_aishell4_eval.py` and a report against the existing 20-sample results.**

### Task 6: Full validation and controlled evaluation

**Files:**
- Modify: `docs/WEEKLY_PROGRESS.md` or `WEEKLY_PROGRESS.md` with measured results only after evaluation completes.

- [ ] **Step 1: Run the complete focused suite:** `PYTHONPATH=backend pytest -q tests/test_correction_candidates.py tests/test_deepseek_context_judge.py tests/test_context_resolution.py tests/test_retrace_service.py tests/test_aishell4_eval.py`.
- [ ] **Step 2: Run static validation on modified Python files with `python -m py_compile`.**
- [ ] **Step 3: Run a small offline replay or no-more-than-three-sample audio evaluation before the full 20-sample run.**
- [ ] **Step 4: Compare baseline and new metrics, including harmed revisions; stop and reassess if harm rate increases materially.**
- [ ] **Step 5: Run full 20-sample evaluation only after the small evaluation confirms recall improvement without unacceptable harm.**
- [ ] **Step 6: Record results, limitations, and whether the open-candidate path was actually exercised by real audio.**
