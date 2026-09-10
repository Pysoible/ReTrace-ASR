# ReTrace V2 Agentic Acoustic Recovery Design

**Date:** 2026-09-10  
**Status:** Approved for implementation  
**Target branch:** `feat/open-candidate-fast-retrace-20260907`

## 1. Objective

Build a new ReTrace V2 that produces a visible, reproducible reduction in natural
AISHELL-4 character error rate (CER) while preserving Raw MOSS as an immutable,
independent baseline. V2 is a clean implementation and must not import or reuse
the legacy `backend/asr_agent` resolver.

The primary acceptance target on an unseen AISHELL-4 evaluation set is:

- at least 3% relative pooled CER reduction over Raw MOSS;
- at least 70% revision precision;
- no more than 10% of recordings harmed;
- at least 5% relative error reduction in overlap regions;
- net reductions in both substitutions and deletions/missed speech;
- a result better than direct second-ASR replacement and global ASR fusion.

The implementation must stop before a full evaluation when small-sample evidence
shows that the candidate generator cannot support these targets.

## 2. Scientific Claim

ReTrace V2 is a risk-aware selective speech-retracing agent. It preserves the
first-pass recognizer output, identifies uncertain local regions, adaptively
acquires additional acoustic evidence, and commits a local edit only when the
expected reduction in recognition error outweighs the risk of harm.

This is not an unrestricted language-model post-editor and not a conventional
two-ASR ensemble. The agent decides which evidence tool to invoke, whether to
continue investigating a region, and whether to keep, commit, or roll back a
revision under a compute budget.

## 3. Experimental Boundaries

### 3.1 Model isolation

Every artifact records explicit model roles:

- `baseline_asr`: MOSS-Transcribe-Diarize;
- `evidence_asr`: an independent Mandarin acoustic recognizer, initially
  Paraformer;
- `semantic_ranker`: DeepSeek or a fixed local text model;
- `separation_backend`: GSS/MVDR for multichannel audio, with CSS as a future
  single-channel fallback.

OMNI, when evaluated, is an independent baseline. MOSS and OMNI must never be
chained, share result directories, or appear as a composite backend/model name.

### 3.2 Ground-truth isolation

Ground truth is unavailable to localization, evidence acquisition, candidate
generation, semantic ranking, action selection, and revision submission.
Reference text is read only by an offline scorer after inference artifacts have
been finalized.

Development references may label candidates for controller calibration. This is
ordinary supervised development and must be separated by recording from the
unseen evaluation set. The 20 AISHELL-4 recordings already inspected during
legacy development are development data and cannot be reported as a blind test.

### 3.3 RTTM use

Predicted RTTM may be used by the main system for speaker activity, overlap
localization, and guided source separation. Human or reference RTTM is permitted
only in a separately labeled oracle upper-bound experiment. Every run records
`rttm_source` as `none`, `predicted`, or `oracle`.

## 4. Agent Loop

The agent state contains:

- frozen Raw MOSS segments and timestamps;
- suspicious regions and trigger reasons;
- acquired acoustic hypotheses and their provenance;
- predicted diarization and overlap evidence;
- aligned edit candidates;
- provisional and committed revisions;
- tool-call history and remaining compute budget;
- meeting-local entity and evidence memory.

For each region, the policy selects one action:

- `KEEP`: preserve Raw MOSS for the region;
- `RELISTEN`: run the evidence ASR on the original local audio;
- `EXPAND_WINDOW`: add left/right audio context and listen again;
- `SEPARATE`: invoke overlap separation;
- `RELISTEN_SEPARATED`: recognize one or more separated streams;
- `QUERY_CONTEXT`: retrieve meeting-local entity or phrase evidence;
- `COMPARE_EVIDENCE`: align hypotheses and build edit candidates;
- `COMMIT`: append an accepted local edit;
- `ROLLBACK`: withdraw a provisional edit after contradictory evidence;
- `STOP`: terminate investigation because evidence or budget is exhausted.

The loop is `observe -> choose action -> invoke tool -> update state -> decide`.
It terminates on `COMMIT`, `KEEP`, `STOP`, or budget exhaustion. Tool failure,
invalid alignment, or ambiguous cross-speaker evidence defaults to `KEEP`.

The initial policy is deterministic and auditable. After candidate coverage is
validated, a calibrated monotonic logistic controller may replace thresholds
without changing the action or artifact contracts.

## 5. Components

V2 lives in a new top-level `retrace_v2` package and must not import legacy
resolver code.

### 5.1 Schemas

Stable records include:

- `RawSegment`: model identity, speaker, timestamps, immutable text;
- `SuspiciousWindow`: time range, score, triggers, overlap probability;
- `EvidenceHypothesis`: source model/view, text, timestamps, acoustic scores;
- `EditCandidate`: localized SUB/INS/DEL operations and supporting evidence;
- `AgentAction`: action, reason, cost, state revision;
- `RevisionDecision`: KEEP/COMMIT/ROLLBACK, score, evidence, rejection reason.

### 5.2 Suspicion locator

The locator uses only inference-time signals:

- Raw MOSS versus evidence-ASR disagreement;
- token or segment confidence where available;
- repetitions, truncation indicators, pauses, and boundary instability;
- predicted overlap probability;
- disagreement across audio views.

It proposes investigation regions but cannot edit text.

### 5.3 Evidence tools

The initial evidence ASR is a Mandarin Paraformer model selected for timestamps,
speed, and architectural independence from MOSS. It returns local hypotheses,
scores, and timing metadata.

For multichannel AISHELL-4 audio, overlap-triggered separation uses predicted
speaker activity with GSS/MVDR. Separation is never run on every window. A
single-channel CSS backend is outside the first milestone.

### 5.4 Alignment and candidate graph

Evidence hypotheses are time-aligned to the Raw MOSS span. The candidate graph
represents explicit character edits:

- SUB replaces acoustically unsupported Raw characters;
- INS removes Raw characters that lack acoustic support (CER insertion repair);
- DEL inserts independently supported missed speech (CER deletion repair).

Candidate text must be acoustically attested by at least one evidence view.
The semantic ranker may score a closed candidate set but cannot invent text.
Each region retains Raw MOSS as an explicit no-edit candidate.

### 5.5 Acoustic and semantic gates

The acoustic gate measures score margin, timing consistency, number of
independent supporting views, cross-view agreement, speaker consistency, and
alignment stability.

The semantic gate evaluates only the closed candidate set against nearby
meeting context and meeting-local memory. It cannot convert a failed replacement
into an implicit deletion and cannot override missing acoustic support.

### 5.6 Risk controller

For candidate `c`, the calibrated controller estimates:

`expected_gain(c) = p_improve(c) * estimated_recovery(c)
                  - p_harm(c) * estimated_harm(c)`

with harm weighted more heavily than benefit. The controller commits only when
the expected gain, acoustic margin, and calibration threshold all pass. At most
one compatible edit is selected per overlapping span.

Development labels are derived by scoring each generated candidate against the
reference after inference. Labels are `improve`, `neutral`, and `harm`. Training
and threshold calibration split by complete meeting, never by window.

### 5.7 Meeting-local memory

Memory stores only evidence observed earlier in the same meeting:

- confirmed names, terms, and abbreviations;
- repeated pronunciations linked to acoustic evidence;
- previously rejected candidates and conflict reasons;
- speaker-specific verified forms.

No test reference and no cross-dataset reference-derived memory is allowed.
Persistent long-term memory is deferred until the acoustic core is successful.

## 6. Run Artifacts

Each run writes an append-only, inspectable bundle:

```text
runs/<run_id>/
  manifest.json
  raw_moss.jsonl
  predicted.rttm
  suspicious_windows.jsonl
  acoustic_evidence.jsonl
  candidates.jsonl
  actions.jsonl
  decisions.jsonl
  final_hypothesis.jsonl
  timings.json
  scoring/metrics.json
```

The manifest records model versions, checkpoint identifiers, model roles,
`rttm_source`, data item IDs, configuration hash, Git commit, random seed, device,
and tool availability. Scoring output is physically separated from inference
artifacts.

## 7. Diagnostic Evaluation

Every run reports both final performance and candidate potential:

- Raw pooled CER and S/D/I counts;
- final pooled CER and S/D/I counts;
- oracle-candidate CER, calculated offline with references;
- candidate coverage by error type;
- revision precision and changed-character precision;
- harmful, improved, and unchanged recording counts;
- overlap-region error rate with an explicitly documented denominator;
- action counts, error recovery per tool call, stage latency, and real-time factor.

Oracle-candidate scoring is diagnostic only. If oracle candidates cannot reduce
errors, selector tuning is prohibited because the correct evidence is absent.

## 8. Experiment Stages and Gates

### Stage 0: integrity tests

Validate model identity isolation, GT inaccessibility during inference, RTTM
provenance, edit replay, failure-to-KEEP behavior, and complete audit trails.

### Stage 1: three development recordings

Run three recordings and inspect the candidate funnel and timing. Continue only
if oracle candidates cover at least 10% of Raw errors, the final output removes
at least 20 net character errors, and no large-span harmful rewrite occurs.

If oracle recovery is below 10%, improve localization, evidence ASR, windowing,
or separation. If oracle recovery is high but final gain is below 1% relative,
improve the action policy or risk controller.

### Stage 2: inspected 20-recording development set

Use the existing inspected recordings for feature selection, controller
calibration, thresholds, ablations, and runtime optimization. Freeze code,
configuration, data IDs, and a commit before evaluation.

### Stage 3: unseen 20-recording AISHELL-4 evaluation

Run the frozen system once. The result enters the paper only if it satisfies the
acceptance criteria in Section 1. No tuning on these recordings is allowed.

### Stage 4: second meeting corpus

After AISHELL-4 succeeds, run the unchanged system on AliMeeting or another
natural Mandarin meeting corpus to support a generality claim.

## 9. Baselines and Ablations

Required baselines are:

- Raw MOSS;
- independent Paraformer alone;
- MOSS plus unrestricted semantic post-editing;
- global MOSS/Paraformer fusion;
- fixed pipeline that invokes all evidence tools;
- legacy ReTrace;
- ReTrace V2 agent;
- OMNI as a completely independent baseline when included.

Required ablations remove separation, acoustic gating, semantic gating,
meeting-local memory, learned risk calibration, or adaptive action selection.
The fixed-pipeline versus agent comparison uses an equal or reported compute
budget.

## 10. Runtime Constraints

The system uses progressive evidence acquisition:

1. cheap localization and disagreement analysis;
2. local evidence-ASR calls only on suspicious windows;
3. separation only after overlap or cross-speaker conflict is observed.

The default investigation budget covers at most a configurable fraction of the
recording, initially 20%. Actions are prioritized by expected error reduction
per unit cost. Timing is recorded for localization, relistening, separation,
alignment, semantic scoring, decision making, and total execution.

## 11. Failure Handling

- Missing audio, unavailable checkpoints, or identity mismatch fails the run
  before inference rather than silently switching models.
- Evidence-ASR timeout, separation failure, malformed RTTM, or invalid alignment
  produces a recorded failure and `KEEP` for the affected span.
- Cross-speaker contamination blocks COMMIT unless independent same-speaker
  evidence resolves it.
- Semantic evidence alone cannot authorize a text change.
- Overlapping edits are resolved before replay; unresolved conflicts keep Raw.
- Evaluation refuses to score a run whose inference manifest exposes a
  reference path or whose model-role metadata is incomplete.

## 12. Paper Reporting

Table 1 reports pooled CER, S/D/I counts, relative error reduction, overlap-region
error rate, revision precision, harmful-recording rate, and real-time factor.
The main comparison must show that the agent beats direct model replacement,
global fusion, and a compute-matched fixed pipeline.

A candidate-funnel table reports suspicious windows, generated candidates,
oracle-improving candidates, committed revisions, correct revisions, and harmful
revisions. An action analysis reports which errors are recovered by relistening,
window expansion, separation, context retrieval, and rollback.

The legacy natural result of 14.1533% to 14.1516% CER is motivation for V2, not
evidence of effectiveness. Controlled error injection is a capability probe and
must not replace natural evaluation.
