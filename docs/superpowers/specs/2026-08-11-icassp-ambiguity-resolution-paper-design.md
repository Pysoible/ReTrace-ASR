# ReTrace ICASSP Ambiguity-Resolution Paper Design

**Date:** 2026-08-11  
**Target:** ICASSP 2027, four technical pages plus an optional references-only fifth page  
**Manuscript:** `/Users/panyibo/ReTrace-ICASSP-Paper`

## 1. Objective and Scope

Rewrite the manuscript around the current ReTrace implementation rather than the obsolete paper narrative. The paper studies a specific streaming-ASR problem: an early recognition hypothesis can remain ambiguous until a later conversational turn supplies discriminative evidence. ReTrace treats this setting as temporal evidence arbitration, where later evidence may support the old interpretation, establish a genuine state change, coexist with it, or justify revising it.

The paper will not claim a new dataset, treat generic LLM use as an innovation, describe the DeepSeek judge as locally deployed, or invent experimental results. It will use public datasets, reproducible evaluation slices, and explicit result placeholders until experiments have been executed.

## 2. Paper Positioning

### Proposed title

**ReTrace: Temporal Memory and Selective Re-listening for Ambiguity Resolution in Streaming ASR**

### Central claim

Streaming ASR ambiguity is a temporal evidence-arbitration problem. A reliable system must preserve uncertain earlier observations, relate later evidence to prior beliefs, selectively recover acoustic evidence when text is insufficient, and support reversible historical revision.

### Research gap

Most contextual ASR and error-correction systems use preceding context to improve the current utterance. ReTrace addresses the reverse temporal direction: later turns can reveal that an earlier ASR interpretation was wrong. Unlike unconstrained transcript rewriting, ReTrace separates semantic coexistence, genuine temporal change, and mutually exclusive conflict before deciding whether any historical text should change.

### Contributions

1. **Temporal belief-revision formulation.** ReTrace formalizes streaming ASR ambiguity as relation-aware arbitration between immutable observations and evolving beliefs. New evidence can support, replace, defer, or coexist with an earlier interpretation.
2. **Dual-timescale provenance-aware memory.** Short-term memory retains recent and dependent turns, working beliefs, and open hypotheses; long-term memory stores only stable, source-traceable beliefs supported across sessions or verified against audio.
3. **Selective acoustic verification with reversible revision.** ReTrace requests historical audio only for mutually exclusive ambiguity, records changes in a versioned event ledger, and allows later evidence to revise or roll back prior decisions.

## 3. Method Design

The method section title is **ReTrace: Temporal Evidence Arbitration**. It uses research-oriented subsection titles rather than product-feature labels.

### 3.1 Streaming ASR as Temporal Belief Revision

At turn `t`, the first-pass recognizer emits an immutable observation `o_t`. The visible transcript is a projection rather than an overwritten source:

`Y_t = Replay(O_{1:t}, E_t)`,

where `E_t` is an append-only sequence of revision events. Given memory `M_t` and the newest observation, the context judge produces a judgment class, belief proposals, and focused relationships to earlier evidence. The judgment classes are `CONSISTENT`, `NOVEL`, `CONFLICT`, and `UNCERTAIN`.

The action space contains `KEEP_OLD`, `ACCEPT_NEW`, `DEFER`, `REVISE_HISTORY`, and `ROLLBACK`. `COEXIST` is represented as a relationship outcome that preserves both compatible beliefs rather than overwriting one.

### 3.2 Dual-Timescale Provenance-Aware Memory

Short-term memory contains recent turns, dependency-retrieved turns, working beliefs, and unresolved hypotheses. Retrieval is sparse: the agent receives relevant dependencies in addition to a bounded recent window rather than an unbounded transcript dump.

Long-term memory contains stable beliefs and their provenance. A belief is consolidated only when its confidence reaches the configured threshold and it is either supported by at least two independent sessions or verified against audio. Superseded beliefs remain auditable instead of being destructively erased.

### 3.3 Relation-Aware Arbitration with Selective Acoustic Verification

For each focused discrepancy, the judge predicts one of three temporal-semantic relationships:

- `COEXIST`: both statements may be true, so no historical rewrite is allowed.
- `TEMPORAL_CHANGE`: the new statement represents a later state, so the new state is accepted while the historical record is retained.
- `MUTUALLY_EXCLUSIVE`: the two interpretations cannot both be true for the relevant time, so targeted historical acoustic verification is permitted.

The verifier receives only the localized historical audio and constrained alternatives relevant to the conflict. This creates a direct accuracy-cost comparison against always-relisten systems.

The calibrated revision policy uses context confidence, audio confidence, the margin between audio-supported alternatives, memory support, and independent-source count. A historical revision requires available acoustic evidence, a valid top proposal, a sufficient audio margin, and calibrated probability above the development-set threshold `tau_r`.

### 3.4 Version-Consistent Transcript Revision and Rollback

Every analysis and event carries the version of the evidence from which it was derived. The ledger replays immutable ASR observations plus active events to produce the current transcript. If later verified evidence invalidates the cause of an earlier revision, ReTrace appends a rollback event and reprojects the transcript. Stale analyses are rejudged rather than committed against a newer state.

## 4. Data and Evaluation Protocol

### Public datasets

- **MagicData-RAMC** is the primary conversational Mandarin benchmark because it contains long conversations, timestamps, multiple domains, and mobile-channel speech.
- **AliMeeting** evaluates robustness under Mandarin meeting acoustics and multi-speaker interaction.

No new corpus or manual ambiguity annotation is introduced.

### Later-supported error slice

The full public test sets remain the basis for headline CER and safety results. A reproducible analysis slice isolates cases where later context can plausibly disambiguate an earlier error:

1. Run a fixed first-pass Qwen ASR sequentially over each public test conversation.
2. Align each early hypothesis with its reference and identify incorrect reference tokens or spans.
3. Include an early error when its correct reference form occurs in a later reference turn within a fixed horizon `H`.
4. Freeze the extraction script, normalization, horizon, and ASR outputs before evaluating correction systems.

This subset must be called a **later-supported error slice** or **ambiguity proxy slice**, not a human-annotated ambiguity dataset. Conclusions must be reported alongside full-test-set results because repeated reference forms are only a proxy for genuine conversational disambiguation.

## 5. Baselines and Fairness

All systems share the same first-pass ASR outputs, underlying language/audio models where applicable, candidate scope, and context-token budget. They differ only in the information and decision mechanisms available to them.

1. **Raw ASR:** unchanged Qwen first-pass transcripts.
2. **Current-turn LLM:** correction of the current utterance without memory.
3. **Sliding-context LLM:** direct correction using the latest `K` textual turns.
4. **Memory-only:** dual-timescale memory and context judgment without historical audio verification.
5. **Always Re-listen:** acoustic verification for every nominated discrepancy, providing an upper-cost comparison.
6. **ReTrace:** relation arbitration, dual-timescale memory, selective verification, and reversible event-ledger revision.

Published systems such as ClozeGER and RAMC-Corr will be discussed in related work. They enter the experimental table only if their code, model inputs, and evaluation conditions can be reproduced fairly; the main study must not depend on unavailable implementations.

## 6. Ablations

The main ablation table removes one mechanism at a time:

1. **w/o Short-term Memory**
2. **w/o Long-term Memory**
3. **w/o Relation Arbitration** (treat all conflicts as mutually exclusive)
4. **w/o Selective Re-listening** (text and memory evidence only)
5. **w/o Revalidation & Rollback** (committed revisions are frozen)

The always-relisten baseline separately quantifies the quality-cost effect of selection.

## 7. Hyperparameters

Only behaviorally meaningful hyperparameters are analyzed:

- Recent-memory window `K` in `{2, 4, 8, 16}` turns.
- Historical-revision threshold `tau_r` in `{0.65, 0.72, 0.78, 0.85, 0.90}`.
- Acoustic-evidence margin `delta_a` in `{0.05, 0.10, 0.15, 0.20, 0.25}`.

The long-term-memory confidence threshold remains fixed at the implementation default of `0.85`, together with the independent-session or audio-verification requirement. Hyperparameters are selected once on development data and frozen for test evaluation. Threshold selection optimizes `F_2` to reflect the stated preference for high error-recovery recall while tolerating a small number of incorrect edits.

A compact sensitivity plot shows the effect of `K`, `tau_r`, and `delta_a` on quality, overcorrection, and re-listening cost. Learned or heuristic calibration weights must be reported accurately according to the final experimental implementation; they must not be presented as trained parameters unless a training protocol exists.

## 8. Metrics

- **Overall quality:** final character error rate (CER) and relative CER reduction.
- **Ambiguity resolution:** precision, recall, and `F_2` on the later-supported error slice.
- **Revision safety:** overcorrection rate on initially correct content, rollback count, and rollback success rate.
- **Temporal behavior:** mean revision delay measured in turns and time-to-resolution on the analysis slice.
- **Computation:** fraction and duration of re-listened audio, judge calls per audio hour, and real-time factor (RTF).

The main result table reports full-set CER, ambiguity recall or `F_2`, overcorrection, and re-listening fraction. A compact figure or secondary table reports latency and cost. No metric is populated until its experiment has been run.

## 9. Manuscript Structure and Page Budget

1. **Introduction** (approximately 0.6 page): ambiguity motivation, temporal-direction gap, and contributions.
2. **Related Work and Problem Formulation** (approximately 0.45 page): contextual ASR, ASR error correction, acoustic grounding, memory, and formal task definition.
3. **ReTrace: Temporal Evidence Arbitration** (approximately 1.5 pages): four method subsections, one system figure, equations, and a compact algorithm.
4. **Experimental Setup** (approximately 0.6 page): datasets, slice construction, baselines, metrics, and implementation details.
5. **Results and Analysis** (approximately 0.8 page): main results, ablations, sensitivity, and cost-quality trade-off.
6. **Conclusion and Limitations** (approximately 0.2 page): conclusions and explicit limitations.

The optional fifth page is references only. The final balance may shift during typesetting, but technical content must remain within four pages.

## 10. Figures and Tables

- **Figure 1:** a left-to-right temporal example and method overview. An early ambiguous ASR observation remains open; a later turn triggers relation arbitration; only a mutually exclusive conflict targets historical audio; the ledger revises or rolls back the projected transcript.
- **Algorithm 1:** per-turn memory retrieval, context judgment, relationship gating, selective verification, calibrated decision, ledger append, replay, and consolidation.
- **Table 1:** full-test and later-supported-slice comparison across all baselines.
- **Table 2:** compact mechanism ablation table.
- **Figure 2 or compact panel:** hyperparameter and quality-cost sensitivity, included only if space and real results permit.

The figure must visually distinguish immutable observations, mutable beliefs, memory stores, audio access, and event-ledger projection. It must not imply that every turn is segmented, corrected, or re-listened to.

## 11. Literature Plan

Expand the bibliography from four entries to approximately 20--25 carefully selected primary sources. Coverage includes:

- retrospective speech recognition and later-evidence revision;
- contextual and retrieval-augmented ASR;
- text-only and multimodal ASR error correction;
- ClozeGER, HyPoradise, and recent ASR correction benchmarks;
- memory-augmented correction, including RAMC-Corr;
- public dataset papers for MagicData-RAMC and AliMeeting;
- calibration or selective-prediction work only where directly used by the method.

Claims about each prior method must follow its published temporal direction and input assumptions. In particular, memory-assisted correction of current or future hypotheses must not be described as equivalent to backward historical revision.

## 12. Implementation Truthfulness and Result Policy

The manuscript is grounded in the current repository implementation:

- The context judge is model/API based. The current DeepSeek adapter uses an external client configured for the public DeepSeek API, so the paper must not call it a locally deployed model unless the implementation and experiments change.
- Raw ASR observations remain immutable; revision is represented through events and replay.
- Long-term consolidation requires confidence plus independent-session support or audio verification.
- Relationship arbitration gates historical audio access.
- Version checks, revalidation, and rollback are part of the claimed method and therefore require direct tests and experiment logging.

Before submission, every quantitative placeholder must be replaced by script-generated results. If an experiment is not completed, its claim, table row, and comparative conclusion must be removed rather than estimated.

## 13. Acceptance Criteria

The rewritten manuscript is acceptable when:

1. The title, abstract, introduction, method, experiments, and conclusion consistently frame the task as ambiguity resolution through temporal evidence arbitration.
2. Every claimed mechanism maps to current code or to an explicitly planned experiment-support change.
3. The paper contains no obsolete bigram/entity heuristics, no generic local-first fallback narrative, and no incorrect local-DeepSeek claim.
4. Dataset, baseline, ablation, hyperparameter, metric, and fairness protocols are reproducible and do not require a newly collected dataset.
5. All citations are real, primary where possible, and semantically support the surrounding claims.
6. Tables contain explicit non-numeric placeholders until real runs are available; prose draws no unsupported performance conclusion.
7. The compiled PDF fits the ICASSP page policy, has legible figures and tables, and passes a final visual and reference-integrity review.
