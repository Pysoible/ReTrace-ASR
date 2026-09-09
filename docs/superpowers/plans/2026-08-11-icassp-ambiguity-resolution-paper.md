# ICASSP Ambiguity-Resolution Paper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute inline; the user has requested that this work not be split across many subagents.

**Goal:** Replace the obsolete ReTrace manuscript with a compile-ready ICASSP 2027 paper grounded in the current temporal-memory, selective-re-listening, versioned-revision implementation and a reproducible public-data evaluation protocol.

**Architecture:** Preserve the standalone `main.tex` plus `body/*.tex` manuscript structure, while separating the method overview into an editable TikZ source. Work in a temporary mirror because the manuscript directory is outside the writable Git workspace and is not itself a Git repository; compile and inspect the mirror, then synchronize only verified source and build artifacts back to `/Users/panyibo/ReTrace-ICASSP-Paper`.

**Tech Stack:** LaTeX, ICASSP/IEEE author template, BibTeX, TikZ, `latexmk`, Poppler (`pdfinfo`, `pdftotext`, `pdftoppm`), shell-based consistency checks.

---

## File Map

All manuscript paths below are relative to the staged working directory `/private/tmp/ReTrace-ICASSP-Paper-work` and are synchronized to `/Users/panyibo/ReTrace-ICASSP-Paper` only after verification.

- Modify `main.tex`: document class, packages, title, abstract, keywords, section order, bibliography balancing.
- Modify `body/introduction.tex`: ambiguity-first motivation, research gap, contribution claims.
- Modify `body/related.tex`: contextual ASR, ASR correction, acoustic grounding, memory-assisted correction, retrospective recognition.
- Modify `body/problem.tex`: formal temporal belief-revision task and action space.
- Modify `body/method.tex`: four approved research-style subsections, equations, calibrated policy, algorithmic flow.
- Modify `body/experiment.tex`: public datasets, later-supported slice, baselines, fairness, metrics, statistical protocol.
- Modify `body/analysis.tex`: result-table shells, ablations, sensitivity protocol, planned error analysis, limitations.
- Modify `body/conclusion.tex`: concise contribution summary without unsupported empirical claims.
- Create `figures/retrace_overview.tex`: editable two-column TikZ temporal-evidence diagram.
- Modify `retrace.bib`: verified 20--25-entry primary-source bibliography.
- Modify `README.md`: build command, result-population policy, and source-of-truth warning.

The ReTrace code is not modified by this plan. It is used as implementation evidence, especially:

- `backend/asr_agent/retrace.py`
- `backend/asr_agent/memory.py`
- `backend/asr_agent/context_judge.py`
- `backend/asr_agent/resolver.py`
- `backend/asr_agent/calibration.py`
- `backend/asr_agent/ledger.py`
- `backend/asr_agent/metrics.py`

### Task 1: Stage the Manuscript and Freeze the Implementation Truth Set

**Files:**
- Read: `/Users/panyibo/ReTrace-ICASSP-Paper/**`
- Read: `/Users/panyibo/ReTrace-ASR/backend/asr_agent/*.py`
- Create staging mirror: `/private/tmp/ReTrace-ICASSP-Paper-work/**`

- [ ] **Step 1: Record the original manuscript state**

Run:

```bash
find /Users/panyibo/ReTrace-ICASSP-Paper -maxdepth 2 -type f -print0 | sort -z | xargs -0 shasum > /private/tmp/retrace-paper-before.sha1
```

Expected: the checksum file lists every current source and build artifact without modifying the manuscript.

- [ ] **Step 2: Create a clean working mirror**

Run:

```bash
test ! -e /private/tmp/ReTrace-ICASSP-Paper-work
mkdir -p /private/tmp/ReTrace-ICASSP-Paper-work
rsync -a --exclude '.DS_Store' /Users/panyibo/ReTrace-ICASSP-Paper/ /private/tmp/ReTrace-ICASSP-Paper-work/
```

Expected: the first command confirms that no stale working copy exists; `/private/tmp/ReTrace-ICASSP-Paper-work/main.tex` and all seven existing `body/*.tex` files then exist.

- [ ] **Step 3: Capture code-backed claims before writing**

Run:

```bash
rg -n "MemoryPacket|recent_limit|confidence_threshold|MUTUALLY_EXCLUSIVE|COEXIST|TEMPORAL_CHANGE|REVISE_HISTORY|ROLLBACK|audio_margin|select_revision_threshold|mean_resolution_latency" /Users/panyibo/ReTrace-ASR/backend/asr_agent
```

Expected: matches demonstrate dual-timescale retrieval, `recent_limit=8`, long-memory threshold `0.85`, three relationship classes, selective audio policy, revision/rollback events, development threshold selection, and revision metrics.

- [ ] **Step 4: Establish forbidden obsolete claims**

Use this exact rejection list for all later checks:

```text
local-first
deterministic nomination
open re-transcription route
entity fallback
weak-case DeepSeek adjudication
locally deployed DeepSeek
new ambiguity dataset
```

Expected: none of these phrases appears as a claimed component of the rewritten method.

### Task 2: Verify the Author Format and Build the Bibliography

**Files:**
- Modify: `main.tex`
- Modify: `retrace.bib`

- [ ] **Step 1: Verify the current ICASSP 2027 format from the official author page**

Check the official publishing instructions and author kit. Record in the manuscript comments that technical content is limited to four pages and that an optional fifth page may contain references only. If the official kit supplies a mandatory LaTeX class, copy that class and sample settings into the staged directory; otherwise retain `IEEEtran` with the `conference` option.

Expected: `main.tex` follows the official current template rather than relying on an unverified recollection of prior ICASSP formatting.

- [ ] **Step 2: Replace the bibliography with verified primary sources**

Build `retrace.bib` using publication pages, proceedings pages, arXiv records only for works without a proceedings version, and official dataset repositories. Include and cite the following coverage set, removing an entry only when its metadata cannot be verified:

```text
retrospective_asr_2026  Retrospective Speech Recognition
ramc_corr_2026          Ontology Memory-Augmented ASR Correction
clozeger_2024           Listen Again and Choose the Right Answer / ClozeGER
asrec_benchmark_2025    ASR-EC Benchmark
hyporadise_2023         HyPoradise
salm_2024               It's Never Too Late / speech-augmented LM correction
rag_asr_2025            Contextual ASR with Retrieval-Augmented LLM
seal_2025               Acoustic-conditioned LLM error correction
speech_rescore_2025     Speech recognition rescoring with speech-text foundation models
audio_grounding_2025    Can Large Audio-Language Models Truly Hear?
failing_forward_2025    LLM-based ASR error correction analysis
deragec_2025            Retrieval-augmented ASR error correction
contextual_asr_2024     Contextual speech recognition
magicdata_ramc          MagicData-RAMC dataset
alimeeting_2022         AliMeeting / M2MeT dataset
aishell4_2021           AISHELL-4 background dataset
wenetspeech_2022        WenetSpeech dataset used by correction benchmarks
aishell1_2017           AISHELL-1 dataset used by correction benchmarks
thchs30_2015            THCHS-30 dataset used by correction benchmarks
selective_prediction    calibration/selective prediction source used for threshold policy
```

Expected: 20--25 unique entries, no duplicate titles, resolvable DOI/proceedings/arXiv/official URLs, and no fabricated page numbers or venues.

- [ ] **Step 3: Run bibliography integrity checks**

Run:

```bash
rg -n '^@' retrace.bib
rg -n 'title=|author=|year=' retrace.bib
```

Expected: every entry has a title, author list, year, and stable publication locator; dataset descriptions cite their paper or official repository.

### Task 3: Rewrite the Front Matter, Introduction, Related Work, and Formulation

**Files:**
- Modify: `main.tex`
- Modify: `body/introduction.tex`
- Modify: `body/related.tex`
- Modify: `body/problem.tex`

- [ ] **Step 1: Rewrite the front matter**

Set the exact title to:

```latex
\title{ReTrace: Temporal Memory and Selective Re-listening for Ambiguity Resolution in Streaming ASR}
```

Write a 140--170-word abstract with this logic: early ASR ambiguity may be irresolvable at emission time; later turns supply evidence; ReTrace uses provenance-aware dual-timescale memory, relation-aware arbitration, selective historical acoustic verification, and reversible event replay; evaluation uses public Mandarin conversational corpora and measures quality, safety, temporal behavior, and cost. Do not claim numerical superiority before results exist.

Use these keywords:

```latex
\begin{IEEEkeywords}
streaming ASR, ambiguity resolution, conversational memory, retrospective correction, selective re-listening
\end{IEEEkeywords}
```

- [ ] **Step 2: Rewrite the introduction around the temporal-direction gap**

Use three compact paragraphs:

1. Explain why names, terms, homophones, ellipsis, and referential expressions may remain ambiguous until later dialogue arrives.
2. Contrast prior-history-to-current correction with ReTrace's later-evidence-to-earlier revision; explain why direct rewriting risks overcorrection when statements coexist or describe real temporal change.
3. Present the three approved contributions using the exact concepts “temporal belief revision,” “dual-timescale provenance-aware memory,” and “selective acoustic verification with reversible revision.”

Expected: the introduction never presents generic LLM use, handcrafted terms, or a new corpus as a contribution.

- [ ] **Step 3: Rewrite related work by research distinction**

Organize `body/related.tex` into dense paragraphs rather than many shallow subsections:

```text
Contextual ASR and retrieval: prior context improves the current hypothesis.
ASR error correction and acoustic grounding: text correction is fluent but can overcorrect; audio-conditioned methods restore signal evidence.
Memory and retrospective recognition: memory-assisted current-turn correction differs from backward historical revision; Retrospective Speech Recognition is the closest temporal precedent.
ReTrace distinction: explicit relationship arbitration, selective historical audio access, and versioned rollback.
```

Expected: RAMC-Corr is accurately described as memory-assisted correction with causally available history, not as the same backward-revision task.

- [ ] **Step 4: Replace the problem formulation**

Define immutable observations and projected transcripts:

```latex
o_t=(h_t,q_t,\tau_t,m_t), \qquad
Y_t=\operatorname{Replay}(O_{1:t},E_t).
```

Define the agent mapping:

```latex
(z_t,B_t,F_t)=J(o_t,M_t^{S},M_t^{L}),
```

where `z_t` belongs to `CONSISTENT`, `NOVEL`, `CONFLICT`, or `UNCERTAIN`; each focus relation belongs to `COEXIST`, `TEMPORAL_CHANGE`, or `MUTUALLY_EXCLUSIVE`; and the decision belongs to `KEEP_OLD`, `ACCEPT_NEW`, `DEFER`, `REVISE_HISTORY`, or `ROLLBACK`.

State that the objective jointly minimizes final CER, unresolved supported errors, overcorrection, revision delay, and re-listened audio. Avoid inventing a trained end-to-end loss when the current system is policy based.

- [ ] **Step 5: Check terminology consistency**

Run:

```bash
rg -n "local-first|open re-transcription|entity fallback|degenerate|weak-case|locally deployed" main.tex body/{introduction,related,problem}.tex
```

Expected: no matches.

### Task 4: Write the Core Method and Editable Overview Figure

**Files:**
- Modify: `body/method.tex`
- Create: `figures/retrace_overview.tex`
- Modify: `main.tex`

- [ ] **Step 1: Add only the packages required by the figure and compact layout**

Add `tikz`, `microtype`, and the TikZ libraries `arrows.meta`, `positioning`, `fit`, and `calc` if they compile under the official template. Keep all text at legible IEEE sizes; do not use manual negative vertical spacing to force page count.

- [ ] **Step 2: Create the temporal-evidence overview**

Implement `figures/retrace_overview.tex` as a two-column-width TikZ figure with this left-to-right flow:

```text
t-2 early audio + immutable ASR observation
        -> short-term memory / open hypothesis
t later observation
        -> relation-aware temporal arbitration
        -> COEXIST: preserve both
        -> TEMPORAL_CHANGE: accept new state, retain history
        -> MUTUALLY_EXCLUSIVE: targeted historical re-listening
        -> calibrated action
        -> append-only event ledger -> replayed transcript
                                  <- later evidence can trigger rollback
long-term provenance memory supplies stable support to arbitration
```

Use neutral grayscale plus one restrained blue accent and one restrained red conflict accent, solid arrows for data flow, dashed arrows for conditional audio access, and a curved return arrow for rollback. The early ambiguous span and its later disambiguating evidence must be readable without relying on the caption.

Caption:

```latex
\caption{ReTrace treats streaming ASR ambiguity as temporal evidence arbitration. Later observations are compared with short- and long-term memory; only mutually exclusive conflicts permit targeted access to historical audio. Accepted changes are appended to a versioned ledger, allowing replay and rollback without mutating first-pass observations.}
```

- [ ] **Step 3: Write Section 3.1, Streaming ASR as Temporal Belief Revision**

Explain immutable observations, evolving beliefs, judgment outputs, relationship labels, and the event-projected transcript. Connect the formal variables directly to the per-turn agent flow and cite the closest retrospective-recognition work without claiming identical mechanisms.

- [ ] **Step 4: Write Section 3.2, Dual-Timescale Provenance-Aware Memory**

Define:

```latex
M_t^S=\{R_t,D_t,W_t,H_t\}, \qquad M_t^L=\{b_i,p_i,v_i\}_{i=1}^{N_t},
```

where `R_t` is the bounded recent window, `D_t` dependency-retrieved turns, `W_t` working beliefs, `H_t` open hypotheses, and each long-term item stores belief, provenance, and status/version. State the implemented consolidation condition:

```latex
\operatorname{stable}(b_i) \Leftarrow c_i\geq\tau_m \land
(|\mathcal{S}_i|\geq 2 \lor \operatorname{audioVerified}(b_i)),
```

with `tau_m=0.85` in the current implementation.

- [ ] **Step 5: Write Section 3.3, Relation-Aware Arbitration with Selective Acoustic Verification**

Explain all three relations and make the audio gate explicit:

```latex
g(f)=\mathbb{1}[r_f=\textsc{MutuallyExclusive}].
```

Define the feature vector and calibrated probability:

```latex
x_f=[c_{ctx},c_{aud},\Delta_{aud},s_{mem},n_{src}],\qquad
p_f=\sigma(w^\top x_f+b).
```

State that a revision requires historical audio, top-candidate agreement, `Delta_aud >= delta_a`, and `p_f >= tau_r`; otherwise the system defers. Describe weights as bootstrap policy weights unless a fitted calibration experiment is actually performed.

- [ ] **Step 6: Write Section 3.4, Version-Consistent Transcript Revision and Rollback**

Describe append-only events, supersession links, replay, idempotent event identifiers, stale-version rejudgment, and automatic rollback when newer verified evidence reverses the cause of an earlier edit. Use a compact algorithm block or numbered in-text procedure with this exact order:

```text
observe -> retrieve memory -> judge -> apply beliefs -> arbitrate relation
-> optionally verify localized audio -> calibrate action -> append event
-> replay transcript -> consolidate stable beliefs
```

- [ ] **Step 7: Verify method-to-code traceability**

Run:

```bash
rg -n "Streaming ASR as Temporal Belief Revision|Dual-Timescale Provenance-Aware Memory|Relation-Aware Arbitration with Selective Acoustic Verification|Version-Consistent Transcript Revision and Rollback" body/method.tex
rg -n "COEXIST|TEMPORAL_CHANGE|MUTUALLY_EXCLUSIVE|0\.85|ROLLBACK|Replay" body/method.tex
```

Expected: all four approved subsection titles and all implemented mechanism terms are present exactly once in their defining context.

### Task 5: Write the Reproducible Experimental Protocol

**Files:**
- Modify: `body/experiment.tex`

- [ ] **Step 1: Specify datasets without claiming a new corpus**

Describe MagicData-RAMC as the primary Mandarin conversational benchmark and AliMeeting as the meeting-acoustics/multi-speaker robustness benchmark. Cite exact official statistics only after confirming them from primary sources. State that all splits remain conversation-disjoint and that no manual ambiguity annotations are introduced.

- [ ] **Step 2: Define the later-supported error slice algorithmically**

Write the extraction rule in reproducible form:

```text
Run fixed Qwen ASR sequentially.
Normalize hypotheses and references with one frozen Chinese-text policy.
Align early hypotheses to references and collect incorrect reference spans.
Include an error when its reference form occurs in a later reference turn within horizon H.
Freeze H, normalization, alignments, and ASR outputs using development data only.
```

Call it a “later-supported error slice” and explicitly state that it is an automatic ambiguity proxy, not a human-grounded ambiguity label.

- [ ] **Step 3: Specify the six fair baselines**

Use this exact order and capability ladder:

```text
Raw Qwen ASR
Current-turn LLM
Sliding-context LLM
Memory-only
Always Re-listen
ReTrace
```

State that all systems share first-pass output, underlying model versions, candidate scope, context-token budget, decoding settings, and turn order. Only memory access, relation arbitration, historical audio access policy, and reversible revision differ.

- [ ] **Step 4: Specify metrics and statistical testing**

Report full-test CER and relative CER reduction; slice precision, recall, and `F_2`; overcorrection on initially correct content; rollback count and success; mean revision delay in turns; re-listened audio fraction and duration; judge calls per audio hour; and RTF. Use paired bootstrap resampling over conversations with 95% confidence intervals for primary quality differences.

- [ ] **Step 5: Specify development-only hyperparameter selection**

Document the approved sweeps:

```text
K:       2, 4, 8, 16
tau_r:   0.65, 0.72, 0.78, 0.85, 0.90
delta_a: 0.05, 0.10, 0.15, 0.20, 0.25
tau_m:   fixed at 0.85
```

Select `tau_r` by development-set `F_2` under the declared overcorrection cap and freeze all settings before test evaluation.

- [ ] **Step 6: Assert result honesty**

Run:

```bash
rg -n "outperform|significant|state-of-the-art|improves by|reduces .*%" body/experiment.tex
```

Expected: no unsupported empirical-result language.

### Task 6: Build Result, Ablation, Sensitivity, and Limitation Sections

**Files:**
- Modify: `body/analysis.tex`
- Modify: `body/conclusion.tex`

- [ ] **Step 1: Create a compact main-results table**

Use rows for all six baselines and columns for full-set CER, slice `F_2`, overcorrection, re-listened audio percentage, and RTF. Keep numeric cells as `--` until script-generated results exist. Caption must say that entries are populated from the frozen evaluation pipeline, not imply any ranking.

- [ ] **Step 2: Create the six-row mechanism ablation table**

Use rows:

```text
Full ReTrace
w/o Short-term Memory
w/o Long-term Memory
w/o Relation Arbitration
w/o Selective Re-listening
w/o Revalidation & Rollback
```

Use columns CER, slice recall, overcorrection, and re-listened audio percentage. If width is tight, abbreviate only in column headers and define abbreviations in the caption.

- [ ] **Step 3: Describe the sensitivity study without fabricating curves**

Reserve a compact panel only if real experiment output is available at execution time. Otherwise describe the three sweeps in prose and omit an empty figure; an unpopulated chart wastes technical-page space and communicates no evidence.

- [ ] **Step 4: Define result-analysis questions in advance**

Organize analysis around four falsifiable questions:

```text
Does later evidence improve earlier ambiguous spans on the full stream and proxy slice?
Does relation arbitration prevent coexistence and temporal change from becoming overcorrections?
How much historical audio does selective verification save relative to Always Re-listen?
Do revalidation and rollback recover from premature revisions?
```

Do not write answers until measurements exist.

- [ ] **Step 5: Write limitations and conclusion**

State four limitations: the automatic slice is not human ambiguity annotation; repeated later forms do not always establish the same referent; the context judge can misclassify temporal relationships; and strictly homophonic distinctions may remain acoustically irresolvable. Conclude with the method contributions and evaluation scope only, without performance adjectives.

### Task 7: Compile, Audit, and Visually Inspect the Paper

**Files:**
- Verify: all staged manuscript sources
- Generate: `main.pdf`, BibTeX and LaTeX build artifacts

- [ ] **Step 1: Clean-build the staged paper**

Run:

```bash
latexmk -C main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit code 0 and `main.pdf` produced.

- [ ] **Step 2: Fail on unresolved references or citations**

Run:

```bash
rg -n "Undefined control sequence|Citation .* undefined|Reference .* undefined|There were undefined references|multiply defined" main.log main.blg
```

Expected: no matches. Review overfull boxes individually; fix any visible overflow and all overfull boxes caused by tables, equations, or figure labels.

- [ ] **Step 3: Audit forbidden and unsupported claims**

Run:

```bash
rg -ni "local-first|deterministic nomination|open re-transcription|entity fallback|weak-case DeepSeek|locally deployed DeepSeek|we construct .*dataset|state-of-the-art|statistically significant" main.tex body retrace.bib
```

Expected: no obsolete method claims, new-dataset claims, or unsupported result claims.

- [ ] **Step 4: Check page policy and fifth-page contents**

Run:

```bash
pdfinfo main.pdf | rg '^Pages:'
```

Expected: at most five pages. If the result is `Pages: 5`, additionally run:

```bash
pdftotext -f 5 -l 5 main.pdf -
```

Expected: the extracted fifth-page text contains bibliography entries only, apart from running headers/footers allowed by the template.

- [ ] **Step 5: Render every page for visual QA**

Run:

```bash
mkdir -p /private/tmp/retrace-paper-pages
pdftoppm -png -r 160 main.pdf /private/tmp/retrace-paper-pages/page
```

Inspect every rendered page. Expected: no clipped text, overlapping columns, illegible figure labels, table overflow, isolated headings, excessive white space, or reference spill beyond page five. In Figure 1, the ambiguity timeline, three relationships, conditional audio branch, ledger, and rollback arrow must remain legible at normal page scale.

- [ ] **Step 6: Check section and citation balance**

Run:

```bash
pdftotext -layout main.pdf /private/tmp/retrace-paper.txt
rg -n "Introduction|Related Work|Problem Formulation|Temporal Evidence Arbitration|Experimental Setup|Results|Conclusion|References" /private/tmp/retrace-paper.txt
```

Expected: all required sections appear in order; Method has the largest technical allocation; references are cited in body text and do not exist solely as uncited bibliography padding.

### Task 8: Synchronize the Verified Manuscript and Rebuild in Place

**Files:**
- Modify: `/Users/panyibo/ReTrace-ICASSP-Paper/**`

- [ ] **Step 1: Review the staged delta before synchronization**

Run:

```bash
diff -ru --exclude '.DS_Store' --exclude 'main.aux' --exclude 'main.bbl' --exclude 'main.blg' --exclude 'main.fdb_latexmk' --exclude 'main.fls' --exclude 'main.log' --exclude 'main.pdf' /Users/panyibo/ReTrace-ICASSP-Paper /private/tmp/ReTrace-ICASSP-Paper-work
```

Expected: differences are limited to the planned manuscript sources, `figures/retrace_overview.tex`, and README.

- [ ] **Step 2: Confirm the source directory has not changed during editing**

Run the original checksum command again to `/private/tmp/retrace-paper-current.sha1` and compare it with `/private/tmp/retrace-paper-before.sha1`.

```bash
find /Users/panyibo/ReTrace-ICASSP-Paper -maxdepth 2 -type f -print0 | sort -z | xargs -0 shasum > /private/tmp/retrace-paper-current.sha1
diff -u /private/tmp/retrace-paper-before.sha1 /private/tmp/retrace-paper-current.sha1
```

Expected: no differences. If files changed, stop synchronization and merge those user changes into the staged copy first.

- [ ] **Step 3: Synchronize the verified working copy**

Run:

```bash
rsync -a --exclude '.DS_Store' /private/tmp/ReTrace-ICASSP-Paper-work/ /Users/panyibo/ReTrace-ICASSP-Paper/
```

Expected: verified source and compiled PDF are present in the manuscript directory; no unrelated file is deleted because `--delete` is not used.

- [ ] **Step 4: Rebuild from the final location**

Run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf | rg '^Pages:'
rg -n "Citation .* undefined|Reference .* undefined|There were undefined references" main.log
```

Expected: build succeeds in `/Users/panyibo/ReTrace-ICASSP-Paper`, page count remains at most five, and there are no unresolved citations or references.

- [ ] **Step 5: Report remaining empirical work precisely**

Final handoff must identify the rewritten source files, compiled PDF path, page count, and all numeric cells that intentionally remain `--` because local datasets and ASR models are unavailable. It must not describe the manuscript as submission-ready until those real experiments have been run and populated.
