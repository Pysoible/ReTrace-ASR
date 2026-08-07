# Full ICASSP Manuscript Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the standalone ReTrace manuscript into an approximately four-page ICASSP technical paper whose algorithmic claims match the current local-first sparse implementation.

**Architecture:** Keep `main.tex` as the IEEE conference entry point and organize technical material in focused `body/*.tex` files. The paper will make resolver paths explicit: deterministic nomination, localized candidate audio verification, weak-case DeepSeek escalation, degenerate open re-transcription, and separately reported symbolic entity fallback.

**Tech Stack:** LaTeX (`IEEEtran`, pdfLaTeX, BibTeX), standalone Overleaf folder at `/Users/panyibo/ReTrace-ICASSP-Paper`.

---

## File structure

- `/Users/panyibo/ReTrace-ICASSP-Paper/main.tex`: title, abstract, keywords, section order, bibliography.
- `/Users/panyibo/ReTrace-ICASSP-Paper/body/introduction.tex`: problem, motivation, contributions.
- `/Users/panyibo/ReTrace-ICASSP-Paper/body/related.tex`: directly relevant ASR and audio-language-model work.
- `/Users/panyibo/ReTrace-ICASSP-Paper/body/problem.tex`: formal streaming state, evidence objects, resolver outputs.
- `/Users/panyibo/ReTrace-ICASSP-Paper/body/method.tex`: local-first routing, local audio window, strong/weak gates, open recovery, audit semantics, cost analysis.
- `/Users/panyibo/ReTrace-ICASSP-Paper/body/experiment.tex`: data construction, baselines, metrics, ablations, statistics, blank result templates.
- `/Users/panyibo/ReTrace-ICASSP-Paper/body/analysis.tex`: worked trace, failure taxonomy, limitations.
- `/Users/panyibo/ReTrace-ICASSP-Paper/body/conclusion.tex`: concise conclusion and empirical next step.
- `/Users/panyibo/ReTrace-ICASSP-Paper/retrace.bib`: citations used by the expanded text.

### Task 1: Add a formal problem definition

**Files:**
- Create: `/Users/panyibo/ReTrace-ICASSP-Paper/body/problem.tex`
- Modify: `/Users/panyibo/ReTrace-ICASSP-Paper/main.tex`

- [ ] **Step 1: Write a failing structural check**

Create a temporary shell assertion that requires the new input and resolver-path notation:

```bash
rg -F '\\input{body/problem}' /Users/panyibo/ReTrace-ICASSP-Paper/main.tex
rg -F 'r_t \in' /Users/panyibo/ReTrace-ICASSP-Paper/body/problem.tex
```

- [ ] **Step 2: Run the check to verify it fails**

Run the two commands above. Expected: the first command has no match because `problem.tex` is not yet included.

- [ ] **Step 3: Add formal notation**

Define immutable observation $o_t=(h_t,\mathcal{N}_t,q_t,\tau_t)$, later raw evidence, a decision-time candidate set, and resolver labels `audio-semantic-gate`, `audio-llm-confirm`, `audio-open-relisten`, and symbolic fallback. State that candidate ambiguity and degenerate recovery are different tasks.

- [ ] **Step 4: Run the structural check**

Run the commands from Step 1. Expected: both commands print a matching line.

### Task 2: Expand the method into explicit resolver paths

**Files:**
- Modify: `/Users/panyibo/ReTrace-ICASSP-Paper/body/method.tex`

- [ ] **Step 1: Write a failing method-content check**

```bash
for term in 'Local-First' 'Strong and Weak Audio Gates' 'Degenerate-Transcript Recovery' 'Complexity and Cost'; do
  rg -F "$term" /Users/panyibo/ReTrace-ICASSP-Paper/body/method.tex || exit 1
done
```

- [ ] **Step 2: Run the check to verify it fails**

Run the loop. Expected: at least one required subsection title is absent.

- [ ] **Step 3: Expand the method**

Add four subsections: deterministic local nomination; local-window construction and candidate scoring; strong gate ($\geq0.70$, margin $\geq0.10$) versus weak top-candidate escalation ($\geq0.55$); and degenerate open re-transcription. Add a complexity paragraph defining $K=|\mathcal{C}_i|$, re-listened seconds, and LLM calls per session.

- [ ] **Step 4: Run the method-content check**

Run the loop from Step 1. Expected: all four headings are found.

### Task 3: Make the experimental protocol reproducible and publication-ready

**Files:**
- Modify: `/Users/panyibo/ReTrace-ICASSP-Paper/body/experiment.tex`

- [ ] **Step 1: Write a failing protocol-content check**

```bash
for term in 'Dataset Construction' 'Annotation Protocol' 'Statistical Testing' 'DeepSeek calls per session'; do
  rg -F "$term" /Users/panyibo/ReTrace-ICASSP-Paper/body/experiment.tex || exit 1
done
```

- [ ] **Step 2: Run the check to verify it fails**

Run the loop. Expected: the current concise protocol lacks at least one heading.

- [ ] **Step 3: Add executable experiment details**

Specify session segmentation, retained audio/timestamp records, ambiguity and degeneracy labels, double annotation with adjudication, exact baseline families, CER/WER, ARA, revision precision, over-correction, recovery accuracy, audio seconds, LLM calls, bootstrap confidence intervals, paired significance testing, and resolver-specific ablations. Keep all result cells as `--`.

- [ ] **Step 4: Run the protocol-content check**

Run the loop from Step 1. Expected: all headings are found and no numeric result claims are introduced.

### Task 4: Add analysis, failure modes, and limitations

**Files:**
- Create: `/Users/panyibo/ReTrace-ICASSP-Paper/body/analysis.tex`
- Modify: `/Users/panyibo/ReTrace-ICASSP-Paper/main.tex`

- [ ] **Step 1: Write a failing structural check**

```bash
rg -F '\\input{body/analysis}' /Users/panyibo/ReTrace-ICASSP-Paper/main.tex
rg -F 'Failure Modes' /Users/panyibo/ReTrace-ICASSP-Paper/body/analysis.tex
```

- [ ] **Step 2: Run the check to verify it fails**

Run the commands. Expected: neither file inclusion nor the new section exists.

- [ ] **Step 3: Add a worked trace and limitations**

Provide a non-numeric example showing later evidence, local audio scores, strong-gate revision, and abstention. Enumerate unsupported semantic evidence, timestamp misalignment, audio-verifier error, candidate-set miss, and degenerate re-transcription error. Explicitly disclose the symbolic fallback as non-audio.

- [ ] **Step 4: Run the structural check**

Run the commands from Step 1. Expected: both match.

### Task 5: Expand citations and compile the final manuscript

**Files:**
- Modify: `/Users/panyibo/ReTrace-ICASSP-Paper/retrace.bib`
- Modify: `/Users/panyibo/ReTrace-ICASSP-Paper/main.tex`
- Modify: all files under `/Users/panyibo/ReTrace-ICASSP-Paper/body/`

- [ ] **Step 1: Run an initial build**

```bash
cd /Users/panyibo/ReTrace-ICASSP-Paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: successful build, with any page-count, undefined-citation, or overfull-box issues visible in the log.

- [ ] **Step 2: Add only cited, relevant bibliography records**

Add primary sources needed for the expanded related-work and evaluation discussion. Do not add unreferenced bibliography padding.

- [ ] **Step 3: Repair LaTeX warnings and control length**

Use short equations, split wide tables, and concise paragraphs until the body is approximately four IEEE two-column pages before references. Do not reduce content by deleting method-path disclosures.

- [ ] **Step 4: Run final verification**

```bash
cd /Users/panyibo/ReTrace-ICASSP-Paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
if rg -F 'Overfull \\hbox' main.log || rg -F 'undefined references' main.log || rg -F 'undefined on input' main.log; then exit 1; fi
pdfinfo main.pdf | rg '^Pages:'
```

Expected: compilation succeeds, no undefined citations/references or overfull boxes, and a page count near four before references.
