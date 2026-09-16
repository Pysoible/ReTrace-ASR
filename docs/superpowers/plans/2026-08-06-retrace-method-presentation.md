# ReTrace Method Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver an editable template-following PPTX whose method section precisely explains the current evidence-grounded ReTrace implementation for a paper presentation.

**Architecture:** Import the provided 16-slide PPTX and preserve its existing masters, layouts, typography and recurring chrome. Reuse suitable existing slides to replace the old method material with five connected method slides: overview, reflection, verification, event replay, and a timeline example; maintain speaker notes with an internal-code provenance block.

**Tech Stack:** `@oai/artifact-tool` JavaScript ES modules, presentation template-following scripts, PowerPoint export/render utilities.

---

### Task 1: Audit the source deck and map template frames

**Files:**
- Create: `/Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-audit.txt`
- Create: `/Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-frame-map.json`
- Create: `/Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/deviation-log.txt`

- [ ] **Step 1: Inspect all source slide renders and structural metadata**

Run:

```bash
node /Users/panyibo/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/template_following_scripts/inspect_template_deck.mjs \
  --workspace /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method \
  --pptx /Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_含备注.pptx
```

Expected: `template-inspect/source-slides`, layout JSON, manifest and inspection NDJSON are present for all 16 slides.

- [ ] **Step 2: Write a five-slide map using existing content frames**

Map the method outputs to source slides with editable title/body regions: overview, reflection, verification, replay and worked example. List every reused source shape ID as a rewrite target and record every omitted source slide.

- [ ] **Step 3: Build and inspect the starter deck**

Run:

```bash
node /Users/panyibo/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/template_following_scripts/prepare_template_starter_deck.mjs \
  --workspace /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method \
  --pptx /Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_含备注.pptx \
  --map /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-frame-map.json \
  --out /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-starter.pptx \
  --preview-dir /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-starter-preview \
  --layout-dir /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-starter-layout \
  --contact-sheet /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-starter-contact-sheet.png
```

Expected: a 16-slide starter deck and a passing frame-map validation.

### Task 2: Edit copied method slides through artifact-tool

**Files:**
- Create: `/Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/edit-retrace-method.mjs`
- Modify: `/Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-starter.pptx`
- Create: `/Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_方法详说.pptx`

- [ ] **Step 1: Define code-grounded slide copy and notes**

Use these claims exactly: raw `Turn.raw_text` is immutable; `reflect_timeline` returns evidence-anchored proposals only; `_reflect` rejects invalid temporal ordering, absent quoted evidence, absent spans, identical edits and scores below `0.7`; `_replay` derives `current_text` from raw turns plus active `RevisionEvent`s; `undo_revision` deactivates a revision and replays.

- [ ] **Step 2: Import the starter and replace only mapped inherited content**

Use `PresentationFile.importPptx(await FileBlob.load(starterPptx))`; resolve mapped slide/shape IDs; rewrite mapped text and fill existing visual regions with simple native shapes/connectors. The RHR slide must show proposal fields; EPV must show the five controller gates; EBR must show raw observations, events and replay; the example must use one visual timeline and label the evidence quote.

- [ ] **Step 3: Export the editable PPTX and source notes**

Export with `PresentationFile.exportPptx(presentation)` to `/Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_方法详说.pptx`. Preserve or add `[Sources]` notes explaining that technical claims are derived from the local code files `backend/asr_agent/retrace.py` and `backend/asr_agent/integrations/deepseek.py`.

### Task 3: Render and validate the final deck

**Files:**
- Create: `/Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/final-render/`
- Create: `/Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/final-layout/`

- [ ] **Step 1: Render every final slide and inspect the montage plus each method slide**

Run:

```bash
python3 /Users/panyibo/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/container_tools/render_slides.py \
  /Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_方法详说.pptx
python3 /Users/panyibo/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/container_tools/create_montage.py \
  --input_dir /Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_方法详说 \
  --output_file /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/final-montage.png
```

Expected: all slides render; method titles remain one line; no text is clipped or overlaps unexpectedly.

- [ ] **Step 2: Run slide canvas overflow validation**

Run:

```bash
python3 /Users/panyibo/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/container_tools/slides_test.py \
  /Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_方法详说.pptx
```

Expected: no overflow warnings.

- [ ] **Step 3: Run template fidelity validation**

Run:

```bash
node /Users/panyibo/.codex/plugins/cache/openai-primary-runtime/presentations/26.802.11031/skills/presentations/template_following_scripts/check_template_fidelity.mjs \
  --workspace /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method \
  --starter-pptx /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-starter.pptx \
  --final-pptx /Users/panyibo/Documents/Codex/2026-08-04/new-chat/outputs/ReTrace-ASR_论文汇报_方法详说.pptx \
  --map /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-frame-map.json \
  --starter-layout-dir /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/template-starter-layout \
  --final-layout-dir /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method/final-layout \
  --edit-dir /Users/panyibo/Documents/Codex/2026-08-04/https-chatgpt-com-c-web-57ba31ba/.tmp-ppt-method
```

Expected: no unplanned master/layout/template deviations and no unfilled inherited placeholders.
