# Complete MOSS ReTrace Evaluation Design

## Goal

Make the MOSS-Transcribe-Diarize experiment complete, model-pure, measurable,
and fast enough to evaluate on the 20-file AISHELL-4 slice. ReTrace must be able
to produce evidence-gated revisions without calling Qwen/Omni at any stage.

The experiment compares two modes over the same complete MOSS first pass:

- `baseline`: return the complete MOSS transcript without Judge, memory,
  verifier, relisten, or revision processing.
- `retrace`: reuse the same MOSS transcript, run bounded contextual candidate
  discovery, verify candidates using a local crop transcribed by MOSS, and
  commit only revisions that pass the existing strict evidence policy.

Ground-truth TextGrid content is available only to offline scoring and failure
analysis. It must never enter prompts, memory, candidate generation, audio
verification, or online decisions.

## Confirmed Failure Modes

The server investigation on 2026-09-08 established the following facts for
`L_R003S01C02.flac`:

- The file is 2,362.945 seconds, eight-channel FLAC, and 272.785 MiB.
- Default vLLM limits rejected it for file size, missing audio decode support,
  maximum duration, and encoder-cache capacity before model inference.
- With those deployment limits corrected, the existing request returned HTTP
  200 in 36 seconds but covered only 884.94 seconds because the client sent
  `max_new_tokens`, which the transcription schema ignores.
- The installed vLLM transcription schema uses `max_completion_tokens`.
- With `max_completion_tokens=32768`, MOSS returned 465 timestamped segments,
  20,136 characters, and coverage through 2,347.34 seconds in 85 seconds.
- The current MOSS path binds neither a MOSS verifier nor a MOSS relistener.
  Strict resolution therefore defers Judge candidates with
  `verifier_unavailable`, so a nominal ReTrace run can produce no transcript
  changes.
- Calling DeepSeek once for every short MOSS segment would require roughly 465
  calls for this one recording and is not an acceptable evaluation path.

## Alternatives Considered

### A. Complete transcript, batched Judge, MOSS-local verification (selected)

Use one complete MOSS first pass, analyze bounded groups of turns with DeepSeek,
and transcribe only focused audio crops with MOSS for verification. This keeps
the experiment model-pure, permits actual evidence-gated revisions, and bounds
API cost. Local verification is not an independent-model vote; reports must
describe it as same-model focused re-observation.

### B. Minimal request fix with per-segment Judge

Only fix `max_completion_tokens` and retain one DeepSeek call per MOSS segment.
This is simple but makes long meetings unnecessarily slow and expensive, and it
does not solve the missing MOSS verifier.

### C. Semantic-only MOSS ReTrace

Allow DeepSeek to rewrite without audio evidence or keep strict mode and defer
all candidates. The former weakens the evidence contract and risks
overcorrection; the latter is safe but cannot demonstrate correction benefit.

## Architecture

### 1. Complete MOSS first pass

`moss_asr.py` sends `max_completion_tokens`, configured by
`MOSS_MAX_COMPLETION_TOKENS` with a default of 32768. The deprecated
`MOSS_MAX_NEW_TOKENS` name is accepted only as a compatibility fallback and is
never sent to vLLM.

The adapter derives `duration_sec` from the response usage object when the
top-level duration is absent. It parses timestamped segments and reports:

- `covered_until_sec`: greatest valid segment end;
- `coverage_ratio`: `covered_until_sec / duration_sec`;
- `truncated`: true when coverage falls below the configured threshold;
- `parse_tail`: any non-whitespace output remaining after the last complete
  segment, bounded in length for diagnostics.

The default completeness threshold is 0.98. A small trailing-silence allowance
is therefore accepted, while the previously observed 0.37 coverage is not.
Baseline and ReTrace responses expose the same completeness record. Evaluation
refuses to score an incomplete first pass as a valid model result.

### 2. Analysis windows without changing scoring turns

MOSS timestamped segments remain immutable ReTrace Turns so timestamps,
speakers, revisions, and offline TextGrid alignment stay precise. DeepSeek calls
are scheduled in bounded analysis windows rather than once per segment.

An analysis window closes when any configured bound is reached:

- 10 newly observed turns;
- 180 transcript characters; or
- 30 seconds of covered audio.

One Judge request audits all new turns in the window plus a bounded history
digest and may target any concrete Turn ID in that window or prior history.
The final partial window is always analyzed. The existing final session audit
remains one bounded call and must not repeat already audited candidates.

The batch contract returns zero or more normalized correction candidates. A
malformed response is retried at most once under the existing protocol; a
transport failure activates the existing cooldown and records the skipped
window. It must not fall back to one call per turn.

These bounds are configuration values so later ablations can compare quality
and cost, but the above values are the correctness-first defaults for the first
1-3 sample run.

### 3. MOSS-only focused audio verifier

The MOSS integration provides a verifier and relistener whose identity is:

```text
backend=moss-transcribe-diarize
model=MOSS-Transcribe-Diarize
```

For a proposed correction, the verifier:

1. Uses the target Turn timestamps and the existing focused-window calculation.
2. Crops that window from the source audio into a temporary mono 16 kHz WAV.
3. Sends the crop to the same MOSS transcription endpoint with a small bounded
   completion budget.
4. Removes MOSS timestamps and speaker tags from the local transcript.
5. Scores only the supplied closed candidate set using deterministic normalized
   character and pinyin similarity against the focused transcript.
6. Returns scores for exactly the supplied candidates, including `[DELETE]`
   when requested.

The temporary file is deleted in a `finally` block. Failure to decode, crop,
transcribe, align, or produce an unambiguous candidate returns a structured
failure and leaves the original text unchanged.

The verifier is same-model focused re-observation, not an independent acoustic
model. Candidate audit evidence and experimental reports state this explicitly.
The existing strict confidence and margin gates remain authoritative; DeepSeek
confidence alone can never commit a replacement.

### 4. Model-pure routing

Backend selection constructs a complete tool bundle rather than selecting only
the first-pass function:

- MOSS first pass receives only MOSS verifier and MOSS relistener identities.
- Qwen/Omni first pass receives only Qwen/Omni verifier and relistener identities.
- Missing same-family tools fail closed with `verifier_unavailable` or
  `relistener_unavailable`.
- Family mismatch fails closed before executing the tool.

No environment variable, default import, or exception fallback may route a MOSS
Turn to Qwen/Omni. Provenance is recorded for first pass, Judge, verifier, and
relistener separately.

### 5. DeepSeek and long-term memory

DeepSeek remains the semantic Context Judge, not an ASR baseline. Its model name
is recorded under the Judge role and must never replace or modify the MOSS
first-pass identity.

When no DeepSeek API key is configured, integration status reports `ready=false`
and a ReTrace evaluation is marked `judge_unavailable`; it must not be presented
as a full ReTrace effectiveness result.

Memory stays file-backed and scoped by experiment namespace plus concrete
first-pass backend/model. Windowed analysis updates the same session working
memory and the same model-scoped long-term repository. Reports expose the
memory path, scope, stable/provisional counts, last operation, and error state.
Baseline mode performs no memory reads or writes.

### 6. Deployment preflight

A server preflight command checks, without transcribing the full dataset:

- MOSS endpoint health and served model identity;
- audio decode dependency availability;
- configured upload and duration limits;
- encoder capacity for the longest selected file;
- configured completion-token budget;
- DeepSeek readiness without printing the API key;
- writable session, memory, result, and temporary directories;
- first-pass/verifier/relistener family equality.

The documented 118 launch profile uses:

```text
VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=1024
VLLM_MAX_AUDIO_DECODE_DURATION_S=3600
--max-num-batched-tokens 32768
```

The preflight fails before evaluation when a selected sample cannot fit these
limits. HTTP health alone is not considered sufficient readiness.

## Experiment Flow

For each sample, run baseline and ReTrace from the same complete MOSS first-pass
artifact. The artifact is immutable and cacheable by audio checksum, served
model identity, and inference configuration. ReTrace must not call MOSS again
for the full recording; only candidate-local crops are additional calls.

The rollout order is:

1. One sample: validate completeness, identities, DeepSeek readiness, memory
   persistence, candidate audit, and all stage timings.
2. Three samples: inspect every committed and harmed revision and identify the
   genuinely slow stage.
3. Twenty samples: run only after the first three contain no incomplete first
   passes, cross-model calls, silent Judge fallback, or unaudited revision.

Resume uses durable per-sample artifacts and never reuses an artifact whose
model identity, completion budget, or completeness threshold differs.

## Metrics

The primary 2-3 reported measures are:

1. **Effective Character Error Reduction (ECER):** net reference-character
   edits removed by ReTrace divided by baseline edits, computed only on the
   identical complete and scorable sample set.
2. **Revision precision:** improved committed revisions divided by all committed
   revisions; harmed and neutral revisions are reported separately.
3. **Candidate-to-correction yield:** improved committed revisions divided by
   validated candidates, with discovered, verifier-attempted, verified,
   deferred, and rejected counts shown as the audit funnel.

Raw and final CER remain supporting diagnostics, not the sole headline metric.
AISHELL-4 whole-record character metrics are labeled diagnostic unless the
alignment/scoring protocol satisfies the selected official scoring definition.

Cost reporting includes first-pass seconds, Judge calls and seconds, candidate
build time, verifier calls and seconds, relisten calls and seconds, ledger time,
request total, real-time factor, and peak candidate count. Baseline and ReTrace
use the same first-pass time so the incremental ReTrace cost is explicit.

## Error Handling and Audit Requirements

- HTTP 200 with incomplete timestamp coverage is an incomplete ASR result, not
  success.
- A missing DeepSeek key, cooldown fallback, malformed Judge response, and
  Judge retry are individually counted.
- Every normalized candidate receives exactly one terminal audit state for each
  evaluation attempt: committed, kept, deferred, or rejected.
- Every committed revision names the source candidate, evidence Turn IDs,
  verifier identity, local audio window, scores, threshold, and margin.
- Raw MOSS text and raw Turns are immutable. Revisions affect only the projected
  current text through the ledger.
- Failed memory consolidation is non-fatal but visible in the result.

## Testing

Focused tests cover:

- the multipart request sends `max_completion_tokens` and never
  `max_new_tokens`;
- duration extraction from vLLM `usage.seconds`;
- complete, truncated, malformed-tail, and trailing-silence coverage cases;
- evaluation rejects incomplete first-pass artifacts;
- window grouping and one Judge call per closed window;
- batch candidates preserve target Turn IDs and deduplicate correctly;
- a MOSS Turn invokes only a MOSS-identified verifier/relistener;
- crop cleanup on success and failure;
- candidate scoring returns the exact closed set and safely defers ambiguous
  output;
- baseline and ReTrace reuse the same cached complete first pass;
- missing DeepSeek readiness invalidates a full ReTrace effectiveness run;
- model-scoped memory persists across MOSS sessions and remains isolated from
  Qwen/Omni;
- stage timing and audit counts reconcile with actual calls.

After focused tests pass, run the existing integration suites and `git diff
--check`. Server acceptance then runs one sample followed by three samples with
saved JSON evidence before any 20-sample claim.

## Non-goals

- Mixing MOSS first-pass recognition with a Qwen/Omni verifier or relistener.
- Treating DeepSeek as either ASR baseline.
- Claiming that same-model focused re-observation is independent acoustic
  evidence.
- Optimizing thresholds against AISHELL-4 test references.
- Claiming metric improvement before baseline/ReTrace results exist on the same
  complete sample set.
- Running all 20 samples before the 1-3 sample audit is clean.
