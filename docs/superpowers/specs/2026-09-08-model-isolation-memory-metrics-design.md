# Model-Isolated ReTrace, Durable Memory, and Task-Aligned Metrics

## Problem

MOSS and Qwen3-Omni are independent experimental baselines. The current public
repository has only a Qwen-Omni audio adapter and uses it as the default targeted
verifier and open relisten implementation. A server-side MOSS response can be
labelled `backend=moss-transcribe-diarize` while the global Qwen model path is
still displayed as its model. This makes provenance ambiguous and can silently
turn a MOSS run into a mixed MOSS/Omni pipeline.

Long-term memory is also effectively isolated per uploaded audio because the
audio endpoints reset each session with `memory_scope=bound_session`. Stable
beliefs therefore cannot be reused by later audio sessions. Consolidation and
retrieval failures are currently converted to empty results without observable
status, so a permission or JSON error looks identical to "no memory learned".

The experiment needs model-pure pipelines, durable but isolated memory, and a
small metric set aligned with retrospective correction rather than relying on
corpus-wide CER alone.

## Confirmed Experimental Boundary

- MOSS first-pass recognition may use only MOSS for targeted verification and
  open relisten.
- Qwen3-Omni first-pass recognition may use only Qwen3-Omni for targeted
  verification and open relisten.
- There is no cross-model fallback. If the matching verifier or relistener is
  unavailable, the candidate is deferred with an explicit reason.
- MOSS and Omni never share long-term memory.
- Sessions produced by the same backend and concrete model reuse one durable
  memory scope across audio files.
- Pure baseline runs do not invoke ReTrace, memory, Judge, verifier, or relisten.

## Considered Approaches

### A. Relabel the existing mixed pipeline

Keep Qwen as the verifier for every backend but display it separately. This is
easy and useful for a hybrid system experiment, but it violates the confirmed
baseline boundary and is rejected.

### B. Model-aware routing with fail-closed compatibility

Attach a concrete identity to every audio stage and route verification/relisten
only to an implementation registered for the first-pass identity. Missing
implementations defer instead of falling back. This works with the current
repository while allowing a private MOSS adapter to register its own functions.
This is the selected approach.

### C. Separate MOSS and Omni service deployments

Run distinct API processes and code configurations. This provides strong
operational isolation but duplicates deployment and still needs structured
provenance and memory namespaces. It may be used operationally later, but it is
not required by this code change.

## Stage Identity Contract

Introduce a serializable model identity with these fields:

```json
{
  "backend": "qwen-omni-vllm",
  "model": "Qwen3_Omni_30B",
  "role": "first_pass"
}
```

Roles are `first_pass`, `context_judge`, `acoustic_sentinel`,
`targeted_verifier`, and `open_relistener`. The session/result provenance is a
map keyed by role. `backend` and `model` always describe the same stage; the UI
must not combine a first-pass backend with another stage's model.

The ASR adapter must return both `backend` and `model`. For compatibility, a
missing model is represented as `unknown`, never inferred from another adapter's
global configuration. The Qwen adapter derives its own model name from its
configured checkpoint. A private MOSS adapter supplies its own identity.

## Model-Pure Verification and Relisten

The resolver and relisten path receive a backend-aware callable rather than
assuming the global Qwen implementation. Routing uses normalized backend family:

- `qwen-omni-*` routes only to a Qwen verifier/relistener.
- `moss-*` routes only to a registered MOSS verifier/relistener.
- unknown or mismatched identities return a structured unavailable result.

Candidate audit evidence records the attempted verifier identity. A mismatch is
recorded as `candidate_stage=deferred`, `verifier_attempted=false`, with a
machine-readable `verifier_unavailable` or `verifier_backend_mismatch` reason.
No transcript change is allowed on that path.

Because the public repository contains no MOSS implementation, its built-in
behavior is deliberately fail-closed. Server-specific MOSS code integrates by
registering MOSS functions; it must not reuse the Qwen default.

## Durable Model-Scoped Memory

Audio sessions derive memory scope from the normalized first-pass identity, not
the audio filename:

```text
model--<safe-backend>--<safe-model>--<short-stable-hash>
```

The hash is derived only from backend and model identity and prevents collisions
after safe-name truncation. All audio sessions from the same model share the
scope. MOSS and Omni necessarily receive different scopes. An optional explicit
experiment namespace may prefix the scope so train/dev/test or ablation runs do
not contaminate one another.

Existing text-only API sessions retain their explicit/default scope. Existing
session JSON remains loadable. Existing per-session memory files are not merged
automatically because doing so could leak evaluation data; migration must be an
explicit offline operation.

## Memory Observability

Expose memory state without returning secrets:

- current `memory_scope`;
- model identity bound to the scope;
- number of provisional, stable, and superseded beliefs;
- last consolidation outcome and promoted count;
- storage error category and message when a read/write fails.

Memory repository errors must remain non-fatal to transcription, but they can no
longer be silently indistinguishable from an empty store. Session observability
and integration status include the memory status.

## Experiment Modes

Every audio request/result declares one of:

- `baseline`: first-pass ASR only; no ReTrace components or memory access.
- `retrace`: model-pure Judge/candidate/verifier/relisten pipeline enabled.

The response carries the selected mode and complete stage provenance. Evaluation
scripts reject or separately classify results with missing identity, mismatched
verifier family, or memory leakage across model scopes.

## Primary Metrics

Only three task-facing metrics are primary:

1. **Later-Evidence Correction Recall (LECR)**: correctly repaired errors whose
   correct form is supported by later turns or prior stable memory, divided by
   all such baseline errors. Eligibility is derived offline from baseline, GT,
   and later/prior evidence, never from generated candidates.
2. **Revision Precision (RP)**: committed revisions that reduce local edit
   distance to GT divided by all committed revisions.
3. **Entity Consistency Error Rate (ECER)**: incorrect aliases or spellings of a
   gold entity divided by all mentions of that entity.

Corpus CER remains a secondary safety result and is not presented as the primary
measure of retrospective correction. Metrics are reported separately for MOSS
and Omni and for cold-memory versus warm-memory ReTrace runs.

## Timing and Audit

Each sample records wall-clock duration for first pass, acoustic sentinel,
Context Judge, candidate construction, targeted verification, open relisten,
ledger commit, and end to end. Reports include p50/p95 latency and call counts.
Candidate audit remains the causal trace connecting discovery, verification,
deferral, and commit.

## Testing

Tests must prove:

- a MOSS result can never acquire the configured Qwen model as first-pass model;
- a MOSS turn cannot call a Qwen verifier or relistener;
- a missing MOSS verifier safely defers and is audited;
- Qwen routes only to Qwen functions;
- identical backend/model identities share a memory scope across sessions;
- MOSS and Omni scopes differ and do not retrieve each other's beliefs;
- storage failures appear in observability without breaking transcription;
- old session JSON remains readable;
- baseline mode produces no Judge, verifier, relisten, or memory writes;
- metric fixtures compute LECR, RP, and ECER without counting candidate audits as
  committed revisions.

## Non-Goals

- Implementing a MOSS model adapter that is absent from this repository.
- Automatically migrating old per-session memory into shared model memory.
- Optimizing latency before component timing identifies the actual bottleneck.
- Claiming that a hybrid MOSS+Omni system is a pure MOSS result.
