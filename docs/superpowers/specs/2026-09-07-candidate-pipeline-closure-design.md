# Candidate Pipeline Closure Design

**Date:** 2026-09-07

**Status:** Approved for implementation

## Goal

Make the open-candidate correction work measurable and reachable in the real ReTrace path. Every correction source must use one normalized candidate contract, one evidence resolver, and one auditable stage trail. The repair must preserve immutable raw ASR text and strict local-audio verification.

## Baseline and Rollout

Quality evaluation starts with the MOSS normal-turn fast path disabled. `ASR_FAST_NORMAL_TURNS` defaults to false until candidate recall and revision harm have been measured. The fast path remains available as an explicit opt-in experiment, but it cannot be used for the first correctness baseline.

The repository must expose enough status to prove which code and policy are running: fast-path state, strict-revision state, acoustic-disagreement state, and homophone dependency availability. MOSS is treated as a supported source only when the ASR adapter actually returns `backend=moss-transcribe-diarize`; tests must exercise that boundary.

## Unified Candidate Contract

`CorrectionCandidate` is the only proposal representation at the ReTrace boundary. These sources produce candidates:

- `history_homophone`
- `domain_entity`
- `semantic_open`
- `acoustic_diff`
- `relisten_open`

Candidates are validated and deduplicated by `(target_turn_id, span, candidate)`. Deduplication merges source labels, evidence turn IDs, semantic confidence, and bounded audio coordinates.

DeepSeek output is normalized from both `focus` and top-level `candidates` into this contract. The prompt explicitly defines the top-level candidate schema. JSON shape must not change the safety policy: an absent-history replacement is always marked `semantic_open`, including when the model returned it inside `focus` without a source.

## Processing Flow

For each analyzed turn:

1. Collect candidates from the normalized Context Judge output, bounded history homophones, acoustic disagreement, and bounded relisten differences.
2. Validate target turn, source span, language/script compatibility, evidence IDs, and audio bounds.
3. Deduplicate the candidate pool.
4. Convert each candidate to a `FocusProposal` only at the resolver adapter boundary.
5. Send every replacement through `EvidenceResolver`.
6. Append an audit event for the candidate outcome, including discovery source, verifier attempt/result, and rejection reason.
7. Append a revision event only when the resolver commits.

The existing `direct_context_event` path and direct relisten revision construction are removed. They may nominate candidates but may not mutate projected text themselves.

## Evidence Policy

All replacement candidates require bounded historical audio and closed-set verification. The verified set contains the original span, proposed replacement, and `[DELETE]` when applicable.

`semantic_open` does not require pre-existing Paraformer disagreement. Paraformer disagreement is an independent positive feature, not a hard prerequisite. Strict mode keeps the existing initial thresholds:

- context confidence at least `0.85`
- proposed audio score at least `0.85`
- audio margin at least `0.30`
- calibrated evidence score at least the configured revision threshold

Missing audio, malformed verifier output, non-winning replacement, or insufficient evidence yields `DEFER` or `KEEP_OLD` and leaves text unchanged.

Whole-window relisten remains limited to true degeneration/coverage recovery. For a healthy turn, a relisten difference is converted into bounded local candidates and cannot directly replace the whole transcript.

## Audit and Metrics

Every candidate produces a durable candidate audit record with:

- candidate ID and source labels
- target/source text and target turn
- evidence turn IDs
- validation result
- verifier call status
- audio score and margin when available
- final outcome and rejection rationale

Candidate metrics are computed from these records, not inferred from committed revision events. Reports include discovered, validated, verified, committed, deferred, and rejected counts by source. `candidate_to_verifier_rate` uses validated candidates as its denominator; `verified_candidate_rate` uses verifier attempts as its denominator. Candidate recall remains explicitly unavailable until post-inference reference alignment supplies a valid denominator.

## Dependencies and Failure Visibility

`pypinyin`, `requests`, and the Levenshtein package used by evaluation are declared in project metadata and lock data. Homophone discovery no longer silently disappears: missing optional/runtime support is visible in integration status and candidate audit rationale.

DeepSeek transport cooldown remains fail-safe, but its status and fallback reason must remain observable. No Ground Truth is included in online candidate generation, context prompts, or resolver input.

## Tests and Acceptance Criteria

Tests must cover:

- MOSS normal turns call the Context Judge by default; fast mode works only when explicitly enabled.
- Semantic-open candidates absent from history reach local verification and commit with decisive audio, with or without Paraformer disagreement.
- Ambiguous semantic-open audio leaves text unchanged.
- History, semantic, acoustic, and relisten proposals all traverse the normalized candidate adapter and resolver.
- Candidate audit events retain sources, evidence IDs, verifier status, scores, and rejection rationale.
- Repeated spans are changed only at the resolver-selected occurrence.
- Coverage-risk recovery retains the previously expected supported behavior.
- MOSS backend labeling is exercised through the server boundary.
- All tests are top-level and collected by pytest.
- The focused suite, full suite, Python compilation, and diff checks pass before evaluation.

After automated verification, run no more than three real samples first. The run is acceptable only if candidate events prove the open path was exercised and no material revision harm appears. A full evaluation must wait for that checkpoint.

## Out of Scope

- Lowering strict thresholds before measurement
- Feeding Ground Truth into online processing
- Replacing the ASR checkpoint
- Frontend redesign
- Speaker-diarization redesign
- Large-scale evaluation before the three-sample gate passes
