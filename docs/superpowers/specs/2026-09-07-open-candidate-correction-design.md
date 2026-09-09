# Open Candidate Correction Design

**Date:** 2026-09-07

**Status:** Proposed

## Goal

Increase correction recall for semantically obvious ASR errors whose correct wording has not appeared in history, while preserving high precision through local closed-set audio verification and auditable revision events.

## Problem

The current pipeline is effectively history-closed:

```text
current ASR text -> historical same-pronunciation candidate -> context judgment -> audio verification -> revision
```

This makes historical occurrence a practical prerequisite for correction. It also has two information-loss points: homophone discovery currently scans a concatenated history string, which can create cross-turn candidates, and fallback focus construction can discard candidate source turn IDs. Acoustic disagreement and open retranscription are separate paths from semantic candidate discovery and do not share one submission contract.

## Design Principles

- Optimize end-to-end correction quality, not architectural purity.
- Candidate recall may be open-world; revision submission may not be open-world.
- Every submitted replacement must be supported by local audio evidence or an explicit bounded deletion/coverage proof.
- Ground Truth is offline evaluation only and never enters candidate generation, Judge input, resolver input, or online audio correction.
- Historical occurrence is evidence, never a correctness requirement.
- Candidate frequency must not decide correctness.
- Preserve language/script compatibility and immutable raw ASR text.
- Every candidate and revision must retain source, evidence turn IDs, audio window, scores, and decision rationale.

## Target Flow

```text
ASR observation
    |
    v
Candidate sources
    |-- history homophone/entity candidates
    |-- semantic open-world candidates
    |-- acoustic disagreement candidates
    |-- bounded relisten candidates
    v
CandidatePool
    |
    v
Context ranking and conflict detection
    |
    v
Closed-set local audio verification
    |
    v
EvidenceResolver
    |
    v
RevisionLedger + RevisionEvent
```

## Candidate Contract

Introduce a normalized candidate record at the ReTrace boundary:

```python
@dataclass
class CorrectionCandidate:
    target_turn_id: str
    span: str
    candidate: str
    source: str
    evidence_turn_ids: list[str]
    rationale: str = ""
    semantic_confidence: float = 0.0
    audio_required: bool = True
    audio_start_sec: float | None = None
    audio_end_sec: float | None = None
```

Allowed `source` values:

- `history_homophone`
- `domain_entity`
- `semantic_open`
- `acoustic_diff`
- `relisten_open`

The candidate contract is internal and should be convertible to the existing `FocusProposal` without breaking public APIs during the first implementation slice.

## Candidate Sources

### History candidates

Scan each historical turn independently. Never concatenate turn text before generating candidates. Preserve the turn IDs that contain the candidate. Candidate discovery must remain bounded by span width and a configurable result limit.

### Semantic open candidates

When the Judge identifies a semantically unsuitable span but history has no suitable candidate, it may propose a same-language candidate from context, domain knowledge, or its own bounded world knowledge. The candidate must be marked `semantic_open` and must still pass local audio verification before submission.

### Acoustic candidates

Convert Paraformer disagreement spans and bounded retranscription replacements into candidates rather than sending them directly to separate resolver paths. Acoustic output is candidate evidence, not an automatic replacement.

### Relisten candidates

Open-vocabulary relisten may propose a replacement only for the bounded audio window that produced it. It must carry the source audio interval and remain subject to the same revision safety gate.

## Ranking and Verification

- Deduplicate candidates by `(target_turn_id, span, candidate)` while merging evidence IDs and source labels.
- Reject candidates that change language/script, contain malformed spans, or target text not present in the turn.
- Prefer candidates with semantic evidence, acoustic evidence, and explicit source turn IDs, but do not use occurrence count as the decision rule.
- For replacements, verify a closed set containing the original span, proposed candidate, and `[DELETE]` where supported by the existing resolver.
- Preserve current strict audio thresholds for the first slice: proposed score at least `0.85` and margin at least `0.30`.
- Missing audio, invalid verifier output, or ambiguous scores produces `DEFER`/`KEEP_OLD`, never an online rewrite.

## Revision Events

Extend revision evidence without breaking existing event consumers:

```text
evidence:
  candidate_source:semantic_open
  candidate_source:history_homophone
  evidence_turn:t011
  audio_verified:0.91/0.08
action: REVISE_CURRENT | REVISE_HISTORY | DEFER | KEEP_OLD
resolver: evidence-resolver
```

Raw text remains immutable. Ledger replay must use the event's complete `after_text` for full-text corrections and must preserve repeated-span replacements.

## Scope Boundaries

In scope:

- CandidatePool or equivalent normalization at the existing ReTrace/DeepSeek/Resolver boundary.
- Per-turn homophone discovery and evidence preservation.
- Open semantic candidate acceptance into the same closed-set resolver path.
- Acoustic/relisten candidate normalization.
- Focused metrics and regression tests for recall, precision, and false revisions.

Out of scope for this slice:

- Frontend redesign.
- Ground Truth in online processing.
- New ASR model training or checkpoint changes.
- Speaker separation redesign.
- Removing all Judge calls or making the model freely rewrite transcripts.
- Changing strict audio verification thresholds before measuring the new candidate recall.

## Success Criteria

On the existing focused regression set:

- Historical homophone corrections continue to pass.
- Cross-turn false candidates are not generated.
- Open candidates can reach audio verification without appearing in history.
- Ambiguous or unsupported audio leaves text unchanged.
- Revision events include candidate source and evidence provenance.
- Existing context, resolver, and ReTrace tests remain green.

On offline evaluation:

- Report candidate recall, candidate-to-verifier rate, verified-candidate rate, committed revision count, corrected-turn count, harmed-turn count, neutral-turn count, raw/final CER, and net edits removed.
- Compare against the current 20-sample baseline without feeding GT into inference.
