# Agent-First Real-Time Memory Revision Design

## Objective

ReTrace-ASR will become a fully autonomous, real-time retrospective ASR agent. Each new ASR turn is interpreted against short-term conversational memory and long-term cross-session memory. The agent decides whether the turn is consistent, novel, conflicting, or uncertain before invoking tokenization, entity extraction, candidate generation, or audio re-listening.

The core research contribution is temporal belief revision: later evidence may make a new interpretation more plausible than the current one and trigger a reversible correction to one or more earlier turns. The system prioritizes correction recall and accepts a small amount of over-correction, while preserving every raw ASR observation and supporting automatic rollback.

## Design Principles

1. Raw ASR observations are immutable.
2. Memory is versioned evidence, not a static term dictionary.
3. New information does not automatically beat old information, and old memory does not automatically override new information.
4. The agent nominates conflicts and interpretations; calibrated evidence gates execute revisions.
5. Segmentation, named-entity recognition, and surface-form comparison run only when requested for a suspicious turn.
6. Every accepted revision is an append-only event that can be superseded or rolled back.
7. The real-time ASR path remains available when the agent, long-term memory, or an integration fails.
8. The system is fully automatic and never requires user confirmation.

## Architecture

The system is divided into six components with explicit interfaces.

### ASR Observer

The observer emits an immutable turn containing raw text, timestamps, speaker information when available, acoustic confidence, N-best candidates, and an addressable audio window. It immediately publishes the raw text to the real-time transcript.

### Memory Retriever

The retriever selects the recent conversational window, unresolved hypotheses, dependency-linked historical turns, and semantically relevant long-term beliefs. It must not place the complete conversation or complete long-term store into the agent context.

### Context Judge Agent

The judge receives the current observation and retrieved memory. Before any full-sentence token scan, it assigns one of four outcomes:

- `CONSISTENT`: the observation is coherent with available evidence.
- `NOVEL`: the observation introduces plausible new information.
- `CONFLICT`: the new and current interpretations cannot all be true as represented.
- `UNCERTAIN`: the available evidence cannot distinguish the interpretations.

Only `CONFLICT` and `UNCERTAIN` cause the agent to identify suspicious spans and request targeted tools.

### Evidence Resolver

The resolver constructs and compares explicit alternatives, including:

- the old interpretation is correct and the current ASR is wrong;
- the new interpretation is correct and a historical ASR turn is wrong;
- both expressions are valid and refer to different entities;
- both expressions are valid at different times because the underlying fact changed.

It gathers targeted acoustic, contextual, short-term memory, and long-term memory evidence. The agent returns structured features and evidence references; it does not directly mutate transcript text.

### Revision Ledger

The ledger records `ACCEPT_NEW`, `KEEP_OLD`, `REVISE_CURRENT`, `REVISE_HISTORY`, `COEXIST`, `DEFER`, and `ROLLBACK` events. Current transcript text is replayed from immutable observations and active events. Events carry the session version, affected turns, replacement span, evidence references, resolver output, and superseded event IDs.

### Memory Consolidator

The consolidator updates short-term memory after each turn and asynchronously promotes stable beliefs into long-term memory. Promotion is automatic but uses a stronger evidence gate than a session-local transcript revision. Long-term updates never block real-time ASR publication.

## Memory Model

### Short-Term Memory

Short-term memory is scoped to one live session and contains:

- `RecentContext`: recent immutable observations and current rendered text;
- `WorkingBeliefs`: current entities, events, relations, topics, and temporal claims;
- `OpenHypotheses`: competing interpretations that remain active;
- `DependencyIndex`: links from beliefs and revisions to their supporting turns and audio windows.

An open hypothesis contains an ID, affected turn IDs, the proposed interpretation, alternatives, supporting and contradicting evidence, audio windows, calibrated score, status, creation version, and last evaluated version.

### Long-Term Memory

Long-term memory stores stable cross-session knowledge rather than complete transcripts. It contains versioned entities, aliases, relations, facts, temporal validity, confidence, independent support counts, source sessions, and supersession links.

When new evidence contradicts a long-term belief, the system distinguishes ASR error, aliasing, multiple entities, and real-world change. A new version may supersede the old version, but the old record remains auditable. Updating long-term memory does not automatically rewrite every historical transcript; each transcript revision requires its own evidence decision.

## Real-Time Decision Flow

1. Publish and persist the immutable ASR observation.
2. Retrieve recent context, unresolved hypotheses, dependency-linked turns, and relevant long-term beliefs.
3. Ask the Context Judge for `CONSISTENT`, `NOVEL`, `CONFLICT`, or `UNCERTAIN`.
4. Accept consistent or plausible novel information into short-term memory without full-sentence candidate extraction.
5. For conflicts or uncertainty, construct old-correct, new-correct, coexistence, and temporal-change hypotheses.
6. Request only the tools needed for the affected spans: tokenization, entity analysis, current audio re-listening, historical audio re-listening, closed-set ASR verification, or additional memory retrieval.
7. Calibrate evidence scores and apply the action policy.
8. Append a revision, defer, coexistence, or rollback event.
9. Recompute only beliefs, turns, and hypotheses linked by the dependency index.
10. Consolidate sufficiently stable beliefs into long-term memory asynchronously.

## Threshold Policy

Business logic does not contain a single hand-written correction threshold. A held-out development set calibrates evidence features into comparable hypothesis probabilities. Policy uses four separately calibrated gates:

- `T_suspect` favors recall and starts deeper analysis;
- `T_relisten` favors recall and permits additional targeted audio work;
- `T_revise` permits a reversible session-local transcript revision;
- `T_long_memory` requires the strongest evidence before cross-session promotion or supersession.

A revision also requires a calibrated margin between the best and second-best hypotheses. Thresholds are selected on the development set using recall-weighted `F2` or `F3` together with the observed over-correction rate. The untouched test set is never used for threshold selection.

Before enough representative data exists, thresholds are explicitly marked as bootstrap configuration. They are not presented as learned values or research conclusions.

## Scheduling and Consistency

The runtime has a fast path and a slow path.

The fast path publishes raw ASR immediately, writes the observation, and updates the recent-turn index. The slow path performs agent judgment, retrieval, targeted re-listening, calibration, revision, and consolidation without blocking subsequent turns.

Every slow-path task carries the session version from which it was derived. If the session advances before completion, its decision cannot be committed directly. The system revalidates it against the latest affected state. Tasks concerning the same hypothesis are coalesced, and only dependency-linked state is recomputed.

## Failure Handling

- Agent timeout or malformed output appends or retains `DEFER` and schedules reconsideration on later evidence.
- Long-term memory failure degrades to short-term memory without blocking ASR.
- Audio re-listening failure preserves alternatives and does not authorize a text-only rewrite.
- A stale asynchronous decision is not committed; reusable evidence may be attached to the next evaluation.
- Event IDs and expected session versions make writes idempotent.
- Later contradictory evidence creates `ROLLBACK` or a superseding revision instead of deleting history.
- Service restart reconstructs visible text, working state, and active dependencies from immutable observations and events.

## Removal of Global N-Gram Nomination

The current `_COMMON_BIGRAMS`, generic CJK blocking sets, and global sliding n-gram scan must leave the primary decision path. They are artifacts of nominating entity-like spans before understanding whether a sentence is suspicious. The new flow makes the Context Judge responsible for deciding whether analysis is necessary.

Small lexical lists may remain only as optional, versioned features inside a targeted tool. They cannot independently nominate a revision, suppress a memory conflict, or authorize a transcript change.

## Evaluation Dataset

Evaluation uses continuous multi-turn speech with synchronized audio, raw ASR output, reference transcription, speaker and turn order, and annotations linking later disambiguating evidence to affected historical turns.

The dataset includes correct observations, immediate errors, future-resolved errors, distinct entities with similar names, real-world temporal changes, automatic rollback cases, helpful long-term memories, and stale or incorrect long-term memories.

Train, development, and test partitions are separated by conversation, speaker, and core entity. An entity used for calibration cannot reappear as the decisive entity in the test partition.

## Baselines and Ablations

Evaluation compares:

1. Raw ASR.
2. Static dictionary correction.
3. The existing deterministic n-gram and surface-similarity pipeline.
4. The Context Judge without memory.
5. The agent with short-term memory only.
6. The agent with short-term and long-term memory.
7. The complete agent without audio re-listening.
8. The complete real-time ReTrace agent.

These ablations isolate the contributions of contextual judgment, short-term memory, long-term memory, and acoustic verification.

## Metrics and Success Criteria

Reported metrics include final CER/WER, error-detection recall, revision precision and recall, `F2/F3`, over-correction rate, historical revision success rate, evidence-to-revision latency, rollback success rate, long-term memory contamination rate, agent calls per turn, re-listened audio seconds, and end-to-end latency.

Results include recall-versus-over-correction curves, confidence intervals, and paired significance tests. The design succeeds when the complete system reduces final test-set error relative to raw ASR and the deterministic baseline, and obtains higher historical revision recall at a matched over-correction operating point. This comparison, rather than performance on a small hand-written example set, validates the memory-driven retrospective mechanism.

## Engineering Verification

Unit tests cover observation immutability, hypothesis transitions, dependency selection, score-policy boundaries, memory versioning, event replay, supersession, and rollback. Integration tests cover short-term and long-term retrieval, closed-set audio verification, asynchronous version conflicts, idempotency, and restart recovery. End-to-end tests stream multi-turn sessions and verify that later evidence can revise historical text, reverse a mistaken revision, and update long-term memory without user input.

## Implementation Scope

The first implementation replaces the nomination and decision core while preserving the existing FastAPI boundary, ASR integration adapters, immutable raw-turn principle, audio-window verification capability, and append-only audit behavior. The large `retrace.py` module should be split along the six component boundaries above as part of this work. Frontend changes are limited to consuming the new event and memory states needed to observe real-time revisions; redesigning the product interface is outside this scope.
