# ReTrace-ASR Clean-Slate Design

## Product boundary

ReTrace-ASR is an evidence-grounded, retrospective conversational ASR system. It consumes Qwen-Omni observations and revises prior interpretations only when later conversational evidence justifies the change. It contains only the components necessary for that task.

## Backend

`ReTraceService` owns a `VersionedTranscriptState` per session. Each submitted turn stores immutable ASR observation, surface-text candidates, entity candidates, decision state, and auditable revisions. `EvidenceResolver` evaluates a deferred prior span only against strictly later turns, using ASR uncertainty, textual candidates, entity aliases, roles, organisations, timestamps, and supplied speaker context.

Entity memory is divided into `quarantine` (unresolved candidate facts) and `verified` (facts supported by a committed revision or user confirmation). The public actions are `KEEP`, `REVISE_TEXT`, `REVISE_ENTITY`, `DEFER`, and `CLARIFY`.

The API exposes session creation/inspection, entity profile updates, text observation processing, audio observation ingestion, confirmation, and audit history. The sole MCP capability is `retrace_process_turn`.

## Frontend

The default screen is a dark, research-grade session workspace. The left pane is a chronological transcript with status markers, candidates, and links from a later evidence turn to the earlier revised span. The right pane is a decision explainer: evidence source turn, text and entity candidates, scores, threshold/margin, action, and immutable audit log. No dictionary, term expansion, TTS, document, chat, or legacy tool views exist.

## Migration

Delete all superseded application components and replace the audio interface with a Qwen-Omni observation adapter that preserves raw observations.

## Acceptance criteria

- No application surface depends on superseded components.
- A later turn can revise text or entity binding; ties remain deferred.
- Raw ASR observations remain immutable and every revision exposes its source evidence.
- Frontend renders the timeline and decision explainer from the new API contract.
