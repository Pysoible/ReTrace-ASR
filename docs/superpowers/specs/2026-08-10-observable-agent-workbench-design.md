# Observable Agent Workbench Design

## Goal

Turn the frontend from a transcript showcase into a realtime research workbench that makes ReTrace-ASR's temporal memory protocol directly observable without adding human confirmation controls.

## Information Architecture

The first viewport uses a compact application header followed by a three-column workspace:

1. **Transcript timeline** shows immutable raw ASR, current projection, revisions, and rollback state.
2. **Agent trace** shows the latest judgment, confidence, decision action, evidence, version, and the five live protocol stages.
3. **Memory inspector** separates short-term context from long-term stable beliefs and exposes belief provenance, confidence, validity, and supersession.

The input composer remains at the bottom of the workspace. Empty states explain what evidence is absent and never imply that a bundled benchmark or ASR model is available.

## Data Contract

Every session response includes a read-only `observability` object assembled from persisted protocol state:

- `latest_analysis`: outcome, confidence, rationale, analyzed turn, observed version, analyzed session version, and revalidation state.
- `short_term`: recent turns, dependency-linked turns, working beliefs, and active hypotheses.
- `long_term`: relevant stable beliefs retrieved for the latest turn.

`WorkingHypothesis` persists its relationship (`MUTUALLY_EXCLUSIVE`, `COEXIST`, or `TEMPORAL_CHANGE`) so the UI can explain why two interpretations compete or coexist.

## Behavior

All ledger actions are visible, including `KEEP_OLD`, `ACCEPT_NEW`, `COEXIST`, `DEFER`, revisions, and rollbacks. Selecting a turn filters the trace to relevant decisions. There are no confirmation, undo, or manual correction controls.

The stage rail is data-driven: observation is complete when a turn exists; memory and judgment reflect the latest analysis; relisten is active only when audio evidence is present; replay reflects ledger events.

## Visual Direction

Use a compact, work-focused interface with white and cool-gray surfaces, dark neutral text, teal for memory/context, amber for revisions, red for rejected text, and blue for version metadata. Avoid an oversized hero, decorative gradients, nested cards, and static marketing copy.

## Verification

- Contract tests verify observability and relationship persistence.
- Frontend source tests verify memory tabs, all autonomous decision labels, version metadata, and removal of stale demo copy.
- TypeScript build and Python test suite must pass.
- Browser screenshots cover desktop and mobile, with no overlaps or clipped controls.
