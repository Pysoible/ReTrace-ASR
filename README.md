# ReTrace-ASR

ReTrace-ASR is an evidence-grounded retrospective agent for conversational ASR. Qwen-Omni produces immutable audio observations; later turns can provide evidence for revising an earlier uncertain span while preserving the complete audit trail.

## Method

1. Qwen-Omni transcribes audio into silence-aware chunks. Each chunk is an ordered `Turn` with immutable `raw_text`.
2. Low-confidence text or entity alternatives enter `quarantine_memory` as hypotheses, never as facts.
3. The ReTrace controller evaluates only later-turn evidence and chooses `KEEP`, `REVISE_TEXT`, `REVISE_ENTITY`, `DEFER`, or `CLARIFY`.
4. A successful decision appends a `RevisionEvent` with its evidence, score and source turn. `verified_memory` records only promoted entities.
5. `UNDO_REVISION` appends a new event and restores the prior display state; no ASR observation or previous audit event is deleted.

## Run

```bash
uv sync --group dev
cd frontend && npm install && npm run build && cd ..
uv run uvicorn asr_agent.server:app --reload
```

For GPU audio transcription, copy `.env.example` to `.env`, set `ASR_AUDIO_ENABLED=1`, and point `ASR_MODEL_PATH` to a Qwen-Omni checkpoint.

## APIs

- `POST /api/sessions/{id}/turns` — append a text ASR observation with optional confidence/candidate maps.
- `POST /api/sessions/{id}/audio/upload` — upload one audio file; its chunks become ordered turns in one session.
- `GET /api/sessions/{id}` — retrieve raw observations, memory layers and persisted revision events.
- `POST /api/sessions/{id}/revisions/{event_id}/undo` — append an auditable undo event.
- `GET /api/integrations/status` — Qwen adapter readiness.

The Studio UI renders the current subtitle, original text, subsequent evidence, event state and undo operation.
