# ReTrace-ASR

ReTrace-ASR is an evidence-grounded retrospective agent for conversational ASR. It begins with only immutable Qwen-Omni observations; as later turns arrive, the agent autonomously reinterprets earlier turns and records every justified revision.

## Method

1. Qwen-Omni transcribes audio into silence-aware chunks. Each chunk is an ordered `Turn` with immutable `raw_text`.
2. Before reflection, ReTrace marks suspicious spans from ASR confidence and N-best disagreement. Each span retains competing candidates, an evidence packet and a conservative decision state: `WAIT`, `ASK_USER` or `COMMIT`. Audio re-listening and memory retrieval are exposed as evidence requirements, not claimed as executed tools.
3. Each new turn triggers a DeepSeek `REFLECT` action over earlier raw turns. It can discover a previously unnoticed ambiguity; no user-supplied entity list is required.
4. A proposal must identify the prior span, replacement interpretation, later evidence turn IDs and verbatim evidence quotes. The controller rejects any quote that cannot be located in a later raw turn; already-revised display text is never evidence.
5. Accepted proposals append `RevisionEvent`s. High-risk hypotheses require explicit operator confirmation before any revision. Visible subtitles and memory are replayed from immutable raw turns plus active events.
6. `UNDO_REVISION` deactivates an event and replays state; it never deletes an ASR observation or audit record.

## Run

```bash
uv sync --group dev
cd frontend && npm install && npm run build && cd ..
uv run uvicorn asr_agent.server:app --reload
```

For GPU audio transcription, copy `.env.example` to `.env`, set `ASR_AUDIO_ENABLED=1`, and point `ASR_MODEL_PATH` to a Qwen-Omni checkpoint.
Set `DEEPSEEK_API_KEY` to enable autonomous DeepSeek reflection. DeepSeek can propose a new interpretation but can never directly overwrite transcript text: the controller requires later quoted evidence before it records a revision.

## APIs

- `POST /api/sessions/{id}/turns` — append a raw text ASR observation; later turns autonomously trigger reflection.
- `POST /api/sessions/{id}/hypotheses/{turn_id}/confirm` — audibly record an operator's choice of an existing candidate, or retain the original span.
- `POST /api/sessions/{id}/audio/upload` — upload one audio file; its chunks become ordered turns in one session.
- `GET /api/sessions/{id}` — retrieve raw observations, memory layers and persisted revision events.
- `POST /api/sessions/{id}/revisions/{event_id}/undo` — append an auditable undo event.
- `GET /api/integrations/status` — Qwen adapter readiness.

The Studio UI renders the current subtitle, original text, subsequent evidence, event state and undo operation.

## ICASSP-oriented evaluation

For a turn-level reference set, evaluate the event stream with
`asr_agent.metrics.evaluate_revisions`. It reports committed-revision count,
revision precision, over-correction rate, revision coverage, and mean
resolution latency in turns. These metrics separate final transcript quality
from the safety and latency of retrospective corrections.
