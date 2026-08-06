# ReTrace-ASR

ReTrace-ASR is an evidence-grounded retrospective agent for conversational ASR. It begins with only immutable Qwen-Omni observations; as later turns arrive, the agent autonomously reinterprets earlier turns and records every justified revision.

## Method

1. Qwen-Omni transcribes audio into silence-aware chunks. Each chunk is an ordered `Turn` with immutable `raw_text`.
2. Each new turn triggers a DeepSeek `REFLECT` action over earlier raw turns. It can discover a previously unnoticed ambiguity; no user-supplied entity list is required.
3. A proposal must identify the prior span, replacement interpretation, later evidence turn IDs and verbatim evidence quotes. The controller rejects any quote that cannot be located in a later raw turn.
4. Accepted proposals append `RevisionEvent`s. Visible subtitles and memory are replayed from immutable raw turns plus active events, so a later reinterpretation may supersede an earlier one.
5. `UNDO_REVISION` deactivates an event and replays state; it never deletes an ASR observation or audit record.

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
- `POST /api/sessions/{id}/audio/upload` — upload one audio file; its chunks become ordered turns in one session.
- `GET /api/sessions/{id}` — retrieve raw observations, memory layers and persisted revision events.
- `POST /api/sessions/{id}/revisions/{event_id}/undo` — append an auditable undo event.
- `GET /api/integrations/status` — Qwen adapter readiness.

The Studio UI renders the current subtitle, original text, subsequent evidence, event state and undo operation.
