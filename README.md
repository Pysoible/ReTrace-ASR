# ReTrace-ASR

Evidence-grounded retrospective revision for conversational ASR. Qwen-Omni observations are stored immutably; later conversational evidence can revise a prior text span or bind a same-name entity while preserving a complete audit record.

## Core concepts

- Versioned transcript state
- Text candidate and entity candidate hypotheses
- Future-context evidence only
- Verified and quarantine entity memory
- `KEEP`, `REVISE_TEXT`, `REVISE_ENTITY`, `DEFER`, and `CLARIFY`

## Run

```bash
uv sync
cd frontend && npm install && npm run build && cd ..
uv run uvicorn asr_agent.server:app --reload
```

Submit entity profiles with `PUT /api/sessions/{session_id}/entities`, then submit Qwen-Omni observations with `POST /api/sessions/{session_id}/turns`. The browser workspace shows the versioned timeline and the evidence behind every revision.
