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

## Integrations (same stack as ASR_agent)

| Backend | How ReTrace connects |
|---------|----------------------|
| **DeepSeek** | Reuses sibling `ASR_domainterms` (`llm_client` over Huawei WS / public HTTP). Load via `ASR_DOMAINTERMS_ROOT` + that repo's `.env`. |
| **Qwen-Omni ASR** | Local ms-swift `VllmEngine` two-pass (Pass1 → dictionary retrieve → Pass2), gated by `ASR_AUDIO_ENABLED=1`. |

Useful endpoints:

- `GET /api/integrations/status` — DeepSeek / Qwen readiness
- `POST /api/sessions/{id}/turns` — text observation; set `use_llm=true` for DeepSeek deferred revise, `correct_with_llm=true` for DeepSeek text correction
- `POST /api/sessions/{id}/audio` — Qwen two-pass ASR then ReTrace revise (`{"turn_id","audio",...}`)
- `POST /api/integrations/deepseek/correct` — standalone DeepSeek text correction

Long audio is automatically split (silence-aware, default `ASR_CHUNK_MAX_SEC=15`) before Qwen-Omni. **One audio file = one session**; each chunk becomes a turn (`t001`, `t002`, …) so later context can revise earlier turns. Session id defaults from the audio filename (`audio_<name>`).

See `.env.example`. DeepSeek credentials stay in `ASR_domainterms/.env` (do not duplicate keys here).
