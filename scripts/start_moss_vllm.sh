#!/usr/bin/env bash
set -euo pipefail

: "${MOSS_MODEL_PATH:?Set MOSS_MODEL_PATH to the MOSS-Transcribe-Diarize directory}"
: "${MOSS_LOG_PATH:?Set MOSS_LOG_PATH to an explicit log file}"

if [[ ! -d "$MOSS_MODEL_PATH" ]]; then
  echo "MOSS model directory does not exist: $MOSS_MODEL_PATH" >&2
  exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -ltn | awk '{print $4}' | grep -Eq '(^|:)8010$'; then
  echo "Port 8010 is already listening; refusing to start another MOSS service." >&2
  exit 1
fi

mkdir -p "$(dirname "$MOSS_LOG_PATH")"
MOSS_VLLM_BIN="${MOSS_VLLM_BIN:-vllm}"

nohup env \
  CUDA_VISIBLE_DEVICES="${MOSS_GPU_INDEX:-4}" \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_MAX_AUDIO_CLIP_FILESIZE_MB=1024 \
  VLLM_MAX_AUDIO_DECODE_DURATION_S=3600 \
  "$MOSS_VLLM_BIN" serve "$MOSS_MODEL_PATH" \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8010 \
  --served-model-name MOSS-Transcribe-Diarize \
  --max-num-batched-tokens 32768 \
  >"$MOSS_LOG_PATH" 2>&1 &

echo "Started MOSS vLLM pid=$! log=$MOSS_LOG_PATH"
