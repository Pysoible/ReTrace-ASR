#!/usr/bin/env bash
# Run ReTrace-ASR on the ModelArts GPU conda env (ms-swift + vLLM).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
set -a
[ -f "$ROOT/.env" ] && . "$ROOT/.env"
set +a

export ASR_AUDIO_ENABLED="${ASR_AUDIO_ENABLED:-1}"
# VAD coverage is always available to the agent. The 840M second ASR is an
# optional offline/low-throughput evidence source: enabling it on Qwen's already
# saturated GPUs can stall the realtime transcription path.
export ASR_ACOUSTIC_DISAGREEMENT="${ASR_ACOUSTIC_DISAGREEMENT:-1}"
export ASR_ACOUSTIC_DEVICE="${ASR_ACOUSTIC_DEVICE:-cpu}"
# Optional local HuggingFace embedding checkpoint for dense memory retrieval.
# Leave empty for the dependency-free BM25 + char n-gram hybrid retriever.
export ASR_MEMORY_EMBEDDING_MODEL="${ASR_MEMORY_EMBEDDING_MODEL:-}"
export ASR_DOMAINTERMS_ROOT="${ASR_DOMAINTERMS_ROOT:-$ROOT/../ASR_domainterms}"
export ASR_MODEL_PATH="${ASR_MODEL_PATH:-/home/ma-user/work/dataset/sjk_data/sjk/model_demo/checkpoint-793-merged}"
export ASR_AUDIO_DIR="${ASR_AUDIO_DIR:-/home/ma-user/work/dataset/sjk_data/ASR_audio}"
export ASR_GPU="${ASR_GPU:-0,1}"
export ASR_GPU_MEM_UTIL="${ASR_GPU_MEM_UTIL:-0.85}"
export ASR_MAX_MODEL_LEN="${ASR_MAX_MODEL_LEN:-4096}"
export ASR_MAX_NUM_SEQS="${ASR_MAX_NUM_SEQS:-1}"
export ASR_ENFORCE_EAGER="${ASR_ENFORCE_EAGER:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Qwen-Omni audio infer is unstable on vLLM V1 in this env; force V0.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

PY="${RETRACE_PYTHON:-/home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python}"
PY_ENV_ROOT="$(cd "$(dirname "$PY")/.." && pwd)"
export LD_LIBRARY_PATH="$PY_ENV_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/home/ma-user/ms-swift:${ROOT}/backend${PYTHONPATH:+:$PYTHONPATH}"

exec "$PY" -m uvicorn asr_agent.server:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
