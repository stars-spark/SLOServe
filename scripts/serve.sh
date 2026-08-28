#!/usr/bin/env bash
# Launch the pinned single-GPU vLLM OpenAI-compatible server.
#
# The launch arguments are NOT hand-typed here: they are derived from the versioned
# configuration by `sloserve serve-command`, so the running server always matches
# configs/base.yaml (model/tokenizer commit, port, context, GPU memory, concurrency).
#
# Usage: scripts/serve.sh [CONFIG]   (default: configs/base.yaml)
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Serve strictly from the local, immutable snapshot verified in Week 1 step 2.
export HF_HOME="${HF_HOME:-/home/jiale/storage/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# Use vLLM's native PyTorch top-k/top-p sampler instead of flashinfer's sampler.
# The flashinfer sampler JIT-compiles a CUDA kernel at first use via `ninja`+`nvcc`,
# which is not part of this pinned, offline single-GPU environment. The native
# sampler needs no runtime compilation and keeps startup reproducible.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Derive the exact argv from the typed config (one arg per line).
mapfile -t SERVE_ARGS < <(uv run sloserve serve-command --config "$CONFIG" --format lines)

echo "vllm ${SERVE_ARGS[*]}" >&2
exec "$REPO_ROOT/.venv-vllm/bin/vllm" "${SERVE_ARGS[@]}"
