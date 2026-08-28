# Environment check

Checked on 2026-08-26T09:41:49+08:00 from the native Ubuntu installation. These observations
are environment facts, not benchmark results.

## Observed Ubuntu environment

- OS: Ubuntu 26.04 LTS.
- Kernel: 7.0.0-30-generic.
- Formal repository path: `/home/jiale/Desktop/SLOServe`.
- Formal repository filesystem: `ext4` on `/dev/nvme0n1p6` through the `/home` mount. At the
  environment check, `/home` had 82G total with 48G available.
- Source/backup checkout: `/run/media/jiale/AI_Local/SLOServe` on the `ntfs3` volume mounted from
  `/dev/nvme0n1p3`. It remains unsuitable for the formal vLLM environment or project virtual
  environments.
- Available ext4 locations:
  - `/home`: `/dev/nvme0n1p6`, 82G total with 48G available.
  - `/home/jiale/storage`: `/dev/nvme0n1p7`, 49G total with 32G available.
- Root filesystem: `/dev/nvme0n1p5`, 95G total with 25G available.
- Secure Boot: disabled.

## Observed Ubuntu GPU and CUDA visibility

- GPU: NVIDIA GeForce RTX 4080 Laptop GPU.
- GPU count: 1.
- Total GPU memory reported by `nvidia-smi`: 12,282 MiB.
- Compute capability: 8.9.
- Driver: 580.178.04.
- CUDA API level reported by `nvidia-smi`: 13.0. This is the driver-supported CUDA API level,
  not necessarily the user-space toolkit used by Python packages.
- CUDA toolkit compiler: `nvcc` 12.8.93 from `/usr/local/cuda/bin/nvcc`.
- CUDA/NVML driver libraries are visible through the dynamic linker cache, including
  `libcuda.so` and `libnvidia-ml.so.1`.
- At check time, no inference workload was running. `nvidia-smi` showed 14 MiB GPU memory in
  use, with GNOME Shell listed as the only GPU process.

## Observed Ubuntu Python and tools

- `python` and `python3` currently resolve to a Miniconda CPython 3.13.13 environment.
- `/usr/bin/python3` resolves to CPython 3.14.4.
- Project `.python-version`: 3.11.14.
- `uv`: 0.11.14 at `/home/jiale/.local/bin/uv`.
- Managed CPython 3.11.14 is now present at
  `/home/jiale/.local/share/uv/python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11`.
- The project CPU development environment is `/home/jiale/Desktop/SLOServe/.venv`, created by
  `uv` from managed CPython 3.11.14 on ext4.
- Git: 2.53.0.
- PyTorch and vLLM are intentionally absent from the CPU development `.venv` and installed in
  the separate `.venv-vllm` serving environment.
- `uv lock --check` resolved the existing lock successfully on Ubuntu. During this check, `uv`
  downloaded the missing managed CPython 3.11.14 interpreter required by `.python-version`;
  no vLLM, PyTorch, or benchmark dependency installation was performed.
- From the formal ext4 workspace, `uv lock --check`, `uv run ruff check .`,
  `uv run ruff format --check .`, and `uv run pytest` all passed. Pytest collected and passed all
  8 existing tests.
- `uv run sloserve config-check --config configs/base.yaml` passed and reported the configured
  model, FCFS policy, fixed random seed, and repetition count.

## Observed vLLM serving environment

- Environment: `/home/jiale/Desktop/SLOServe/.venv-vllm` on ext4, using managed CPython 3.11.14.
- Direct version pins and the rebuild command are recorded in
  `requirements/vllm-cu130.txt`.
- vLLM: 0.27.1.
- PyTorch: 2.13.0+cu130; `torch.version.cuda` reports 13.0.
- Selected CUDA user-space packages include `cuda-toolkit` 13.0.3.0,
  `nvidia-cuda-runtime` 13.0.96, `nvidia-cudnn-cu13` 9.20.0.48,
  `nvidia-cublas` 13.1.1.3, and `nvidia-nccl-cu13` 2.29.7.
- `uv pip check` checked 196 installed packages and reported them compatible.
- A package/runtime smoke check imported PyTorch and vLLM, detected one CUDA device, identified
  the RTX 4080 Laptop GPU with compute capability 8.9, and completed a small CUDA matrix
  multiplication. This is an installation check, not a benchmark or vLLM serving result.

## Pinned model snapshot

- Repository: `Qwen/Qwen3-0.6B`.
- Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Tokenizer revision: `c1899de289a04d12100db370d81485cdf75e47ca` from the same snapshot.
- License reported by the Hugging Face model metadata: Apache-2.0.
- Snapshot path:
  `/home/jiale/storage/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`.
- Download command:
  `env -u ALL_PROXY -u all_proxy .venv-vllm/bin/hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca`.
  The two proxy variables are unset only for this command because their `socks://` scheme is not
  accepted by the installed `httpx`; the configured HTTPS proxy remains active.
- `model.safetensors`: 1,503,300,328 bytes; SHA256
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`, matching the
  Hugging Face file metadata.
- The snapshot contains all 10 repository files with no broken cache links. Offline loading
  identified `Qwen3ForCausalLM`, loaded the tokenizer, and read 311 safetensors entries without
  loading the model into GPU memory.

## vLLM single-GPU server launch (verified 2026-08-27)

The pinned OpenAI-compatible server was started and exercised from the formal ext4 workspace.
These are functional smoke observations, not a benchmark.

- Launch command (versioned): `scripts/serve.sh configs/base.yaml`. It does not hand-type engine
  flags; it derives them from the typed config through `sloserve serve-command` so the running
  server always matches `configs/base.yaml`.
- Conservative launch parameters: `--host 127.0.0.1 --port 8000 --dtype bfloat16
  --max-model-len 4096 --gpu-memory-utilization 0.5 --max-num-seqs 16 --enforce-eager`, with model
  and tokenizer both pinned to commit `c1899de289a04d12100db370d81485cdf75e47ca`.
- The launcher serves offline (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`,
  `HF_HOME=/home/jiale/storage/huggingface`) from the verified snapshot.
- Engine report at load: available KV cache 4.47 GiB, GPU KV cache size 41,808 tokens, maximum
  concurrency for 4,096 tokens per request 10.21x. `nvidia-smi` after load: 6,136 MiB used, 0%
  utilization while idle.
- After `SIGTERM`, no `.venv-vllm` vLLM process remained, the endpoint refused connections, and
  the GPU returned to 14 MiB used with no compute apps.

### Two Python 3.11 environment fixes required before the server starts

Both are reproducible and captured in-repo; neither invents or alters model behaviour.

1. `flashinfer-python==0.6.16.post3` (a hard-pinned vLLM dependency) fails to import on CPython
   3.11 because `flashinfer/comm/fd_exchange.py` uses the annotation `array.array[int]`, and
   `array.array` only became subscriptable in CPython 3.12. vLLM imports this module during
   engine warm-up, so a stock launch crashes with
   `TypeError: type 'array.array' is not subscriptable`. `scripts/patch_flashinfer_py311.py`
   idempotently inserts the missing `from __future__ import annotations` into that one file,
   which defers the annotation without changing runtime behaviour and keeps the exact pinned
   dependency set (so `uv pip check` stays consistent). Run it after any `.venv-vllm` rebuild on
   Python < 3.12.
2. vLLM's default sampler would use flashinfer's top-k/top-p path, which JIT-compiles a CUDA
   kernel at first use via `ninja` (and `nvcc`), neither of which is part of this offline pinned
   environment; the launch otherwise fails with `FileNotFoundError: 'ninja'`. `scripts/serve.sh`
   exports `VLLM_USE_FLASHINFER_SAMPLER=0` so vLLM uses its native PyTorch sampler, which needs
   no runtime compilation. The server log confirms: "FlashInfer top-p/top-k sampling disabled via
   VLLM_USE_FLASHINFER_SAMPLER=0".

## Ubuntu execution decision

The Ubuntu GPU, driver, CUDA visibility, Secure Boot state, and formal ext4 workspace are
sufficient for the single-GPU MVP. The project was copied to `/home/jiale/Desktop/SLOServe` with
its files and Git metadata intact. The isolated PyTorch/vLLM environment has passed package,
dependency, import, and basic CUDA execution checks; a real model server has not yet been started.

The CPU project environment checks and the pinned serving software/model environment required for
Week 1 steps 1 and 2 are complete. Before starting any experiment:

- Validate one non-streaming and one streaming request before any performance run.

Checked on 2026-08-25 from the Windows installation. These observations are development-host
facts, not benchmark results.

## Observed Windows environment

- OS: Windows 11 Pro for Workstations Insider Preview, build 26220.
- GPU: NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB, compute capability 8.9.
- Driver: 591.91; `nvidia-smi` reports CUDA 13.1 as the driver's supported API level.
- Locally installed CUDA toolkit: 11.8 (`nvcc` 11.8.89). This is distinct from the
  `nvidia-smi` CUDA value.
- Python: 3.14.3 and 3.10 are installed; Python 3.11 is not yet installed globally.
- Project environment: `uv` installed managed CPython 3.11.14 in `.venv`; `uv.lock`
  records the exact resolved CPU-development dependencies.
- PyTorch and vLLM: not installed in the checked Windows Python environment.
- WSL: only the Docker Desktop distribution is registered; it is not a development Ubuntu
  environment.
- Docker client is installed, but the Docker engine was not running during the check.
- Windows `E:` (`AI_Local`) is a 200 GB NTFS volume with 87.2 GB free at check time. It is
  suitable for transfer/backup, but the active Ubuntu repository and virtual environments
  should live on ext4 to avoid NTFS permission, case-sensitivity, symlink, and small-file issues.

## Execution decision

The machine has a user-reported native Ubuntu dual-boot installation. Native Ubuntu will be
the reference environment for vLLM serving and formal GPU experiments. Windows remains usable
for CPU-only development and unit tests.

At the time of the Windows check, the Ubuntu driver, GPU visibility, Python, kernel, CUDA
runtime, PyTorch, vLLM, and model revision were still unknown. The current Ubuntu facts are
recorded in the 2026-08-26 section above.
