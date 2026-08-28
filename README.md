# SLOServe

SLOServe is an **external admission, queueing, and routing layer in front of vLLM** for
mixed interactive and batch inference requests. The MVP does not modify vLLM's internal
token scheduler.

The project will compare FCFS, static priority, and SLO-aware scheduling with aging on a
single GPU before considering any multi-GPU or KV-cache-aware extensions. `PLAN.md` is the
scope and acceptance source of truth.

## Current status

The repository currently contains the validated project configuration, the common scheduling
policy interface, and the FCFS ordering primitive. It does **not** yet contain a running router,
workload generator, vLLM deployment, or performance results.

## Development setup

Python 3.11 and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run ruff check .
uv run pytest
uv run sloserve config-check --config configs/base.yaml
```

Formal GPU experiments will run under native Ubuntu. vLLM is not supported natively on
Windows; follow the official [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
after the Ubuntu driver and CUDA environment has been recorded.

## Repository layout

```text
configs/                  Versioned experiment parameters
src/sloserve/router/      External request models and policies
src/sloserve/workload/    Workload generation (planned)
src/sloserve/experiments/ Experiment runners and sweeps (planned)
src/sloserve/analysis/    Analysis and plots (planned)
tests/                    Unit and smoke tests
results/                  Raw data, processed summaries, and figures
report/                   Technical report sources
docs/                     Environment checks and progress log
```

No performance claim belongs in this repository unless it can be regenerated from committed
raw JSON/CSV experiment data and a versioned experiment configuration.

The concrete bootstrap sequence and acceptance checks are tracked in
[`docs/WEEK1.md`](docs/WEEK1.md).
