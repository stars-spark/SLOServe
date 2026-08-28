# SLOServe contributor instructions

Read `PLAN.md` completely before changing code. Treat it as the project scope and acceptance
standard. Check `docs/PROGRESS.md` and `docs/WEEK1.md` before starting the next task.

## Scope

- The MVP is an external admission, queueing, and routing layer in front of a real vLLM
  OpenAI-compatible server. Never describe it as a modification to vLLM's internal token
  scheduler.
- Finish and validate the single-GPU MVP before multi-GPU routing, KV-cache awareness, or vLLM
  internal extensions.
- This is an AI infrastructure project, not a chat UI or an API-wrapper demo.

## Engineering rules

- Use managed CPython 3.11.14, `pyproject.toml`, `uv.lock`, type annotations, pytest, and Ruff.
- Keep runtime and experiment parameters in versioned configuration files.
- Keep scheduling policies behind the common interface in `src/sloserve/router/policies/`.
- Separate workload generation, routing, metrics, experiment execution, and analysis.
- Make small changes. After each module, run its focused tests, then Ruff and the full test suite.
- After each project step, append a study note to `docs/LEARNING_NOTES.md` before moving on. Record
  the problem, why the step was needed, key commands or code, success signals, failure diagnosis,
  observed result, and next step. Keep the notes factual and useful for review.
- Preserve unrelated files and existing user changes. Do not delete raw experiment data.

## Experiment integrity

- Never invent results, GPU conditions, completed features, or performance improvements.
- Every numerical claim must be reproducible from saved request-level JSONL/CSV, GPU samples,
  an experiment config, and environment/version metadata.
- Use warmup, fixed random seeds, and at least three repetitions for measured configurations.
- Retain failed and negative runs and explain them.
- Report throughput, TTFT, TPOT, P50/P95/P99 latency, SLO attainment, failures/timeouts, queue
  waiting/fairness, and GPU utilization/memory together.

## Required checks

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
