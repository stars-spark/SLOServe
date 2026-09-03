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

## Results (Week 3)

Week 3 evaluates the external router on one NVIDIA GeForce RTX 4080 Laptop GPU with
`Qwen/Qwen3-0.6B`. SLOServe does not modify vLLM's internal token scheduler. The complete methods,
session boundaries, and interpretation are in [`docs/REPORT.md`](docs/REPORT.md).

Below capacity (`rps≤2` with the balanced mix), all three policies are equivalent because little
or nothing queues. Output-token throughput plateaus around 430 tok/s at `rps=3.0`, and queue onset
is between `rps=1.5` and `2.0`. The rate-ladder FCFS/static rows are from an initial session; the
SLO-aware rows are from the post-fix session. At `rps=3.0`, interactive SLO is 0.12 for FCFS, 1.00
for static priority, and 0.51 for SLO-aware. At `rps=4.0`, the values are 0.09, 0.85, and 0.32;
static priority's batch SLO falls to 0.72 while SLO-aware retains 0.99.

![Throughput-latency trade-off](results/figures/throughput-latency.png)

![Request rate versus end-to-end P99](results/figures/rate-p99.png)

![SLO attainment versus request rate](results/figures/slo-attainment-rate.png)

The headline robustness experiment uses `rps=3.0`, three policies, and six independent seeds.
FCFS records interactive SLO 0.57±0.31, overall SLO 0.80±0.14, TTFT P99
4.27±1.66 s, end-to-end P99 6.86±1.60 s, throughput 438±16 tok/s, and Jain
0.87±0.15. Static priority records interactive 0.98±0.01, overall 0.99±0.00,
TTFT P99 5.28±2.02 s, end-to-end P99 7.94±2.16 s, throughput 433±14 tok/s,
and Jain 1.00±0.00. SLO-aware records interactive 0.92±0.07, overall
0.96±0.03, TTFT P99 4.03±1.15 s, end-to-end P99 6.55±1.18 s, throughput
446±19 tok/s, and Jain 1.00±0.01. Batch SLO is 1.00 for all three.

![Saturation robustness](results/figures/saturation-robustness.png)

At `rps=2.0`, differences emerge when interactive requests are the minority. At interactive
fraction 0.2, interactive SLO is 0.48 for FCFS (Jain 0.89), 1.00 for static priority, and 0.97 for
SLO-aware. At fraction 0.5 the values are 1.00, 0.97, and 1.00; at fraction 0.8 all three are 1.00.

The `rps=3.0` SLO-aware ablation gives interactive SLO 0.60 for the full policy, 0.47 without
slack, 0.46 without the length estimate, and 0.92 without aging. Removing slack or the normalized
shortest-job term hurts interactive attainment; disabling aging exposes that the class-blind hard
tier caps SLO differentiation under saturation.

The aging sweep reports interactive SLO / longest queue wait of 0.85 / 3.86 s at 3 s, 0.97 /
7.07 s at 10 s, 0.96 / 6.44 s at 30 s, and 0.88 / 8.25 s at ∞. At 30 s the tier never fires, so
the 0.96 versus 0.88 difference from ∞ is pure vLLM execution-timing noise of approximately 0.08.

![Aging threshold trade-off](results/figures/aging-tradeoff.png)

The single-seed engine sweep gives FCFS / SLO-aware interactive SLO of 0.22 / 0.50 at
`max_num_seqs=8`, 0.19 / 0.55 at `max_num_seqs=16`, 0.36 / 0.55 at
`max_num_seqs=32`, and 0.12 / 0.77 with chunked prefill enabled at
`max_num_seqs=16`. SLO-aware leads FCFS for every tested engine configuration, but finer engine
trends are within saturation noise.

![Engine-parameter sensitivity](results/figures/engine-params.png)

Static priority has the strongest and most stable interactive SLO, but also the worst tail latency
and batch starvation under deeper overload. SLO-aware does **not** beat static priority on
interactive SLO. It is the balanced choice across latency, throughput, fairness, and batch
protection.

The saturation knee is session-sensitive. The same SLO-aware configuration (`aging=3 s`,
`rps=3.0`) produced interactive SLO approximately 0.51/0.55/0.60 in one session, 0.85 in the aging
sweep, and 0.92±0.07 in the robustness study. The fixed seed fixes workload generation but not
vLLM continuous-batching timing. Direction-level conclusions are robust; no single-run saturation
number is definitive.

## Results (Week 4)

Week 4 adds bursty Poisson arrivals and a multi-level generalization of the aging rule. A single
Poisson session at `rps=3.0` reproduces the Week 3 policy ordering on interactive SLO (static 0.86,
SLO-aware 0.28, FCFS 0.14). Multi-level aging is then evaluated properly, sweeping aging levels
K in {1, 2, 3, 5} over six seeds each. It shows **no** interactive-SLO benefit: K=1 (the binary
policy) is highest at 0.48±0.20 and no larger K improves on it, with the differences inside the
run-to-run noise. Higher K instead trades interactive urgency for a shorter worst-case wait
(9.0 s → 7.6 s) and higher batch SLO (0.956 → 0.991). The governing knob for SLO differentiation
remains the aging threshold, not the number of tiers. Single-seed data would have misread this as a
real K effect; the multi-seed sweep corrects it.

![Multi-level aging under Poisson arrivals](results/figures/multilevel-aging.png)

## Results (Week 5)

Week 5 tests whether the aging ceiling threshold — identified in Weeks 3–4 as the knob that governs
SLO differentiation — should be made adaptive. An adaptive ceiling that floats with live queue
congestion (`clamp(margin * median_queue_wait, floor, cap)`) is swept against fixed thresholds
(3 s, 10 s, 30 s) over six seeds under Poisson arrivals. The adaptive ceiling **fails** on both
axes: interactive SLO 0.39 (adaptive) versus 0.91 (fixed 10–30 s), with a longer worst-case wait —
floating on the median in-queue wait is a destabilizing signal that Poisson bursts move the wrong
way. The useful byproduct is about the fixed knob: the 3 s threshold behind the Week-3 saturation
variance is simply mis-set. Raising it to 10–30 s lifts interactive SLO (0.83 → 0.91) and cuts the
run-to-run standard deviation from 0.25 to 0.08–0.10 — so much of that "irreducible" variance was a
knife-edge threshold, not execution-timing noise. The lever is real; the right way to pull it here
is a calibrated constant, not a controller.

![Adaptive ceiling versus fixed thresholds](results/figures/adaptive-ceiling.png)

## Reproducing the experiments

Use Python 3.11.14 and the locked development environment:

```bash
uv sync
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Start the pinned vLLM server separately from a versioned server configuration, then run a
router-side sweep against that already-running endpoint:

```bash
scripts/serve.sh configs/base.yaml
uv run sloserve sweep --config configs/sweeps/expB-rate.yaml
```

The sweep command writes request-level JSONL/CSV, completeness reports, configuration hashes,
environment metadata, and aggregate `sweep-results.csv`/`sweep-results.json` under `results/raw/`.
Use the refresh configurations `expB-slo.yaml` and `expC-slo.yaml` for the corrected SLO-aware
rows, and `expA-sat.yaml`, `expE-sat.yaml`, `expF-aging.yaml`, and `expG-robust.yaml` for the
saturation studies. Experiment D changes server-side parameters, so each
`serve-D-*.yaml`/`expD-*.yaml` pair requires a separate vLLM restart.

Regenerate the standard figures from an aggregate CSV and the additional Week 3 figures from the
saved sweep CSVs:

```bash
uv run sloserve plot --results results/raw/<experiment>/sweep-results.csv --out results/figures
uv run python scripts/plot_week3_extra.py
```

The figures are auto-generated from raw data: plotting reads the saved aggregate CSVs, which are
derived from retained request-level facts. Do not compare or merge the stale SLO-aware rows from
the first Week 3 run; FCFS and static-priority rows from that session are unaffected.
