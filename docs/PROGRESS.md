# Project progress

## 2026-08-25 — Phase 1 bootstrap

- Read `PLAN.md` and adopted it as the project scope and acceptance standard.
- Confirmed that the repository initially contained only `PLAN.md` and had no Git metadata.
- Checked the Windows host, GPU, CUDA toolkit, Python, WSL, Docker, PyTorch, and vLLM state.
- Chose native Ubuntu dual boot as the formal vLLM and benchmark environment.
- Added a Python 3.11 project skeleton, typed YAML configuration, a unified policy interface,
  an FCFS primitive, CLI config validation, pytest tests, and Ruff configuration.
- Locked the CPU development environment to managed CPython 3.11.14 and 14 resolved packages.
- Verified `uv.lock`, Ruff lint, Ruff formatting, the config-check CLI, and 8 pytest tests.
- Initialized a local Git repository on branch `main`; no commit or remote was created.
- Added repository-level `AGENTS.md` so contributor sessions on Ubuntu inherit the project boundaries
  and validation rules.

No vLLM service or performance experiment has been run yet. Therefore there are no throughput,
latency, SLO, fairness, or GPU utilization results to report.

## 2026-08-26 — Week 1 step 1 complete

- Verified Ubuntu 26.04 LTS, kernel 7.0.0-30-generic, ext4 storage, and disabled Secure Boot.
- Verified one RTX 4080 Laptop GPU with 12,282 MiB, driver 580.178.04, compute capability 8.9,
  driver-reported CUDA API 13.0, and CUDA toolkit compiler 12.8.93.
- Copied the repository and Git metadata from the NTFS transfer location to the formal ext4
  workspace at `/home/jiale/Desktop/SLOServe` and verified file checksums.
- Created the CPU development `.venv` from managed CPython 3.11.14.
- Passed the lockfile check, Ruff lint and format checks, all 8 pytest tests, and the CLI config
  check from the formal workspace.

At the completion of step 1, PyTorch and vLLM had not yet been installed. They were installed in
the separate environment recorded in step 2 below. No GPU service or performance experiment was
started during the environment step.

## 2026-08-26 — Week 1 step 2 complete

- Created the isolated `.venv-vllm` environment from managed CPython 3.11.14 on ext4.
- Installed and pinned vLLM 0.27.1 with PyTorch 2.13.0+cu130.
- Recorded the rebuild inputs in `requirements/vllm-cu130.txt`.
- Verified dependency compatibility, vLLM import and CLI version, CUDA visibility, and a small
  PyTorch CUDA matrix multiplication on the RTX 4080 Laptop GPU.
- Pinned both model and tokenizer to `Qwen/Qwen3-0.6B` commit
  `c1899de289a04d12100db370d81485cdf75e47ca` in the validated base configuration.
- Downloaded the fixed snapshot to the ext4 Hugging Face cache, matched the safetensors SHA256 to
  the repository metadata, and loaded the config, tokenizer, and safetensors index offline.

No vLLM model server or performance experiment has been run. Week 1 step 3 is the next task.

## 2026-08-27 — Week 1 step 3 complete

- Added a versioned `server` section to `configs/base.yaml` (host, port, GPU memory utilization,
  max context, max sequences, dtype, enforce-eager) with typed `ServerConfig` validation and
  cross-checks that the server address matches `backend.base_url` and that `max_num_seqs` does not
  undercut the external router's `max_in_flight`.
- Added the `sloserve serve-command` CLI that emits the deterministic `vllm serve` command from
  the config, plus `scripts/serve.sh` which launches the server from that command so no engine
  flag is hand-typed. Extended the config and CLI tests; lockfile, Ruff lint, Ruff format, and 15
  pytest tests pass.
- Started the pinned single-GPU vLLM OpenAI-compatible server and verified `/v1/models`, one
  non-streaming request, and one streaming request (first chunk plus `[DONE]` terminator). Raw
  startup log, GPU memory sample, and request responses are saved under
  `results/raw/week1-step3-smoke/`.
- Diagnosed and fixed two Python 3.11 environment blockers before the server would start: a
  flashinfer import failure (`array.array[int]` unsubscriptable on 3.11), patched idempotently by
  `scripts/patch_flashinfer_py311.py`; and a `ninja`-based JIT sampler compile, avoided by setting
  `VLLM_USE_FLASHINFER_SAMPLER=0` in the launcher. Both fixes are reproducible and recorded in
  `docs/ENVIRONMENT.md`.
- Stopped the server and confirmed no leftover vLLM process, a refused endpoint, and the GPU back
  to 14 MiB used at 0% utilization.

This step is a functional smoke check only. No throughput, TTFT, TPOT, latency, SLO, fairness, or
sustained GPU-utilization result was measured, and none should be inferred from it. Week 1 step 4
(the async load generator) is next.

## 2026-08-27 — Week 1 step 4 minimum slice complete

- Added a deterministic request generator that reuses `RequestEnvelope`/`RequestClass`, seeds a
  local RNG from `workload.random_seed`, samples class-specific token ranges, and derives each
  deadline from the configured end-to-end SLO.
- Defined warmups as extra prefixed requests: `warmup_requests + total_requests` envelopes share
  one deterministic RNG stream and one contiguous sequence beginning at zero; the first measured
  request therefore has `sequence_id == warmup_requests`.
- Separated arrival scheduling from request generation. Fixed-rate arrivals use
  `i / request_rate_rps`; Poisson and burst configurations fail explicitly with
  `NotImplementedError` and remain future work.
- Added an async backend protocol, an in-memory configurable fake backend, and an FCFS dispatcher
  that schedules relative arrivals, uses `router.max_in_flight`, applies
  `backend.request_timeout_s`, and returns minimal success/error/timeout/cancelled results.
- Added 11 workload tests covering determinism, profile bounds/deadlines, warmup numbering, fixed
  arrivals, unfinished arrival types, bounded concurrency, FCFS order, all result states, arrival
  waiting, external cancellation propagation, cleanup, and subsequent capacity reuse.
- Passed the lockfile check, Ruff lint and format checks, and all 26 pytest tests.

This slice uses only the in-memory backend and does not define the Week 1 step 5 metrics schema or
make performance claims. No real HTTP benchmark or GPU experiment was run.

## 2026-08-27 — Week 1 step 5 minimum slice complete

- Defined the request-event and offline-metric contract in `docs/METRICS.md` before implementation,
  including all five timestamps, nearest-rank percentiles, SLO rules, boundary behavior, and the
  required attainment-gap and Jain fairness formulas.
- Added an immutable request record that reuses `RequestClass` and `DispatchStatus`, can be built
  from `RequestEnvelope`, carries a deterministic full-config SHA-256 and caller-supplied environment
  version, and validates terminal event ordering without inventing runtime metadata.
- Added JSONL fact-source and corresponding derived CSV read/write support. The unified writer uses
  `MetricsConfig.output_directory`, `save_jsonl`, and `save_csv`; JSONL and CSV round-trip tests compare
  the complete typed records.
- Added pure offline calculations for TTFT, conditional TPOT, end-to-end and queue percentiles,
  longest wait, output-token throughput, all terminal-state counts, overall/per-class SLO attainment,
  attainment gap, and Jain fairness. Empty input, all failures, one-token TPOT exclusion, inclusive
  SLO boundaries, and zero-duration throughput are covered explicitly.
- Recomputed a six-record hand-calculated sample from saved JSONL and verified nearest-rank P95,
  `2/6` SLO attainment, 1.5-second longest wait, 1.125 output token/s, and Jain index 0.9. These are
  arithmetic fixtures only, not observed service-performance results.
- Passed 9 focused metrics tests, lockfile validation, Ruff lint and formatting, and all 35 tests.

This slice is offline only. It does not connect request events to the dispatcher or a real HTTP
stream, observe a first-token signal, sample a GPU, or produce benchmark results. Week 1 step 6 is
the next integration boundary.

## 2026-08-28 — Week 1 step 6 minimum slice complete

- Added an external `AdmissionQueue` that takes its waiting capacity, in-flight limit, and total
  request timeout from the existing versioned router/backend configuration. It uses the shared
  `SchedulingPolicy` interface and `FcfsPolicy`; it does not alter vLLM's internal token scheduler.
- Serialized queue mutation and slot reservation under one async lock. Each free slot selects the
  first request returned by the policy, and only waiting requests count against `queue_capacity`;
  a full waiting queue returns `REJECTED` immediately without starting backend work.
- Applied one enqueue-to-terminal timeout across both states. A waiting timeout removes the entry;
  an in-flight timeout cancels and awaits the backend task. Caller cancellation propagates after
  the same cleanup, while queue shutdown resolves affected submissions as `CANCELLED` and awaits
  all internal tasks.
- Appended only `REJECTED = "rejected"` to `DispatchStatus`; no dispatcher logic or analysis
  counters were changed. Added 9 pure in-memory `FakeBackend` tests for FCFS/tie ordering, bounded
  concurrency and queue capacity, rejection, both timeout states, both cancellation states, mixed
  terminal outcomes with subsequent capacity reuse, and clean shutdown.
- Passed the focused 9 admission tests, lockfile validation, Ruff lint and formatting, and all 44
  pytest tests.

This is a functional queue/admission validation only. It is not connected to real HTTP/vLLM,
`RequestRecord`, first-token events, analysis rejection counts, or GPU sampling, and it contains no
service-performance measurements. Week 1 step 7 is the next integration boundary.

## 2026-08-28 — Week 1 step 7a-1 HTTP backend complete

- Added an `httpx` OpenAI-compatible streaming backend with request-keyed first-token, token-count,
  status, finish-reason, and error telemetry. Four `MockTransport` tests validate SSE success,
  usage fallback, HTTP errors, and unchanged `AdmissionQueue` integration entirely offline.

This is only the HTTP backend component of step 7. No benchmark runner, GPU sampling, completeness
report, real vLLM request, or performance measurement was run; Week 1 step 7 remains in progress.

## 2026-08-28 — Week 1 step 7a-2 offline benchmark pipeline complete

- Extended request facts and serialization for undispatched terminal records, added rejected
  analysis counts, and excluded undispatched records from queue-wait calculations.
- Added an injectable benchmark runner that schedules open-loop arrivals through the external
  admission queue, joins telemetry before request IDs repeat, persists warmup and formal facts,
  calculates formal-only metrics, and writes a structured completeness report.
- Pure offline tests cover two repetitions, deterministic replay, warmup exclusion, a persisted
  rejected request, JSONL/CSV round trips, and explicit missing reasons for unsampled GPU fields.
- Passed 14 focused tests, lockfile validation, Ruff lint and format checks, and all 54 tests.

This remains pipeline validation only. No GPU sample, real environment metadata, CLI benchmark,
real vLLM request, or performance measurement was produced; Week 1 step 7 remains in progress.

## 2026-08-29 — Week 1 step 7a-3 offline integration complete

- Added an injectable, concurrent single-GPU `nvidia-smi` sampler with shared-clock timestamps,
  clean task cancellation, skipped failed ticks, and explicit missing power rather than fabricated
  zeroes.
- Added deterministic environment metadata capture with injected readers and explicit
  `unavailable` markers, plus a no-I/O CLI assembly factory and `sloserve benchmark --config` for
  an already-running vLLM server.
- Limited benchmark runner changes to nullable GPU power, correct missing/partial power reporting,
  and one sampler context spanning all repetitions; sampler samples take precedence over direct
  injected samples.
- Added fully offline fake-based GPU, metadata, benchmark, and CLI coverage. The focused 12 tests
  and the first full run of 61 tests passed without invoking real GPU, vLLM, or network resources.

This completes 7a-3 only. It is pipeline validation with fixture values, not a performance result.
Week 1 step 7 remains in progress until 7b runs the CLI against a real single-GPU vLLM service.

## 2026-08-29 — Week 1 step 7 (7b) complete — Week 1 done

- Started the pinned single-GPU vLLM server (`scripts/serve.sh`), then ran
  `sloserve benchmark --config configs/base.yaml` against it end to end.
- Produced 75 request records (3 repetitions × [5 warmup + 20 formal]) as JSONL fact source plus
  derived CSV, and a completeness report, all under `results/raw/week1-step7-smoke/`.
- All 60 formal requests succeeded (0 error/timeout/cancelled/rejected); terminal counts sum to the
  request count. The completeness report has no unexplained missing values: TTFT/TPOT/end-to-end/
  queue P50-P95-P99, token throughput, overall and per-class SLO attainment, longest wait, fairness
  (gap 0.0, Jain 1.0), and concurrently sampled GPU utilization/memory/power (power was readable on
  this laptop GPU). GPU samples and request timestamps share one monotonic clock.
- Stopped the server and confirmed no leftover vLLM process, a refused endpoint, and the GPU back to
  14 MiB used.

This is a low-load pipeline validation only: queue waits are near zero and every request meets its
SLO by design, so these numbers must NOT be read as a policy-performance result. Observed values are
reproducible from the saved JSONL, GPU samples, config hash, and metadata. Known gap: the captured
`env_version` marks vllm/torch as `unavailable` because the CLI runs in the dev `.venv`; the real
serving versions (vLLM 0.27.1, torch 2.13.0+cu130) are recorded in `docs/ENVIRONMENT.md`.

Week 1 is complete: the single-GPU external FCFS admission/queueing/routing MVP is built and
validated end to end, with a reproducible, fully-populated raw data pipeline. No policy comparison
or performance claim has been made.

## 2026-08-29 — Week 2 W2-1 static priority complete

- Added `StaticPriorityPolicy` behind the common external scheduling interface: interactive
  requests rank before batch requests, with FCFS arrival/sequence ordering within each class.
- Added configuration-driven policy construction for FCFS and static priority; SLO-aware remains
  an explicit W2-2 `NotImplementedError`.
- Wired only `run_benchmark()` queue construction to inject `build_policy(config)`, so the existing
  CLI inherits `config.router.policy` switching while the base configuration stays FCFS.
- Added offline policy, factory, and benchmark integration tests. The integration holds one batch
  in the sole in-flight slot and confirms a later interactive request starts before an earlier
  waiting batch when capacity becomes available.
- Passed 10 focused tests, lockfile validation, Ruff lint and formatting, and all 67 pytest tests.

This slice did not access a GPU, network, or real vLLM, did not modify vLLM's internal scheduler,
and makes no policy-performance claim. SLO-aware scoring, aging, normalization, and estimated
service time remain for W2-2.

## 2026-08-31 — Week 2 complete: W2-2/W2-3a/W2-3b

- W2-2: SLO-aware policy (normalized cost/slack/waiting score + two-tier hard aging) behind the
  common interface; config factory now builds all three policies. Offline tests + 4 checks green.
- W2-3a: offline multi-policy correctness runner + starvation analysis (`sloserve correctness`,
  `analyze_policy`, `StarvationVerdict`) with a contention config. 81 pytest tests pass.
- W2-3b: real single-GPU run at two load points (3 repetitions each, fixed seed), raw data under
  `results/raw/week2-step3-correctness/{overload,concurrent}/`:
  - concurrent (mif=4, rps=2.0): all three policies WITHIN_BOUND, 0 rejected, depth 3–4 — no
    obvious starvation, the no-starvation acceptance evidence.
  - overload (mif=1, rps=1.5): all BOUND_EXCEEDED (system saturated, not a scheduling defect);
    static_priority honestly starves batch (interactive 3.4s vs batch 73.0s), FCFS is class-blind
    (~70s both), slo_aware ages to fairness (~74s both, no class singled out).
  - GPU util mean ~40–44%, memory peak 6136 MiB, power mean ~79–83 W.

Week 2 acceptance met: three policies switch by configuration alone, all tests pass, and the
concurrent load point shows no obvious request starvation. This is single-GPU correctness
validation, not a policy-performance comparison. Note: the sandboxed build environment cannot
access the GPU (`Failed to infer device type`), so the real-GPU runs were executed directly. Next: Week 3 full
evaluation (experiments A–E, throughput–latency / rate–P99 / SLO-attainment charts, report).

## 2026-09-02 — Week 3 complete: full evaluation and write-up

- Completed the single-GPU rate, workload-mix, SLO-aware ablation, aging-threshold,
  engine-parameter, and six-seed saturation-robustness experiments. Request-level JSONL/CSV,
  completeness reports, config hashes, environment metadata, and aggregate sweep CSV/JSON remain
  under `results/raw/`; all six figures are regenerated from saved CSVs.
- Corrected the SLO-aware dimensional inconsistency found during Week 3: service length is now a
  seconds-based estimate, `cost_weight` applies to normalized shortest-job service time, and slack
  is real remaining-time slack. All SLO-aware rows from the first Week 3 session were treated as
  stale and rerun; FCFS/static-priority rows were unaffected.
- At rps=3.0 across six independent seeds, interactive SLO was 0.57±0.31 for FCFS,
  0.98±0.01 for static priority, and 0.92±0.07 for SLO-aware. Corresponding Jain fairness was
  0.87±0.15, 1.00±0.00, and 1.00±0.01; batch SLO was 1.00 for all three.
- The honest conclusion is not that SLO-aware wins interactive SLO. Static priority is strongest
  and most stable there, but has the worst tail latency and begins starving batch at deeper
  overload. SLO-aware is the balanced choice across tail latency, throughput, fairness, and batch
  protection; its class-blind hard-aging tier trades bounded waiting against SLO differentiation.
- Saturation-knee values are session-sensitive because fixed seeds determine workload generation,
  not nondeterministic vLLM continuous-batching timing. Direction-level conclusions are robust;
  no individual saturation result is definitive.
- Added `docs/REPORT.md`, appended Week 3 results and reproduction instructions to `README.md`, and
  documented limitations: one GPU, one small model, fixed arrivals, no multi-GPU or KV-cache-aware
  routing, coarse token-seconds length estimates, high saturation variance, and single-seed
  experiment D.

## 2026-09-02 — Week 4 complete: Poisson arrivals and multi-level aging

- Added reproducible Poisson arrivals and generalized hard aging to 1/2/3/5 graduated tiers while
  preserving exact K=1 behavior. CPU checks and the real-GPU 24-point six-seed sweep passed.
- Multi-level aging did not improve interactive SLO: K=1 reached 0.48±0.20; K=2/3/5 reached
  0.28±0.13, 0.36±0.23, and 0.37±0.23. Higher K shortened the worst wait and improved batch SLO,
  so it is a fairness-versus-urgency control rather than an SLO recovery mechanism.
- The result is retained as a negative finding with request-level evidence under `results/raw/`.

## 2026-09-03 — Week 5 complete: adaptive aging ceiling

- Implemented and tested a median-queue-wait adaptive ceiling with floor/cap bounds, then compared
  fixed 3/10/30-second thresholds and two adaptive caps over six seeds (30 real-GPU points).
- The adaptive variants failed on both objectives: interactive SLO was 0.39–0.40 versus 0.83–0.91
  for fixed thresholds, and their longest waits were worse. Poisson bursts inject many zero-wait
  requests, pulling the median and threshold in the wrong direction.
- The useful result was configuration diagnosis: increasing the fixed threshold from 3 seconds to
  10–30 seconds raised interactive SLO and reduced run-to-run variance substantially.

## 2026-09-03/04 — Week 6 handoff: expK-A interpretation withdrawn; expK-B resumed

- Added a realistic heavy-tailed output-length mixture, an independently trained bucket-median
  predictor, and true/advertised/learned information boundaries. The first expK-A run was invalid
  because benchmark reclocking dropped the new fields; this was fixed, regression-tested, and the
  full 18-point sweep was rerun rather than hiding or reusing contaminated data.
- The field-preserving expK-A rerun observed interactive SLO 0.74±0.17 for the sampled target-cap
  source, 0.65±0.18 for the advertised constant, and 0.51±0.17 for the learned predictor. A later
  raw-fact audit found target-to-realized-length correlation only ~0.33 and just 202/1080 formal
  requests reached their target. Since vLLM treats `max_tokens` as a ceiling, the oracle/true-length
  interpretation is withdrawn; retain these numbers only as a target-cap-proxy diagnostic.
- Commit `979dfa4` adds default-off learned max-token clipping, preserves original target versus
  backend cap in raw facts, reports queue mean plus cap/truncation/token-reduction costs, keeps old
  JSONL/CSV readable, adds a qualitative M/G/1 module, and defines the 24-point six-seed
  `configs/sweeps/expK-B-clipping.yaml`. All 130 tests and required checks pass.
- Added default-off `realistic_length.force_exact_output_tokens`; the backend then sends matching
  vLLM `min_tokens` and effective `max_tokens`. A real probe confirmed ordinary cap 512 stopped at
  423 tokens, while forced mode realized 512 with `finish_reason=length`. The first seven cap-proxy
  points are preserved under `week6-expK-B-invalid-cap-proxy/` and must not enter final analysis.
- Exact-length rps=1.2 was ~3x over capacity, so the overloaded partial is preserved separately and
  the stable operating point was recalibrated to rps=0.17. The tractable final design is four arms
  × three seeds. Nine of twelve points (no-clip, 1536, 1024) are complete; the three 512 points are
  resuming under `week6-expK-B-remainder/`. Do not publish a clipping conclusion until they finish,
  are merged, and all required checks pass.
- The attempted Notion overview correction after the valid expK-A rerun failed because one table
  row no longer matched. The repository docs and commit `ad58133` contain the authoritative result;
  update the Notion W6 overview from these facts when that connector is next available.
- ExpK-B is complete: all twelve points (four caps × three seeds) finished under the recalibrated
  rps=0.17 operating point and the thermally constrained power profile, peaking at 63 °C GPU and
  76 °C CPU. Clipping to 512 tokens cuts mean queue wait 10.6x (2.78 → 0.26 s) and raises
  interactive SLO attainment from 0.26 to 0.73, at the cost of truncating 43.1% of requests by
  ~1019 tokens each and losing 36% of token throughput. Mean service falls 1.75x while E[S²] falls
  2.94x, confirming the variance-driven mechanism rather than a capacity effect.
- The M/G/1 module now takes an explicit `concurrency` argument read from `max_in_flight`, because
  the literal single-server model reported utilization above 2.6 on every arm and refused to
  predict while the measured system was plainly stable. Service times are divided by concurrency;
  this is not M/G/c and over-predicts wait roughly two-fold, so it is documented and used only as a
  qualitative trend check.
- Section L of `docs/REPORT.md`, the README Week-6 paragraph, and
  `results/figures/clipping-tradeoff.png` now carry this result with its boundaries. Absolute
  latencies in section L are not comparable with expK-A because the power profile changed between
  them; only within-section comparisons are valid.
