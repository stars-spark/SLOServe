"""Command-line entry point for SLOServe development utilities."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import httpx

from sloserve.analysis.plot_results import plot_sweep_results
from sloserve.config import ExperimentConfig, load_config
from sloserve.experiments.benchmark import BenchmarkResult, TelemetryBackend, run_benchmark
from sloserve.experiments.correctness import (
    CorrectnessReport,
    run_correctness_experiment,
)
from sloserve.experiments.gpu import GpuSampler, NvidiaSmiSampler
from sloserve.experiments.metadata import capture_environment
from sloserve.experiments.sweep import (
    SweepPoint,
    SweepResult,
    load_sweep_config,
    run_sweep,
)
from sloserve.workload.http_backend import HttpStreamingBackend


@dataclass(frozen=True, slots=True)
class BenchmarkRuntime:
    """Constructed benchmark dependencies; creating them performs no I/O."""

    config: ExperimentConfig
    client: httpx.AsyncClient
    backend: TelemetryBackend
    gpu_sampler: GpuSampler
    clock: Callable[[], float]
    env_version: str


@dataclass(frozen=True, slots=True)
class CorrectnessRuntime:
    """Lazy per-policy dependencies for a correctness experiment."""

    config: ExperimentConfig
    make_backend: Callable[[], AbstractAsyncContextManager[TelemetryBackend]]
    make_gpu_sampler: Callable[[], GpuSampler]
    clock: Callable[[], float]
    env_version: str


@dataclass(frozen=True, slots=True)
class SweepRuntime:
    """Lazy per-point dependencies for a single-GPU evaluation sweep."""

    config: ExperimentConfig
    points: tuple[SweepPoint, ...]
    make_backend: Callable[[], AbstractAsyncContextManager[TelemetryBackend]]
    make_gpu_sampler: Callable[[], GpuSampler]
    clock: Callable[[], float]
    env_version: str


def build_benchmark_runtime(
    config: ExperimentConfig,
    *,
    env_version: str,
    clock: Callable[[], float] = time.monotonic,
    client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    sampler_factory: Callable[..., GpuSampler] = NvidiaSmiSampler,
) -> BenchmarkRuntime:
    """Assemble configured benchmark dependencies without starting I/O."""
    client = client_factory(
        base_url=config.backend.base_url,
        timeout=config.backend.request_timeout_s,
    )
    backend = HttpStreamingBackend(
        backend_config=config.backend,
        client=client,
        clock=clock,
    )
    gpu_sampler = sampler_factory(
        interval_s=config.metrics.gpu_sample_interval_s,
        clock=clock,
    )
    return BenchmarkRuntime(
        config=config,
        client=client,
        backend=backend,
        gpu_sampler=gpu_sampler,
        clock=clock,
        env_version=env_version,
    )


def build_correctness_runtime(
    config: ExperimentConfig,
    *,
    env_version: str,
    clock: Callable[[], float] = time.monotonic,
    client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    sampler_factory: Callable[..., GpuSampler] = NvidiaSmiSampler,
) -> CorrectnessRuntime:
    """Assemble lazy factories without starting client or sampler I/O."""

    @asynccontextmanager
    async def make_backend() -> AsyncIterator[TelemetryBackend]:
        client = client_factory(
            base_url=config.backend.base_url,
            timeout=config.backend.request_timeout_s,
        )
        async with client:
            yield HttpStreamingBackend(
                backend_config=config.backend,
                client=client,
                clock=clock,
            )

    def make_gpu_sampler() -> GpuSampler:
        return sampler_factory(
            interval_s=config.metrics.gpu_sample_interval_s,
            clock=clock,
        )

    return CorrectnessRuntime(
        config=config,
        make_backend=make_backend,
        make_gpu_sampler=make_gpu_sampler,
        clock=clock,
        env_version=env_version,
    )


def build_sweep_runtime(
    config: ExperimentConfig,
    points: Sequence[SweepPoint],
    *,
    env_version: str,
    clock: Callable[[], float] = time.monotonic,
    client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    sampler_factory: Callable[..., GpuSampler] = NvidiaSmiSampler,
) -> SweepRuntime:
    """Assemble fresh per-point client and sampler factories without starting I/O."""

    @asynccontextmanager
    async def make_backend() -> AsyncIterator[TelemetryBackend]:
        client = client_factory(
            base_url=config.backend.base_url,
            timeout=config.backend.request_timeout_s,
        )
        async with client:
            yield HttpStreamingBackend(
                backend_config=config.backend,
                client=client,
                clock=clock,
            )

    def make_gpu_sampler() -> GpuSampler:
        return sampler_factory(
            interval_s=config.metrics.gpu_sample_interval_s,
            clock=clock,
        )

    return SweepRuntime(
        config=config,
        points=tuple(points),
        make_backend=make_backend,
        make_gpu_sampler=make_gpu_sampler,
        clock=clock,
        env_version=env_version,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="sloserve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("config-check", help="validate an experiment YAML file")
    check_parser.add_argument("--config", required=True, help="path to the YAML config")

    serve_parser = subparsers.add_parser(
        "serve-command",
        help="print the pinned vllm serve command derived from config",
    )
    serve_parser.add_argument("--config", required=True, help="path to the YAML config")
    serve_parser.add_argument(
        "--format",
        choices=["shell", "json", "lines"],
        default="shell",
        help="shell: one quoted command; json: argv list; lines: one arg per line",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="run the external FCFS benchmark against an already-running vLLM server",
    )
    benchmark_parser.add_argument("--config", required=True, help="path to the YAML config")
    correctness_parser = subparsers.add_parser(
        "correctness",
        help="run the single-GPU multi-policy correctness experiment",
    )
    correctness_parser.add_argument("--config", required=True, help="path to the YAML config")
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="run a single-GPU router-side evaluation sweep",
    )
    sweep_parser.add_argument("--config", required=True, help="path to the sweep YAML config")
    plot_parser = subparsers.add_parser(
        "plot",
        help="generate Week 3 figures from aggregated sweep CSV results",
    )
    plot_parser.add_argument("--results", required=True, help="path to sweep-results.csv")
    plot_parser.add_argument("--out", required=True, help="output figures directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a SLOServe development command."""
    args = build_parser().parse_args(argv)
    if args.command == "config-check":
        config = load_config(args.config)
        summary = {
            "model": config.backend.model,
            "model_revision": config.backend.model_revision,
            "policy": config.router.policy.value,
            "random_seed": config.workload.random_seed,
            "repetitions": config.workload.repetitions,
            "tokenizer_revision": config.backend.tokenizer_revision,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "serve-command":
        config = load_config(args.config)
        serve_args = config.vllm_serve_args()
        if args.format == "json":
            print(json.dumps(["vllm", *serve_args]))
        elif args.format == "lines":
            print("\n".join(serve_args))
        else:
            print(shlex.join(["vllm", *serve_args]))
        return 0
    if args.command == "benchmark":
        config = load_config(args.config)
        runtime = build_benchmark_runtime(
            config,
            env_version=capture_environment(config),
        )
        result = asyncio.run(_run_benchmark_runtime(runtime))
        print(f"completeness_report={result.completeness_report_path}")
        print(json.dumps(_benchmark_summary(result), sort_keys=True))
        return 0
    if args.command == "correctness":
        config = load_config(args.config)
        runtime = build_correctness_runtime(
            config,
            env_version=capture_environment(config),
        )
        report = asyncio.run(_run_correctness_runtime(runtime))
        report_path = config.metrics.output_directory / "correctness-report.json"
        print(f"correctness_report={report_path}")
        print(json.dumps(_correctness_summary(report), sort_keys=True))
        return 0
    if args.command == "sweep":
        definition = load_sweep_config(args.config)
        runtime = build_sweep_runtime(
            definition.base_config,
            definition.points,
            env_version=capture_environment(definition.base_config),
        )
        result = asyncio.run(_run_sweep_runtime(runtime))
        print(f"sweep_results_csv={result.csv_path}")
        print(json.dumps(_sweep_summary(result), sort_keys=True))
        return 0
    if args.command == "plot":
        written = plot_sweep_results(args.results, args.out)
        for path in written:
            print(f"figure={path}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


async def _run_benchmark_runtime(runtime: BenchmarkRuntime) -> BenchmarkResult:
    async with runtime.client:
        return await run_benchmark(
            config=runtime.config,
            backend=runtime.backend,
            env_version=runtime.env_version,
            clock=runtime.clock,
            gpu_sampler=runtime.gpu_sampler,
        )


async def _run_correctness_runtime(runtime: CorrectnessRuntime) -> CorrectnessReport:
    return await run_correctness_experiment(
        config=runtime.config,
        make_backend=runtime.make_backend,
        env_version=runtime.env_version,
        make_gpu_sampler=runtime.make_gpu_sampler,
        clock=runtime.clock,
    )


async def _run_sweep_runtime(runtime: SweepRuntime) -> SweepResult:
    return await run_sweep(
        base_config=runtime.config,
        points=runtime.points,
        make_backend=runtime.make_backend,
        env_version=runtime.env_version,
        make_gpu_sampler=runtime.make_gpu_sampler,
        clock=runtime.clock,
    )


def _benchmark_summary(result: BenchmarkResult) -> dict[str, int | str]:
    return {
        "scope": result.completeness_report.scope_note,
        "formal_requests": result.metrics.request_count,
        "success": result.metrics.success_count,
        "error": result.metrics.error_count,
        "timeout": result.metrics.timeout_count,
        "cancelled": result.metrics.cancelled_count,
        "rejected": result.metrics.rejected_count,
    }


def _correctness_summary(report: CorrectnessReport) -> dict[str, object]:
    return {
        "scope": report.scope_note,
        "policies": [
            {
                "policy_name": policy.policy_name,
                "verdict": policy.verdict,
                "max_queue_wait_s": policy.max_queue_wait_s,
                "aging_bound_s": policy.aging_bound_s,
                "max_queue_depth": policy.max_queue_depth,
            }
            for policy in report.policies
        ],
    }


def _sweep_summary(result: SweepResult) -> dict[str, object]:
    return {
        "points": len(result.rows),
        "policies": sorted({str(row["policy"]) for row in result.rows}),
    }


if __name__ == "__main__":
    raise SystemExit(main())
