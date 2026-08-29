"""Command-line entry point for SLOServe development utilities."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx

from sloserve.config import ExperimentConfig, load_config
from sloserve.experiments.benchmark import BenchmarkResult, TelemetryBackend, run_benchmark
from sloserve.experiments.gpu import GpuSampler, NvidiaSmiSampler
from sloserve.experiments.metadata import capture_environment
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


if __name__ == "__main__":
    raise SystemExit(main())
