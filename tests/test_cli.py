"""Smoke tests for the installed command-line interface."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sloserve.cli import build_benchmark_runtime, build_parser, main
from sloserve.config import load_config
from sloserve.experiments.benchmark import GpuSample
from sloserve.workload.http_backend import HttpStreamingBackend

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "base.yaml")


def test_config_check_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["config-check", "--config", str(PROJECT_ROOT / "configs" / "base.yaml")])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"policy": "fcfs"' in captured.out
    assert '"model_revision": "c1899de289a04d12100db370d81485cdf75e47ca"' in captured.out
    assert '"tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca"' in captured.out


def test_serve_command_shell(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["serve-command", "--config", BASE_CONFIG])

    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("vllm serve Qwen/Qwen3-0.6B")
    assert "--revision c1899de289a04d12100db370d81485cdf75e47ca" in out
    assert "--enforce-eager" in out


def test_serve_command_json_is_argv(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["serve-command", "--config", BASE_CONFIG, "--format", "json"])

    assert exit_code == 0
    argv = json.loads(capsys.readouterr().out)
    assert argv[0] == "vllm"
    assert argv[1] == "serve"
    assert "--gpu-memory-utilization" in argv


def test_benchmark_command_parsing() -> None:
    args = build_parser().parse_args(["benchmark", "--config", BASE_CONFIG])

    assert args.command == "benchmark"
    assert args.config == BASE_CONFIG


def test_benchmark_runtime_factory_uses_config_without_io() -> None:
    config = load_config(BASE_CONFIG)
    calls: dict[str, dict[str, Any]] = {}

    def unreachable_handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("factory test must not send a network request")

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        calls["client"] = kwargs
        return httpx.AsyncClient(transport=httpx.MockTransport(unreachable_handler), **kwargs)

    class FakeSampler:
        def __init__(self, **kwargs: Any) -> None:
            calls["sampler"] = kwargs

        @property
        def samples(self) -> tuple[GpuSample, ...]:
            return ()

    def clock() -> float:
        return 123.0

    runtime = build_benchmark_runtime(
        config,
        env_version="offline-env",
        clock=clock,
        client_factory=client_factory,
        sampler_factory=FakeSampler,
    )

    assert runtime.config is config
    assert runtime.env_version == "offline-env"
    assert runtime.clock is clock
    assert isinstance(runtime.backend, HttpStreamingBackend)
    assert calls["client"] == {
        "base_url": config.backend.base_url,
        "timeout": config.backend.request_timeout_s,
    }
    assert calls["sampler"] == {
        "interval_s": config.metrics.gpu_sample_interval_s,
        "clock": clock,
    }
    asyncio.run(runtime.client.aclose())
