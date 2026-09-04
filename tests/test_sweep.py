"""Offline tests for single-GPU evaluation sweep orchestration."""

from __future__ import annotations

import asyncio
import csv
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from sloserve.cli import build_parser, build_sweep_runtime
from sloserve.config import (
    AdaptiveClipSignal,
    ArrivalProcess,
    ClipSource,
    ExperimentConfig,
    LengthModel,
    LengthSource,
    SchedulerPolicyName,
    load_config,
)
from sloserve.experiments.sweep import (
    SWEEP_RESULT_COLUMNS,
    SweepPoint,
    config_for_point,
    load_sweep_config,
    run_sweep,
)
from sloserve.metrics import read_request_records_jsonl
from sloserve.router.models import RequestEnvelope
from sloserve.workload.backend import FakeBackend
from sloserve.workload.http_backend import HttpCallTelemetry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicTime:
    """Advance benchmark arrival sleeps and park admission timeout sleeps."""

    def __init__(self) -> None:
        self.now_s = 0.0

    def clock(self) -> float:
        return self.now_s

    async def sleep(self, delay_s: float) -> None:
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else ""
        if task_name.startswith("sloserve-benchmark-arrival-"):
            self.now_s += delay_s
            await asyncio.sleep(0)
            return
        await asyncio.get_running_loop().create_future()


class TelemetryFakeBackend(FakeBackend):
    """In-memory backend with benchmark-compatible request telemetry."""

    def __init__(self, clock: Any) -> None:
        super().__init__()
        self._clock = clock
        self.telemetry: dict[str, HttpCallTelemetry] = {}

    async def send(self, request: RequestEnvelope) -> None:
        await super().send(request)
        self.telemetry[request.request_id] = HttpCallTelemetry(
            first_token_time_s=self._clock(),
            output_tokens=2,
        )


def _config(tmp_path: Path) -> ExperimentConfig:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    workload = config.workload.model_copy(
        update={
            "request_rate_rps": 100.0,
            "total_requests": 2,
            "warmup_requests": 0,
            "repetitions": 1,
        }
    )
    router = config.router.model_copy(update={"max_in_flight": 1, "queue_capacity": 4})
    metrics = config.metrics.model_copy(update={"output_directory": tmp_path})
    return config.model_copy(update={"workload": workload, "router": router, "metrics": metrics})


def test_sweep_uses_fresh_backends_applies_overrides_and_round_trips(tmp_path: Path) -> None:
    config = _config(tmp_path)
    deterministic_time = DeterministicTime()
    entered: list[TelemetryFakeBackend] = []
    exited: list[TelemetryFakeBackend] = []

    @asynccontextmanager
    async def make_backend() -> AsyncIterator[TelemetryFakeBackend]:
        backend = TelemetryFakeBackend(deterministic_time.clock)
        entered.append(backend)
        try:
            yield backend
        finally:
            exited.append(backend)

    points = (
        SweepPoint(label="slow-fcfs", policy="fcfs", request_rate_rps=100.0),
        SweepPoint(
            label="fast-slo",
            policy="slo_aware",
            request_rate_rps=200.0,
            interactive_fraction=0.25,
            max_in_flight=2,
            aging_threshold_s=8.0,
            cost_weight=2.0,
            slack_weight=0.0,
            waiting_weight=3.0,
            disable_length_estimate=True,
            length_source="advertised",
            length_estimator_path="unused-predictor.json",
            aging_levels=3,
        ),
    )

    result = asyncio.run(
        run_sweep(
            base_config=config,
            points=points,
            make_backend=make_backend,
            env_version="offline-test",
            clock=deterministic_time.clock,
            sleep=deterministic_time.sleep,
        )
    )

    assert len(result.rows) == 2
    assert len({id(backend) for backend in entered}) == 2
    assert exited == entered
    assert set(result.rows[0]) == set(SWEEP_RESULT_COLUMNS)
    assert set(result.rows[1]) == set(SWEEP_RESULT_COLUMNS)
    assert result.rows[0]["request_rate_rps"] == 100.0
    assert result.rows[1]["request_rate_rps"] == 200.0
    assert result.rows[1]["policy"] == SchedulerPolicyName.SLO_AWARE.value
    assert result.rows[1]["interactive_fraction"] == 0.25
    assert result.rows[1]["max_in_flight"] == 2
    assert result.rows[1]["aging_threshold_s"] == 8.0
    assert result.rows[1]["slack_weight"] == 0.0
    assert result.rows[1]["disable_length_estimate"] is True
    assert result.rows[1]["length_source"] == LengthSource.ADVERTISED.value
    assert result.rows[1]["length_estimator_path"] == "unused-predictor.json"
    assert result.rows[1]["length_model"] == LengthModel.UNIFORM_CAP.value
    assert result.rows[1]["prompt_kinds"] is None
    assert result.rows[1]["aging_levels"] == 3

    assert result.csv_path == tmp_path / "sweep-results.csv"
    assert result.json_path == tmp_path / "sweep-results.json"
    with result.csv_path.open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    assert len(csv_rows) == 2
    assert tuple(csv_rows[0]) == SWEEP_RESULT_COLUMNS
    assert csv_rows[0]["request_rate_rps"] == "100.0"
    assert csv_rows[1]["request_rate_rps"] == "200.0"
    assert json.loads(result.json_path.read_text(encoding="utf-8")) == list(result.rows)

    first_records = read_request_records_jsonl(tmp_path / "sweep-000-slow-fcfs.jsonl")
    second_records = read_request_records_jsonl(tmp_path / "sweep-001-fast-slo.jsonl")
    assert first_records[1].arrival_time_s - first_records[0].arrival_time_s == pytest.approx(0.01)
    assert second_records[1].arrival_time_s - second_records[0].arrival_time_s == pytest.approx(
        0.005
    )
    assert (tmp_path / "sweep-000-slow-fcfs.csv").is_file()
    assert (tmp_path / "sweep-001-fast-slo.csv").is_file()


def test_versioned_sweep_configs_load_expected_matrices() -> None:
    expected_counts = {
        "expA-policy.yaml": 3,
        "expA-sat.yaml": 3,
        "expB-rate.yaml": 18,
        "expB-slo.yaml": 6,
        "expC-mix.yaml": 9,
        "expC-slo.yaml": 3,
        "expE-ablation.yaml": 4,
        "expE-sat.yaml": 4,
        "expH-poisson.yaml": 3,
        "expI-multilevel.yaml": 4,
        "expK-A-length-source.yaml": 18,
        "expK-B-clipping.yaml": 12,
        "expL-adaptive-clipping.yaml": 24,
    }

    definitions = {
        name: load_sweep_config(PROJECT_ROOT / "configs" / "sweeps" / name)
        for name in expected_counts
    }

    assert {name: len(definition.points) for name, definition in definitions.items()} == (
        expected_counts
    )
    assert all(
        definition.base_config.workload.repetitions >= 3 for definition in definitions.values()
    )
    ablations = definitions["expE-ablation.yaml"].points
    no_length = next(point for point in ablations if point.label == "no-length-estimate")
    assert no_length.disable_length_estimate is True
    for name in ("expA-sat.yaml", "expE-sat.yaml"):
        assert definitions[name].base_config.workload.request_rate_rps == 3.0
    assert all(
        point.policy is SchedulerPolicyName.SLO_AWARE
        for name in ("expB-slo.yaml", "expC-slo.yaml")
        for point in definitions[name].points
    )
    for name in ("expH-poisson.yaml", "expI-multilevel.yaml"):
        assert definitions[name].base_config.workload.arrival_process is ArrivalProcess.POISSON
    assert [point.aging_levels for point in definitions["expI-multilevel.yaml"].points] == [
        1,
        2,
        3,
        5,
    ]
    length_source = definitions["expK-A-length-source.yaml"]
    assert length_source.base_config.workload.length_model is LengthModel.REALISTIC
    assert length_source.base_config.workload.realistic_length is not None
    assert [point.length_source for point in length_source.points[:6]] == [LengthSource.TRUE] * 6
    assert [point.length_source for point in length_source.points[6:12]] == [
        LengthSource.ADVERTISED
    ] * 6
    assert [point.length_source for point in length_source.points[12:]] == [
        LengthSource.LEARNED
    ] * 6
    clipping = definitions["expK-B-clipping.yaml"]
    assert clipping.base_config.router.policy is SchedulerPolicyName.FCFS
    assert clipping.base_config.admission.clip_source is ClipSource.LEARNED
    assert [point.clip_enabled for point in clipping.points[:3]] == [False] * 3
    assert [point.clip_enabled for point in clipping.points[3:]] == [True] * 9
    assert [point.clip_max_tokens for point in clipping.points[3:]] == (
        [1536] * 3 + [1024] * 3 + [512] * 3
    )


def test_expl_sweep_is_the_frozen_24_point_asymmetric_matrix() -> None:
    definition = load_sweep_config(
        PROJECT_ROOT / "configs" / "sweeps" / "expL-adaptive-clipping.yaml"
    )
    configured = [config_for_point(definition.base_config, point) for point in definition.points]
    labels = [point.label for point in definition.points]
    seeds = {20250825, 11, 202}

    assert len(labels) == 24
    assert len(set(labels)) == 24
    high = [
        (point, config)
        for point, config in zip(definition.points, configured, strict=True)
        if point.label.startswith("high-")
    ]
    low = [
        (point, config)
        for point, config in zip(definition.points, configured, strict=True)
        if point.label.startswith("low-")
    ]
    assert len(high) == 18
    assert len(low) == 6

    expected_high_arms = {
        "no-clip",
        "fixed-512",
        "adaptive-q-default",
        "adaptive-q-fast",
        "adaptive-q-conservative",
        "adaptive-rho-fixed-capacity",
    }
    expected_low_arms = {"no-clip", "adaptive-q-default"}

    def arm(label: str, load: str) -> str:
        prefix = f"{load}-"
        return label.removeprefix(prefix).rsplit("-s", 1)[0]

    assert {arm(point.label, "high") for point, _ in high} == expected_high_arms
    assert {arm(point.label, "low") for point, _ in low} == expected_low_arms
    for load, points, expected_arms in (
        ("high", high, expected_high_arms),
        ("low", low, expected_low_arms),
    ):
        for expected_arm in expected_arms:
            assert {
                config.workload.random_seed
                for point, config in points
                if arm(point.label, load) == expected_arm
            } == seeds

    for config in configured:
        assert config.router.policy is SchedulerPolicyName.FCFS
        assert config.router.max_in_flight == 4
        assert config.workload.repetitions == 3
        assert config.workload.total_requests == 24
        assert config.workload.warmup_requests == 4
        assert config.workload.length_model is LengthModel.REALISTIC
        assert config.workload.realistic_length is not None
        assert config.workload.realistic_length.force_exact_output_tokens is True
        assert config.admission.clip_source is ClipSource.LEARNED

    adaptive = {
        arm(point.label, "high"): config.admission
        for point, config in high
        if "adaptive-" in point.label
    }
    assert adaptive["adaptive-q-default"].adaptive_clip_signal is (
        AdaptiveClipSignal.QUEUE_DEPTH_EWMA
    )
    assert adaptive["adaptive-q-default"].adaptive_clip_tighten_thresholds == (1.0, 2.0, 3.0)
    assert adaptive["adaptive-q-default"].adaptive_clip_relax_thresholds == (0.25, 0.75, 1.5)
    assert adaptive["adaptive-q-default"].adaptive_clip_ewma_tau_s == 6.0
    assert adaptive["adaptive-q-default"].adaptive_clip_tighten_hold_s == 4.0
    assert adaptive["adaptive-q-default"].adaptive_clip_relax_hold_s == 30.0
    assert adaptive["adaptive-q-default"].adaptive_clip_capacity_rps is None
    assert adaptive["adaptive-q-fast"].adaptive_clip_tighten_hold_s == 2.0
    assert adaptive["adaptive-q-conservative"].adaptive_clip_tighten_thresholds == (
        2.0,
        3.0,
        4.0,
    )
    assert adaptive["adaptive-q-conservative"].adaptive_clip_relax_thresholds == (
        0.5,
        1.5,
        2.5,
    )
    assert adaptive["adaptive-q-conservative"].adaptive_clip_ewma_tau_s == 12.0
    assert adaptive["adaptive-q-conservative"].adaptive_clip_tighten_hold_s == 4.0
    assert adaptive["adaptive-q-conservative"].adaptive_clip_relax_hold_s == 45.0
    rho = adaptive["adaptive-rho-fixed-capacity"]
    assert rho.adaptive_clip_signal is AdaptiveClipSignal.RHO_HAT
    assert rho.adaptive_clip_tighten_thresholds == (0.70, 0.85, 1.00)
    assert rho.adaptive_clip_relax_thresholds == (0.60, 0.75, 0.90)
    assert rho.adaptive_clip_ewma_tau_s == 10.0
    assert rho.adaptive_clip_tighten_hold_s == 4.0
    assert rho.adaptive_clip_relax_hold_s == 10.0
    assert rho.adaptive_clip_capacity_rps == 0.230
    assert all(config.adaptive_clip_caps == (1536, 1024, 512) for config in adaptive.values())


def test_sweep_cli_parser_and_runtime_factories_are_lazy_and_fresh() -> None:
    config_path = str(PROJECT_ROOT / "configs" / "sweeps" / "expA-policy.yaml")
    args = build_parser().parse_args(["sweep", "--config", config_path])
    assert args.command == "sweep"
    assert args.config == config_path

    definition = load_sweep_config(config_path)
    clients: list[httpx.AsyncClient] = []
    sampler_calls: list[dict[str, Any]] = []

    def unreachable_handler(_: httpx.Request) -> httpx.Response:
        raise AssertionError("factory wiring test must not send a network request")

    def client_factory(**kwargs: Any) -> httpx.AsyncClient:
        client = httpx.AsyncClient(transport=httpx.MockTransport(unreachable_handler), **kwargs)
        clients.append(client)
        return client

    class FakeSampler:
        def __init__(self, **kwargs: Any) -> None:
            sampler_calls.append(kwargs)

        @property
        def samples(self) -> tuple[object, ...]:
            return ()

    runtime = build_sweep_runtime(
        definition.base_config,
        definition.points,
        env_version="offline-env",
        client_factory=client_factory,
        sampler_factory=FakeSampler,
    )
    assert clients == []
    assert sampler_calls == []

    async def enter_backends() -> None:
        async with runtime.make_backend() as first:
            assert first.telemetry == {}
        async with runtime.make_backend() as second:
            assert second.telemetry == {}
        assert first is not second

    asyncio.run(enter_backends())
    first_sampler = runtime.make_gpu_sampler()
    second_sampler = runtime.make_gpu_sampler()

    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.is_closed for client in clients)
    assert first_sampler is not second_sampler
    assert len(sampler_calls) == 2
