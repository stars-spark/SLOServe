"""Offline tests for multi-policy correctness and starvation analysis."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx

from sloserve.cli import build_correctness_runtime, build_parser
from sloserve.config import ExperimentConfig, SchedulerPolicyName, load_config
from sloserve.experiments.correctness import (
    StarvationVerdict,
    analyze_policy,
    run_correctness_experiment,
)
from sloserve.metrics.records import RequestRecord
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.backend import FakeBackend
from sloserve.workload.dispatcher import DispatchStatus
from sloserve.workload.http_backend import HttpCallTelemetry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_HASH = "a" * 64


class DeterministicTime:
    """Advance benchmark arrival sleeps while parking queue timeout sleeps."""

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
    """Existing in-memory fake plus benchmark-compatible telemetry."""

    def __init__(self, clock: Any) -> None:
        super().__init__()
        self._clock = clock
        self.telemetry: dict[str, HttpCallTelemetry] = {}

    async def send(self, request: RequestEnvelope) -> None:
        await super().send(request)
        now_s = self._clock()
        self.telemetry[request.request_id] = HttpCallTelemetry(
            first_token_time_s=now_s,
            output_tokens=1,
        )


def _config(tmp_path: Path) -> ExperimentConfig:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    workload = config.workload.model_copy(
        update={
            "request_rate_rps": 1000.0,
            "total_requests": 2,
            "warmup_requests": 0,
            "repetitions": 1,
        }
    )
    router = config.router.model_copy(update={"max_in_flight": 1, "queue_capacity": 4})
    metrics = config.metrics.model_copy(update={"output_directory": tmp_path})
    return config.model_copy(update={"workload": workload, "router": router, "metrics": metrics})


def _record(
    sequence_id: int,
    *,
    request_class: RequestClass,
    arrival_time_s: float,
    enqueue_time_s: float,
    dispatch_time_s: float | None,
    completion_time_s: float,
    status: DispatchStatus | str = DispatchStatus.SUCCESS,
) -> RequestRecord:
    typed_status = cast(DispatchStatus, status)
    is_success = typed_status is DispatchStatus.SUCCESS
    return RequestRecord(
        request_id=f"request-{sequence_id}",
        sequence_id=sequence_id,
        request_class=request_class,
        arrival_time_s=arrival_time_s,
        enqueue_time_s=enqueue_time_s,
        dispatch_time_s=dispatch_time_s,
        first_token_time_s=(dispatch_time_s if is_success else None),
        completion_time_s=completion_time_s,
        input_tokens=8,
        output_tokens=(1 if is_success else 0),
        status=typed_status,
        error_type=None,
        policy_name=SchedulerPolicyName.SLO_AWARE,
        config_hash=CONFIG_HASH,
        repetition_index=0,
        env_version="offline-test",
    )


def _hand_computed_records() -> tuple[RequestRecord, ...]:
    return (
        _record(
            0,
            request_class=RequestClass.INTERACTIVE,
            arrival_time_s=0.0,
            enqueue_time_s=0.0,
            dispatch_time_s=0.0,
            completion_time_s=1.0,
        ),
        _record(
            1,
            request_class=RequestClass.BATCH,
            arrival_time_s=0.5,
            enqueue_time_s=0.5,
            dispatch_time_s=3.5,
            completion_time_s=5.5,
        ),
        _record(
            2,
            request_class=RequestClass.INTERACTIVE,
            arrival_time_s=1.0,
            enqueue_time_s=1.0,
            dispatch_time_s=4.0,
            completion_time_s=4.5,
        ),
    )


def test_analysis_exact_queue_wait_depth_service_and_class_observations() -> None:
    analysis = analyze_policy(
        _hand_computed_records(),
        policy_name=SchedulerPolicyName.SLO_AWARE.value,
        aging_threshold_s=3.0,
        bound_margin_s=1.0,
    )

    assert analysis.formal_count == 3
    assert analysis.dispatched_count == 3
    assert analysis.max_queue_wait_s == 3.0
    assert analysis.max_queue_depth == 2
    assert analysis.observed_max_service_s == 2.0
    assert analysis.aging_bound_s == 4.0
    class_waits = {summary.request_class: summary for summary in analysis.class_waits}
    assert class_waits[RequestClass.INTERACTIVE.value].max_queue_wait_s == 3.0
    assert class_waits[RequestClass.INTERACTIVE.value].p95_queue_wait_s == 3.0
    assert class_waits[RequestClass.BATCH.value].max_queue_wait_s == 3.0
    assert class_waits[RequestClass.BATCH.value].p95_queue_wait_s == 3.0
    assert analysis.verdict is StarvationVerdict.WITHIN_BOUND


def test_bound_exceeded_when_a_contended_wait_is_over_a_plus_b() -> None:
    records = _hand_computed_records()
    over_bound = replace(
        records[2],
        dispatch_time_s=5.1,
        first_token_time_s=5.1,
        completion_time_s=5.6,
    )

    analysis = analyze_policy(
        (*records[:2], over_bound),
        policy_name="static_priority",
        aging_threshold_s=3.0,
        bound_margin_s=1.0,
    )

    assert analysis.max_queue_depth == 2
    assert analysis.max_queue_wait_s == 4.1
    assert analysis.within_aging_bound is False
    assert analysis.verdict is StarvationVerdict.BOUND_EXCEEDED


def test_no_contention_takes_precedence_when_queue_depth_is_below_two() -> None:
    # A single record that actually waited (enqueue < dispatch) so depth is a truthful 1;
    # record 0 has enqueue == dispatch (zero-width wait) and would read as depth 0.
    analysis = analyze_policy(
        _hand_computed_records()[1:2],
        policy_name=SchedulerPolicyName.FCFS.value,
        aging_threshold_s=3.0,
        bound_margin_s=1.0,
    )

    assert analysis.max_queue_depth == 1
    assert analysis.within_aging_bound is True
    assert analysis.verdict is StarvationVerdict.NO_CONTENTION


def test_terminal_unfinished_and_rejected_counts_are_separate() -> None:
    success = _hand_computed_records()[0]
    rejected = _record(
        1,
        request_class=RequestClass.BATCH,
        arrival_time_s=1.0,
        enqueue_time_s=1.0,
        dispatch_time_s=None,
        completion_time_s=1.0,
        status=DispatchStatus.REJECTED,
    )
    unfinished = _record(
        2,
        request_class=RequestClass.BATCH,
        arrival_time_s=1.5,
        enqueue_time_s=1.5,
        dispatch_time_s=2.0,
        completion_time_s=2.5,
        status="running",
    )

    terminal = analyze_policy(
        (success, rejected),
        policy_name="fcfs",
        aging_threshold_s=3.0,
    )
    incomplete = analyze_policy(
        (success, rejected, unfinished),
        policy_name="fcfs",
        aging_threshold_s=3.0,
    )

    assert terminal.terminal_count == 2
    assert terminal.all_terminal is True
    assert terminal.unfinished_count == 0
    assert terminal.dispatched_count == 1
    assert terminal.rejected_count == 1
    assert incomplete.terminal_count == 2
    assert incomplete.all_terminal is False
    assert incomplete.unfinished_count == 1
    assert incomplete.dispatched_count == 2
    assert incomplete.rejected_count == 1


def test_runner_uses_fresh_backends_and_writes_policy_outputs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    deterministic_time = DeterministicTime()
    factory_calls = 0
    entered: list[TelemetryFakeBackend] = []
    exited: list[TelemetryFakeBackend] = []

    @asynccontextmanager
    async def make_backend() -> AsyncIterator[TelemetryFakeBackend]:
        nonlocal factory_calls
        factory_calls += 1
        backend = TelemetryFakeBackend(deterministic_time.clock)
        entered.append(backend)
        try:
            yield backend
        finally:
            exited.append(backend)

    report = asyncio.run(
        run_correctness_experiment(
            config=config,
            make_backend=make_backend,
            env_version="offline-test",
            clock=deterministic_time.clock,
            sleep=deterministic_time.sleep,
        )
    )

    expected_policies = [policy.value for policy in SchedulerPolicyName]
    assert [policy.policy_name for policy in report.policies] == expected_policies
    assert factory_calls == 3
    assert len({id(backend) for backend in entered}) == 3
    assert exited == entered
    for policy_name in expected_policies:
        assert (tmp_path / f"correctness-{policy_name}.jsonl").is_file()
        assert (tmp_path / f"correctness-{policy_name}.csv").is_file()

    report_path = tmp_path / "correctness-report.json"
    persisted = json.loads(report_path.read_text(encoding="utf-8"))
    assert persisted["config_hash"] == report.config_hash
    assert [policy["policy_name"] for policy in persisted["policies"]] == expected_policies
    assert "非策略性能比较" in persisted["scope_note"]


def test_correctness_cli_parser_and_runtime_factories_are_lazy_and_fresh() -> None:
    config_path = str(PROJECT_ROOT / "configs" / "correctness.yaml")
    args = build_parser().parse_args(["correctness", "--config", config_path])
    assert args.command == "correctness"
    assert args.config == config_path

    config = load_config(config_path)
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

    runtime = build_correctness_runtime(
        config,
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
    assert sampler_calls == [
        {"interval_s": config.metrics.gpu_sample_interval_s, "clock": runtime.clock},
        {"interval_s": config.metrics.gpu_sample_interval_s, "clock": runtime.clock},
    ]
