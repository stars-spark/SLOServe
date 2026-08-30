"""Multi-policy single-GPU correctness experiment: behavior and no-starvation checks.

This orchestrates the existing benchmark pipeline once per scheduling policy and derives a
starvation analysis from the raw request records. It is a correctness experiment, not a
performance comparison: it checks that policies switch by configuration alone and that no request
is starved, using the (a) hard-aging wait bound and (b) terminal-status criteria. It never claims
throughput or latency improvements.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from sloserve.config import ExperimentConfig, SchedulerPolicyName
from sloserve.experiments.benchmark import TelemetryBackend, run_benchmark
from sloserve.metrics.config_hash import experiment_config_hash
from sloserve.metrics.records import RequestRecord
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus

if TYPE_CHECKING:
    from sloserve.experiments.gpu import GpuSampler

_DEFAULT_POLICIES: tuple[SchedulerPolicyName, ...] = (
    SchedulerPolicyName.FCFS,
    SchedulerPolicyName.STATIC_PRIORITY,
    SchedulerPolicyName.SLO_AWARE,
)
_SCOPE_NOTE = "单 GPU 正确性实验：行为与防饥饿检查，非策略性能比较"  # noqa: RUF001
_REPORT_FILENAME = "correctness-report.json"
_TERMINAL_STATUSES = frozenset(
    {
        DispatchStatus.SUCCESS,
        DispatchStatus.ERROR,
        DispatchStatus.TIMEOUT,
        DispatchStatus.CANCELLED,
        DispatchStatus.REJECTED,
    }
)

BackendFactory = Callable[[], AbstractAsyncContextManager[TelemetryBackend]]
GpuSamplerFactory = Callable[[], "GpuSampler"]


class StarvationVerdict(StrEnum):
    """Interpretation of one policy's no-starvation evidence under contention."""

    NO_CONTENTION = "no_contention"
    WITHIN_BOUND = "within_bound"
    BOUND_EXCEEDED = "bound_exceeded"


@dataclass(frozen=True, slots=True)
class ClassWaitSummary:
    """Per-class queue-wait observation (dispatch minus arrival) for context."""

    request_class: str
    dispatched_count: int
    max_queue_wait_s: float | None
    p95_queue_wait_s: float | None


@dataclass(frozen=True, slots=True)
class PolicyCorrectness:
    """Starvation-focused correctness facts for one scheduling policy."""

    policy_name: str
    formal_count: int
    terminal_count: int
    all_terminal: bool
    dispatched_count: int
    rejected_count: int
    unfinished_count: int
    max_queue_wait_s: float
    max_queue_depth: int
    observed_max_service_s: float
    aging_threshold_s: float
    bound_margin_s: float
    aging_bound_s: float
    within_aging_bound: bool
    class_waits: tuple[ClassWaitSummary, ...]
    verdict: StarvationVerdict


@dataclass(frozen=True, slots=True)
class CorrectnessReport:
    """Combined correctness evidence across every evaluated policy."""

    scope_note: str
    config_hash: str
    env_version: str
    bound_margin_s: float
    policies: tuple[PolicyCorrectness, ...]


def _nearest_rank(sorted_values: Sequence[float], proportion: float) -> float | None:
    """Return the nearest-rank percentile, or None for an empty population."""
    if not sorted_values:
        return None
    rank = math.ceil(proportion * len(sorted_values))
    return sorted_values[rank - 1]


def _max_queue_depth(records: Sequence[RequestRecord]) -> int:
    """Return the peak count of requests simultaneously waiting before dispatch."""
    events: list[tuple[float, int]] = []
    for record in records:
        if record.dispatch_time_s is None:
            continue
        events.append((record.enqueue_time_s, 1))
        events.append((record.dispatch_time_s, -1))
    # Departures (-1) sort before arrivals (+1) at equal times so [enqueue, dispatch) is half-open.
    events.sort(key=lambda event: (event[0], event[1]))
    depth = 0
    peak = 0
    for _, delta in events:
        depth += delta
        peak = max(peak, depth)
    return peak


def _class_wait_summary(
    dispatched: Sequence[RequestRecord], request_class: RequestClass
) -> ClassWaitSummary:
    """Summarize dispatch-minus-arrival waits for one request class."""
    waits = sorted(
        record.dispatch_time_s - record.arrival_time_s
        for record in dispatched
        if record.request_class is request_class
        if record.dispatch_time_s is not None
    )
    return ClassWaitSummary(
        request_class=request_class.value,
        dispatched_count=len(waits),
        max_queue_wait_s=waits[-1] if waits else None,
        p95_queue_wait_s=_nearest_rank(waits, 0.95),
    )


def _verdict(
    *,
    max_queue_depth: int,
    all_terminal: bool,
    unfinished_count: int,
    within_aging_bound: bool,
) -> StarvationVerdict:
    """Decide the starvation verdict from contention and the wait bound."""
    if max_queue_depth < 2:
        return StarvationVerdict.NO_CONTENTION
    if all_terminal and unfinished_count == 0 and within_aging_bound:
        return StarvationVerdict.WITHIN_BOUND
    return StarvationVerdict.BOUND_EXCEEDED


def analyze_policy(
    records: Sequence[RequestRecord],
    *,
    policy_name: str,
    aging_threshold_s: float,
    bound_margin_s: float = 2.0,
) -> PolicyCorrectness:
    """Derive starvation-focused correctness facts from one policy's formal records."""
    if not math.isfinite(aging_threshold_s) or aging_threshold_s <= 0:
        raise ValueError("aging_threshold_s must be finite and positive")
    if not math.isfinite(bound_margin_s) or bound_margin_s < 0:
        raise ValueError("bound_margin_s must be finite and non-negative")

    formal = list(records)
    formal_count = len(formal)
    terminal_count = sum(1 for record in formal if record.status in _TERMINAL_STATUSES)
    all_terminal = terminal_count == formal_count
    unfinished_count = formal_count - terminal_count

    dispatched = [record for record in formal if record.dispatch_time_s is not None]
    dispatched_count = len(dispatched)
    rejected_count = sum(1 for record in formal if record.status is DispatchStatus.REJECTED)

    waits = [record.dispatch_time_s - record.arrival_time_s for record in dispatched]
    max_queue_wait_s = max(waits) if waits else 0.0
    services = [record.completion_time_s - record.dispatch_time_s for record in dispatched]
    observed_max_service_s = max(services) if services else 0.0
    max_queue_depth = _max_queue_depth(formal)

    aging_bound_s = aging_threshold_s + bound_margin_s
    within_aging_bound = max_queue_wait_s <= aging_bound_s
    class_waits = tuple(
        _class_wait_summary(dispatched, request_class) for request_class in RequestClass
    )

    return PolicyCorrectness(
        policy_name=policy_name,
        formal_count=formal_count,
        terminal_count=terminal_count,
        all_terminal=all_terminal,
        dispatched_count=dispatched_count,
        rejected_count=rejected_count,
        unfinished_count=unfinished_count,
        max_queue_wait_s=max_queue_wait_s,
        max_queue_depth=max_queue_depth,
        observed_max_service_s=observed_max_service_s,
        aging_threshold_s=aging_threshold_s,
        bound_margin_s=bound_margin_s,
        aging_bound_s=aging_bound_s,
        within_aging_bound=within_aging_bound,
        class_waits=class_waits,
        verdict=_verdict(
            max_queue_depth=max_queue_depth,
            all_terminal=all_terminal,
            unfinished_count=unfinished_count,
            within_aging_bound=within_aging_bound,
        ),
    )


def _with_policy(config: ExperimentConfig, policy: SchedulerPolicyName) -> ExperimentConfig:
    """Return a copy of the config that selects a specific scheduling policy."""
    router = config.router.model_copy(update={"policy": policy})
    return config.model_copy(update={"router": router})


def correctness_report_path(output_directory: Path) -> Path:
    """Return the JSON report path for a given metrics output directory."""
    return output_directory / _REPORT_FILENAME


def _write_report(report: CorrectnessReport, output_directory: Path) -> Path:
    """Persist the combined correctness report as JSON."""
    output_directory.mkdir(parents=True, exist_ok=True)
    path = correctness_report_path(output_directory)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(
            asdict(report), output, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True
        )
        output.write("\n")
    return path


async def run_correctness_experiment(
    *,
    config: ExperimentConfig,
    make_backend: BackendFactory,
    env_version: str,
    policies: Sequence[SchedulerPolicyName] = _DEFAULT_POLICIES,
    bound_margin_s: float = 2.0,
    make_gpu_sampler: GpuSamplerFactory | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> CorrectnessReport:
    """Run the benchmark once per policy and derive combined correctness evidence.

    A fresh backend is entered per policy because the fixed seed reuses request ids, so a shared
    backend would let telemetry bleed across policies. This function never raises on
    ``BOUND_EXCEEDED`` or ``NO_CONTENTION``; a human interprets the report, and Static Priority may
    legitimately exceed the bound because it has no aging.
    """
    if not env_version:
        raise ValueError("env_version must not be empty")
    if not math.isfinite(bound_margin_s) or bound_margin_s < 0:
        raise ValueError("bound_margin_s must be finite and non-negative")

    results: list[PolicyCorrectness] = []
    for policy in policies:
        policy_config = _with_policy(config, policy)
        gpu_sampler = make_gpu_sampler() if make_gpu_sampler is not None else None
        async with make_backend() as backend:
            benchmark_result = await run_benchmark(
                config=policy_config,
                backend=backend,
                env_version=env_version,
                clock=clock,
                sleep=sleep,
                gpu_sampler=gpu_sampler,
                file_stem=f"correctness-{policy.value}",
            )
        results.append(
            analyze_policy(
                benchmark_result.formal_records,
                policy_name=policy.value,
                aging_threshold_s=config.slo_aware.aging_threshold_s,
                bound_margin_s=bound_margin_s,
            )
        )

    report = CorrectnessReport(
        scope_note=_SCOPE_NOTE,
        config_hash=experiment_config_hash(config),
        env_version=env_version,
        bound_margin_s=bound_margin_s,
        policies=tuple(results),
    )
    _write_report(report, config.metrics.output_directory)
    return report
