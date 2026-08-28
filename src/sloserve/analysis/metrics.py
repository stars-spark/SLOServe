"""Pure functions for request-level latency, SLO, throughput, and fairness metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from sloserve.config import RequestProfileConfig, WorkloadConfig
from sloserve.metrics.records import RequestRecord
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus


@dataclass(frozen=True, slots=True)
class Percentiles:
    """Nearest-rank latency percentiles, or null values for an empty population."""

    p50: float | None
    p95: float | None
    p99: float | None


@dataclass(frozen=True, slots=True)
class AttainmentSummary:
    """SLO observations and their deterministic rate."""

    request_count: int
    met_count: int
    rate: float


@dataclass(frozen=True, slots=True)
class MetricsSummary:
    """Complete Week 1 step 5 offline metric result."""

    request_count: int
    success_count: int
    error_count: int
    timeout_count: int
    cancelled_count: int
    success_output_tokens: int
    wall_clock_window_s: float | None
    token_throughput_per_s: float | None
    ttft_s: Percentiles
    tpot_s: Percentiles
    end_to_end_s: Percentiles
    queue_wait_s: Percentiles
    longest_queue_wait_s: float | None
    slo_overall: AttainmentSummary
    slo_by_class: dict[RequestClass, AttainmentSummary]
    slo_attainment_gap: float
    jain_fairness_index: float


def _nearest_rank(sorted_values: Sequence[float], proportion: float) -> float | None:
    if not sorted_values:
        return None
    rank = math.ceil(proportion * len(sorted_values))
    return sorted_values[rank - 1]


def _percentiles(values: Sequence[float]) -> Percentiles:
    sorted_values = sorted(values)
    return Percentiles(
        p50=_nearest_rank(sorted_values, 0.50),
        p95=_nearest_rank(sorted_values, 0.95),
        p99=_nearest_rank(sorted_values, 0.99),
    )


def _profile_for_class(
    request_class: RequestClass, workload_config: WorkloadConfig
) -> RequestProfileConfig:
    if request_class is RequestClass.INTERACTIVE:
        return workload_config.interactive
    return workload_config.batch


def _meets_slo(record: RequestRecord, workload_config: WorkloadConfig) -> bool:
    if record.status is not DispatchStatus.SUCCESS:
        return False
    first_token_time_s = record.first_token_time_s
    if first_token_time_s is None:
        return False
    profile = _profile_for_class(record.request_class, workload_config)
    ttft_s = first_token_time_s - record.arrival_time_s
    end_to_end_s = record.completion_time_s - record.arrival_time_s
    return ttft_s <= profile.ttft_slo_ms / 1000 and end_to_end_s <= profile.end_to_end_slo_ms / 1000


def _attainment(
    records: Sequence[RequestRecord], workload_config: WorkloadConfig
) -> AttainmentSummary:
    met_count = sum(_meets_slo(record, workload_config) for record in records)
    request_count = len(records)
    rate = met_count / request_count if request_count else 0.0
    return AttainmentSummary(request_count=request_count, met_count=met_count, rate=rate)


def calculate_metrics(
    records: Sequence[RequestRecord], workload_config: WorkloadConfig
) -> MetricsSummary:
    """Calculate documented metrics without I/O or external state."""
    successes = [record for record in records if record.status is DispatchStatus.SUCCESS]
    ttft_values = [
        record.first_token_time_s - record.arrival_time_s
        for record in successes
        if record.first_token_time_s is not None
    ]
    tpot_values = [
        (record.completion_time_s - record.first_token_time_s) / (record.output_tokens - 1)
        for record in successes
        if record.first_token_time_s is not None and record.output_tokens >= 2
    ]
    end_to_end_values = [record.completion_time_s - record.arrival_time_s for record in successes]
    success_queue_wait_values = [
        record.dispatch_time_s - record.arrival_time_s for record in successes
    ]
    all_queue_wait_values = [record.dispatch_time_s - record.arrival_time_s for record in records]

    success_output_tokens = sum(record.output_tokens for record in successes)
    if records:
        wall_clock_window_s = max(record.completion_time_s for record in records) - min(
            record.arrival_time_s for record in records
        )
        token_throughput_per_s = (
            success_output_tokens / wall_clock_window_s if wall_clock_window_s > 0 else None
        )
    else:
        wall_clock_window_s = None
        token_throughput_per_s = None

    by_class_records = {
        request_class: [record for record in records if record.request_class is request_class]
        for request_class in RequestClass
    }
    slo_by_class = {
        request_class: _attainment(class_records, workload_config)
        for request_class, class_records in by_class_records.items()
    }
    class_rates = [summary.rate for summary in slo_by_class.values()]
    rate_sum = sum(class_rates)
    squared_rate_sum = sum(rate * rate for rate in class_rates)
    jain_fairness_index = (
        rate_sum * rate_sum / (len(class_rates) * squared_rate_sum) if squared_rate_sum > 0 else 1.0
    )

    return MetricsSummary(
        request_count=len(records),
        success_count=len(successes),
        error_count=sum(record.status is DispatchStatus.ERROR for record in records),
        timeout_count=sum(record.status is DispatchStatus.TIMEOUT for record in records),
        cancelled_count=sum(record.status is DispatchStatus.CANCELLED for record in records),
        success_output_tokens=success_output_tokens,
        wall_clock_window_s=wall_clock_window_s,
        token_throughput_per_s=token_throughput_per_s,
        ttft_s=_percentiles(ttft_values),
        tpot_s=_percentiles(tpot_values),
        end_to_end_s=_percentiles(end_to_end_values),
        queue_wait_s=_percentiles(success_queue_wait_values),
        longest_queue_wait_s=max(all_queue_wait_values) if all_queue_wait_values else None,
        slo_overall=_attainment(records, workload_config),
        slo_by_class=slo_by_class,
        slo_attainment_gap=max(class_rates) - min(class_rates),
        jain_fairness_index=jain_fairness_index,
    )
