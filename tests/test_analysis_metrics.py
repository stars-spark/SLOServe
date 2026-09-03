"""Hand-calculated and boundary tests for pure offline metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from sloserve.analysis import Percentiles, calculate_metrics
from sloserve.config import (
    ArrivalProcess,
    RequestProfileConfig,
    SchedulerPolicyName,
    TokenRangeConfig,
    WorkloadConfig,
)
from sloserve.metrics import RequestRecord, read_request_records_jsonl, write_request_records_jsonl
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus

CONFIG_HASH = "a" * 64
ENV_VERSION = "test-environment-placeholder"


def _workload() -> WorkloadConfig:
    token_range = TokenRangeConfig(minimum=1, maximum=100)
    return WorkloadConfig(
        arrival_process=ArrivalProcess.FIXED,
        request_rate_rps=1.0,
        total_requests=6,
        interactive_fraction=0.5,
        random_seed=1,
        warmup_requests=0,
        repetitions=1,
        interactive=RequestProfileConfig(
            input_tokens=token_range,
            output_tokens=token_range,
            ttft_slo_ms=1000.0,
            end_to_end_slo_ms=3000.0,
        ),
        batch=RequestProfileConfig(
            input_tokens=token_range,
            output_tokens=token_range,
            ttft_slo_ms=2000.0,
            end_to_end_slo_ms=6000.0,
        ),
    )


def _record(
    sequence_id: int,
    request_class: RequestClass,
    status: DispatchStatus,
    *,
    arrival: float,
    dispatch: float | None,
    first_token: float | None,
    completion: float,
    output_tokens: int,
    requested_output_tokens: int | None = None,
    backend_max_output_tokens: int | None = None,
    finish_reason: str | None = None,
) -> RequestRecord:
    return RequestRecord(
        request_id=f"request-{sequence_id:06d}",
        sequence_id=sequence_id,
        request_class=request_class,
        arrival_time_s=arrival,
        enqueue_time_s=arrival,
        dispatch_time_s=dispatch,
        first_token_time_s=first_token,
        completion_time_s=completion,
        input_tokens=10,
        output_tokens=output_tokens,
        status=status,
        error_type=None if status is DispatchStatus.SUCCESS else status.value,
        policy_name=SchedulerPolicyName.FCFS,
        config_hash=CONFIG_HASH,
        repetition_index=0,
        env_version=ENV_VERSION,
        requested_output_tokens=requested_output_tokens,
        backend_max_output_tokens=backend_max_output_tokens,
        finish_reason=finish_reason,
    )


def _hand_calculated_records() -> tuple[RequestRecord, ...]:
    return (
        _record(
            0,
            RequestClass.INTERACTIVE,
            DispatchStatus.SUCCESS,
            arrival=0.0,
            dispatch=0.1,
            first_token=0.5,
            completion=2.0,
            output_tokens=3,
        ),
        _record(
            1,
            RequestClass.INTERACTIVE,
            DispatchStatus.SUCCESS,
            arrival=1.0,
            dispatch=1.4,
            first_token=2.2,
            completion=4.5,
            output_tokens=2,
        ),
        _record(
            2,
            RequestClass.BATCH,
            DispatchStatus.SUCCESS,
            arrival=2.0,
            dispatch=2.6,
            first_token=3.5,
            completion=7.0,
            output_tokens=4,
        ),
        _record(
            3,
            RequestClass.INTERACTIVE,
            DispatchStatus.ERROR,
            arrival=3.0,
            dispatch=4.0,
            first_token=None,
            completion=5.0,
            output_tokens=0,
        ),
        _record(
            4,
            RequestClass.INTERACTIVE,
            DispatchStatus.TIMEOUT,
            arrival=4.0,
            dispatch=5.2,
            first_token=None,
            completion=6.0,
            output_tokens=0,
        ),
        _record(
            5,
            RequestClass.BATCH,
            DispatchStatus.CANCELLED,
            arrival=5.0,
            dispatch=6.5,
            first_token=None,
            completion=8.0,
            output_tokens=0,
        ),
    )


def test_hand_calculated_metrics_can_be_recomputed_from_jsonl(tmp_path: Path) -> None:
    facts_path = write_request_records_jsonl(
        tmp_path / "hand-sample.jsonl", _hand_calculated_records()
    )

    result = calculate_metrics(read_request_records_jsonl(facts_path), _workload())

    assert result.request_count == 6
    assert result.success_count == 3
    assert result.error_count == 1
    assert result.timeout_count == 1
    assert result.cancelled_count == 1
    assert result.rejected_count == 0
    assert (
        result.success_count
        + result.error_count
        + result.timeout_count
        + result.cancelled_count
        + result.rejected_count
        == result.request_count
    )
    assert result.ttft_s.p50 == pytest.approx(1.2)
    assert result.ttft_s.p95 == pytest.approx(1.5)
    assert result.ttft_s.p99 == pytest.approx(1.5)
    assert result.end_to_end_s == Percentiles(p50=3.5, p95=5.0, p99=5.0)
    assert result.tpot_s.p50 == pytest.approx(3.5 / 3)
    assert result.tpot_s.p95 == pytest.approx(2.3)
    assert result.tpot_s.p99 == pytest.approx(2.3)
    assert result.queue_wait_s.p95 == pytest.approx(0.6)
    assert result.queue_wait_mean_s == pytest.approx((0.1 + 0.4 + 0.6) / 3)
    assert result.longest_queue_wait_s == pytest.approx(1.5)
    assert result.success_output_tokens == 9
    assert result.wall_clock_window_s == 8.0
    assert result.token_throughput_per_s == pytest.approx(1.125)
    assert result.slo_overall.met_count == 2
    assert result.slo_overall.rate == pytest.approx(2 / 6)
    assert result.slo_by_class[RequestClass.INTERACTIVE].rate == pytest.approx(1 / 4)
    assert result.slo_by_class[RequestClass.BATCH].rate == pytest.approx(1 / 2)
    assert result.slo_attainment_gap == pytest.approx(0.25)
    assert result.jain_fairness_index == pytest.approx(0.9)
    assert result.clip_applied_rate is None


def test_empty_input_has_documented_nulls_and_zero_rate_conventions() -> None:
    result = calculate_metrics((), _workload())

    assert result.request_count == 0
    assert result.wall_clock_window_s is None
    assert result.token_throughput_per_s is None
    assert result.ttft_s == Percentiles(None, None, None)
    assert result.tpot_s == Percentiles(None, None, None)
    assert result.longest_queue_wait_s is None
    assert result.queue_wait_mean_s is None
    assert result.slo_overall.rate == 0.0
    assert all(summary.rate == 0.0 for summary in result.slo_by_class.values())
    assert result.slo_attainment_gap == 0.0
    assert result.jain_fairness_index == 1.0


def test_all_failures_have_zero_throughput_and_no_success_latency_percentiles() -> None:
    records = (
        _record(
            0,
            RequestClass.INTERACTIVE,
            DispatchStatus.ERROR,
            arrival=0.0,
            dispatch=0.5,
            first_token=None,
            completion=1.0,
            output_tokens=0,
        ),
        _record(
            1,
            RequestClass.BATCH,
            DispatchStatus.TIMEOUT,
            arrival=1.0,
            dispatch=1.5,
            first_token=None,
            completion=2.0,
            output_tokens=0,
        ),
    )

    result = calculate_metrics(records, _workload())

    assert result.token_throughput_per_s == 0.0
    assert result.ttft_s == Percentiles(None, None, None)
    assert result.end_to_end_s == Percentiles(None, None, None)
    assert result.queue_wait_s == Percentiles(None, None, None)
    assert result.longest_queue_wait_s == 0.5
    assert result.queue_wait_mean_s is None
    assert result.slo_overall.rate == 0.0
    assert result.jain_fairness_index == 1.0


def test_one_token_success_is_excluded_from_tpot_and_slo_boundaries_are_inclusive() -> None:
    record = _record(
        0,
        RequestClass.INTERACTIVE,
        DispatchStatus.SUCCESS,
        arrival=0.0,
        dispatch=0.0,
        first_token=1.0,
        completion=3.0,
        output_tokens=1,
    )

    result = calculate_metrics((record,), _workload())

    assert result.tpot_s == Percentiles(None, None, None)
    assert result.slo_overall.met_count == 1
    assert result.slo_overall.rate == 1.0


def test_zero_wall_clock_window_has_undefined_throughput() -> None:
    record = _record(
        0,
        RequestClass.BATCH,
        DispatchStatus.SUCCESS,
        arrival=0.0,
        dispatch=0.0,
        first_token=0.0,
        completion=0.0,
        output_tokens=2,
    )

    result = calculate_metrics((record,), _workload())

    assert result.wall_clock_window_s == 0.0
    assert result.token_throughput_per_s is None
    assert result.tpot_s == Percentiles(0.0, 0.0, 0.0)


def test_fairness_ignores_absent_request_classes() -> None:
    records = (
        _record(
            0,
            RequestClass.INTERACTIVE,
            DispatchStatus.SUCCESS,
            arrival=0.0,
            dispatch=0.1,
            first_token=0.5,
            completion=2.0,
            output_tokens=3,
        ),
        _record(
            1,
            RequestClass.INTERACTIVE,
            DispatchStatus.ERROR,
            arrival=1.0,
            dispatch=1.1,
            first_token=None,
            completion=2.0,
            output_tokens=0,
        ),
    )

    result = calculate_metrics(records, _workload())

    # Only the class that actually appears is reported and scored for fairness; the absent
    # BATCH class must not be counted as 0.0 attainment and drag Jain/gap toward "unfair".
    assert set(result.slo_by_class) == {RequestClass.INTERACTIVE}
    assert result.slo_by_class[RequestClass.INTERACTIVE].rate == pytest.approx(0.5)
    assert result.slo_attainment_gap == 0.0
    assert result.jain_fairness_index == pytest.approx(1.0)


def test_rejected_count_and_queue_wait_exclude_undispatched_records() -> None:
    records = (
        _record(
            0,
            RequestClass.INTERACTIVE,
            DispatchStatus.SUCCESS,
            arrival=0.0,
            dispatch=0.25,
            first_token=0.5,
            completion=1.0,
            output_tokens=2,
        ),
        _record(
            1,
            RequestClass.BATCH,
            DispatchStatus.REJECTED,
            arrival=1.0,
            dispatch=None,
            first_token=None,
            completion=10.0,
            output_tokens=0,
        ),
    )

    result = calculate_metrics(records, _workload())

    assert result.rejected_count == 1
    assert result.queue_wait_s == Percentiles(0.25, 0.25, 0.25)
    assert result.longest_queue_wait_s == 0.25
    assert (
        result.success_count
        + result.error_count
        + result.timeout_count
        + result.cancelled_count
        + result.rejected_count
        == result.request_count
    )


def test_clipping_metrics_report_latency_gain_with_output_cost() -> None:
    records = (
        _record(
            0,
            RequestClass.INTERACTIVE,
            DispatchStatus.SUCCESS,
            arrival=0.0,
            dispatch=0.2,
            first_token=0.4,
            completion=1.0,
            output_tokens=500,
            requested_output_tokens=1000,
            backend_max_output_tokens=500,
            finish_reason="length",
        ),
        _record(
            1,
            RequestClass.BATCH,
            DispatchStatus.SUCCESS,
            arrival=1.0,
            dispatch=1.4,
            first_token=1.5,
            completion=2.0,
            output_tokens=300,
            requested_output_tokens=400,
            backend_max_output_tokens=400,
            finish_reason="stop",
        ),
    )

    result = calculate_metrics(records, _workload())

    assert result.queue_wait_mean_s == pytest.approx(0.3)
    assert result.clip_applied_count == 1
    assert result.clip_applied_rate == pytest.approx(0.5)
    assert result.realized_truncation_count == 1
    assert result.realized_truncation_rate == pytest.approx(0.5)
    assert result.mean_cap_reduction_tokens == pytest.approx(500.0)
