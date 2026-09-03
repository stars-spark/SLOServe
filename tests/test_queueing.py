"""Tests for the deliberately limited M/G/1 trend baseline."""

import pytest

from sloserve.analysis.queueing import (
    LinearServiceFit,
    estimate_mg1_wait,
    fit_linear_service_time,
)
from sloserve.config import SchedulerPolicyName
from sloserve.metrics import RequestRecord
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus


def _record(sequence_id: int, output_tokens: int, service_time_s: float) -> RequestRecord:
    return RequestRecord(
        request_id=f"r-{sequence_id}",
        sequence_id=sequence_id,
        request_class=RequestClass.BATCH,
        arrival_time_s=float(sequence_id),
        enqueue_time_s=float(sequence_id),
        dispatch_time_s=float(sequence_id),
        first_token_time_s=float(sequence_id),
        completion_time_s=float(sequence_id) + service_time_s,
        input_tokens=32,
        output_tokens=output_tokens,
        status=DispatchStatus.SUCCESS,
        error_type=None,
        policy_name=SchedulerPolicyName.FCFS,
        config_hash="a" * 64,
        repetition_index=0,
        env_version="test",
    )


def test_linear_service_fit_recovers_known_coefficients() -> None:
    records = tuple(
        _record(index, tokens, 0.5 + 0.01 * tokens)
        for index, tokens in enumerate((10, 20, 50, 100))
    )

    fit = fit_linear_service_time(records)

    assert fit.seconds_per_output_token == pytest.approx(0.01)
    assert fit.intercept_s == pytest.approx(0.5)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.sample_count == 4


def test_mg1_matches_formula_and_clipping_reduces_second_moment_and_wait() -> None:
    fit = LinearServiceFit(0.01, 0.1, 1.0, 4)
    original = estimate_mg1_wait((10, 10, 10, 100), arrival_rate_rps=0.5, service_fit=fit)
    clipped = estimate_mg1_wait((10, 10, 10, 20), arrival_rate_rps=0.5, service_fit=fit)

    expected_wait = 0.5 * original.second_moment_service_s2 / (2 * (1 - original.utilization))
    assert original.mean_queue_wait_s == pytest.approx(expected_wait)
    assert clipped.second_moment_service_s2 < original.second_moment_service_s2
    assert clipped.mean_queue_wait_s is not None
    assert original.mean_queue_wait_s is not None
    assert clipped.mean_queue_wait_s < original.mean_queue_wait_s


def test_mg1_marks_unstable_load_without_emitting_infinity() -> None:
    estimate = estimate_mg1_wait(
        (100, 100),
        arrival_rate_rps=2.0,
        service_fit=LinearServiceFit(0.01, 0.1, 1.0, 2),
    )

    assert estimate.stable is False
    assert estimate.mean_queue_wait_s is None


def test_queueing_rejects_insufficient_or_nonphysical_inputs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fit_linear_service_time((_record(0, 10, 1.0),))
    with pytest.raises(ValueError, match="positive"):
        estimate_mg1_wait(
            (0,),
            arrival_rate_rps=1.0,
            service_fit=LinearServiceFit(0.01, 0.1, 1.0, 2),
        )
