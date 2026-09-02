"""Tests for workload arrival schedules."""

from itertools import pairwise
from pathlib import Path

import pytest

from sloserve.config import ArrivalProcess, load_config
from sloserve.workload.arrivals import (
    FixedRateArrivalSchedule,
    PoissonArrivalSchedule,
    arrival_schedule_from_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_rate_arrivals_start_at_zero() -> None:
    schedule = FixedRateArrivalSchedule(request_rate_rps=4.0)

    assert schedule.arrival_times(4) == (0.0, 0.25, 0.5, 0.75)


def test_poisson_same_seed_is_reproducible_and_different_seed_changes_arrivals() -> None:
    first = PoissonArrivalSchedule(request_rate_rps=2.0, seed=17).arrival_times(20)
    replay = PoissonArrivalSchedule(request_rate_rps=2.0, seed=17).arrival_times(20)
    different = PoissonArrivalSchedule(request_rate_rps=2.0, seed=18).arrival_times(20)

    assert first == replay
    assert first != different


def test_poisson_arrivals_start_at_zero_are_non_decreasing_and_have_requested_length() -> None:
    arrivals = PoissonArrivalSchedule(request_rate_rps=3.0, seed=20250825).arrival_times(100)

    assert len(arrivals) == 100
    assert arrivals[0] == 0.0
    assert all(current <= following for current, following in pairwise(arrivals))


def test_poisson_mean_interarrival_matches_inverse_rate_over_large_sample() -> None:
    request_rate_rps = 2.0
    schedule = PoissonArrivalSchedule(request_rate_rps=request_rate_rps, seed=20250825)
    arrivals = schedule.arrival_times(5000)
    intervals = [following - current for current, following in pairwise(arrivals)]

    assert sum(intervals) / len(intervals) == pytest.approx(1 / request_rate_rps, rel=0.10)


def test_poisson_handles_empty_count() -> None:
    assert PoissonArrivalSchedule(2.0, seed=1).arrival_times(0) == ()


def test_poisson_rejects_non_positive_rate_and_negative_count() -> None:
    with pytest.raises(ValueError, match="request_rate_rps"):
        PoissonArrivalSchedule(0.0, seed=1)
    with pytest.raises(ValueError, match="request_count"):
        PoissonArrivalSchedule(2.0, seed=1).arrival_times(-1)


def test_poisson_factory_uses_workload_rate_and_seed() -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload.model_copy(
        update={"arrival_process": ArrivalProcess.POISSON, "request_rate_rps": 2.0}
    )

    schedule = arrival_schedule_from_config(workload)

    assert isinstance(schedule, PoissonArrivalSchedule)
    assert schedule.arrival_times(20) == PoissonArrivalSchedule(
        workload.request_rate_rps, workload.random_seed
    ).arrival_times(20)


def test_unsupported_arrival_process_is_explicit() -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload.model_copy(
        update={"arrival_process": ArrivalProcess.BURST}
    )

    with pytest.raises(ValueError, match=ArrivalProcess.BURST.value):
        arrival_schedule_from_config(workload)
