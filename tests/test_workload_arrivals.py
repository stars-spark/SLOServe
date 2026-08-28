"""Tests for workload arrival schedules."""

from pathlib import Path

import pytest

from sloserve.config import ArrivalProcess, load_config
from sloserve.workload.arrivals import FixedRateArrivalSchedule, arrival_schedule_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_rate_arrivals_start_at_zero() -> None:
    schedule = FixedRateArrivalSchedule(request_rate_rps=4.0)

    assert schedule.arrival_times(4) == (0.0, 0.25, 0.5, 0.75)


@pytest.mark.parametrize("arrival_process", [ArrivalProcess.POISSON, ArrivalProcess.BURST])
def test_unimplemented_arrival_process_is_explicit(arrival_process: ArrivalProcess) -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload.model_copy(
        update={"arrival_process": arrival_process}
    )

    with pytest.raises(NotImplementedError, match=arrival_process.value):
        arrival_schedule_from_config(workload)
