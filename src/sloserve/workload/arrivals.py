"""Arrival schedules for reproducible workload generation."""

from __future__ import annotations

from typing import Protocol

from sloserve.config import ArrivalProcess, WorkloadConfig


class ArrivalSchedule(Protocol):
    """Interface separating arrival timing from request content generation."""

    def arrival_times(self, request_count: int) -> tuple[float, ...]:
        """Return relative arrival times for ``request_count`` requests."""


class FixedRateArrivalSchedule:
    """Generate evenly spaced arrivals at a configured request rate."""

    def __init__(self, request_rate_rps: float) -> None:
        if request_rate_rps <= 0:
            raise ValueError("request_rate_rps must be positive")
        self._request_rate_rps = request_rate_rps

    def arrival_times(self, request_count: int) -> tuple[float, ...]:
        """Return ``i / request_rate_rps`` for sequence numbers starting at zero."""
        if request_count < 0:
            raise ValueError("request_count must be non-negative")
        return tuple(index / self._request_rate_rps for index in range(request_count))


def arrival_schedule_from_config(config: WorkloadConfig) -> ArrivalSchedule:
    """Build the configured arrival schedule, rejecting unfinished process types."""
    if config.arrival_process is ArrivalProcess.FIXED:
        return FixedRateArrivalSchedule(config.request_rate_rps)
    raise NotImplementedError(
        f"{config.arrival_process.value} arrival process is not implemented in Week 1 step 4"
    )
