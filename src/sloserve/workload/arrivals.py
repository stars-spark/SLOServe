"""Arrival schedules for reproducible workload generation."""

from __future__ import annotations

import random
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


class PoissonArrivalSchedule:
    """Generate seeded exponential inter-arrivals at a configured mean rate."""

    def __init__(self, request_rate_rps: float, seed: int) -> None:
        if not request_rate_rps > 0:
            raise ValueError("request_rate_rps must be positive")
        self._request_rate_rps = request_rate_rps
        self._seed = seed

    def arrival_times(self, request_count: int) -> tuple[float, ...]:
        """Return cumulative exponential arrival times beginning at zero."""
        if request_count < 0:
            raise ValueError("request_count must be non-negative")
        if request_count == 0:
            return ()

        rng = random.Random(self._seed)
        arrivals = [0.0]
        for _ in range(1, request_count):
            arrivals.append(arrivals[-1] + rng.expovariate(self._request_rate_rps))
        return tuple(arrivals)


def arrival_schedule_from_config(config: WorkloadConfig) -> ArrivalSchedule:
    """Build a configured arrival schedule and reject unsupported process types."""
    if config.arrival_process is ArrivalProcess.FIXED:
        return FixedRateArrivalSchedule(config.request_rate_rps)
    if config.arrival_process is ArrivalProcess.POISSON:
        return PoissonArrivalSchedule(config.request_rate_rps, config.random_seed)
    raise ValueError(f"unsupported arrival process: {config.arrival_process.value}")
