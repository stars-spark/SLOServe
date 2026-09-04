"""M/G/1 trend baseline for output-length and queueing-delay studies."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from sloserve.metrics.records import RequestRecord
from sloserve.workload.dispatcher import DispatchStatus


@dataclass(frozen=True, slots=True)
class LinearServiceFit:
    """Ordinary least-squares fit of backend service time against output tokens."""

    seconds_per_output_token: float
    intercept_s: float
    r_squared: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class Mg1Estimate:
    """Pollaczek-Khinchine mean-wait estimate for one empirical length sample.

    ``concurrency`` above one selects a deliberately crude bridge to the real system, which
    serves up to ``max_in_flight`` requests at once: every fitted service time is divided by
    ``concurrency``, so the queue is modelled as a single server that is ``concurrency`` times
    faster. This is *not* M/G/c -- Pollaczek-Khinchine has no exact M/G/c form -- and it
    understates waiting because it lets one long request borrow the whole aggregate rate. It is
    used only because the unscaled single-server model is vacuous here: the measured system is
    plainly stable while a c=1 model reports utilization above two and refuses to predict at all.
    """

    arrival_rate_rps: float
    concurrency: int
    mean_service_s: float
    second_moment_service_s2: float
    utilization: float
    mean_queue_wait_s: float | None

    @property
    def stable(self) -> bool:
        """Whether the single-server approximation has utilization below one."""
        return self.utilization < 1.0


def fit_linear_service_time(records: Sequence[RequestRecord]) -> LinearServiceFit:
    """Fit ``service_time = a * output_tokens + c`` from successful dispatches."""
    samples = [
        (float(record.output_tokens), record.completion_time_s - record.dispatch_time_s)
        for record in records
        if record.status is DispatchStatus.SUCCESS
        and record.dispatch_time_s is not None
        and record.output_tokens > 0
    ]
    if len(samples) < 2:
        raise ValueError("service-time fitting requires at least two successful samples")
    token_counts = [tokens for tokens, _ in samples]
    if len(set(token_counts)) < 2:
        raise ValueError("service-time fitting requires at least two distinct output lengths")
    service_times = [service_time for _, service_time in samples]
    token_mean = statistics.fmean(token_counts)
    service_mean = statistics.fmean(service_times)
    centered_square_sum = sum((tokens - token_mean) ** 2 for tokens in token_counts)
    slope = (
        sum(
            (tokens - token_mean) * (service_time - service_mean)
            for tokens, service_time in samples
        )
        / centered_square_sum
    )
    intercept = service_mean - slope * token_mean
    predictions = [slope * tokens + intercept for tokens in token_counts]
    residual_square_sum = sum(
        (observed - predicted) ** 2
        for observed, predicted in zip(service_times, predictions, strict=True)
    )
    total_square_sum = sum((service_time - service_mean) ** 2 for service_time in service_times)
    r_squared = 1.0 - residual_square_sum / total_square_sum if total_square_sum > 0 else 1.0
    return LinearServiceFit(
        seconds_per_output_token=slope,
        intercept_s=intercept,
        r_squared=r_squared,
        sample_count=len(samples),
    )


def estimate_mg1_wait(
    output_lengths: Sequence[int],
    *,
    arrival_rate_rps: float,
    service_fit: LinearServiceFit,
    concurrency: int = 1,
) -> Mg1Estimate:
    """Estimate mean FCFS wait; return null wait when the M/G/1 model is unstable.

    ``concurrency`` scales the fitted service times by ``1 / concurrency``; see ``Mg1Estimate``
    for why that approximation is used and what it costs.
    """
    if not output_lengths:
        raise ValueError("M/G/1 estimation requires at least one output length")
    if arrival_rate_rps <= 0.0 or not math.isfinite(arrival_rate_rps):
        raise ValueError("arrival_rate_rps must be finite and positive")
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    if any(length < 1 for length in output_lengths):
        raise ValueError("output lengths must be positive")
    service_times = [
        (service_fit.seconds_per_output_token * length + service_fit.intercept_s) / concurrency
        for length in output_lengths
    ]
    if any(
        service_time <= 0.0 or not math.isfinite(service_time) for service_time in service_times
    ):
        raise ValueError("fitted service times must be finite and positive")
    mean_service = statistics.fmean(service_times)
    second_moment = statistics.fmean(service_time * service_time for service_time in service_times)
    utilization = arrival_rate_rps * mean_service
    mean_wait = (
        arrival_rate_rps * second_moment / (2.0 * (1.0 - utilization))
        if utilization < 1.0
        else None
    )
    return Mg1Estimate(
        arrival_rate_rps=arrival_rate_rps,
        concurrency=concurrency,
        mean_service_s=mean_service,
        second_moment_service_s2=second_moment,
        utilization=utilization,
        mean_queue_wait_s=mean_wait,
    )
