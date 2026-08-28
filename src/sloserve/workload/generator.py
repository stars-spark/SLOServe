"""Deterministic request generation from versioned workload configuration."""

from __future__ import annotations

import random

from sloserve.config import RequestProfileConfig, WorkloadConfig
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.arrivals import ArrivalSchedule, arrival_schedule_from_config


def generate_requests(
    config: WorkloadConfig,
    *,
    arrival_schedule: ArrivalSchedule | None = None,
) -> tuple[RequestEnvelope, ...]:
    """Generate warmup requests followed by measured requests.

    Warmup requests are extra requests prepended to ``total_requests``. They use sequence IDs
    starting at zero and consume the same deterministic RNG stream. Consequently, measured
    requests start at ``sequence_id == warmup_requests``.
    """
    schedule = arrival_schedule or arrival_schedule_from_config(config)
    request_count = config.warmup_requests + config.total_requests
    arrival_times = schedule.arrival_times(request_count)
    if len(arrival_times) != request_count:
        raise ValueError("arrival schedule returned an unexpected number of timestamps")

    rng = random.Random(config.random_seed)
    requests: list[RequestEnvelope] = []
    for sequence_id, arrival_time_s in enumerate(arrival_times):
        request_class = _sample_request_class(rng, config.interactive_fraction)
        profile = _profile_for_class(config, request_class)
        requests.append(
            RequestEnvelope(
                request_id=f"request-{sequence_id:06d}",
                sequence_id=sequence_id,
                request_class=request_class,
                arrival_time_s=arrival_time_s,
                input_tokens=rng.randint(
                    profile.input_tokens.minimum,
                    profile.input_tokens.maximum,
                ),
                max_output_tokens=rng.randint(
                    profile.output_tokens.minimum,
                    profile.output_tokens.maximum,
                ),
                deadline_time_s=arrival_time_s + profile.end_to_end_slo_ms / 1000.0,
            )
        )
    return tuple(requests)


def _sample_request_class(rng: random.Random, interactive_fraction: float) -> RequestClass:
    if rng.random() < interactive_fraction:
        return RequestClass.INTERACTIVE
    return RequestClass.BATCH


def _profile_for_class(config: WorkloadConfig, request_class: RequestClass) -> RequestProfileConfig:
    if request_class is RequestClass.INTERACTIVE:
        return config.interactive
    return config.batch
