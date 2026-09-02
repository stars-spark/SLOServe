"""SLO-aware scheduling with normalized scoring and hard aging."""

from __future__ import annotations

from sloserve.config import SloAwareConfig
from sloserve.router.models import RequestEnvelope
from sloserve.router.policies.base import SchedulingPolicy

_EPSILON = 1e-9


class SloAwarePolicy(SchedulingPolicy):
    """Rank requests by normalized service time, slack, waiting time, and hard aging."""

    def __init__(self, config: SloAwareConfig) -> None:
        self.input_token_seconds = config.input_token_seconds
        self.output_token_seconds = config.output_token_seconds
        self.cost_weight = config.cost_weight
        self.slack_weight = config.slack_weight
        self.waiting_weight = config.waiting_weight
        self.aging_threshold_s = config.aging_threshold_s
        self.disable_length_estimate = config.disable_length_estimate

    @property
    def name(self) -> str:
        """Return the policy's configuration name."""
        return "slo_aware"

    def priority_key(self, request: RequestEnvelope, *, now_s: float) -> tuple[float | int, ...]:
        """Return a normalized score key with an absolute aged-request tier."""
        budget = request.deadline_time_s - request.arrival_time_s
        if budget <= 0.0:
            budget = _EPSILON

        if self.disable_length_estimate:
            service_time = 0.0
        else:
            service_time = (
                self.input_token_seconds * request.input_tokens
                + self.output_token_seconds * request.max_output_tokens
            )
        slack = request.deadline_time_s - now_s - service_time
        waiting = now_s - request.arrival_time_s

        service_time_norm = service_time / budget
        slack_norm = slack / budget
        waiting_norm = waiting / budget

        score = (
            self.cost_weight * service_time_norm
            + self.slack_weight * slack_norm
            - self.waiting_weight * waiting_norm
        )

        if waiting >= self.aging_threshold_s:
            return (0.0, request.arrival_time_s, float(request.sequence_id))
        return (1.0, score, float(request.sequence_id))
