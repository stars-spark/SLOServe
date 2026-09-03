"""SLO-aware scheduling with normalized scoring and hard aging."""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence

from sloserve.config import LengthSource, SloAwareConfig
from sloserve.router.models import RequestEnvelope
from sloserve.router.policies.base import SchedulingPolicy
from sloserve.workload.length_predictor import LengthPredictor

_EPSILON = 1e-9


class SloAwarePolicy(SchedulingPolicy):
    """Rank requests by normalized scoring within graduated aging tiers."""

    def __init__(self, config: SloAwareConfig) -> None:
        self.input_token_seconds = config.input_token_seconds
        self.output_token_seconds = config.output_token_seconds
        self.cost_weight = config.cost_weight
        self.slack_weight = config.slack_weight
        self.waiting_weight = config.waiting_weight
        self.aging_threshold_s = config.aging_threshold_s
        self.disable_length_estimate = config.disable_length_estimate
        self.aging_levels = config.aging_levels
        self.adaptive_ceiling = config.adaptive_ceiling
        self.ceiling_margin = config.ceiling_margin
        self.ceiling_floor_s = config.ceiling_floor_s
        self.ceiling_cap_s = config.ceiling_cap_s
        self.length_source = config.length_source
        self._length_predictor: LengthPredictor | None = None
        if self.length_source is LengthSource.LEARNED:
            if config.length_estimator_path is None:
                raise ValueError("learned length source requires length_estimator_path")
            self._length_predictor = LengthPredictor.load(config.length_estimator_path)

    @property
    def name(self) -> str:
        """Return the policy's configuration name."""
        return "slo_aware"

    def _estimated_output_tokens(self, request: RequestEnvelope) -> float:
        """Return the output length visible through the configured information boundary."""
        if self.length_source is LengthSource.TRUE:
            return float(request.max_output_tokens)
        if self.length_source is LengthSource.ADVERTISED:
            return float(request.advertised_cap_tokens or request.max_output_tokens)
        if self.length_source is LengthSource.LEARNED:
            if self._length_predictor is None:
                raise RuntimeError("learned length predictor was not loaded")
            return self._length_predictor.predict(
                request.request_class,
                request.input_tokens,
                request.prompt_kind,
                request.advertised_cap_tokens or request.max_output_tokens,
            )
        raise ValueError(f"unsupported length source: {self.length_source}")

    def _key(
        self, request: RequestEnvelope, now_s: float, threshold: float
    ) -> tuple[float | int, ...]:
        """Return the normalized score key for a request under a given aging threshold."""
        budget = request.deadline_time_s - request.arrival_time_s
        if budget <= 0.0:
            budget = _EPSILON

        if self.disable_length_estimate:
            service_time = 0.0
        else:
            service_time = (
                self.input_token_seconds * request.input_tokens
                + self.output_token_seconds * self._estimated_output_tokens(request)
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

        step = threshold / self.aging_levels
        if step <= 0.0:
            raise ValueError("aging tier step must be positive")
        age_tier = min(int(waiting // step), self.aging_levels)
        if age_tier == self.aging_levels:
            return (0.0, request.arrival_time_s, float(request.sequence_id))
        return (
            float(self.aging_levels - age_tier),
            score,
            float(request.sequence_id),
        )

    def priority_key(self, request: RequestEnvelope, *, now_s: float) -> tuple[float | int, ...]:
        """Return a normalized score key with a fixed absolute aged-request tier."""
        return self._key(request, now_s, self.aging_threshold_s)

    def _effective_threshold(self, waits: Sequence[float]) -> float:
        """Clamp a congestion-relative ceiling from the current waiting-time distribution."""
        congestion = statistics.median(waits)
        return min(self.ceiling_cap_s, max(self.ceiling_floor_s, self.ceiling_margin * congestion))

    def order(self, requests: Iterable[RequestEnvelope], *, now_s: float) -> list[RequestEnvelope]:
        """Order requests, floating the aging ceiling with queue congestion when enabled."""
        if not self.adaptive_ceiling:
            return super().order(requests, now_s=now_s)
        materialized = list(requests)
        if not materialized:
            return []
        waits = [now_s - request.arrival_time_s for request in materialized]
        threshold = self._effective_threshold(waits)
        return sorted(materialized, key=lambda request: self._key(request, now_s, threshold))
