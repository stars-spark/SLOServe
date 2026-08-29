"""Static request-class priority scheduling policy."""

from __future__ import annotations

from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.router.policies.base import SchedulingPolicy


class StaticPriorityPolicy(SchedulingPolicy):
    """Prioritize interactive requests while preserving FCFS within each class."""

    @property
    def name(self) -> str:
        """Return the policy's configuration name."""
        return "static_priority"

    def priority_key(self, request: RequestEnvelope, *, now_s: float) -> tuple[float | int, ...]:
        """Return request-class priority followed by deterministic FCFS ordering."""
        del now_s
        class_rank = 0 if request.request_class is RequestClass.INTERACTIVE else 1
        return (class_rank, request.arrival_time_s, request.sequence_id)
