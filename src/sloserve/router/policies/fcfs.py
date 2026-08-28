"""First-come, first-served baseline policy."""

from __future__ import annotations

from sloserve.router.models import RequestEnvelope
from sloserve.router.policies.base import SchedulingPolicy


class FcfsPolicy(SchedulingPolicy):
    """Order requests by arrival time, then deterministic sequence number."""

    @property
    def name(self) -> str:
        """Return the policy's configuration name."""
        return "fcfs"

    def priority_key(self, request: RequestEnvelope, *, now_s: float) -> tuple[float | int, ...]:
        """Return a deterministic FCFS ordering key."""
        del now_s
        return (request.arrival_time_s, request.sequence_id)
