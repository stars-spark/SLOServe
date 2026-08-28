"""Scheduling policy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from sloserve.router.models import RequestEnvelope


class SchedulingPolicy(ABC):
    """Common interface used to rank queued requests."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable configuration name of the policy."""

    @abstractmethod
    def priority_key(self, request: RequestEnvelope, *, now_s: float) -> tuple[float | int, ...]:
        """Return an ascending sort key for a queued request."""

    def order(self, requests: Iterable[RequestEnvelope], *, now_s: float) -> list[RequestEnvelope]:
        """Return requests in dispatch order without mutating the input collection."""
        return sorted(requests, key=lambda request: self.priority_key(request, now_s=now_s))
