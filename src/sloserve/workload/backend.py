"""Async backend boundary and an in-memory controllable test backend."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from sloserve.router.models import RequestEnvelope


class AsyncRequestBackend(Protocol):
    """Boundary for dispatching one request to a future streaming HTTP backend."""

    async def send(self, request: RequestEnvelope) -> None:
        """Send one request and return when its backend interaction is complete."""


class FakeBackendMode(StrEnum):
    """Controllable outcomes supported by :class:`FakeBackend`."""

    SUCCESS = "success"
    ERROR = "error"
    DELAY = "delay"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class FakeBackendPlan:
    """Per-request behavior for the in-memory fake backend."""

    mode: FakeBackendMode = FakeBackendMode.SUCCESS
    delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.delay_s < 0:
            raise ValueError("delay_s must be non-negative")


class FakeBackend:
    """Pure in-memory backend with deterministic, per-request behavior."""

    def __init__(
        self,
        plans: Mapping[str, FakeBackendPlan] | None = None,
        *,
        default_plan: FakeBackendPlan | None = None,
    ) -> None:
        self._plans = dict(plans or {})
        self._default_plan = default_plan or FakeBackendPlan()
        self.started_request_ids: list[str] = []
        self.cancelled_request_ids: list[str] = []
        self.active_count = 0
        self.max_active_count = 0

    async def send(self, request: RequestEnvelope) -> None:
        """Execute the configured behavior without network or filesystem access."""
        plan = self._plans.get(request.request_id, self._default_plan)
        self.started_request_ids.append(request.request_id)
        self.active_count += 1
        self.max_active_count = max(self.max_active_count, self.active_count)
        try:
            if plan.delay_s:
                await asyncio.sleep(plan.delay_s)
            if plan.mode is FakeBackendMode.ERROR:
                raise RuntimeError(f"configured fake backend error for {request.request_id}")
            if plan.mode is FakeBackendMode.CANCELLED:
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            self.cancelled_request_ids.append(request.request_id)
            raise
        finally:
            self.active_count -= 1
