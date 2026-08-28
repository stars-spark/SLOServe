"""Async fixed-arrival request dispatch through the shared scheduling interface."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from sloserve.config import BackendConfig, RouterConfig, SchedulerPolicyName
from sloserve.router.models import RequestEnvelope
from sloserve.router.policies.base import SchedulingPolicy
from sloserve.router.policies.fcfs import FcfsPolicy
from sloserve.workload.backend import AsyncRequestBackend


class DispatchStatus(StrEnum):
    """Minimal completion states produced before the metrics schema exists."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Minimal request identity and outcome from asynchronous dispatch."""

    request_id: str
    sequence_id: int
    status: DispatchStatus
    error_message: str | None = None


class AsyncDispatcher:
    """Dispatch requests at relative arrival times with bounded concurrency."""

    def __init__(
        self,
        *,
        router_config: RouterConfig,
        backend_config: BackendConfig,
        backend: AsyncRequestBackend,
        policy: SchedulingPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if policy is None:
            if router_config.policy is not SchedulerPolicyName.FCFS:
                raise NotImplementedError(
                    f"{router_config.policy.value} dispatch is outside Week 1 step 4"
                )
            policy = FcfsPolicy()
        self._max_in_flight = router_config.max_in_flight
        self._request_timeout_s = backend_config.request_timeout_s
        self._backend = backend
        self._policy = policy
        self._clock = clock
        self._sleep = sleep
        self._capacity = asyncio.Semaphore(self._max_in_flight)

    async def run(self, requests: Sequence[RequestEnvelope]) -> tuple[DispatchResult, ...]:
        """Dispatch all requests, propagating caller cancellation after clean task teardown."""
        start_time_s = self._clock()
        ordered = self._policy.order(requests, now_s=start_time_s)
        tasks: list[asyncio.Task[DispatchResult]] = []

        try:
            for request in ordered:
                delay_s = request.arrival_time_s - (self._clock() - start_time_s)
                if delay_s > 0:
                    await self._sleep(delay_s)
                await self._capacity.acquire()
                tasks.append(asyncio.create_task(self._dispatch_one(request)))

            results = await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return tuple(sorted(results, key=lambda result: result.sequence_id))

    async def _dispatch_one(self, request: RequestEnvelope) -> DispatchResult:
        try:
            try:
                async with asyncio.timeout(self._request_timeout_s):
                    await self._backend.send(request)
            except TimeoutError:
                return self._result(request, DispatchStatus.TIMEOUT)
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                return self._result(request, DispatchStatus.CANCELLED)
            except Exception as exc:
                return self._result(request, DispatchStatus.ERROR, str(exc))
            return self._result(request, DispatchStatus.SUCCESS)
        finally:
            self._capacity.release()

    @staticmethod
    def _result(
        request: RequestEnvelope,
        status: DispatchStatus,
        error_message: str | None = None,
    ) -> DispatchResult:
        return DispatchResult(
            request_id=request.request_id,
            sequence_id=request.sequence_id,
            status=status,
            error_message=error_message,
        )
