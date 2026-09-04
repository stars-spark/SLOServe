"""Bounded external admission queue in front of an asynchronous request backend."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from sloserve.config import AdmissionControlConfig, BackendConfig, RouterConfig, SchedulerPolicyName
from sloserve.router.adaptive_clipping import (
    AdaptiveClipController,
    AdaptiveClipDecision,
    build_adaptive_clip_controller,
)
from sloserve.router.clipping import OutputClipper
from sloserve.router.models import RequestEnvelope
from sloserve.router.policies.base import SchedulingPolicy
from sloserve.router.policies.fcfs import FcfsPolicy
from sloserve.workload.backend import AsyncRequestBackend
from sloserve.workload.dispatcher import DispatchStatus


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    """Terminal admission and backend outcome for one request."""

    request_id: str
    sequence_id: int
    status: DispatchStatus
    enqueue_time_s: float
    dispatch_time_s: float | None
    completion_time_s: float | None
    effective_envelope: RequestEnvelope
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class AdaptiveAdmissionDecision:
    """One dispatch decision joined to the selected request identity."""

    request_id: str
    decision: AdaptiveClipDecision


class _RequestState(StrEnum):
    WAITING = "waiting"
    IN_FLIGHT = "in_flight"
    DONE = "done"


@dataclass(slots=True)
class _QueueEntry:
    envelope: RequestEnvelope
    enqueue_time_s: float
    result_future: asyncio.Future[AdmissionResult]
    state: _RequestState = _RequestState.WAITING
    dispatch_time_s: float | None = None
    backend_task: asyncio.Task[None] | None = None
    runner_task: asyncio.Task[None] | None = None
    timeout_task: asyncio.Task[None] | None = None
    forced_status: DispatchStatus | None = None


class AdmissionQueue:
    """Run a bounded FCFS admission queue outside the backend inference engine."""

    def __init__(
        self,
        *,
        router_config: RouterConfig,
        backend_config: BackendConfig,
        backend: AsyncRequestBackend,
        policy: SchedulingPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        admission_config: AdmissionControlConfig | None = None,
        adaptive_clipper: OutputClipper | None = None,
    ) -> None:
        if policy is None:
            if router_config.policy is not SchedulerPolicyName.FCFS:
                raise NotImplementedError(
                    f"{router_config.policy.value} admission is outside Week 1 step 6"
                )
            policy = FcfsPolicy()

        self._queue_capacity = router_config.queue_capacity
        self._max_in_flight = router_config.max_in_flight
        self._request_timeout_s = backend_config.request_timeout_s
        self._backend = backend
        self._policy = policy
        self._clock = clock
        self._sleep = sleep
        self._adaptive_controller: AdaptiveClipController | None = None
        self._adaptive_clipper: OutputClipper | None = None
        if admission_config is not None and admission_config.adaptive_clip_enabled:
            if adaptive_clipper is None:
                raise ValueError("enabled adaptive clipping requires an OutputClipper")
            self._adaptive_controller = build_adaptive_clip_controller(
                admission_config,
                clock=clock,
            )
            self._adaptive_clipper = adaptive_clipper

        self._condition = asyncio.Condition()
        self._waiting: list[_QueueEntry] = []
        self._entries: dict[int, _QueueEntry] = {}
        self._runner_tasks: set[asyncio.Task[None]] = set()
        self._timeout_tasks: set[asyncio.Task[None]] = set()
        self._started = False
        self._closing = False
        self._closed = False
        self._adaptive_decisions: list[AdaptiveAdmissionDecision] = []

    @property
    def adaptive_decisions(self) -> tuple[AdaptiveAdmissionDecision, ...]:
        """Return dispatch decisions accumulated by an enabled adaptive controller."""
        return tuple(self._adaptive_decisions)

    async def __aenter__(self) -> AdmissionQueue:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc_value, traceback
        await self.aclose()

    async def start(self) -> None:
        """Open the queue for submissions."""
        async with self._condition:
            if self._closed:
                raise RuntimeError("admission queue is closed")
            if self._started:
                return
            self._started = True

    async def submit(self, envelope: RequestEnvelope) -> AdmissionResult:
        """Enqueue one request or reject it immediately when the waiting queue is full."""
        enqueue_time_s = self._clock()
        loop = asyncio.get_running_loop()

        async with self._condition:
            if not self._started or self._closing or self._closed:
                raise RuntimeError("admission queue is not accepting requests")
            if len(self._waiting) >= self._queue_capacity:
                return AdmissionResult(
                    request_id=envelope.request_id,
                    sequence_id=envelope.sequence_id,
                    status=DispatchStatus.REJECTED,
                    enqueue_time_s=enqueue_time_s,
                    dispatch_time_s=None,
                    completion_time_s=self._clock(),
                    effective_envelope=envelope,
                    error_message="admission waiting queue is full",
                )

            entry = _QueueEntry(
                envelope=envelope,
                enqueue_time_s=enqueue_time_s,
                result_future=loop.create_future(),
            )
            self._waiting.append(entry)
            self._entries[id(entry)] = entry
            timeout_task = asyncio.create_task(
                self._watch_timeout(entry),
                name=f"sloserve-admission-timeout-{envelope.sequence_id}",
            )
            entry.timeout_task = timeout_task
            self._timeout_tasks.add(timeout_task)
            timeout_task.add_done_callback(self._timeout_tasks.discard)
            self._admit_available_locked()

        try:
            return await asyncio.shield(entry.result_future)
        except asyncio.CancelledError:
            await self._cancel_from_submitter(entry)
            raise

    async def aclose(self) -> None:
        """Cancel queued and in-flight work, then wait for every internal task to finish."""
        async with self._condition:
            if self._closed:
                return
            if not self._started:
                self._closed = True
                return
            self._closing = True

            for entry in list(self._waiting):
                self._waiting.remove(entry)
                self._complete_locked(
                    entry,
                    DispatchStatus.CANCELLED,
                    error_message="admission queue closed before dispatch",
                )

            for worker_entry in self._in_flight_entries_locked():
                worker_entry.forced_status = DispatchStatus.CANCELLED
                if worker_entry.backend_task is not None:
                    worker_entry.backend_task.cancel()

        try:
            runner_tasks = tuple(self._runner_tasks)
            if runner_tasks:
                await asyncio.gather(*runner_tasks)
            timeout_tasks = tuple(self._timeout_tasks)
            if timeout_tasks:
                await asyncio.gather(*timeout_tasks)
        finally:
            async with self._condition:
                self._closed = True
                self._closing = False

    def _admit_available_locked(self) -> None:
        while (
            not self._closing
            and self._waiting
            and len(self._in_flight_entries_locked()) < self._max_in_flight
        ):
            ordered = self._policy.order(
                (entry.envelope for entry in self._waiting), now_s=self._clock()
            )
            selected_envelope = ordered[0]
            selected_index = next(
                index
                for index, entry in enumerate(self._waiting)
                if entry.envelope is selected_envelope
            )
            entry = self._waiting.pop(selected_index)
            if self._clock() - entry.enqueue_time_s >= self._request_timeout_s:
                self._complete_locked(
                    entry,
                    DispatchStatus.TIMEOUT,
                    error_message=self._timeout_error_message(),
                )
                continue
            if self._adaptive_controller is not None:
                adaptive_clipper = self._adaptive_clipper
                if adaptive_clipper is None:
                    raise RuntimeError("adaptive clipper was not configured")
                stable_waiting_depth = self._stable_waiting_depth_after_selected_locked()
                decision = self._adaptive_controller.update(stable_waiting_depth)
                entry.envelope = adaptive_clipper.apply_adaptive_cap(
                    entry.envelope,
                    decision.selected_cap,
                )
                self._adaptive_decisions.append(
                    AdaptiveAdmissionDecision(
                        request_id=entry.envelope.request_id,
                        decision=decision,
                    )
                )
            entry.state = _RequestState.IN_FLIGHT
            entry.dispatch_time_s = self._clock()
            entry.backend_task = asyncio.create_task(
                self._backend.send(entry.envelope),
                name=f"sloserve-admission-backend-{entry.envelope.sequence_id}",
            )
            runner_task = asyncio.create_task(
                self._run_backend(entry),
                name=f"sloserve-admission-runner-{entry.envelope.sequence_id}",
            )
            entry.runner_task = runner_task
            self._runner_tasks.add(runner_task)
            runner_task.add_done_callback(self._runner_tasks.discard)

    async def _run_backend(self, entry: _QueueEntry) -> None:
        backend_status = DispatchStatus.SUCCESS
        error_message: str | None = None
        backend_task = entry.backend_task
        if backend_task is None:
            raise RuntimeError("in-flight admission entry has no backend task")

        try:
            await backend_task
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                backend_task.cancel()
                await asyncio.gather(backend_task, return_exceptions=True)
                backend_status = DispatchStatus.CANCELLED
                raise
            backend_status = DispatchStatus.CANCELLED
        except Exception as exc:
            backend_status = DispatchStatus.ERROR
            error_message = str(exc)
        finally:
            async with self._condition:
                status = entry.forced_status or backend_status
                if (
                    entry.forced_status is None
                    and self._clock() - entry.enqueue_time_s >= self._request_timeout_s
                ):
                    status = DispatchStatus.TIMEOUT
                if status is DispatchStatus.TIMEOUT:
                    error_message = self._timeout_error_message()
                elif status is DispatchStatus.CANCELLED and error_message is None:
                    error_message = "request was cancelled"
                self._complete_locked(entry, status, error_message=error_message)
                self._admit_available_locked()

    async def _watch_timeout(self, entry: _QueueEntry) -> None:
        deadline_s = entry.enqueue_time_s + self._request_timeout_s
        try:
            while (remaining_s := deadline_s - self._clock()) > 0:
                await self._sleep(remaining_s)
        except asyncio.CancelledError:
            return

        async with self._condition:
            if entry.state is _RequestState.DONE:
                return
            if entry.state is _RequestState.WAITING:
                self._waiting.remove(entry)
                self._complete_locked(
                    entry,
                    DispatchStatus.TIMEOUT,
                    error_message=self._timeout_error_message(),
                )
                return

            entry.forced_status = DispatchStatus.TIMEOUT
            if entry.backend_task is not None:
                entry.backend_task.cancel()

    async def _cancel_from_submitter(self, entry: _QueueEntry) -> None:
        async with self._condition:
            if entry.state is _RequestState.WAITING:
                self._waiting.remove(entry)
                self._complete_locked(
                    entry,
                    DispatchStatus.CANCELLED,
                    error_message="submit caller cancelled before dispatch",
                )
            elif entry.state is _RequestState.IN_FLIGHT:
                entry.forced_status = DispatchStatus.CANCELLED
                if entry.backend_task is not None:
                    entry.backend_task.cancel()

        await asyncio.shield(entry.result_future)

    def _in_flight_entries_locked(self) -> list[_QueueEntry]:
        return [entry for entry in self._entries.values() if entry.state is _RequestState.IN_FLIGHT]

    def _stable_waiting_depth_after_selected_locked(self) -> int:
        """Project waiting depth after every currently free slot is synchronously filled."""
        in_flight_count = len(self._in_flight_entries_locked())
        remaining_free_slots = self._max_in_flight - in_flight_count - 1
        return max(0, len(self._waiting) - remaining_free_slots)

    def _timeout_error_message(self) -> str:
        return f"request exceeded total timeout of {self._request_timeout_s}s"

    def _complete_locked(
        self,
        entry: _QueueEntry,
        status: DispatchStatus,
        *,
        error_message: str | None = None,
    ) -> None:
        if entry.state is _RequestState.DONE:
            return
        entry.state = _RequestState.DONE
        self._entries.pop(id(entry), None)
        current_task = asyncio.current_task()
        if entry.timeout_task is not None and entry.timeout_task is not current_task:
            entry.timeout_task.cancel()
        entry.result_future.set_result(
            AdmissionResult(
                request_id=entry.envelope.request_id,
                sequence_id=entry.envelope.sequence_id,
                status=status,
                enqueue_time_s=entry.enqueue_time_s,
                dispatch_time_s=entry.dispatch_time_s,
                completion_time_s=self._clock(),
                effective_envelope=entry.envelope,
                error_message=error_message,
            )
        )
