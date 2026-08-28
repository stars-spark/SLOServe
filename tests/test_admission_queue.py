"""Tests for the bounded external FCFS admission queue."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest

from sloserve.config import BackendConfig, RouterConfig
from sloserve.router.admission import AdmissionQueue
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.backend import FakeBackend, FakeBackendMode, FakeBackendPlan
from sloserve.workload.dispatcher import DispatchStatus

PINNED_REVISION = "a" * 40


class ManualTime:
    """Controllable monotonic clock and independently wakeable sleeps."""

    def __init__(self) -> None:
        self.now_s = 0.0
        self.sleep_futures: list[asyncio.Future[None]] = []
        self.sleep_deadlines: list[float] = []

    def clock(self) -> float:
        return self.now_s

    async def sleep(self, delay_s: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self.sleep_futures.append(future)
        self.sleep_deadlines.append(self.now_s + delay_s)
        await future

    def wake(self, index: int) -> None:
        self.now_s = max(self.now_s, self.sleep_deadlines[index])
        self.sleep_futures[index].set_result(None)


def _router_config(*, max_in_flight: int = 1, queue_capacity: int = 4) -> RouterConfig:
    return RouterConfig(
        policy="fcfs",
        listen_host="127.0.0.1",
        listen_port=8080,
        max_in_flight=max_in_flight,
        queue_capacity=queue_capacity,
    )


def _backend_config(*, request_timeout_s: float = 1.0) -> BackendConfig:
    return BackendConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="fake-model",
        model_revision=PINNED_REVISION,
        tokenizer_revision=PINNED_REVISION,
        request_timeout_s=request_timeout_s,
    )


def _request(sequence_id: int, *, arrival_time_s: float | None = None) -> RequestEnvelope:
    arrival = float(sequence_id) if arrival_time_s is None else arrival_time_s
    return RequestEnvelope(
        request_id=f"request-{sequence_id}",
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=arrival,
        input_tokens=8,
        max_output_tokens=4,
        deadline_time_s=arrival + 10.0,
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def test_admission_is_strict_fcfs_with_bounded_concurrency() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=0.02)}
        )
        async with AdmissionQueue(
            router_config=_router_config(max_in_flight=1, queue_capacity=3),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=backend,
        ) as queue:
            blocker = asyncio.create_task(queue.submit(_request(0, arrival_time_s=0.0)))
            await _wait_until(lambda: backend.active_count == 1)

            queued = [
                asyncio.create_task(queue.submit(_request(3, arrival_time_s=2.0))),
                asyncio.create_task(queue.submit(_request(1, arrival_time_s=1.0))),
                asyncio.create_task(queue.submit(_request(2, arrival_time_s=1.0))),
            ]
            results = await asyncio.gather(blocker, *queued)

        assert all(result.status is DispatchStatus.SUCCESS for result in results)
        assert backend.started_request_ids == [
            "request-0",
            "request-1",
            "request-2",
            "request-3",
        ]
        assert backend.max_active_count == 1

    asyncio.run(scenario())


def test_admission_accepts_in_flight_plus_waiting_capacity() -> None:
    async def scenario() -> None:
        backend = FakeBackend(default_plan=FakeBackendPlan(delay_s=0.01))
        async with AdmissionQueue(
            router_config=_router_config(max_in_flight=2, queue_capacity=4),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=backend,
        ) as queue:
            results = await asyncio.gather(
                *(queue.submit(_request(sequence_id)) for sequence_id in range(6))
            )

        assert [result.status for result in results] == [DispatchStatus.SUCCESS] * 6
        assert backend.started_request_ids == [f"request-{index}" for index in range(6)]
        assert backend.max_active_count == 2
        assert backend.active_count == 0

    asyncio.run(scenario())


def test_full_waiting_queue_rejects_immediately_without_backend_or_slot() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0)}
        )
        async with AdmissionQueue(
            router_config=_router_config(max_in_flight=1, queue_capacity=1),
            backend_config=_backend_config(request_timeout_s=20.0),
            backend=backend,
        ) as queue:
            blocker = asyncio.create_task(queue.submit(_request(0)))
            await _wait_until(lambda: backend.active_count == 1)
            waiting = asyncio.create_task(queue.submit(_request(1)))
            await asyncio.sleep(0)

            rejected = await queue.submit(_request(2))

            assert rejected.status is DispatchStatus.REJECTED
            assert rejected.dispatch_time_s is None
            assert backend.started_request_ids == ["request-0"]
            waiting.cancel()
            blocker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            with pytest.raises(asyncio.CancelledError):
                await blocker

        assert backend.active_count == 0

    asyncio.run(scenario())


def test_request_times_out_while_waiting_without_dispatch() -> None:
    async def scenario() -> None:
        manual_time = ManualTime()
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0)}
        )
        async with AdmissionQueue(
            router_config=_router_config(max_in_flight=1, queue_capacity=1),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=backend,
            clock=manual_time.clock,
            sleep=manual_time.sleep,
        ) as queue:
            blocker = asyncio.create_task(queue.submit(_request(0)))
            await _wait_until(
                lambda: backend.active_count == 1 and len(manual_time.sleep_futures) == 1
            )
            waiting = asyncio.create_task(queue.submit(_request(1)))
            await _wait_until(lambda: len(manual_time.sleep_futures) == 2)

            manual_time.wake(1)
            result = await waiting

            assert result.status is DispatchStatus.TIMEOUT
            assert result.dispatch_time_s is None
            assert result.completion_time_s == 1.0
            assert backend.started_request_ids == ["request-0"]
            blocker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocker

    asyncio.run(scenario())


def test_request_times_out_in_flight_and_backend_is_awaited() -> None:
    async def scenario() -> None:
        manual_time = ManualTime()
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0)}
        )
        async with AdmissionQueue(
            router_config=_router_config(),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=backend,
            clock=manual_time.clock,
            sleep=manual_time.sleep,
        ) as queue:
            submitted = asyncio.create_task(queue.submit(_request(0)))
            await _wait_until(
                lambda: backend.active_count == 1 and len(manual_time.sleep_futures) == 1
            )

            manual_time.wake(0)
            result = await submitted

            assert result.status is DispatchStatus.TIMEOUT
            assert result.dispatch_time_s == 0.0
            assert backend.cancelled_request_ids == ["request-0"]
            assert backend.active_count == 0

    asyncio.run(scenario())


def test_elapsed_timeout_wins_if_timer_task_has_not_been_scheduled() -> None:
    async def scenario() -> None:
        manual_time = ManualTime()
        backend = FakeBackend(default_plan=FakeBackendPlan(delay_s=0.01))
        async with AdmissionQueue(
            router_config=_router_config(),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=backend,
            clock=manual_time.clock,
            sleep=manual_time.sleep,
        ) as queue:
            submitted = asyncio.create_task(queue.submit(_request(0)))
            await _wait_until(lambda: backend.active_count == 1)

            manual_time.now_s = 1.0
            result = await submitted

            assert result.status is DispatchStatus.TIMEOUT
            assert backend.cancelled_request_ids == []
            assert backend.active_count == 0

    asyncio.run(scenario())


def test_submitter_cancellation_removes_waiting_request() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0)}
        )
        async with AdmissionQueue(
            router_config=_router_config(max_in_flight=1),
            backend_config=_backend_config(request_timeout_s=20.0),
            backend=backend,
        ) as queue:
            blocker = asyncio.create_task(queue.submit(_request(0)))
            await _wait_until(lambda: backend.active_count == 1)
            waiting = asyncio.create_task(queue.submit(_request(1)))
            await asyncio.sleep(0)

            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            assert backend.started_request_ids == ["request-0"]

            blocker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await blocker

    asyncio.run(scenario())


def test_submitter_cancellation_cleans_up_in_flight_backend() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0)}
        )
        async with AdmissionQueue(
            router_config=_router_config(),
            backend_config=_backend_config(request_timeout_s=20.0),
            backend=backend,
        ) as queue:
            submitted = asyncio.create_task(queue.submit(_request(0)))
            await _wait_until(lambda: backend.active_count == 1)

            submitted.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submitted

            assert backend.cancelled_request_ids == ["request-0"]
            assert backend.active_count == 0

    asyncio.run(scenario())


def test_mixed_terminal_results_do_not_leak_capacity() -> None:
    async def scenario() -> None:
        manual_time = ManualTime()
        backend = FakeBackend(
            {
                "request-1": FakeBackendPlan(mode=FakeBackendMode.ERROR),
                "request-2": FakeBackendPlan(mode=FakeBackendMode.CANCELLED),
                "request-3": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0),
                "request-4": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0),
            }
        )
        async with AdmissionQueue(
            router_config=_router_config(max_in_flight=1, queue_capacity=1),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=backend,
            clock=manual_time.clock,
            sleep=manual_time.sleep,
        ) as queue:
            success = await queue.submit(_request(0))
            error = await queue.submit(_request(1))
            backend_cancel = await queue.submit(_request(2))

            timed_out_task = asyncio.create_task(queue.submit(_request(3)))
            await _wait_until(lambda: backend.active_count == 1)
            timeout_sleep_index = len(manual_time.sleep_futures) - 1
            manual_time.wake(timeout_sleep_index)
            timed_out = await timed_out_task

            caller_cancelled = asyncio.create_task(queue.submit(_request(4)))
            await _wait_until(lambda: backend.active_count == 1)
            waiting_cancelled = asyncio.create_task(queue.submit(_request(6)))
            await asyncio.sleep(0)
            rejected = await queue.submit(_request(7))
            waiting_cancelled.cancel()
            caller_cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting_cancelled
            with pytest.raises(asyncio.CancelledError):
                await caller_cancelled

            follow_up = await queue.submit(_request(5))

        assert [success.status, error.status, backend_cancel.status, timed_out.status] == [
            DispatchStatus.SUCCESS,
            DispatchStatus.ERROR,
            DispatchStatus.CANCELLED,
            DispatchStatus.TIMEOUT,
        ]
        assert follow_up.status is DispatchStatus.SUCCESS
        assert rejected.status is DispatchStatus.REJECTED
        assert backend.active_count == 0
        assert backend.max_active_count == 1

    asyncio.run(scenario())


def test_close_cancels_all_work_and_leaves_no_admission_tasks() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0)}
        )
        queue = AdmissionQueue(
            router_config=_router_config(max_in_flight=1),
            backend_config=_backend_config(request_timeout_s=20.0),
            backend=backend,
        )
        await queue.start()
        in_flight = asyncio.create_task(queue.submit(_request(0)))
        await _wait_until(lambda: backend.active_count == 1)
        waiting = asyncio.create_task(queue.submit(_request(1)))
        await asyncio.sleep(0)

        await queue.aclose()
        in_flight_result, waiting_result = await asyncio.gather(in_flight, waiting)

        assert in_flight_result.status is DispatchStatus.CANCELLED
        assert waiting_result.status is DispatchStatus.CANCELLED
        assert backend.active_count == 0
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and task.get_name().startswith("sloserve-admission-")
            and not task.done()
        ]

    asyncio.run(scenario())
