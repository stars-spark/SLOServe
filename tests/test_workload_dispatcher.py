"""Tests for bounded asynchronous request dispatch."""

from __future__ import annotations

import asyncio

import pytest

from sloserve.config import BackendConfig, RouterConfig
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.backend import FakeBackend, FakeBackendMode, FakeBackendPlan
from sloserve.workload.dispatcher import AsyncDispatcher, DispatchStatus

PINNED_REVISION = "a" * 40


def _router_config(max_in_flight: int) -> RouterConfig:
    return RouterConfig(
        policy="fcfs",
        listen_host="127.0.0.1",
        listen_port=8080,
        max_in_flight=max_in_flight,
        queue_capacity=32,
    )


def _backend_config(request_timeout_s: float) -> BackendConfig:
    return BackendConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="fake-model",
        model_revision=PINNED_REVISION,
        tokenizer_revision=PINNED_REVISION,
        request_timeout_s=request_timeout_s,
    )


def _request(sequence_id: int, *, arrival_time_s: float = 0.0) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=f"request-{sequence_id}",
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=arrival_time_s,
        input_tokens=8,
        max_output_tokens=4,
        deadline_time_s=arrival_time_s + 1.0,
    )


def test_dispatch_respects_max_in_flight_and_fcfs_order() -> None:
    async def scenario() -> None:
        backend = FakeBackend(default_plan=FakeBackendPlan(delay_s=0.01))
        dispatcher = AsyncDispatcher(
            router_config=_router_config(max_in_flight=2),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=backend,
        )
        requests = [_request(sequence_id) for sequence_id in range(6)]

        results = await dispatcher.run(requests)

        assert [result.status for result in results] == [DispatchStatus.SUCCESS] * 6
        assert backend.started_request_ids == [request.request_id for request in requests]
        assert backend.max_active_count == 2
        assert backend.active_count == 0

    asyncio.run(scenario())


def test_dispatch_reports_success_error_timeout_and_cancelled() -> None:
    async def scenario() -> None:
        plans = {
            "request-1": FakeBackendPlan(mode=FakeBackendMode.ERROR),
            "request-2": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=0.05),
            "request-3": FakeBackendPlan(mode=FakeBackendMode.CANCELLED),
        }
        backend = FakeBackend(plans)
        dispatcher = AsyncDispatcher(
            router_config=_router_config(max_in_flight=4),
            backend_config=_backend_config(request_timeout_s=0.005),
            backend=backend,
        )

        results = await dispatcher.run([_request(sequence_id) for sequence_id in range(4)])

        assert {result.request_id: result.status for result in results} == {
            "request-0": DispatchStatus.SUCCESS,
            "request-1": DispatchStatus.ERROR,
            "request-2": DispatchStatus.TIMEOUT,
            "request-3": DispatchStatus.CANCELLED,
        }
        error = next(result for result in results if result.status is DispatchStatus.ERROR)
        assert error.error_message is not None
        assert "configured fake backend error" in error.error_message
        assert backend.active_count == 0

    asyncio.run(scenario())


def test_dispatch_waits_for_relative_arrival_time() -> None:
    async def scenario() -> None:
        now_s = 10.0
        observed_delays: list[float] = []

        def clock() -> float:
            return now_s

        async def sleep(delay_s: float) -> None:
            nonlocal now_s
            observed_delays.append(delay_s)
            now_s += delay_s

        dispatcher = AsyncDispatcher(
            router_config=_router_config(max_in_flight=1),
            backend_config=_backend_config(request_timeout_s=1.0),
            backend=FakeBackend(),
            clock=clock,
            sleep=sleep,
        )

        await dispatcher.run([_request(1, arrival_time_s=0.25), _request(0)])

        assert observed_delays == [0.25]

    asyncio.run(scenario())


def test_caller_cancellation_propagates_and_releases_in_flight_capacity() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            {"request-0": FakeBackendPlan(mode=FakeBackendMode.DELAY, delay_s=10.0)}
        )
        dispatcher = AsyncDispatcher(
            router_config=_router_config(max_in_flight=1),
            backend_config=_backend_config(request_timeout_s=20.0),
            backend=backend,
        )
        run_task = asyncio.create_task(dispatcher.run([_request(0)]))
        while backend.active_count == 0:
            await asyncio.sleep(0)

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

        assert backend.active_count == 0
        assert backend.cancelled_request_ids == ["request-0"]
        follow_up = await dispatcher.run([_request(1)])
        assert follow_up[0].status is DispatchStatus.SUCCESS

    asyncio.run(scenario())
