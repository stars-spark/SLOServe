"""Adaptive-cap integration tests at the external dispatch boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from sloserve.config import AdmissionControlConfig, BackendConfig, RouterConfig
from sloserve.router.adaptive_clipping import AdaptiveClipLevel
from sloserve.router.admission import AdmissionQueue
from sloserve.router.clipping import OutputClipper
from sloserve.router.models import RequestClass, RequestEnvelope

PINNED_REVISION = "a" * 40


class BlockingBackend:
    """Record immutable send-time envelopes and release calls explicitly."""

    def __init__(self) -> None:
        self.started: list[RequestEnvelope] = []
        self._release_events: dict[str, asyncio.Event] = {}

    async def send(self, request: RequestEnvelope) -> None:
        self.started.append(request)
        await self._release_events.setdefault(request.request_id, asyncio.Event()).wait()

    def release(self, request_id: str) -> None:
        self._release_events.setdefault(request_id, asyncio.Event()).set()


def _request(sequence_id: int) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=f"request-{sequence_id}",
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=0.0,
        input_tokens=64,
        max_output_tokens=1800,
        deadline_time_s=100.0,
        advertised_cap_tokens=2048,
        prompt_kind="long",
    )


def _router() -> RouterConfig:
    return RouterConfig(
        policy="fcfs",
        listen_host="127.0.0.1",
        listen_port=8080,
        max_in_flight=2,
        queue_capacity=8,
    )


def _backend_config() -> BackendConfig:
    return BackendConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="fake",
        model_revision=PINNED_REVISION,
        tokenizer_revision=PINNED_REVISION,
        request_timeout_s=100.0,
    )


def _adaptive_config() -> AdmissionControlConfig:
    return AdmissionControlConfig(
        clip_enabled=True,
        adaptive_clip_enabled=True,
        adaptive_clip_tighten_hold_s=0.0,
    )


async def _wait_until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def _entry(sequence_id: int) -> object:
    """Build the minimal waiting-list entry the depth projection inspects."""
    from sloserve.router.admission import _QueueEntry

    return _QueueEntry(
        envelope=_request(sequence_id),
        enqueue_time_s=0.0,
        result_future=asyncio.get_event_loop_policy().new_event_loop().create_future(),
    )


def test_adaptive_decision_uses_stable_post_fill_depth_and_never_rewrites_in_flight() -> None:
    async def scenario() -> None:
        backend = BlockingBackend()
        admission_config = _adaptive_config()
        queue = AdmissionQueue(
            router_config=_router(),
            backend_config=_backend_config(),
            backend=backend,
            clock=lambda: 0.0,
            admission_config=admission_config,
            adaptive_clipper=OutputClipper(admission_config),
        )
        await queue.start()
        tasks = [asyncio.create_task(queue.submit(_request(index))) for index in range(5)]
        await _wait_until(lambda: len(backend.started) == 2)

        assert [item.effective_max_output_tokens for item in backend.started] == [1800, 1800]
        assert [item.decision.q_inst for item in queue.adaptive_decisions] == [0, 0]

        backend.release("request-0")
        await _wait_until(lambda: len(backend.started) == 3)

        first, second, third = backend.started
        assert first.effective_max_output_tokens == 1800
        assert second.effective_max_output_tokens == 1800
        assert third.request_id == "request-2"
        assert third.effective_max_output_tokens == 1024
        dispatch_decision = queue.adaptive_decisions[2]
        assert dispatch_decision.request_id == third.request_id
        assert dispatch_decision.decision.q_inst == 2
        assert dispatch_decision.decision.new_level is AdaptiveClipLevel.L2

        for request_id in ("request-1", "request-2", "request-3", "request-4"):
            backend.release(request_id)
        await asyncio.gather(*tasks)
        await queue.aclose()

    asyncio.run(scenario())


def test_idle_slot_absorbs_submit_transient_without_false_depth_one() -> None:
    async def scenario() -> None:
        backend = BlockingBackend()
        admission_config = _adaptive_config()
        async with AdmissionQueue(
            router_config=_router(),
            backend_config=_backend_config(),
            backend=backend,
            clock=lambda: 0.0,
            admission_config=admission_config,
            adaptive_clipper=OutputClipper(admission_config),
        ) as queue:
            task = asyncio.create_task(queue.submit(_request(0)))
            await _wait_until(lambda: len(backend.started) == 1)

            decision = queue.adaptive_decisions[0].decision
            assert decision.q_inst == 0
            assert decision.new_level is AdaptiveClipLevel.L0
            assert backend.started[0].effective_max_output_tokens == 1800

            backend.release("request-0")
            await task

    asyncio.run(scenario())


def test_stable_depth_excludes_waiting_that_free_slots_will_absorb() -> None:
    """Pin the section 1.1 semantics: only unabsorbable backlog counts as depth.

    Reading the raw ``_waiting`` length here would report congestion for requests
    that the very same admission pass is about to dispatch, which is the false
    positive the queue-depth signal exists to avoid. Exercised directly because
    the difference only appears when several requests are queued while more than
    one slot is still free.
    """
    backend = BlockingBackend()
    admission_config = _adaptive_config()
    queue = AdmissionQueue(
        router_config=_router(),
        backend_config=_backend_config(),
        backend=backend,
        clock=lambda: 0.0,
        admission_config=admission_config,
        adaptive_clipper=OutputClipper(admission_config),
    )

    # max_in_flight is 2, so one slot remains free behind the selected request.
    # A single queued request is absorbed by it: stable depth is zero even though
    # the raw waiting list is non-empty.
    queue._waiting = [_entry(1)]
    assert len(queue._waiting) == 1
    assert queue._stable_waiting_depth_after_selected_locked() == 0

    # One more than the free slots can take leaves exactly one request backed up.
    queue._waiting = [_entry(1), _entry(2)]
    assert queue._stable_waiting_depth_after_selected_locked() == 1
