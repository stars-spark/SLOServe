"""Correctness tests for static request-class priority scheduling."""

from __future__ import annotations

from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.router.policies.static_priority import StaticPriorityPolicy


def _request(
    request_id: str,
    sequence_id: int,
    request_class: RequestClass,
    arrival_time_s: float,
) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=request_id,
        sequence_id=sequence_id,
        request_class=request_class,
        arrival_time_s=arrival_time_s,
        input_tokens=64,
        max_output_tokens=32,
        deadline_time_s=arrival_time_s + 10.0,
    )


def test_static_priority_orders_classes_then_fcfs_within_each_class() -> None:
    policy = StaticPriorityPolicy()
    requests = [
        _request("batch-late", 5, RequestClass.BATCH, 3.0),
        _request("interactive-tie-second", 4, RequestClass.INTERACTIVE, 2.0),
        _request("batch-early", 0, RequestClass.BATCH, 0.0),
        _request("interactive-tie-first", 2, RequestClass.INTERACTIVE, 2.0),
        _request("interactive-early", 3, RequestClass.INTERACTIVE, 1.0),
        _request("batch-middle", 1, RequestClass.BATCH, 1.0),
    ]

    ordered = policy.order(requests, now_s=100.0)

    assert [request.request_id for request in ordered] == [
        "interactive-early",
        "interactive-tie-first",
        "interactive-tie-second",
        "batch-early",
        "batch-middle",
        "batch-late",
    ]


def test_static_priority_key_and_name_match_configuration_contract() -> None:
    policy = StaticPriorityPolicy()
    interactive = _request("interactive", 3, RequestClass.INTERACTIVE, 2.0)
    batch = _request("batch", 1, RequestClass.BATCH, 1.0)

    assert policy.name == "static_priority"
    assert policy.priority_key(interactive, now_s=10.0) == (0, 2.0, 3)
    assert policy.priority_key(batch, now_s=10.0) == (1, 1.0, 1)
