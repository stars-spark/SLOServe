"""Correctness tests for the FCFS baseline policy."""

from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.router.policies import FcfsPolicy


def make_request(request_id: str, sequence_id: int, arrival_time_s: float) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=request_id,
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=arrival_time_s,
        input_tokens=64,
        max_output_tokens=32,
        deadline_time_s=arrival_time_s + 10,
    )


def test_fcfs_orders_by_arrival_then_sequence() -> None:
    policy = FcfsPolicy()
    requests = [
        make_request("late", 0, 2.0),
        make_request("tie-second", 2, 1.0),
        make_request("tie-first", 1, 1.0),
    ]

    ordered = policy.order(requests, now_s=3.0)

    assert [request.request_id for request in ordered] == ["tie-first", "tie-second", "late"]


def test_fcfs_does_not_mutate_input() -> None:
    policy = FcfsPolicy()
    requests = [make_request("second", 1, 2.0), make_request("first", 0, 1.0)]
    original = list(requests)

    policy.order(requests, now_s=3.0)

    assert requests == original


def test_request_rejects_deadline_before_arrival() -> None:
    try:
        RequestEnvelope(
            request_id="invalid",
            sequence_id=0,
            request_class=RequestClass.BATCH,
            arrival_time_s=2.0,
            input_tokens=512,
            max_output_tokens=128,
            deadline_time_s=1.0,
        )
    except ValueError as exc:
        assert "deadline_time_s" in str(exc)
    else:
        raise AssertionError("invalid request metadata was accepted")
