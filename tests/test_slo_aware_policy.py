"""Correctness tests for SLO-aware scheduling and hard aging."""

from __future__ import annotations

import math

import pytest

from sloserve.config import SloAwareConfig
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.router.policies.slo_aware import SloAwarePolicy


def _request(
    request_id: str,
    sequence_id: int,
    *,
    arrival_time_s: float = 0.0,
    input_tokens: int = 4,
    max_output_tokens: int = 3,
    deadline_time_s: float = 20.0,
) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=request_id,
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=arrival_time_s,
        input_tokens=input_tokens,
        max_output_tokens=max_output_tokens,
        deadline_time_s=deadline_time_s,
    )


def _policy(
    *,
    input_token_seconds: float = 1.0,
    output_token_seconds: float = 2.0,
    cost_weight: float = 1.0,
    slack_weight: float = 1.0,
    waiting_weight: float = 1.0,
    aging_threshold_s: float = 10.0,
    disable_length_estimate: bool = False,
) -> SloAwarePolicy:
    return SloAwarePolicy(
        SloAwareConfig(
            input_token_seconds=input_token_seconds,
            output_token_seconds=output_token_seconds,
            cost_weight=cost_weight,
            slack_weight=slack_weight,
            waiting_weight=waiting_weight,
            aging_threshold_s=aging_threshold_s,
            disable_length_estimate=disable_length_estimate,
        )
    )


def test_slo_aware_name_matches_configuration_contract() -> None:
    assert _policy().name == "slo_aware"


def test_slo_aware_priority_key_uses_normalized_score() -> None:
    policy = _policy(
        input_token_seconds=0.5,
        output_token_seconds=2.0,
        cost_weight=0.5,
        slack_weight=3.0,
        waiting_weight=4.0,
    )
    request = _request(
        "scored",
        7,
        arrival_time_s=10.0,
        input_tokens=4,
        max_output_tokens=3,
        deadline_time_s=30.0,
    )

    key = policy.priority_key(request, now_s=14.0)

    # budget=30-10=20s; service_time=4*0.5+3*2=8s;
    # slack=30-14-8=8s; waiting=14-10=4s;
    # score=0.5*(8/20) + 3*(8/20) - 4*(4/20) = 0.6.
    assert key[0] == 1.0
    assert key[1] == pytest.approx(0.6)
    assert key[2] == float(request.sequence_id)


def test_smaller_slack_is_dispatched_first() -> None:
    policy = _policy(cost_weight=0.0, slack_weight=1.0, waiting_weight=0.0)
    nearer_deadline = _request("urgent", 1, deadline_time_s=12.0)
    farther_deadline = _request("relaxed", 0, deadline_time_s=24.0)

    ordered = policy.order([farther_deadline, nearer_deadline], now_s=2.0)

    assert [request.request_id for request in ordered] == ["urgent", "relaxed"]


def test_shorter_estimated_service_is_dispatched_first_by_sjf_term() -> None:
    policy = _policy(cost_weight=1.0, slack_weight=0.0, waiting_weight=0.0)
    shorter = _request("shorter", 1, input_tokens=2)
    longer = _request("longer", 0, input_tokens=10)

    ordered = policy.order([longer, shorter], now_s=2.0)

    assert [request.request_id for request in ordered] == ["shorter", "longer"]


def test_interactive_ranks_ahead_of_large_batch_at_equal_waiting() -> None:
    policy = _policy(input_token_seconds=0.0005, output_token_seconds=0.01)
    interactive = _request(
        "interactive",
        1,
        input_tokens=128,
        max_output_tokens=64,
        deadline_time_s=10.0,
    )
    batch = _request(
        "large-batch",
        0,
        input_tokens=1024,
        max_output_tokens=256,
        deadline_time_s=120.0,
    )

    ordered = policy.order([batch, interactive], now_s=2.0)

    # Equal waiting plus the tighter interactive budget produces the lower normalized score.
    assert [request.request_id for request in ordered] == ["interactive", "large-batch"]


def test_disabled_length_estimate_removes_cost_and_service_size_signals() -> None:
    policy = _policy(
        cost_weight=2.0,
        slack_weight=1.0,
        waiting_weight=1.0,
        disable_length_estimate=True,
    )
    length_aware_policy = _policy(cost_weight=2.0, slack_weight=1.0, waiting_weight=1.0)
    earlier_expensive = _request("earlier-expensive", 0, input_tokens=100, max_output_tokens=20)
    later_cheap = _request("later-cheap", 1, input_tokens=1, max_output_tokens=1)

    ordered = policy.order([later_cheap, earlier_expensive], now_s=2.0)
    length_aware_ordered = length_aware_policy.order([later_cheap, earlier_expensive], now_s=2.0)

    assert [request.request_id for request in ordered] == ["earlier-expensive", "later-cheap"]
    assert [request.request_id for request in length_aware_ordered] == [
        "later-cheap",
        "earlier-expensive",
    ]


def test_hard_aging_outranks_fresh_request_with_lower_score() -> None:
    policy = _policy(cost_weight=1.0, slack_weight=0.0, waiting_weight=0.0)
    aged = _request(
        "aged-expensive",
        0,
        arrival_time_s=0.0,
        input_tokens=100,
        deadline_time_s=20.0,
    )
    fresh = _request(
        "fresh-cheap",
        1,
        arrival_time_s=11.0,
        input_tokens=1,
        deadline_time_s=31.0,
    )
    now_s = 12.0

    fresh_score = (1.0 * 1 + 2.0 * 3) / (31.0 - 11.0)
    aged_score_without_hard_aging = (1.0 * 100 + 2.0 * 3) / (20.0 - 0.0)
    assert fresh_score < aged_score_without_hard_aging
    assert policy.order([fresh, aged], now_s=now_s)[0] is aged


def test_aged_bucket_orders_oldest_first_regardless_of_score() -> None:
    policy = _policy(cost_weight=1.0, slack_weight=0.0, waiting_weight=0.0)
    oldest_expensive = _request(
        "oldest-expensive",
        1,
        arrival_time_s=0.0,
        input_tokens=100,
        deadline_time_s=30.0,
    )
    newer_cheap = _request(
        "newer-cheap",
        0,
        arrival_time_s=1.0,
        input_tokens=1,
        deadline_time_s=31.0,
    )

    ordered = policy.order([newer_cheap, oldest_expensive], now_s=12.0)

    assert [request.request_id for request in ordered] == [
        "oldest-expensive",
        "newer-cheap",
    ]


def test_sequence_id_deterministically_breaks_identical_score_ties() -> None:
    policy = _policy()
    first = _request("same-request", 2)
    second = _request("same-request", 5)

    ordered = policy.order([second, first], now_s=2.0)

    assert [request.sequence_id for request in ordered] == [2, 5]


def test_zero_budget_is_guarded_and_returns_finite_key() -> None:
    policy = _policy()
    request = _request(
        "zero-budget",
        3,
        arrival_time_s=5.0,
        deadline_time_s=5.0,
    )

    key = policy.priority_key(request, now_s=5.0)

    assert all(math.isfinite(float(element)) for element in key)
