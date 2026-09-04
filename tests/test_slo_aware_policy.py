"""Correctness tests for SLO-aware scheduling and hard aging."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from sloserve.config import LengthSource, SloAwareConfig
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.router.policies.slo_aware import SloAwarePolicy
from sloserve.workload.length_predictor import LengthPredictor, LengthTrainingPair


def _request(
    request_id: str,
    sequence_id: int,
    *,
    arrival_time_s: float = 0.0,
    input_tokens: int = 4,
    max_output_tokens: int = 3,
    deadline_time_s: float = 20.0,
    advertised_cap_tokens: int | None = None,
    prompt_kind: str | None = None,
) -> RequestEnvelope:
    return RequestEnvelope(
        request_id=request_id,
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=arrival_time_s,
        input_tokens=input_tokens,
        max_output_tokens=max_output_tokens,
        deadline_time_s=deadline_time_s,
        advertised_cap_tokens=advertised_cap_tokens,
        prompt_kind=prompt_kind,
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
    aging_levels: int = 1,
    adaptive_ceiling: bool = False,
    ceiling_margin: float = 1.5,
    ceiling_floor_s: float = 1.0,
    ceiling_cap_s: float = 15.0,
    length_source: LengthSource = LengthSource.TRUE,
    length_estimator_path: str | None = None,
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
            aging_levels=aging_levels,
            adaptive_ceiling=adaptive_ceiling,
            ceiling_margin=ceiling_margin,
            ceiling_floor_s=ceiling_floor_s,
            ceiling_cap_s=ceiling_cap_s,
            length_source=length_source,
            length_estimator_path=length_estimator_path,
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


def test_true_length_source_reproduces_known_priority_key() -> None:
    policy = _policy(
        input_token_seconds=0.5,
        output_token_seconds=2.0,
        cost_weight=0.5,
        slack_weight=3.0,
        waiting_weight=4.0,
        length_source=LengthSource.TRUE,
    )
    request = _request(
        "true-source",
        7,
        arrival_time_s=10.0,
        input_tokens=4,
        max_output_tokens=3,
        deadline_time_s=30.0,
        advertised_cap_tokens=100,
    )

    assert policy.priority_key(request, now_s=14.0) == pytest.approx((1.0, 0.6, 7.0))


def test_advertised_length_source_does_not_read_true_target() -> None:
    policy = _policy(length_source=LengthSource.ADVERTISED)
    shorter = _request("shorter", 0, max_output_tokens=10, advertised_cap_tokens=2048)
    longer = _request("longer", 1, max_output_tokens=1000, advertised_cap_tokens=2048)

    assert policy._estimated_output_tokens(shorter) == 2048.0
    assert policy._estimated_output_tokens(longer) == 2048.0
    assert _policy()._estimated_output_tokens(shorter) != _policy()._estimated_output_tokens(longer)


def test_applied_backend_cap_bounds_service_estimate_without_leaking_unclipped_target() -> None:
    policy = _policy(length_source=LengthSource.ADVERTISED)
    clipped = _request(
        "clipped",
        0,
        max_output_tokens=1800,
        advertised_cap_tokens=2048,
    )
    clipped = replace(clipped, backend_max_output_tokens=1024)
    unclipped = _request(
        "unclipped",
        1,
        max_output_tokens=100,
        advertised_cap_tokens=2048,
    )

    assert policy._estimated_output_tokens(clipped) == 1024.0
    assert policy._estimated_output_tokens(unclipped) == 2048.0


def test_learned_length_source_uses_loaded_predictor_without_true_target(tmp_path: Path) -> None:
    predictor = LengthPredictor.train(
        (
            LengthTrainingPair(RequestClass.INTERACTIVE, 64, "short", 2048, 100),
            LengthTrainingPair(RequestClass.INTERACTIVE, 128, "short", 2048, 300),
            LengthTrainingPair(RequestClass.BATCH, 512, "long", 2048, 900),
        )
    )
    artifact = tmp_path / "predictor.json"
    predictor.save(artifact)
    policy = _policy(
        length_source=LengthSource.LEARNED,
        length_estimator_path=str(artifact),
    )
    shorter = _request(
        "shorter",
        0,
        input_tokens=999,
        max_output_tokens=10,
        advertised_cap_tokens=2048,
        prompt_kind="short",
    )
    longer = _request(
        "longer",
        1,
        input_tokens=999,
        max_output_tokens=1000,
        advertised_cap_tokens=2048,
        prompt_kind="short",
    )

    assert policy._estimated_output_tokens(shorter) == 200.0
    assert policy._estimated_output_tokens(longer) == 200.0


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


def test_one_aging_level_exactly_matches_binary_keys() -> None:
    policy = _policy(aging_threshold_s=10.0, aging_levels=1)
    fresh = _request("fresh", 3, arrival_time_s=1.0, deadline_time_s=21.0)
    aged = _request("aged", 4, arrival_time_s=0.0, deadline_time_s=20.0)

    fresh_key = policy.priority_key(fresh, now_s=5.0)
    aged_key = policy.priority_key(aged, now_s=10.0)

    # service=10, budget=20, slack=6, waiting=4, so score=10/20+6/20-4/20=0.6.
    assert fresh_key[0] == 1.0
    assert fresh_key[1] == pytest.approx(0.6)
    assert fresh_key[2] == 3.0
    assert aged_key == (0.0, 0.0, 4.0)


def test_three_aging_levels_assign_intermediate_tier_with_score_secondary() -> None:
    policy = _policy(aging_threshold_s=12.0, aging_levels=3)
    request = _request("intermediate", 5, arrival_time_s=0.0, deadline_time_s=20.0)

    key = policy.priority_key(request, now_s=5.0)

    # step=4 and waiting=5 select age tier 1, whose primary key is K-1=2.
    assert key[0] == 2.0
    assert key[1] == pytest.approx(0.5)
    assert key[1] != request.arrival_time_s
    assert key[2] == 5.0


def test_requests_in_same_non_ceiling_tier_order_by_score_not_arrival() -> None:
    policy = _policy(
        cost_weight=1.0,
        slack_weight=0.0,
        waiting_weight=0.0,
        aging_threshold_s=12.0,
        aging_levels=3,
    )
    older_longer = _request(
        "older-longer",
        0,
        arrival_time_s=0.0,
        input_tokens=10,
        deadline_time_s=20.0,
    )
    newer_shorter = _request(
        "newer-shorter",
        1,
        arrival_time_s=1.0,
        input_tokens=1,
        deadline_time_s=21.0,
    )

    ordered = policy.order([older_longer, newer_shorter], now_s=6.0)

    assert [request.request_id for request in ordered] == ["newer-shorter", "older-longer"]


def test_multilevel_ceiling_orders_oldest_first_regardless_of_score() -> None:
    policy = _policy(
        cost_weight=1.0,
        slack_weight=0.0,
        waiting_weight=0.0,
        aging_threshold_s=12.0,
        aging_levels=3,
    )
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

    oldest_key = policy.priority_key(oldest_expensive, now_s=13.0)
    newer_key = policy.priority_key(newer_cheap, now_s=13.0)
    ordered = policy.order([newer_cheap, oldest_expensive], now_s=13.0)

    assert oldest_key[0] == 0.0
    assert newer_key[0] == 0.0
    assert [request.request_id for request in ordered] == [
        "oldest-expensive",
        "newer-cheap",
    ]


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


def test_adaptive_ceiling_disabled_matches_fixed_priority_key_order() -> None:
    policy = _policy(adaptive_ceiling=False)
    oldest_expensive = _request(
        "oldest-expensive", 1, arrival_time_s=0.0, input_tokens=100, deadline_time_s=30.0
    )
    newer_cheap = _request(
        "newer-cheap", 0, arrival_time_s=1.0, input_tokens=1, deadline_time_s=31.0
    )
    queue = [newer_cheap, oldest_expensive]

    ordered = policy.order(queue, now_s=12.0)
    fixed = sorted(queue, key=lambda request: policy.priority_key(request, now_s=12.0))

    assert [r.request_id for r in ordered] == [r.request_id for r in fixed]


def test_effective_threshold_clamps_to_floor_when_queue_is_fresh() -> None:
    policy = _policy(adaptive_ceiling=True, ceiling_margin=1.5, ceiling_floor_s=1.0)
    assert policy._effective_threshold([0.0, 0.0, 0.0]) == pytest.approx(1.0)


def test_effective_threshold_clamps_to_cap_under_heavy_congestion() -> None:
    policy = _policy(adaptive_ceiling=True, ceiling_margin=1.5, ceiling_cap_s=15.0)
    assert policy._effective_threshold([100.0, 100.0]) == pytest.approx(15.0)


def test_effective_threshold_scales_with_median_wait_inside_the_clamp() -> None:
    policy = _policy(
        adaptive_ceiling=True, ceiling_margin=1.5, ceiling_floor_s=1.0, ceiling_cap_s=30.0
    )
    # median([2, 4, 12]) = 4, 1.5 * 4 = 6, inside [1, 30].
    assert policy._effective_threshold([2.0, 4.0, 12.0]) == pytest.approx(6.0)


def test_adaptive_ceiling_preserves_slo_order_that_a_low_fixed_threshold_destroys() -> None:
    # cost/waiting off: score = slack / budget, so a near-deadline request is most urgent.
    kwargs = dict(cost_weight=0.0, slack_weight=1.0, waiting_weight=0.0, aging_levels=1)
    old_a = _request(
        "old-a", 0, arrival_time_s=0.0, input_tokens=10, max_output_tokens=3, deadline_time_s=60.0
    )
    old_b = _request(
        "old-b", 1, arrival_time_s=0.0, input_tokens=10, max_output_tokens=3, deadline_time_s=60.0
    )
    fresh = _request(
        "fresh-urgent",
        2,
        arrival_time_s=5.0,
        input_tokens=1,
        max_output_tokens=1,
        deadline_time_s=8.0,
    )
    queue = [old_a, old_b, fresh]

    fixed = _policy(aging_threshold_s=3.0, **kwargs)
    adaptive = _policy(
        aging_threshold_s=3.0,
        adaptive_ceiling=True,
        ceiling_margin=1.5,
        ceiling_floor_s=1.0,
        ceiling_cap_s=30.0,
        **kwargs,
    )

    fixed_order = [r.request_id for r in fixed.order(queue, now_s=5.0)]
    adaptive_order = [r.request_id for r in adaptive.order(queue, now_s=5.0)]

    # Fixed low threshold ages both old requests into the class-blind ceiling ahead of the
    # fresh, most-urgent interactive request; the adaptive ceiling floats up (median wait 5,
    # threshold 7.5) so nothing is aged and the urgent request is ranked first by its score.
    assert fixed_order[0] in {"old-a", "old-b"}
    assert adaptive_order[0] == "fresh-urgent"


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
