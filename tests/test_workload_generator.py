"""Tests for deterministic request generation."""

import statistics
from collections import Counter
from pathlib import Path

import pytest

from sloserve.config import LengthModel, RealisticLengthConfig, WorkloadConfig, load_config
from sloserve.router.models import RequestClass
from sloserve.workload.generator import generate_requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_same_config_and_seed_generate_identical_request_sequence() -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload

    first = generate_requests(workload)
    second = generate_requests(workload)

    assert first == second


def test_warmups_are_extra_prefixed_requests_with_contiguous_ids() -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload.model_copy(
        update={"warmup_requests": 3, "total_requests": 5, "request_rate_rps": 2.0}
    )

    requests = generate_requests(workload)

    assert len(requests) == 8
    assert [request.sequence_id for request in requests] == list(range(8))
    assert [request.request_id for request in requests] == [
        f"request-{sequence_id:06d}" for sequence_id in range(8)
    ]
    assert requests[3].sequence_id == workload.warmup_requests
    assert [request.arrival_time_s for request in requests] == [
        sequence_id / workload.request_rate_rps for sequence_id in range(8)
    ]


def test_generated_sizes_and_deadlines_use_the_selected_profile() -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload

    for request in generate_requests(workload):
        profile = (
            workload.interactive
            if request.request_class is RequestClass.INTERACTIVE
            else workload.batch
        )
        assert profile.input_tokens.minimum <= request.input_tokens <= profile.input_tokens.maximum
        assert (
            profile.output_tokens.minimum
            <= request.max_output_tokens
            <= profile.output_tokens.maximum
        )
        assert request.deadline_time_s == (
            request.arrival_time_s + profile.end_to_end_slo_ms / 1000.0
        )


def test_interactive_fraction_controls_request_class() -> None:
    workload = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload.model_copy(
        update={"warmup_requests": 0, "total_requests": 4, "interactive_fraction": 1.0}
    )

    assert all(
        request.request_class is RequestClass.INTERACTIVE for request in generate_requests(workload)
    )


def test_uniform_cap_preserves_legacy_rng_draw_order() -> None:
    workload = WorkloadConfig.model_validate(
        {
            **load_config(PROJECT_ROOT / "configs" / "base.yaml").workload.model_dump(),
            "random_seed": 12345,
            "warmup_requests": 0,
            "total_requests": 8,
            "length_model": "uniform_cap",
        }
    )
    requests = generate_requests(workload)

    observed = [
        (
            request.request_class.value,
            request.input_tokens,
            request.max_output_tokens,
        )
        for request in requests
    ]
    assert observed == [
        ("interactive", 66, 70),
        ("batch", 908, 266),
        ("interactive", 105, 79),
        ("interactive", 174, 65),
        ("interactive", 108, 110),
        ("interactive", 154, 126),
        ("batch", 698, 399),
        ("batch", 1356, 427),
    ]
    assert all(request.advertised_cap_tokens is None for request in requests)
    assert all(request.prompt_kind is None for request in requests)
    assert not any(request.force_exact_output_tokens for request in requests)


def test_realistic_lengths_are_heavy_tailed_and_prompt_kind_is_partially_predictive() -> None:
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload
    workload = WorkloadConfig.model_validate(
        {
            **base.model_dump(),
            "length_model": LengthModel.REALISTIC,
            "realistic_length": RealisticLengthConfig().model_dump(),
            "warmup_requests": 0,
            "total_requests": 10_000,
        }
    )

    requests = generate_requests(workload)
    realistic = workload.realistic_length
    assert realistic is not None
    lengths = [request.max_output_tokens for request in requests]
    kind_counts = Counter(request.prompt_kind for request in requests)
    by_kind = {
        kind: [request.max_output_tokens for request in requests if request.prompt_kind == kind]
        for kind in realistic.kinds
    }

    assert max(lengths) > 2 * statistics.median(lengths)
    assert all(request.prompt_kind is not None for request in requests)
    assert all(request.advertised_cap_tokens == 2048 for request in requests)
    assert not any(request.force_exact_output_tokens for request in requests)
    assert all(len(set(kind_lengths)) > 1 for kind_lengths in by_kind.values())
    expected_fractions = dict(
        zip(
            realistic.kinds,
            realistic.kind_fractions,
            strict=True,
        )
    )
    for kind, expected in expected_fractions.items():
        assert kind_counts[kind] / len(requests) == pytest.approx(expected, abs=0.03)


def test_realistic_workload_can_force_sampled_output_targets() -> None:
    base = load_config(PROJECT_ROOT / "configs" / "base.yaml").workload
    realistic = RealisticLengthConfig(force_exact_output_tokens=True)
    workload = WorkloadConfig.model_validate(
        {
            **base.model_dump(),
            "length_model": LengthModel.REALISTIC,
            "realistic_length": realistic.model_dump(),
        }
    )

    assert all(request.force_exact_output_tokens for request in generate_requests(workload))
