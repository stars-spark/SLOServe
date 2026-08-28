"""Tests for deterministic request generation."""

from pathlib import Path

from sloserve.config import load_config
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
