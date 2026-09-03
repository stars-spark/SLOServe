"""Validation tests for shared router request models."""

import pytest

from sloserve.router.models import RequestClass, RequestEnvelope


def test_request_envelope_rejects_non_positive_advertised_cap() -> None:
    with pytest.raises(ValueError, match="advertised_cap_tokens must be positive"):
        RequestEnvelope(
            request_id="invalid-cap",
            sequence_id=0,
            request_class=RequestClass.INTERACTIVE,
            arrival_time_s=0.0,
            input_tokens=64,
            max_output_tokens=32,
            deadline_time_s=10.0,
            advertised_cap_tokens=0,
        )


def test_request_envelope_validates_and_exposes_backend_cap() -> None:
    request = RequestEnvelope(
        request_id="clipped",
        sequence_id=0,
        request_class=RequestClass.BATCH,
        arrival_time_s=0.0,
        input_tokens=64,
        max_output_tokens=1500,
        deadline_time_s=10.0,
        backend_max_output_tokens=1024,
    )

    assert request.effective_max_output_tokens == 1024
    assert request.clip_applied is True

    with pytest.raises(ValueError, match="must not exceed"):
        RequestEnvelope(
            request_id="invalid-backend-cap",
            sequence_id=0,
            request_class=RequestClass.BATCH,
            arrival_time_s=0.0,
            input_tokens=64,
            max_output_tokens=100,
            deadline_time_s=10.0,
            backend_max_output_tokens=101,
        )
