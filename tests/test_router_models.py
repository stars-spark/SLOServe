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
