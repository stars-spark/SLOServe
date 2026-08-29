"""Typed request-level facts used for offline metrics."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from sloserve.config import SchedulerPolicyName
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.dispatcher import DispatchStatus

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RequestRecord:
    """One terminal request record suitable for JSONL fact storage."""

    request_id: str
    sequence_id: int
    request_class: RequestClass
    arrival_time_s: float
    enqueue_time_s: float
    dispatch_time_s: float | None
    first_token_time_s: float | None
    completion_time_s: float
    input_tokens: int
    output_tokens: int
    status: DispatchStatus
    error_type: str | None
    policy_name: SchedulerPolicyName
    config_hash: str
    repetition_index: int
    env_version: str

    def __post_init__(self) -> None:
        """Reject records that cannot represent a coherent event timeline."""
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.input_tokens < 1:
            raise ValueError("input_tokens must be positive")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.repetition_index < 0:
            raise ValueError("repetition_index must be non-negative")
        if not self.env_version:
            raise ValueError("env_version must not be empty")
        if not _SHA256_PATTERN.fullmatch(self.config_hash):
            raise ValueError("config_hash must be a lowercase SHA-256 hex digest")
        if self.error_type == "":
            raise ValueError("error_type must be null or a non-empty string")

        timestamps = (self.arrival_time_s, self.enqueue_time_s, self.completion_time_s)
        if not all(math.isfinite(timestamp) for timestamp in timestamps):
            raise ValueError("timestamps must be finite")
        if any(timestamp < 0 for timestamp in timestamps):
            raise ValueError("timestamps must be non-negative")
        if self.dispatch_time_s is not None and not math.isfinite(self.dispatch_time_s):
            raise ValueError("dispatch_time_s must be finite when present")
        if self.first_token_time_s is not None and not math.isfinite(self.first_token_time_s):
            raise ValueError("first_token_time_s must be finite when present")
        if self.enqueue_time_s < self.arrival_time_s:
            raise ValueError("enqueue_time_s must not precede arrival_time_s")
        if self.dispatch_time_s is None:
            if self.first_token_time_s is not None:
                raise ValueError("records without dispatch cannot have first_token_time_s")
            if self.status is DispatchStatus.SUCCESS:
                raise ValueError("success records require dispatch_time_s")
            if self.completion_time_s < self.enqueue_time_s:
                raise ValueError("completion_time_s must not precede enqueue_time_s")
        else:
            if self.dispatch_time_s < self.enqueue_time_s:
                raise ValueError("dispatch_time_s must not precede enqueue_time_s")
            if self.completion_time_s < self.dispatch_time_s:
                raise ValueError("completion_time_s must not precede dispatch_time_s")
            if self.first_token_time_s is not None:
                if self.first_token_time_s < self.dispatch_time_s:
                    raise ValueError("first_token_time_s must not precede dispatch_time_s")
                if self.completion_time_s < self.first_token_time_s:
                    raise ValueError("completion_time_s must not precede first_token_time_s")
                if self.output_tokens < 1:
                    raise ValueError("a first token requires output_tokens to be positive")
        if self.status is DispatchStatus.SUCCESS:
            if self.dispatch_time_s is None:
                raise ValueError("success records require dispatch_time_s")
            if self.first_token_time_s is None:
                raise ValueError("success records require first_token_time_s")
            if self.output_tokens < 1:
                raise ValueError("success records require output_tokens to be positive")

    @classmethod
    def from_envelope(
        cls,
        envelope: RequestEnvelope,
        *,
        enqueue_time_s: float,
        dispatch_time_s: float | None,
        first_token_time_s: float | None,
        completion_time_s: float,
        output_tokens: int,
        status: DispatchStatus,
        error_type: str | None,
        policy_name: SchedulerPolicyName,
        config_hash: str,
        repetition_index: int,
        env_version: str,
    ) -> RequestRecord:
        """Build a record from the shared request envelope and terminal facts."""
        return cls(
            request_id=envelope.request_id,
            sequence_id=envelope.sequence_id,
            request_class=envelope.request_class,
            arrival_time_s=envelope.arrival_time_s,
            enqueue_time_s=enqueue_time_s,
            dispatch_time_s=dispatch_time_s,
            first_token_time_s=first_token_time_s,
            completion_time_s=completion_time_s,
            input_tokens=envelope.input_tokens,
            output_tokens=output_tokens,
            status=status,
            error_type=error_type,
            policy_name=policy_name,
            config_hash=config_hash,
            repetition_index=repetition_index,
            env_version=env_version,
        )
