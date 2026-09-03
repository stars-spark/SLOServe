"""Core request models shared by scheduling policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RequestClass(StrEnum):
    """Workload classes used by the MVP experiments."""

    INTERACTIVE = "interactive"
    BATCH = "batch"


@dataclass(frozen=True, slots=True)
class RequestEnvelope:
    """Scheduling metadata for one request before it reaches vLLM."""

    request_id: str
    sequence_id: int
    request_class: RequestClass
    arrival_time_s: float
    input_tokens: int
    max_output_tokens: int
    deadline_time_s: float
    advertised_cap_tokens: int | None = None
    prompt_kind: str | None = None
    backend_max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid metadata at the queue boundary."""
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.arrival_time_s < 0:
            raise ValueError("arrival_time_s must be non-negative")
        if self.input_tokens < 1:
            raise ValueError("input_tokens must be positive")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.deadline_time_s < self.arrival_time_s:
            raise ValueError("deadline_time_s must not precede arrival_time_s")
        if self.advertised_cap_tokens is not None and self.advertised_cap_tokens < 1:
            raise ValueError("advertised_cap_tokens must be positive")
        if self.backend_max_output_tokens is not None:
            if self.backend_max_output_tokens < 1:
                raise ValueError("backend_max_output_tokens must be positive")
            if self.backend_max_output_tokens > self.max_output_tokens:
                raise ValueError("backend_max_output_tokens must not exceed the requested target")

    @property
    def effective_max_output_tokens(self) -> int:
        """Return the max-token value that should be sent to the backend."""
        return self.backend_max_output_tokens or self.max_output_tokens

    @property
    def clip_applied(self) -> bool:
        """Whether admission reduced the request's backend output limit."""
        return self.effective_max_output_tokens < self.max_output_tokens
