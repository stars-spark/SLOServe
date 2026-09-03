"""Streaming HTTP backend for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from sloserve.config import BackendConfig
from sloserve.router.models import RequestEnvelope
from sloserve.workload.backend import AsyncRequestBackend

_DETERMINISTIC_PROMPT = "Generate a deterministic response."


@dataclass(slots=True)
class HttpCallTelemetry:
    """Request-level observations emitted by one streaming HTTP call."""

    first_token_time_s: float | None = None
    output_tokens: int = 0
    http_status: int | None = None
    finish_reason: str | None = None
    error: str | None = None


class HttpStreamingBackend(AsyncRequestBackend):
    """Send requests to the vLLM OpenAI-compatible streaming endpoint."""

    def __init__(
        self,
        *,
        backend_config: BackendConfig,
        client: httpx.AsyncClient,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._url = f"{backend_config.base_url.rstrip('/')}/chat/completions"
        self._model = backend_config.model
        self._request_timeout_s = backend_config.request_timeout_s
        self._client = client
        self._clock = clock
        self.telemetry: dict[str, HttpCallTelemetry] = {}

    async def send(self, request: RequestEnvelope) -> None:
        """Stream one completion and retain joinable request-level telemetry."""
        call_telemetry = HttpCallTelemetry()
        self.telemetry[request.request_id] = call_telemetry
        content_delta_count = 0
        usage_output_tokens: int | None = None

        body = {
            "model": self._model,
            "stream": True,
            "messages": [{"role": "user", "content": _DETERMINISTIC_PROMPT}],
            "max_tokens": request.effective_max_output_tokens,
            "temperature": 0,
            "stream_options": {"include_usage": True},
        }

        try:
            async with self._client.stream(
                "POST",
                self._url,
                json=body,
                timeout=self._request_timeout_s,
            ) as response:
                call_telemetry.http_status = response.status_code
                if not response.is_success:
                    await response.aread()
                    response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ")
                    if data == "[DONE]":
                        break

                    chunk: Any = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise ValueError("SSE data must decode to a JSON object")

                    choices = chunk.get("choices")
                    if isinstance(choices, list) and choices:
                        choice = choices[0]
                        if isinstance(choice, dict):
                            delta = choice.get("delta")
                            if isinstance(delta, dict):
                                content = delta.get("content")
                                if isinstance(content, str) and content:
                                    content_delta_count += 1
                                    # Without a usage block, one non-empty content delta is
                                    # counted as one output token. This is only a fallback
                                    # chunk-counting convention, not tokenizer-accurate usage.
                                    call_telemetry.output_tokens = content_delta_count
                                    if call_telemetry.first_token_time_s is None:
                                        call_telemetry.first_token_time_s = self._clock()

                            finish_reason = choice.get("finish_reason")
                            if isinstance(finish_reason, str):
                                call_telemetry.finish_reason = finish_reason

                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        completion_tokens = usage.get("completion_tokens")
                        if (
                            isinstance(completion_tokens, int)
                            and not isinstance(completion_tokens, bool)
                            and completion_tokens >= 0
                        ):
                            usage_output_tokens = completion_tokens
                            call_telemetry.output_tokens = completion_tokens

                call_telemetry.output_tokens = (
                    usage_output_tokens if usage_output_tokens is not None else content_delta_count
                )
        except Exception as exc:
            call_telemetry.error = str(exc)
            raise
