"""Offline tests for the vLLM-compatible streaming HTTP backend."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from sloserve.config import BackendConfig, RouterConfig
from sloserve.router.admission import AdmissionQueue
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.dispatcher import DispatchStatus
from sloserve.workload.http_backend import HttpStreamingBackend

PINNED_REVISION = "a" * 40


def _backend_config() -> BackendConfig:
    return BackendConfig(
        base_url="http://vllm.invalid/v1",
        model="test-model",
        model_revision=PINNED_REVISION,
        tokenizer_revision=PINNED_REVISION,
        request_timeout_s=5.0,
    )


def _router_config() -> RouterConfig:
    return RouterConfig(
        policy="fcfs",
        listen_host="127.0.0.1",
        listen_port=8080,
        max_in_flight=1,
        queue_capacity=2,
    )


def _request(request_id: str = "request-0") -> RequestEnvelope:
    return RequestEnvelope(
        request_id=request_id,
        sequence_id=0,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=0.0,
        input_tokens=8,
        max_output_tokens=4,
        deadline_time_s=10.0,
    )


def _sse_response(request: httpx.Request, events: list[dict[str, Any]]) -> httpx.Response:
    lines = [f"data: {json.dumps(event)}\n\n" for event in events]
    lines.append("data: [DONE]\n\n")
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(lines).encode(),
        request=request,
    )


def test_streaming_success_records_usage_and_first_token() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert str(request.url) == "http://vllm.invalid/v1/chat/completions"
            body = json.loads(request.content)
            assert body == {
                "model": "test-model",
                "stream": True,
                "messages": [{"role": "user", "content": "Generate a deterministic response."}],
                "max_tokens": 4,
                "temperature": 0,
                "stream_options": {"include_usage": True},
            }
            return _sse_response(
                request,
                [
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": "hello"}}]},
                    {"choices": [{"delta": {"content": " world"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 3}},
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = HttpStreamingBackend(
                backend_config=_backend_config(), client=client, clock=lambda: 123.5
            )
            await backend.send(_request())

        telemetry = backend.telemetry["request-0"]
        assert telemetry.first_token_time_s == 123.5
        assert telemetry.output_tokens == 3
        assert telemetry.http_status == 200
        assert telemetry.finish_reason == "stop"
        assert telemetry.error is None

    asyncio.run(scenario())


def test_streaming_without_usage_falls_back_to_content_delta_count() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _sse_response(
                request,
                [
                    {"choices": [{"delta": {"content": "one"}}]},
                    {"choices": [{"delta": {"content": ""}}]},
                    {"choices": [{"delta": {"content": "two"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                ],
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = HttpStreamingBackend(backend_config=_backend_config(), client=client)
            await backend.send(_request())

        telemetry = backend.telemetry["request-0"]
        assert telemetry.output_tokens == 2
        assert telemetry.finish_reason == "length"

    asyncio.run(scenario())


def test_backend_sends_clipped_cap_without_overwriting_original_target() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert json.loads(request.content)["max_tokens"] == 2
            return _sse_response(
                request,
                [
                    {"choices": [{"delta": {"content": "token"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "length"}]},
                ],
            )

        request = replace(_request(), backend_max_output_tokens=2)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = HttpStreamingBackend(backend_config=_backend_config(), client=client)
            await backend.send(request)

        assert request.max_output_tokens == 4
        assert backend.telemetry[request.request_id].finish_reason == "length"

    asyncio.run(scenario())


def test_http_error_raises_and_records_status_and_error() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="backend failed", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = HttpStreamingBackend(backend_config=_backend_config(), client=client)
            with pytest.raises(httpx.HTTPStatusError):
                await backend.send(_request())

        telemetry = backend.telemetry["request-0"]
        assert telemetry.http_status == 500
        assert telemetry.error is not None
        assert "500 Internal Server Error" in telemetry.error

    asyncio.run(scenario())


def test_admission_queue_joins_success_with_http_telemetry() -> None:
    async def scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return _sse_response(
                request,
                [
                    {"choices": [{"delta": {"role": "assistant"}}]},
                    {"choices": [{"delta": {"content": "token"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [], "usage": {"completion_tokens": 1}},
                ],
            )

        def shared_clock() -> float:
            return 50.0

        backend_config = _backend_config()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = HttpStreamingBackend(
                backend_config=backend_config,
                client=client,
                clock=shared_clock,
            )
            async with AdmissionQueue(
                router_config=_router_config(),
                backend_config=backend_config,
                backend=backend,
                clock=shared_clock,
            ) as queue:
                result = await queue.submit(_request("joined-request"))

        assert result.status is DispatchStatus.SUCCESS
        assert result.request_id == "joined-request"
        assert backend.telemetry[result.request_id].first_token_time_s == 50.0

    asyncio.run(scenario())
