"""Length-aware max-token clipping at the external admission boundary."""

from __future__ import annotations

from dataclasses import replace

from sloserve.config import AdmissionControlConfig, ClipSource
from sloserve.router.models import RequestEnvelope
from sloserve.workload.length_predictor import LengthPredictor


class OutputClipper:
    """Apply a backend cap only when visible length information crosses the threshold."""

    def __init__(self, config: AdmissionControlConfig) -> None:
        self._config = config
        self._predictor: LengthPredictor | None = None
        if config.clip_enabled and config.clip_source is ClipSource.LEARNED:
            if config.clip_estimator_path is None:
                raise ValueError("learned clip source requires clip_estimator_path")
            self._predictor = LengthPredictor.load(config.clip_estimator_path)

    def _estimated_output_tokens(self, request: RequestEnvelope) -> float:
        if self._config.clip_source is ClipSource.ADVERTISED:
            return float(request.advertised_cap_tokens or request.max_output_tokens)
        if self._config.clip_source is ClipSource.LEARNED:
            if self._predictor is None:
                raise RuntimeError("learned clip predictor was not loaded")
            return self._predictor.predict(
                request.request_class,
                request.input_tokens,
                request.prompt_kind,
                request.advertised_cap_tokens or request.max_output_tokens,
            )
        raise ValueError(f"unsupported clip source: {self._config.clip_source}")

    def apply(self, request: RequestEnvelope) -> RequestEnvelope:
        """Return a request carrying the backend cap, preserving the original target."""
        if not self._config.clip_enabled:
            return request
        clip_max_tokens = self._config.clip_max_tokens
        if (
            self._estimated_output_tokens(request) <= clip_max_tokens
            or request.max_output_tokens <= clip_max_tokens
        ):
            return request
        return replace(request, backend_max_output_tokens=clip_max_tokens)

    def apply_adaptive_cap(
        self,
        request: RequestEnvelope,
        cap_tokens: int | None,
    ) -> RequestEnvelope:
        """Apply one dispatch-time cap without changing the fixed pre-queue path."""
        if not self._config.adaptive_clip_enabled:
            raise ValueError("adaptive cap application requires adaptive_clip_enabled=true")
        if cap_tokens is None:
            return request
        if cap_tokens < 1:
            raise ValueError("adaptive cap must be positive")
        if (
            self._estimated_output_tokens(request) <= cap_tokens
            or request.max_output_tokens <= cap_tokens
        ):
            return request
        return replace(request, backend_max_output_tokens=cap_tokens)
