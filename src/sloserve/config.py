"""Typed configuration loading for SLOServe."""

from __future__ import annotations

import math
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


class StrictModel(BaseModel):
    """Base model that rejects misspelled or unsupported fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SchedulerPolicyName(StrEnum):
    """Scheduling policies planned by the MVP."""

    FCFS = "fcfs"
    STATIC_PRIORITY = "static_priority"
    SLO_AWARE = "slo_aware"


class TorchDtype(StrEnum):
    """Compute dtypes accepted by the vLLM server launch."""

    AUTO = "auto"
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


class ArrivalProcess(StrEnum):
    """Supported workload arrival processes."""

    FIXED = "fixed"
    POISSON = "poisson"
    BURST = "burst"


class LengthModel(StrEnum):
    """Output-length workload models."""

    UNIFORM_CAP = "uniform_cap"
    REALISTIC = "realistic"


class LengthSource(StrEnum):
    """Output-token estimates available to the SLO-aware policy."""

    TRUE = "true"
    ADVERTISED = "advertised"
    LEARNED = "learned"


class ClipSource(StrEnum):
    """Scheduler-visible estimates allowed to trigger output clipping."""

    ADVERTISED = "advertised"
    LEARNED = "learned"


class AdaptiveClipSignal(StrEnum):
    """Signals accepted by the adaptive output-cap configuration."""

    QUEUE_DEPTH_EWMA = "queue_depth_ewma"
    IN_SYSTEM = "in_system"
    RHO_HAT = "rho_hat"


class BackendConfig(StrictModel):
    """Connection settings for the vLLM OpenAI-compatible backend."""

    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    request_timeout_s: float = Field(gt=0)


class ServerConfig(StrictModel):
    """Conservative single-GPU launch parameters for the vLLM server.

    These are the versioned inputs to a real ``vllm serve`` command. They stay in
    configuration so the launch is reproducible and never depends on hand-typed flags.
    """

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    gpu_memory_utilization: float = Field(gt=0, le=1)
    max_model_len: int = Field(ge=1)
    max_num_seqs: int = Field(ge=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    enable_chunked_prefill: bool | None = None
    enable_prefix_caching: bool | None = None
    dtype: TorchDtype
    enforce_eager: bool


class RouterConfig(StrictModel):
    """External router and admission queue settings."""

    policy: SchedulerPolicyName
    listen_host: str = Field(min_length=1)
    listen_port: int = Field(ge=1, le=65535)
    max_in_flight: int = Field(ge=1)
    queue_capacity: int = Field(ge=1)


class AdmissionControlConfig(StrictModel):
    """Optional output-length clipping applied before dispatch to the backend."""

    clip_enabled: bool = False
    clip_max_tokens: int = Field(default=2048, ge=1)
    clip_source: ClipSource = ClipSource.ADVERTISED
    clip_estimator_path: str | None = None
    adaptive_clip_enabled: bool = False
    adaptive_clip_signal: AdaptiveClipSignal = AdaptiveClipSignal.QUEUE_DEPTH_EWMA
    adaptive_clip_caps: tuple[int, ...] = (1536, 1024, 512)
    adaptive_clip_tighten_thresholds: tuple[float, ...] = (1.0, 2.0, 3.0)
    adaptive_clip_relax_thresholds: tuple[float, ...] = (0.25, 0.75, 1.5)
    adaptive_clip_ewma_tau_s: float = Field(default=6.0, gt=0, allow_inf_nan=False)
    adaptive_clip_tighten_hold_s: float = Field(default=8.0, ge=0, allow_inf_nan=False)
    adaptive_clip_relax_hold_s: float = Field(default=30.0, ge=0, allow_inf_nan=False)
    adaptive_clip_capacity_rps: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def learned_clip_source_has_estimator(self) -> AdmissionControlConfig:
        """Require the learned artifact only when learned clipping is active."""
        if (
            self.clip_enabled
            and self.clip_source is ClipSource.LEARNED
            and self.clip_estimator_path is None
        ):
            raise ValueError("learned clip source requires clip_estimator_path")
        return self

    @model_validator(mode="after")
    def validate_adaptive_clip_contract(self) -> AdmissionControlConfig:
        """Validate the finite four-level cap-controller configuration."""
        caps = self.adaptive_clip_caps
        tighten = self.adaptive_clip_tighten_thresholds
        relax = self.adaptive_clip_relax_thresholds
        if len(caps) != 3:
            raise ValueError("adaptive_clip_caps must contain exactly 3 values for L1-L3")
        if any(cap <= 0 for cap in caps):
            raise ValueError("adaptive_clip_caps must contain only positive values")
        if any(left <= right for left, right in pairwise(caps)):
            raise ValueError("adaptive_clip_caps must be strictly decreasing")
        if min(caps) < 512:
            raise ValueError("minimum adaptive clip cap must be at least 512 tokens")

        if len(tighten) != 3:
            raise ValueError(
                "adaptive_clip_tighten_thresholds must contain exactly 3 values for L1-L3"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in tighten):
            raise ValueError(
                "adaptive_clip_tighten_thresholds must contain finite non-negative values"
            )
        if any(left >= right for left, right in pairwise(tighten)):
            raise ValueError("adaptive_clip_tighten_thresholds must be strictly increasing")

        if len(relax) != 3:
            raise ValueError(
                "adaptive_clip_relax_thresholds must contain exactly 3 values for L1-L3"
            )
        if any(not math.isfinite(value) or value < 0.0 for value in relax):
            raise ValueError(
                "adaptive_clip_relax_thresholds must contain finite non-negative values"
            )
        if any(left >= right for left, right in pairwise(relax)):
            raise ValueError("adaptive_clip_relax_thresholds must be strictly increasing")
        if any(
            relax_value >= tighten_value
            for relax_value, tighten_value in zip(relax, tighten, strict=True)
        ):
            raise ValueError(
                "each adaptive clip relax threshold must be below its tighten threshold"
            )

        if self.adaptive_clip_enabled and not self.clip_enabled:
            raise ValueError("adaptive clipping requires clip_enabled=true")
        if (
            self.adaptive_clip_enabled
            and self.adaptive_clip_signal is AdaptiveClipSignal.RHO_HAT
            and self.adaptive_clip_capacity_rps is None
        ):
            raise ValueError("rho_hat adaptive clipping requires adaptive_clip_capacity_rps")
        return self

    @model_serializer(mode="wrap")
    def serialize_with_legacy_hash_projection(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict[str, object]:
        """Omit dormant adaptive defaults from the JSON used by legacy config hashes."""
        serialized: dict[str, object] = handler(self)
        if info.mode == "json" and not self.adaptive_clip_enabled:
            for field_name in (
                "adaptive_clip_enabled",
                "adaptive_clip_signal",
                "adaptive_clip_caps",
                "adaptive_clip_tighten_thresholds",
                "adaptive_clip_relax_thresholds",
                "adaptive_clip_ewma_tau_s",
                "adaptive_clip_tighten_hold_s",
                "adaptive_clip_relax_hold_s",
                "adaptive_clip_capacity_rps",
            ):
                serialized.pop(field_name, None)
        return serialized


class SloAwareConfig(StrictModel):
    """Service-time estimates and scoring coefficients for the SLO-aware policy."""

    input_token_seconds: float = Field(
        ge=0, description="estimated service time in seconds per input token"
    )
    output_token_seconds: float = Field(
        ge=0, description="estimated service time in seconds per output token"
    )
    cost_weight: float = Field(ge=0)
    slack_weight: float = Field(ge=0)
    waiting_weight: float = Field(ge=0)
    aging_threshold_s: float = Field(gt=0)
    disable_length_estimate: bool = False
    aging_levels: int = Field(default=1, ge=1)
    adaptive_ceiling: bool = False
    ceiling_margin: float = Field(default=1.5, gt=0)
    ceiling_floor_s: float = Field(default=1.0, gt=0)
    ceiling_cap_s: float = Field(default=15.0, gt=0)
    length_source: LengthSource = LengthSource.TRUE
    length_estimator_path: str | None = None

    @model_validator(mode="after")
    def at_least_one_scoring_weight(self) -> SloAwareConfig:
        """Reject a scoring function that cannot distinguish requests."""
        if self.cost_weight + self.slack_weight + self.waiting_weight == 0:
            raise ValueError("at least one SLO-aware scoring weight must be positive")
        return self

    @model_validator(mode="after")
    def ceiling_floor_not_above_cap(self) -> SloAwareConfig:
        """Ensure the adaptive-ceiling clamp interval is non-empty."""
        if self.ceiling_floor_s > self.ceiling_cap_s:
            raise ValueError("ceiling floor must not exceed ceiling cap")
        return self

    @model_validator(mode="after")
    def learned_length_source_has_estimator(self) -> SloAwareConfig:
        """Require a serialized predictor for learned length estimates."""
        if self.length_source is LengthSource.LEARNED and self.length_estimator_path is None:
            raise ValueError("learned length source requires length_estimator_path")
        return self


class TokenRangeConfig(StrictModel):
    """Inclusive token-count range for generated requests."""

    minimum: int = Field(ge=1)
    maximum: int = Field(ge=1)

    @model_validator(mode="after")
    def maximum_is_not_below_minimum(self) -> TokenRangeConfig:
        """Ensure the range can be sampled."""
        if self.maximum < self.minimum:
            raise ValueError("maximum token count must be greater than or equal to minimum")
        return self


class RequestProfileConfig(StrictModel):
    """Token sizes and SLO targets for one request class."""

    input_tokens: TokenRangeConfig
    output_tokens: TokenRangeConfig
    ttft_slo_ms: float = Field(gt=0)
    end_to_end_slo_ms: float = Field(gt=0)


class RealisticLengthConfig(StrictModel):
    """Heavy-tailed output-length mixture with partially predictive prompt kinds."""

    kinds: tuple[str, ...] = ("short", "medium", "long")
    kind_fractions: tuple[float, ...] = (0.5, 0.35, 0.15)
    log_mu_by_kind: tuple[float, ...] = (6.2, 7.0, 7.8)
    log_sigma: float = Field(default=0.7, gt=0)
    advertised_cap_tokens: int = Field(default=2048, ge=1)
    clamp_min: int = Field(default=8, ge=1)
    clamp_max: int = Field(default=2048, ge=1)
    force_exact_output_tokens: bool = False

    @model_validator(mode="after")
    def validate_mixture(self) -> RealisticLengthConfig:
        """Require aligned mixture parameters and a valid probability distribution."""
        if not (len(self.kind_fractions) == len(self.kinds) == len(self.log_mu_by_kind)):
            raise ValueError("kinds, kind_fractions, and log_mu_by_kind must have equal lengths")
        if any(not kind for kind in self.kinds):
            raise ValueError("realistic length kinds must not be empty")
        if any(fraction < 0.0 for fraction in self.kind_fractions):
            raise ValueError("realistic length kind fractions must be non-negative")
        if not math.isclose(sum(self.kind_fractions), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("realistic length kind fractions must sum to 1.0")
        if self.clamp_min > self.clamp_max:
            raise ValueError("realistic length clamp_min must not exceed clamp_max")
        return self


class WorkloadConfig(StrictModel):
    """Reproducible workload generation settings."""

    arrival_process: ArrivalProcess
    request_rate_rps: float = Field(gt=0)
    total_requests: int = Field(ge=1)
    interactive_fraction: float = Field(ge=0, le=1)
    random_seed: int = Field(ge=0)
    warmup_requests: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    interactive: RequestProfileConfig
    batch: RequestProfileConfig
    length_model: LengthModel = LengthModel.UNIFORM_CAP
    realistic_length: RealisticLengthConfig | None = None

    @model_validator(mode="after")
    def realistic_model_has_configuration(self) -> WorkloadConfig:
        """Require mixture parameters only when the realistic model is selected."""
        if self.length_model is LengthModel.REALISTIC and self.realistic_length is None:
            raise ValueError("realistic length model requires realistic_length")
        return self


class MetricsConfig(StrictModel):
    """Raw-result and GPU sampling settings."""

    output_directory: Path
    gpu_sample_interval_s: float = Field(gt=0)
    save_jsonl: bool
    save_csv: bool

    @model_validator(mode="after")
    def at_least_one_raw_format(self) -> MetricsConfig:
        """Require a raw result format so no run is silently discarded."""
        if not self.save_jsonl and not self.save_csv:
            raise ValueError("at least one raw result format must be enabled")
        return self


class ExperimentConfig(StrictModel):
    """Top-level validated experiment configuration."""

    backend: BackendConfig
    server: ServerConfig
    router: RouterConfig
    admission: AdmissionControlConfig = AdmissionControlConfig()
    slo_aware: SloAwareConfig
    workload: WorkloadConfig
    metrics: MetricsConfig

    @model_validator(mode="after")
    def server_matches_backend_url(self) -> ExperimentConfig:
        """Keep the launched server address consistent with the client base URL."""
        parts = urlsplit(self.backend.base_url)
        if parts.hostname != self.server.host or parts.port != self.server.port:
            raise ValueError(
                "server host/port must match backend.base_url host/port "
                f"({self.server.host}:{self.server.port} vs "
                f"{parts.hostname}:{parts.port})"
            )
        return self

    @model_validator(mode="after")
    def server_admits_router_concurrency(self) -> ExperimentConfig:
        """Ensure the external router, not the engine, is the admission bottleneck."""
        if self.server.max_num_seqs < self.router.max_in_flight:
            raise ValueError(
                "server.max_num_seqs must be >= router.max_in_flight so the external "
                f"router bounds concurrency ({self.server.max_num_seqs} < "
                f"{self.router.max_in_flight})"
            )
        return self

    def vllm_serve_args(self) -> list[str]:
        """Build the deterministic ``vllm`` argument vector for the pinned server.

        The returned list starts with the ``serve`` subcommand, so a launcher runs
        ``vllm <args>``. Model and tokenizer are pinned to immutable commits.
        """
        args = [
            "serve",
            self.backend.model,
            "--revision",
            self.backend.model_revision,
            "--tokenizer-revision",
            self.backend.tokenizer_revision,
            "--served-model-name",
            self.backend.model,
            "--host",
            self.server.host,
            "--port",
            str(self.server.port),
            "--gpu-memory-utilization",
            str(self.server.gpu_memory_utilization),
            "--max-model-len",
            str(self.server.max_model_len),
            "--max-num-seqs",
            str(self.server.max_num_seqs),
            "--dtype",
            self.server.dtype.value,
        ]
        if self.server.enforce_eager:
            args.append("--enforce-eager")
        if self.server.max_num_batched_tokens is not None:
            args.extend(["--max-num-batched-tokens", str(self.server.max_num_batched_tokens)])
        if self.server.enable_chunked_prefill is not None:
            args.append(
                "--enable-chunked-prefill"
                if self.server.enable_chunked_prefill
                else "--no-enable-chunked-prefill"
            )
        if self.server.enable_prefix_caching is not None:
            args.append(
                "--enable-prefix-caching"
                if self.server.enable_prefix_caching
                else "--no-enable-prefix-caching"
            )
        return args


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML experiment configuration file."""
    config_path = Path(path)
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError("top-level configuration must be a mapping")
    return ExperimentConfig.model_validate(raw)
