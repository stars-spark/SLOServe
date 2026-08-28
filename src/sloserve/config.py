"""Typed configuration loading for SLOServe."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    dtype: TorchDtype
    enforce_eager: bool


class RouterConfig(StrictModel):
    """External router and admission queue settings."""

    policy: SchedulerPolicyName
    listen_host: str = Field(min_length=1)
    listen_port: int = Field(ge=1, le=65535)
    max_in_flight: int = Field(ge=1)
    queue_capacity: int = Field(ge=1)


class SloAwareConfig(StrictModel):
    """Configurable coefficients for the future SLO-aware policy."""

    input_token_cost: float = Field(gt=0)
    output_token_cost: float = Field(gt=0)
    cost_weight: float = Field(ge=0)
    slack_weight: float = Field(ge=0)
    waiting_weight: float = Field(ge=0)
    aging_threshold_s: float = Field(gt=0)

    @model_validator(mode="after")
    def at_least_one_scoring_weight(self) -> SloAwareConfig:
        """Reject a scoring function that cannot distinguish requests."""
        if self.cost_weight + self.slack_weight + self.waiting_weight == 0:
            raise ValueError("at least one SLO-aware scoring weight must be positive")
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
