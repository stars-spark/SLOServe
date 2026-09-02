"""Tests for experiment configuration validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sloserve.config import (
    BackendConfig,
    ExperimentConfig,
    SloAwareConfig,
    TokenRangeConfig,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _base_config_dict() -> dict[str, Any]:
    text = (PROJECT_ROOT / "configs" / "base.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_base_config_loads() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    assert config.router.policy.value == "fcfs"
    assert config.backend.model == "Qwen/Qwen3-0.6B"
    assert config.backend.model_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert config.backend.tokenizer_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert config.workload.random_seed == 20250825
    assert config.workload.repetitions == 3
    assert config.slo_aware.input_token_seconds == 0.0005
    assert config.slo_aware.output_token_seconds == 0.01


def test_token_range_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError, match="maximum token count"):
        TokenRangeConfig(minimum=256, maximum=64)


def test_slo_aware_service_time_estimates_allow_zero() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    slo_aware = SloAwareConfig.model_validate(
        {
            **config.slo_aware.model_dump(),
            "input_token_seconds": 0.0,
            "output_token_seconds": 0.0,
        }
    )

    assert slo_aware.input_token_seconds == 0.0
    assert slo_aware.output_token_seconds == 0.0


def test_config_loader_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level configuration"):
        load_config(config_path)


def test_config_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("unknown: true\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_backend_rejects_floating_model_revision() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    with pytest.raises(ValidationError, match="model_revision"):
        BackendConfig.model_validate({**config.backend.model_dump(), "model_revision": "main"})


def test_server_defaults_are_conservative() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    assert config.server.host == "127.0.0.1"
    assert config.server.port == 8000
    assert config.server.gpu_memory_utilization == 0.5
    assert config.server.max_model_len == 4096
    assert config.server.max_num_batched_tokens is None
    assert config.server.enable_chunked_prefill is None
    assert config.server.enable_prefix_caching is None
    assert config.server.dtype.value == "bfloat16"
    assert config.server.enforce_eager is True


def test_vllm_serve_args_pin_model_and_tokenizer() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    args = config.vllm_serve_args()

    assert args[0] == "serve"
    assert args[1] == "Qwen/Qwen3-0.6B"
    commit = "c1899de289a04d12100db370d81485cdf75e47ca"
    assert args[args.index("--revision") + 1] == commit
    assert args[args.index("--tokenizer-revision") + 1] == commit
    assert args[args.index("--port") + 1] == "8000"
    assert args[args.index("--gpu-memory-utilization") + 1] == "0.5"
    assert "--enforce-eager" in args


def test_vllm_serve_args_express_experiment_d_server_parameters() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    server = config.server.model_copy(
        update={
            "max_num_batched_tokens": 2048,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
        }
    )

    args = config.model_copy(update={"server": server}).vllm_serve_args()

    assert args[args.index("--max-num-batched-tokens") + 1] == "2048"
    assert "--enable-chunked-prefill" in args
    assert "--no-enable-prefix-caching" in args


def test_server_port_must_match_backend_url() -> None:
    raw = _base_config_dict()
    raw["server"]["port"] = 9001

    with pytest.raises(ValidationError, match=r"must match backend\.base_url"):
        ExperimentConfig.model_validate(raw)


def test_server_must_admit_router_concurrency() -> None:
    raw = _base_config_dict()
    raw["server"]["max_num_seqs"] = 4
    raw["router"]["max_in_flight"] = 16

    with pytest.raises(ValidationError, match="max_num_seqs must be >="):
        ExperimentConfig.model_validate(raw)
