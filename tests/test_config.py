"""Tests for experiment configuration validation."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sloserve.config import (
    AdaptiveClipSignal,
    AdmissionControlConfig,
    BackendConfig,
    ExperimentConfig,
    LengthModel,
    LengthSource,
    RealisticLengthConfig,
    SloAwareConfig,
    TokenRangeConfig,
    load_config,
)
from sloserve.metrics.config_hash import experiment_config_hash

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
    assert config.slo_aware.aging_levels == 1
    assert config.slo_aware.length_source is LengthSource.TRUE
    assert config.slo_aware.length_estimator_path is None
    assert config.admission == AdmissionControlConfig()
    assert config.admission.adaptive_clip_enabled is False
    assert config.admission.adaptive_clip_signal is AdaptiveClipSignal.QUEUE_DEPTH_EWMA
    assert config.admission.adaptive_clip_caps == (1536, 1024, 512)
    assert config.admission.adaptive_clip_tighten_thresholds == (1.0, 2.0, 3.0)
    assert config.admission.adaptive_clip_relax_thresholds == (0.25, 0.75, 1.5)
    assert config.admission.adaptive_clip_ewma_tau_s == 6.0
    assert config.admission.adaptive_clip_tighten_hold_s == 8.0
    assert config.admission.adaptive_clip_relax_hold_s == 30.0
    assert config.admission.adaptive_clip_capacity_rps is None
    assert config.workload.length_model is LengthModel.UNIFORM_CAP
    assert config.workload.realistic_length is None


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


def test_realistic_length_model_requires_configuration() -> None:
    raw = _base_config_dict()
    raw["workload"]["length_model"] = "realistic"

    with pytest.raises(ValidationError, match="requires realistic_length"):
        ExperimentConfig.model_validate(raw)


def test_realistic_length_configuration_validates_mixture_and_clamps() -> None:
    with pytest.raises(ValidationError, match="equal lengths"):
        RealisticLengthConfig(kind_fractions=(0.5, 0.5))
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        RealisticLengthConfig(kind_fractions=(0.4, 0.3, 0.2))
    with pytest.raises(ValidationError, match="clamp_min"):
        RealisticLengthConfig(clamp_min=32, clamp_max=16)


def test_learned_length_source_requires_estimator_path() -> None:
    raw = _base_config_dict()
    raw["slo_aware"]["length_source"] = "learned"

    with pytest.raises(ValidationError, match="requires length_estimator_path"):
        ExperimentConfig.model_validate(raw)


def test_enabled_learned_clipping_requires_its_own_estimator_path() -> None:
    raw = _base_config_dict()
    raw["admission"]["clip_enabled"] = True
    raw["admission"]["clip_source"] = "learned"

    with pytest.raises(ValidationError, match="requires clip_estimator_path"):
        ExperimentConfig.model_validate(raw)

    # A dormant learned source must preserve old configs without loading an artifact.
    raw["admission"]["clip_enabled"] = False
    config = ExperimentConfig.model_validate(raw)
    assert config.admission.clip_enabled is False


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("adaptive_clip_caps", [1536, 1024], "exactly 3 values"),
        ("adaptive_clip_caps", [1536, 0, -1], "only positive values"),
        ("adaptive_clip_caps", [1536, 1536, 512], "strictly decreasing"),
        ("adaptive_clip_caps", [1536, 1024, 511], "at least 512"),
        ("adaptive_clip_tighten_thresholds", [1.0, 2.0], "exactly 3 values"),
        (
            "adaptive_clip_tighten_thresholds",
            [1.0, float("nan"), 3.0],
            "finite non-negative",
        ),
        ("adaptive_clip_tighten_thresholds", [-1.0, 2.0, 3.0], "finite non-negative"),
        ("adaptive_clip_tighten_thresholds", [1.0, 1.0, 3.0], "strictly increasing"),
        ("adaptive_clip_relax_thresholds", [0.25, 0.75], "exactly 3 values"),
        (
            "adaptive_clip_relax_thresholds",
            [0.25, 0.75, float("inf")],
            "finite non-negative",
        ),
        ("adaptive_clip_relax_thresholds", [-0.1, 0.75, 1.5], "finite non-negative"),
        ("adaptive_clip_relax_thresholds", [0.25, 0.25, 1.5], "strictly increasing"),
        ("adaptive_clip_relax_thresholds", [0.25, 0.75, 3.0], "below its tighten"),
        ("adaptive_clip_ewma_tau_s", 0.0, "greater than 0"),
        ("adaptive_clip_ewma_tau_s", float("inf"), "finite number"),
        ("adaptive_clip_tighten_hold_s", -1.0, "greater than or equal to 0"),
        ("adaptive_clip_tighten_hold_s", float("nan"), "finite number"),
        ("adaptive_clip_tighten_hold_s", float("inf"), "finite number"),
        ("adaptive_clip_relax_hold_s", -1.0, "greater than or equal to 0"),
        ("adaptive_clip_relax_hold_s", float("inf"), "finite number"),
        ("adaptive_clip_capacity_rps", 0.0, "greater than 0"),
    ],
)
def test_adaptive_clip_configuration_rejects_invalid_values(
    field_name: str, value: object, message: str
) -> None:
    raw = _base_config_dict()
    raw["admission"][field_name] = value

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(raw)


def test_adaptive_clipping_requires_fixed_clipping_to_be_enabled() -> None:
    raw = _base_config_dict()
    raw["admission"]["adaptive_clip_enabled"] = True

    with pytest.raises(ValidationError, match="requires clip_enabled=true"):
        ExperimentConfig.model_validate(raw)


def test_enabled_rho_hat_requires_positive_capacity() -> None:
    raw = _base_config_dict()
    raw["admission"].update(
        {
            "clip_enabled": True,
            "adaptive_clip_enabled": True,
            "adaptive_clip_signal": "rho_hat",
        }
    )

    with pytest.raises(ValidationError, match="requires adaptive_clip_capacity_rps"):
        ExperimentConfig.model_validate(raw)

    raw["admission"]["adaptive_clip_capacity_rps"] = 0.23
    config = ExperimentConfig.model_validate(raw)
    assert config.admission.adaptive_clip_capacity_rps == 0.23


def test_disabled_adaptive_defaults_preserve_legacy_config_hash() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    assert experiment_config_hash(config) == (
        "520091cbcb2669534c962b25c86c8f6fd23e7cd1e3191f7afe36a672b53201a6"
    )
    json_admission = config.admission.model_dump(mode="json")
    assert set(json_admission) == {
        "clip_enabled",
        "clip_max_tokens",
        "clip_source",
        "clip_estimator_path",
    }
    assert "adaptive_clip_enabled" in config.admission.model_dump()

    dormant_admission = AdmissionControlConfig.model_validate(
        {
            **config.admission.model_dump(),
            "adaptive_clip_caps": [2048, 1536, 512],
            "adaptive_clip_tighten_thresholds": [2.0, 3.0, 4.0],
            "adaptive_clip_relax_thresholds": [0.5, 1.5, 2.5],
            "adaptive_clip_ewma_tau_s": 12.0,
            "adaptive_clip_tighten_hold_s": 9.0,
            "adaptive_clip_relax_hold_s": 45.0,
        }
    )
    dormant = ExperimentConfig.model_validate(
        {**config.model_dump(), "admission": dormant_admission.model_dump()}
    )
    assert experiment_config_hash(dormant) == experiment_config_hash(config)


def test_enabled_adaptive_configuration_is_included_in_config_hash() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    enabled_admission = AdmissionControlConfig.model_validate(
        {
            **config.admission.model_dump(),
            "clip_enabled": True,
            "adaptive_clip_enabled": True,
        }
    )
    enabled = ExperimentConfig.model_validate(
        {**config.model_dump(), "admission": enabled_admission.model_dump()}
    )

    assert experiment_config_hash(enabled) != experiment_config_hash(config)
    assert enabled.admission.model_dump(mode="json")["adaptive_clip_enabled"] is True
