"""Offline tests for deterministic environment metadata."""

from __future__ import annotations

import sys
from pathlib import Path

from sloserve.config import load_config
from sloserve.experiments.metadata import capture_environment
from sloserve.metrics.config_hash import experiment_config_hash

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _unavailable_reader() -> str:
    raise LookupError("intentionally unavailable")


def test_capture_environment_uses_readers_and_marks_missing_versions() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")

    metadata = capture_environment(
        config,
        driver_reader=lambda: "580.178.04\n",
        vllm_reader=_unavailable_reader,
        torch_reader=lambda: "",
    )

    python_version = ".".join(str(part) for part in sys.version_info[:3])
    assert metadata == ";".join(
        (
            f"python={python_version}",
            "driver=580.178.04",
            f"model_revision={config.backend.model_revision}",
            f"config_hash={experiment_config_hash(config)}",
            "vllm=unavailable",
            "torch=unavailable",
        )
    )
