"""Deterministic capture of benchmark environment metadata."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from collections.abc import Callable

from sloserve.config import ExperimentConfig
from sloserve.metrics.config_hash import experiment_config_hash

_UNAVAILABLE = "unavailable"


def capture_environment(
    config: ExperimentConfig,
    *,
    driver_reader: Callable[[], str] = lambda: _read_driver_version(),
    vllm_reader: Callable[[], str] = lambda: _read_package_version("vllm"),
    torch_reader: Callable[[], str] = lambda: _read_package_version("torch"),
) -> str:
    """Return compact metadata, marking every unreadable optional field unavailable."""
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    fields = (
        ("python", python_version),
        ("driver", _read_or_unavailable(driver_reader)),
        ("model_revision", config.backend.model_revision),
        ("config_hash", experiment_config_hash(config)),
        ("vllm", _read_or_unavailable(vllm_reader)),
        ("torch", _read_or_unavailable(torch_reader)),
    )
    return ";".join(f"{name}={value}" for name, value in fields)


def _read_driver_version() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("nvidia-smi driver query returned no rows")
    return lines[0]


def _read_package_version(package: str) -> str:
    return importlib.metadata.version(package)


def _read_or_unavailable(reader: Callable[[], str]) -> str:
    try:
        value = reader().strip()
    except Exception:
        return _UNAVAILABLE
    return value if value else _UNAVAILABLE
