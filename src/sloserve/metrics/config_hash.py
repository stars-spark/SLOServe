"""Deterministic experiment-configuration identifiers."""

from __future__ import annotations

import hashlib
import json

from sloserve.config import ExperimentConfig


def experiment_config_hash(config: ExperimentConfig) -> str:
    """Return a SHA-256 hash of the complete stable JSON configuration."""
    serialized = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
