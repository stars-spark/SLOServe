"""Tests for configuration-driven scheduling policy construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from sloserve.config import ExperimentConfig, SchedulerPolicyName, load_config
from sloserve.router.policies.factory import build_policy
from sloserve.router.policies.fcfs import FcfsPolicy
from sloserve.router.policies.static_priority import StaticPriorityPolicy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config_with_policy(policy: SchedulerPolicyName) -> ExperimentConfig:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    router = config.router.model_copy(update={"policy": policy})
    return config.model_copy(update={"router": router})


def test_build_policy_returns_fcfs_for_fcfs_configuration() -> None:
    policy = build_policy(_config_with_policy(SchedulerPolicyName.FCFS))

    assert isinstance(policy, FcfsPolicy)


def test_build_policy_returns_static_priority_for_static_configuration() -> None:
    policy = build_policy(_config_with_policy(SchedulerPolicyName.STATIC_PRIORITY))

    assert isinstance(policy, StaticPriorityPolicy)


def test_build_policy_leaves_slo_aware_for_w2_2() -> None:
    with pytest.raises(NotImplementedError, match="SLO-aware policy is implemented in W2-2"):
        build_policy(_config_with_policy(SchedulerPolicyName.SLO_AWARE))
