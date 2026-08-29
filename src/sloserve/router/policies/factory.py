"""Scheduling policy construction from experiment configuration."""

from __future__ import annotations

from sloserve.config import ExperimentConfig, SchedulerPolicyName
from sloserve.router.policies.base import SchedulingPolicy
from sloserve.router.policies.fcfs import FcfsPolicy
from sloserve.router.policies.slo_aware import SloAwarePolicy
from sloserve.router.policies.static_priority import StaticPriorityPolicy


def build_policy(config: ExperimentConfig) -> SchedulingPolicy:
    """Build the configured external admission scheduling policy."""
    if config.router.policy is SchedulerPolicyName.FCFS:
        return FcfsPolicy()
    if config.router.policy is SchedulerPolicyName.STATIC_PRIORITY:
        return StaticPriorityPolicy()
    if config.router.policy is SchedulerPolicyName.SLO_AWARE:
        return SloAwarePolicy(config.slo_aware)
    raise ValueError(f"unsupported scheduling policy: {config.router.policy}")
