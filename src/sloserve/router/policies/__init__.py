"""Scheduling policy implementations with one common interface."""

from sloserve.router.policies.base import SchedulingPolicy
from sloserve.router.policies.fcfs import FcfsPolicy

__all__ = ["FcfsPolicy", "SchedulingPolicy"]
