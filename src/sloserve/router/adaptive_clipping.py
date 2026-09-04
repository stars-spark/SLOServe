"""Pure queue-depth controller for adaptive output-token caps."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, StrEnum

from sloserve.config import AdaptiveClipSignal, AdmissionControlConfig


class AdaptiveClipLevel(IntEnum):
    """Discrete clipping levels ordered from no clipping to the tightest cap."""

    L0 = 0
    L1 = 1
    L2 = 2
    L3 = 3


class AdaptiveClipTrigger(StrEnum):
    """Auditable reasons for one controller decision."""

    NO_CHANGE = "no_change"
    TIGHTEN_THRESHOLD = "tighten_threshold"
    RELAX_HOLD_STARTED = "relax_hold_started"
    RELAX_HOLD_PENDING = "relax_hold_pending"
    RELAX_HOLD_RESET = "relax_hold_reset"
    RELAX_HOLD_ELAPSED = "relax_hold_elapsed"


@dataclass(frozen=True, slots=True)
class AdaptiveClipDecision:
    """In-memory fact describing one queue-depth controller update."""

    decision_time_s: float
    q_inst: int
    q_bar: float
    old_level: AdaptiveClipLevel
    new_level: AdaptiveClipLevel
    selected_cap: int | None
    trigger_reason: AdaptiveClipTrigger


class AdaptiveClipController:
    """Apply fast tightening and held, one-level-at-a-time relaxation."""

    def __init__(
        self,
        config: AdmissionControlConfig,
        *,
        clock: Callable[[], float],
    ) -> None:
        if not config.adaptive_clip_enabled:
            raise ValueError("adaptive clip controller requires adaptive_clip_enabled=true")
        if config.adaptive_clip_signal is not AdaptiveClipSignal.QUEUE_DEPTH_EWMA:
            raise ValueError(
                "queue-depth controller requires adaptive_clip_signal=queue_depth_ewma"
            )
        self._clock = clock
        self._caps = config.adaptive_clip_caps
        self._tighten_thresholds = config.adaptive_clip_tighten_thresholds
        self._relax_thresholds = config.adaptive_clip_relax_thresholds
        self._tau_s = config.adaptive_clip_ewma_tau_s
        self._relax_hold_s = config.adaptive_clip_relax_hold_s
        self._level = AdaptiveClipLevel.L0
        self._q_inst = 0
        self._q_bar = 0.0
        self._last_update_s: float | None = None
        self._relax_since_s: float | None = None

    @property
    def level(self) -> AdaptiveClipLevel:
        """Return the current discrete clipping level."""
        return self._level

    @property
    def q_inst(self) -> int:
        """Return the latest stable external waiting depth."""
        return self._q_inst

    @property
    def q_bar(self) -> float:
        """Return the wall-clock weighted queue-depth EWMA."""
        return self._q_bar

    @property
    def pressure(self) -> float:
        """Return the dual-time-scale pressure used by the state machine."""
        return max(float(self._q_inst), self._q_bar)

    @property
    def selected_cap(self) -> int | None:
        """Return the current backend cap, or ``None`` for the no-clip level."""
        if self._level is AdaptiveClipLevel.L0:
            return None
        return self._caps[int(self._level) - 1]

    def update(self, q_inst: int) -> AdaptiveClipDecision:
        """Integrate elapsed queue depth, then update the discrete clipping level."""
        if isinstance(q_inst, bool) or not isinstance(q_inst, int) or q_inst < 0:
            raise ValueError("q_inst must be a non-negative integer")
        now_s = float(self._clock())
        if not math.isfinite(now_s):
            raise ValueError("clock must return a finite timestamp")
        if self._last_update_s is not None:
            delta_t = now_s - self._last_update_s
            if delta_t < 0.0:
                raise ValueError("clock must be monotonic")
            decay = math.exp(-delta_t / self._tau_s)
            self._q_bar = decay * self._q_bar + (1.0 - decay) * self._q_inst
        self._last_update_s = now_s
        self._q_inst = q_inst

        old_level = self._level
        trigger = self._update_level(now_s)
        return AdaptiveClipDecision(
            decision_time_s=now_s,
            q_inst=self._q_inst,
            q_bar=self._q_bar,
            old_level=old_level,
            new_level=self._level,
            selected_cap=self.selected_cap,
            trigger_reason=trigger,
        )

    def _update_level(self, now_s: float) -> AdaptiveClipTrigger:
        target_level = AdaptiveClipLevel(bisect_right(self._tighten_thresholds, self.pressure))
        if target_level > self._level:
            self._level = target_level
            self._relax_since_s = None
            return AdaptiveClipTrigger.TIGHTEN_THRESHOLD

        if self._level is AdaptiveClipLevel.L0:
            self._relax_since_s = None
            return AdaptiveClipTrigger.NO_CHANGE

        relax_threshold = self._relax_thresholds[int(self._level) - 1]
        if self.pressure >= relax_threshold:
            if self._relax_since_s is not None:
                self._relax_since_s = None
                return AdaptiveClipTrigger.RELAX_HOLD_RESET
            return AdaptiveClipTrigger.NO_CHANGE

        if self._relax_since_s is None:
            self._relax_since_s = now_s
            trigger = AdaptiveClipTrigger.RELAX_HOLD_STARTED
        else:
            trigger = AdaptiveClipTrigger.RELAX_HOLD_PENDING

        if now_s - self._relax_since_s >= self._relax_hold_s:
            self._level = AdaptiveClipLevel(int(self._level) - 1)
            self._relax_since_s = None
            return AdaptiveClipTrigger.RELAX_HOLD_ELAPSED
        return trigger


def build_adaptive_clip_controller(
    config: AdmissionControlConfig,
    *,
    clock: Callable[[], float],
) -> AdaptiveClipController | None:
    """Construct an enabled queue-depth controller without touching the clock when disabled."""
    if not config.adaptive_clip_enabled:
        return None
    return AdaptiveClipController(config, clock=clock)
