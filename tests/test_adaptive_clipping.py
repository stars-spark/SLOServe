"""Tests for the pure adaptive output-cap controller."""

from __future__ import annotations

import math

import pytest

from sloserve.config import AdmissionControlConfig
from sloserve.router.adaptive_clipping import (
    AdaptiveClipController,
    AdaptiveClipLevel,
    AdaptiveClipTrigger,
    build_adaptive_clip_controller,
)


class FakeClock:
    """Manually controlled monotonic clock for deterministic state-machine tests."""

    def __init__(self, now_s: float = 0.0) -> None:
        self.now_s = now_s
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return self.now_s

    def advance(self, delta_s: float) -> None:
        self.now_s += delta_s


def _enabled_config(**updates: object) -> AdmissionControlConfig:
    return AdmissionControlConfig.model_validate(
        {
            **AdmissionControlConfig().model_dump(),
            "clip_enabled": True,
            "adaptive_clip_enabled": True,
            **updates,
        }
    )


def _controller(clock: FakeClock, **updates: object) -> AdaptiveClipController:
    controller = build_adaptive_clip_controller(_enabled_config(**updates), clock=clock)
    assert controller is not None
    return controller


def test_burst_pressure_is_monotonic_as_unabsorbed_arrivals_accumulate() -> None:
    clock = FakeClock()
    controller = _controller(clock)

    decisions = [controller.update(depth) for depth in (0, 1, 2, 3, 4, 5)]
    pressures = [max(float(decision.q_inst), decision.q_bar) for decision in decisions]

    assert pressures == sorted(pressures)
    assert [decision.new_level for decision in decisions] == [
        AdaptiveClipLevel.L0,
        AdaptiveClipLevel.L1,
        AdaptiveClipLevel.L2,
        AdaptiveClipLevel.L3,
        AdaptiveClipLevel.L3,
        AdaptiveClipLevel.L3,
    ]


def test_immediate_zero_wait_arrivals_do_not_dilute_pressure_or_level() -> None:
    baseline_clock = FakeClock()
    baseline = _controller(baseline_clock)
    baseline.update(2)
    baseline_clock.advance(6.0)
    baseline.update(2)
    baseline.update(0)

    inserted_clock = FakeClock()
    inserted = _controller(inserted_clock)
    inserted.update(2)
    inserted_clock.advance(6.0)
    inserted.update(2)
    inserted.update(0)

    for _ in range(5):
        inserted_clock.advance(1.0)
        inserted.update(0)
    inserted_clock.advance(1.0)
    inserted_decision = inserted.update(0)

    baseline_clock.advance(6.0)
    baseline_decision = baseline.update(0)

    inserted_pressure = max(float(inserted_decision.q_inst), inserted_decision.q_bar)
    baseline_pressure = max(float(baseline_decision.q_inst), baseline_decision.q_bar)
    assert inserted_pressure == pytest.approx(baseline_pressure)
    assert inserted_decision.new_level == baseline_decision.new_level
    assert inserted_decision.new_level is AdaptiveClipLevel.L2


def test_wall_clock_ewma_is_independent_of_event_count() -> None:
    one_event_clock = FakeClock()
    one_event = _controller(one_event_clock)
    one_event.update(4)
    one_event_clock.advance(10.0)
    one_event_decision = one_event.update(4)

    many_events_clock = FakeClock()
    many_events = _controller(many_events_clock)
    many_events.update(4)
    for _ in range(100):
        many_events_clock.advance(0.1)
        many_events_decision = many_events.update(4)

    expected = 4.0 * (1.0 - math.exp(-10.0 / 6.0))
    assert one_event_decision.q_bar == pytest.approx(expected)
    assert many_events_decision.q_bar == pytest.approx(expected)
    assert many_events_decision.q_bar == pytest.approx(one_event_decision.q_bar)


@pytest.mark.parametrize(
    ("depth", "expected_level", "expected_cap"),
    [
        (0, AdaptiveClipLevel.L0, None),
        (1, AdaptiveClipLevel.L1, 1536),
        (2, AdaptiveClipLevel.L2, 1024),
        (3, AdaptiveClipLevel.L3, 512),
    ],
)
def test_tighten_threshold_equality_enters_corresponding_level(
    depth: int,
    expected_level: AdaptiveClipLevel,
    expected_cap: int | None,
) -> None:
    controller = _controller(FakeClock())

    decision = controller.update(depth)

    assert decision.new_level is expected_level
    assert decision.selected_cap == expected_cap


def test_burst_directly_jumps_from_l0_to_tightest_level() -> None:
    controller = _controller(FakeClock())

    decision = controller.update(5)

    assert decision.old_level is AdaptiveClipLevel.L0
    assert decision.new_level is AdaptiveClipLevel.L3
    assert decision.selected_cap == 512
    assert decision.trigger_reason is AdaptiveClipTrigger.TIGHTEN_THRESHOLD


def test_relax_threshold_equality_does_not_start_hold() -> None:
    clock = FakeClock()
    controller = _controller(clock, adaptive_clip_relax_thresholds=(0.0, 0.75, 1.5))
    controller.update(1)

    decision = controller.update(0)

    assert decision.q_bar == 0.0
    assert decision.new_level is AdaptiveClipLevel.L1
    assert decision.trigger_reason is AdaptiveClipTrigger.NO_CHANGE


def test_relaxation_is_one_level_per_hold_and_restarts_each_level_timer() -> None:
    clock = FakeClock()
    controller = _controller(clock, adaptive_clip_relax_hold_s=2.0)
    controller.update(3)
    started_l3 = controller.update(0)

    clock.advance(2.0)
    relaxed_to_l2 = controller.update(0)
    started_l2 = controller.update(0)
    clock.advance(2.0)
    relaxed_to_l1 = controller.update(0)
    started_l1 = controller.update(0)
    clock.advance(2.0)
    relaxed_to_l0 = controller.update(0)

    assert started_l3.trigger_reason is AdaptiveClipTrigger.RELAX_HOLD_STARTED
    assert relaxed_to_l2.new_level is AdaptiveClipLevel.L2
    assert relaxed_to_l2.selected_cap == 1024
    assert started_l2.new_level is AdaptiveClipLevel.L2
    assert started_l2.trigger_reason is AdaptiveClipTrigger.RELAX_HOLD_STARTED
    assert relaxed_to_l1.new_level is AdaptiveClipLevel.L1
    assert relaxed_to_l1.selected_cap == 1536
    assert started_l1.new_level is AdaptiveClipLevel.L1
    assert started_l1.trigger_reason is AdaptiveClipTrigger.RELAX_HOLD_STARTED
    assert relaxed_to_l0.new_level is AdaptiveClipLevel.L0
    assert relaxed_to_l0.selected_cap is None
    assert clock.now_s == 3 * 2.0


def test_pressure_recovery_resets_relax_hold_evidence() -> None:
    clock = FakeClock()
    controller = _controller(clock, adaptive_clip_relax_hold_s=10.0)
    controller.update(1)
    controller.update(0)
    clock.advance(9.0)
    assert controller.update(0).new_level is AdaptiveClipLevel.L1

    reset = controller.update(1)
    controller.update(0)
    clock.advance(9.0)
    still_l1 = controller.update(0)

    assert reset.trigger_reason is AdaptiveClipTrigger.RELAX_HOLD_RESET
    assert still_l1.new_level is AdaptiveClipLevel.L1


def test_decision_exposes_all_required_in_memory_fact_fields() -> None:
    clock = FakeClock(7.5)
    controller = _controller(clock)

    decision = controller.update(2)

    assert decision.decision_time_s == 7.5
    assert decision.q_inst == 2
    assert decision.q_bar == 0.0
    assert decision.old_level is AdaptiveClipLevel.L0
    assert decision.new_level is AdaptiveClipLevel.L2
    assert decision.selected_cap == 1024
    assert decision.trigger_reason is AdaptiveClipTrigger.TIGHTEN_THRESHOLD


def test_fixed_low_load_trace_stays_at_l0() -> None:
    clock = FakeClock()
    controller = _controller(clock)
    decisions = []

    for sequence_id in range(24):
        clock.now_s = sequence_id / 0.08
        decisions.append(controller.update(0))

    assert all(decision.q_inst == 0 for decision in decisions)
    assert all(decision.q_bar == 0.0 for decision in decisions)
    assert all(decision.new_level is AdaptiveClipLevel.L0 for decision in decisions)
    assert all(decision.selected_cap is None for decision in decisions)


def test_disabled_default_does_not_construct_controller_or_read_clock() -> None:
    clock = FakeClock()

    controller = build_adaptive_clip_controller(AdmissionControlConfig(), clock=clock)

    assert controller is None
    assert clock.calls == 0


def test_controller_rejects_invalid_depth_and_non_monotonic_clock() -> None:
    clock = FakeClock(2.0)
    controller = _controller(clock)

    with pytest.raises(ValueError, match="non-negative integer"):
        controller.update(-1)
    controller.update(0)
    clock.now_s = 1.0
    with pytest.raises(ValueError, match="monotonic"):
        controller.update(0)
