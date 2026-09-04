"""CPU fixtures for expL audit aggregation and every preregistered failure gate."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from sloserve.analysis.expl import (
    DecisionFact,
    ExpLIntegrityError,
    LowLoadEvidence,
    PreRegistrationFailure,
    check_benefit_retention,
    check_depth_eligibility,
    check_directionality,
    check_latency_benefit,
    check_low_load_gate,
    check_thrash,
    check_utility_recovery,
    load_and_reconcile_decisions,
    read_decision_facts,
    reconcile_point,
    replay_expkb_depth_eligibility,
    summarize_controller_health,
)
from sloserve.config import SchedulerPolicyName
from sloserve.experiments.sweep import config_for_point, load_sweep_config
from sloserve.metrics import RequestRecord
from sloserve.router.adaptive_clipping import AdaptiveClipLevel, AdaptiveClipTrigger
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 202, 20250825)


def _seed_values(value: float) -> dict[int, float]:
    return dict.fromkeys(SEEDS, value)


def _decision(
    time_s: float,
    old_level: AdaptiveClipLevel,
    new_level: AdaptiveClipLevel,
    *,
    q_inst: int = 0,
    q_bar: float = 0.0,
    repetition_index: int = 0,
    request_id: str | None = None,
) -> DecisionFact:
    caps = {
        AdaptiveClipLevel.L0: None,
        AdaptiveClipLevel.L1: 1536,
        AdaptiveClipLevel.L2: 1024,
        AdaptiveClipLevel.L3: 512,
    }
    return DecisionFact(
        request_id=request_id or f"r-{time_s}",
        repetition_index=repetition_index,
        decision_time_s=time_s,
        q_inst=q_inst,
        q_bar=q_bar,
        old_level=old_level,
        new_level=new_level,
        selected_cap=caps[new_level],
        trigger_reason=AdaptiveClipTrigger.NO_CHANGE,
    )


def _record(
    *,
    request_id: str = "request-000004",
    sequence_id: int = 4,
    repetition_index: int = 0,
    backend_cap: int = 10,
) -> RequestRecord:
    return RequestRecord(
        request_id=request_id,
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=0.0,
        enqueue_time_s=0.0,
        dispatch_time_s=1.0,
        first_token_time_s=1.1,
        completion_time_s=2.0,
        input_tokens=16,
        output_tokens=backend_cap,
        status=DispatchStatus.SUCCESS,
        error_type=None,
        policy_name=SchedulerPolicyName.FCFS,
        config_hash="a" * 64,
        repetition_index=repetition_index,
        env_version="fixture",
        requested_output_tokens=10,
        backend_max_output_tokens=backend_cap,
        finish_reason="length",
    )


def _low_load_evidence() -> LowLoadEvidence:
    return LowLoadEvidence(
        clip_applied_count=0,
        realized_truncation_count=0,
        every_backend_cap_matches_request=True,
        every_decision_is_l0=True,
        paired_trace_equal=True,
        no_clip_throughput_by_seed=_seed_values(100.0),
        adaptive_throughput_by_seed=_seed_values(100.0),
    )


def test_low_load_gate_passes_zero_cost_trace_and_equivalence_interval() -> None:
    result = check_low_load_gate(_low_load_evidence())

    assert result["paired_token_throughput_ratio"]["low_95"] == 1.0
    assert result["paired_token_throughput_ratio"]["high_95"] == 1.0


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"clip_applied_count": 1}, "clip_applied_count"),
        ({"realized_truncation_count": 1}, "realized_truncation_count"),
        ({"every_backend_cap_matches_request": False}, "backend cap"),
        ({"every_decision_is_l0": False}, "decision is not L0"),
        ({"paired_trace_equal": False}, "traces differ"),
        ({"adaptive_throughput_by_seed": _seed_values(90.0)}, "throughput-ratio"),
    ],
)
def test_low_load_gate_rejects_every_registered_violation(
    updates: dict[str, object], match: str
) -> None:
    with pytest.raises(PreRegistrationFailure, match=match):
        check_low_load_gate(replace(_low_load_evidence(), **updates))


def test_depth_eligibility_gate_replays_real_expkb_trace_and_reaches_l3() -> None:
    replay = replay_expkb_depth_eligibility(PROJECT_ROOT / "results" / "raw" / "week6-expK-B")

    assert replay["eligible_levels"] == {"L1": True, "L2": True, "L3": True}
    assert replay["l3_reachable"] is True
    assert replay["peak_depth_by_seed_suffix"] == {"1": 5, "2": 3, "3": 3}
    assert replay["threshold_time_s"]["3"] == pytest.approx(110.96182093699463)
    assert check_depth_eligibility(replay, tighten_hold_s=4.0)["l3_reachable"] is True


@pytest.mark.parametrize("failure", ["missing_level", "l3_unreachable", "hold_too_long"])
def test_depth_eligibility_gate_rejects_empty_or_unresponsive_configuration(
    failure: str,
) -> None:
    replay = replay_expkb_depth_eligibility(PROJECT_ROOT / "results" / "raw" / "week6-expK-B")
    replay = copy.deepcopy(replay)
    tighten_hold_s = 4.0
    if failure == "missing_level":
        replay["eligible_levels"]["L3"] = False
    elif failure == "l3_unreachable":
        # Eligibility and reachability are separate conditions: a level can spend
        # time above its threshold and still never be entered once the hold is
        # applied. Flipping only reachability keeps the earlier check satisfied,
        # so this is what actually exercises the L3 guard -- the regression that
        # an unreachable tightest level is exactly what the first threshold
        # ladder got wrong.
        replay["l3_reachable"] = False
    else:
        tighten_hold_s = 100.0

    with pytest.raises(PreRegistrationFailure):
        check_depth_eligibility(replay, tighten_hold_s=tighten_hold_s)


def test_latency_benefit_gate_passes_paired_seed_improvements() -> None:
    result = check_latency_benefit(
        _seed_values(10.0),
        _seed_values(5.0),
        _seed_values(0.2),
        _seed_values(0.6),
    )

    assert result["mean_queue_wait_reduction"]["low_95"] == 5.0
    assert result["interactive_slo_increase"]["low_95"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("adaptive_wait", "adaptive_slo"),
    [(10.0, 0.6), (5.0, 0.2), (11.0, 0.6)],
)
def test_latency_benefit_gate_rejects_zero_or_wrong_direction(
    adaptive_wait: float, adaptive_slo: float
) -> None:
    with pytest.raises(PreRegistrationFailure, match="latency benefit"):
        check_latency_benefit(
            _seed_values(10.0),
            _seed_values(adaptive_wait),
            _seed_values(0.2),
            _seed_values(adaptive_slo),
        )


def test_benefit_retention_gate_passes_both_frozen_ratios() -> None:
    result = check_benefit_retention(
        _seed_values(10.0),
        _seed_values(2.0),
        _seed_values(3.0),
        _seed_values(0.2),
        _seed_values(0.8),
        _seed_values(0.7),
    )

    assert result["wait_retention"] == 0.875
    assert result["slo_retention"] == pytest.approx(5.0 / 6.0)


@pytest.mark.parametrize(
    ("fixed_wait", "adaptive_wait", "fixed_slo", "adaptive_slo", "match"),
    [
        (2.0, 5.0, 0.8, 0.7, "below"),
        (11.0, 3.0, 0.8, 0.7, "uninterpretable"),
        (2.0, 3.0, 0.1, 0.7, "uninterpretable"),
    ],
)
def test_benefit_retention_gate_rejects_threshold_and_nonpositive_denominators(
    fixed_wait: float,
    adaptive_wait: float,
    fixed_slo: float,
    adaptive_slo: float,
    match: str,
) -> None:
    with pytest.raises(PreRegistrationFailure, match=match):
        check_benefit_retention(
            _seed_values(10.0),
            _seed_values(fixed_wait),
            _seed_values(adaptive_wait),
            _seed_values(0.2),
            _seed_values(fixed_slo),
            _seed_values(adaptive_slo),
        )


@pytest.mark.parametrize(
    ("adaptive_throughput", "adaptive_truncation"),
    [(12.0, 0.5), (10.0, 0.3)],
)
def test_utility_recovery_gate_passes_either_stable_recovery(
    adaptive_throughput: float, adaptive_truncation: float
) -> None:
    result = check_utility_recovery(
        _seed_values(10.0),
        _seed_values(adaptive_throughput),
        _seed_values(0.5),
        _seed_values(adaptive_truncation),
    )

    assert set(result) == {
        "token_throughput_gain",
        "realized_truncation_rate_reduction",
    }


def test_utility_recovery_gate_rejects_no_new_tradeoff() -> None:
    with pytest.raises(PreRegistrationFailure, match="neither"):
        check_utility_recovery(
            _seed_values(10.0),
            _seed_values(10.0),
            _seed_values(0.5),
            _seed_values(0.5),
        )


def test_thrash_gate_passes_held_reverse_transitions_below_switch_ceiling() -> None:
    decisions = (
        _decision(5.0, AdaptiveClipLevel.L0, AdaptiveClipLevel.L2, q_inst=2),
        _decision(35.0, AdaptiveClipLevel.L2, AdaptiveClipLevel.L1),
        _decision(40.0, AdaptiveClipLevel.L1, AdaptiveClipLevel.L2, q_inst=2),
    )
    result = check_thrash(
        decisions,
        formal_dispatch_count=40,
        tighten_hold_s=4.0,
        relax_hold_s=30.0,
    )

    assert result["transition_count"] == 3
    assert result["tighten_relax_retighten_cycles_s"] == [35.0]


@pytest.mark.parametrize("failure", ["short_relax", "short_tighten", "too_many_switches"])
def test_thrash_gate_rejects_hold_violations_and_switch_instability(failure: str) -> None:
    if failure == "short_relax":
        decisions = (
            _decision(0.0, AdaptiveClipLevel.L0, AdaptiveClipLevel.L1),
            _decision(29.0, AdaptiveClipLevel.L1, AdaptiveClipLevel.L0),
        )
        dispatch_count = 100
    elif failure == "short_tighten":
        decisions = (
            _decision(0.0, AdaptiveClipLevel.L1, AdaptiveClipLevel.L0),
            _decision(3.0, AdaptiveClipLevel.L0, AdaptiveClipLevel.L1),
        )
        dispatch_count = 100
    else:
        decisions = tuple(
            _decision(
                float(index * 31),
                AdaptiveClipLevel.L0 if index % 2 == 0 else AdaptiveClipLevel.L1,
                AdaptiveClipLevel.L1 if index % 2 == 0 else AdaptiveClipLevel.L0,
            )
            for index in range(11)
        )
        dispatch_count = 100

    with pytest.raises(PreRegistrationFailure):
        check_thrash(
            decisions,
            formal_dispatch_count=dispatch_count,
            tighten_hold_s=4.0,
            relax_hold_s=30.0,
        )


def test_directionality_gate_passes_monotone_depth_response() -> None:
    decisions = (
        _decision(0.0, AdaptiveClipLevel.L0, AdaptiveClipLevel.L0, q_inst=0, q_bar=0.0),
        _decision(1.0, AdaptiveClipLevel.L0, AdaptiveClipLevel.L1, q_inst=1, q_bar=0.2),
    )

    assert check_directionality(decisions) == {"anomaly_count": 0}


def test_directionality_gate_rejects_looser_response_as_depth_rises() -> None:
    decisions = (
        _decision(0.0, AdaptiveClipLevel.L2, AdaptiveClipLevel.L2, q_inst=1, q_bar=3.0),
        _decision(1.0, AdaptiveClipLevel.L2, AdaptiveClipLevel.L1, q_inst=2, q_bar=1.0),
    )

    with pytest.raises(PreRegistrationFailure, match="q_inst rose"):
        check_directionality(decisions)


def _reconciliation_row() -> dict[str, str]:
    return {
        "request_count": "1",
        "success_count": "1",
        "error_count": "0",
        "timeout_count": "0",
        "cancelled_count": "0",
        "rejected_count": "0",
        "success_output_tokens": "10",
        "wall_clock_window_s": "2.0",
        "token_throughput_per_s": "5.0",
        "end_to_end_p50_s": "2.0",
        "end_to_end_p95_s": "2.0",
        "end_to_end_p99_s": "2.0",
        "queue_wait_p50_s": "1.0",
        "queue_wait_p95_s": "1.0",
        "queue_wait_p99_s": "1.0",
        "queue_wait_mean_s": "1.0",
        "longest_queue_wait_s": "1.0",
        "clip_applied_count": "0",
        "clip_applied_rate": "0.0",
        "realized_truncation_count": "0",
        "realized_truncation_rate": "0.0",
        "mean_cap_reduction_tokens": "",
        "slo_overall_rate": "1.0",
        "slo_interactive_rate": "1.0",
        "slo_batch_rate": "",
    }


def _expl_point_config():
    definition = load_sweep_config(
        PROJECT_ROOT / "configs" / "sweeps" / "expL-adaptive-clipping.yaml"
    )
    point = next(point for point in definition.points if point.label == "high-no-clip-s1")
    return config_for_point(definition.base_config, point)


def test_raw_aggregate_reconciliation_passes_matching_facts() -> None:
    result = reconcile_point(_reconciliation_row(), (_record(),), _expl_point_config())

    assert result["success_output_tokens"] == 10
    assert result["realized_truncation_count"] == 0


def test_raw_aggregate_reconciliation_hard_fails_on_mismatch() -> None:
    row = _reconciliation_row()
    row["success_output_tokens"] = "11"

    with pytest.raises(ExpLIntegrityError, match="aggregate mismatch"):
        reconcile_point(row, (_record(),), _expl_point_config())


def _decision_json(request_id: str = "request-000004") -> str:
    return json.dumps(
        {
            "request_id": request_id,
            "repetition_index": 0,
            "decision_time_s": 1.0,
            "q_inst": 0,
            "q_bar": 0.0,
            "old_level": "L0",
            "new_level": "L0",
            "selected_cap": None,
            "trigger_reason": "no_change",
        },
        separators=(",", ":"),
    )


def test_decision_reconciliation_accepts_one_to_one_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "sweep-000-arm-adaptive-cap-decisions.jsonl"
    path.write_text(_decision_json() + "\n", encoding="utf-8")

    decisions = load_and_reconcile_decisions(tmp_path, "arm", (_record(),))

    assert len(decisions) == 1
    assert decisions[0].new_level is AdaptiveClipLevel.L0


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unmatched"])
def test_decision_reconciliation_hard_fails_missing_duplicate_or_unmatched(
    tmp_path: Path, failure: str
) -> None:
    path = tmp_path / "sweep-000-arm-adaptive-cap-decisions.jsonl"
    if failure == "duplicate":
        line = _decision_json()
        path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    elif failure == "unmatched":
        path.write_text(_decision_json("wrong-request") + "\n", encoding="utf-8")

    with pytest.raises((ExpLIntegrityError, FileNotFoundError)):
        load_and_reconcile_decisions(tmp_path, "arm", (_record(),))


def test_controller_health_reports_residence_entries_cycles_and_response() -> None:
    records = (
        _record(request_id="r-0", sequence_id=4),
        replace(
            _record(request_id="r-1", sequence_id=5),
            arrival_time_s=2.0,
            enqueue_time_s=2.0,
            dispatch_time_s=5.0,
            first_token_time_s=5.1,
            completion_time_s=6.0,
        ),
    )
    decisions = (
        _decision(
            1.0,
            AdaptiveClipLevel.L0,
            AdaptiveClipLevel.L0,
            request_id="r-0",
            q_inst=1,
        ),
        _decision(
            5.0,
            AdaptiveClipLevel.L0,
            AdaptiveClipLevel.L1,
            request_id="r-1",
            q_inst=2,
        ),
    )

    result = summarize_controller_health(
        records,
        decisions,
        warmup_requests=4,
        tighten_hold_s=4.0,
        relax_hold_s=30.0,
    )

    assert result["level_residence"]["L0"]["time_s"] == 4.0
    assert result["level_residence"]["L1"]["entry_count"] == 1
    assert result["first_positive_depth_to_first_tighten"]["p50_s"] == 4.0


def test_read_decision_facts_rejects_duplicate_identity(tmp_path: Path) -> None:
    line = _decision_json()
    path = tmp_path / "decisions.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ExpLIntegrityError, match="duplicate"):
        read_decision_facts(path)
