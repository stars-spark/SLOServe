"""Audited aggregation and preregistered failure gates for expL."""

from __future__ import annotations

import csv
import itertools
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, TypeAlias

from sloserve.analysis.metrics import MetricsSummary, calculate_metrics
from sloserve.analysis.queueing import estimate_mg1_wait, fit_linear_service_time
from sloserve.config import ExperimentConfig
from sloserve.experiments.sweep import (
    SweepDefinition,
    SweepPoint,
    config_for_point,
    load_sweep_config,
)
from sloserve.metrics import RequestRecord, read_request_records_jsonl
from sloserve.router.adaptive_clipping import AdaptiveClipLevel, AdaptiveClipTrigger
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus

SeedValues: TypeAlias = Mapping[int, float]
_LABEL_PATTERN = re.compile(r"^(low|high)-(.+)-s\d+$")
_EXPECTED_LOADS = {"low": 0.08, "high": 0.17}
_LEVELS = tuple(AdaptiveClipLevel)
_RECONCILED_FIELDS = (
    "request_count",
    "success_count",
    "error_count",
    "timeout_count",
    "cancelled_count",
    "rejected_count",
    "success_output_tokens",
    "wall_clock_window_s",
    "token_throughput_per_s",
    "end_to_end_p50_s",
    "end_to_end_p95_s",
    "end_to_end_p99_s",
    "queue_wait_p50_s",
    "queue_wait_p95_s",
    "queue_wait_p99_s",
    "queue_wait_mean_s",
    "longest_queue_wait_s",
    "clip_applied_count",
    "clip_applied_rate",
    "realized_truncation_count",
    "realized_truncation_rate",
    "mean_cap_reduction_tokens",
    "slo_overall_rate",
    "slo_interactive_rate",
    "slo_batch_rate",
)


class ExpLIntegrityError(ValueError):
    """The persisted aggregate and raw expL facts cannot describe the same run."""


class PreRegistrationFailure(ValueError):
    """A frozen expL success claim has failed or is not interpretable."""


@dataclass(frozen=True, slots=True)
class IntervalEstimate:
    """A seed-cluster point estimate and deterministic percentile interval."""

    estimate: float
    low_95: float
    high_95: float
    seed_count: int


@dataclass(frozen=True, slots=True)
class GateResult:
    """One named preregistered gate and its auditable diagnostics."""

    name: str
    passed: bool
    diagnostic: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DecisionFact:
    """One validated dispatch-side adaptive decision loaded from JSONL."""

    request_id: str
    repetition_index: int
    decision_time_s: float
    q_inst: int
    q_bar: float
    old_level: AdaptiveClipLevel
    new_level: AdaptiveClipLevel
    selected_cap: int | None
    trigger_reason: AdaptiveClipTrigger

    @property
    def pressure(self) -> float:
        """Return the persisted dual-time-scale pressure."""
        return max(float(self.q_inst), self.q_bar)


@dataclass(frozen=True, slots=True)
class LowLoadEvidence:
    """Deterministic semantics and paired-throughput evidence for the low-load gate."""

    clip_applied_count: int
    realized_truncation_count: int
    every_backend_cap_matches_request: bool
    every_decision_is_l0: bool
    paired_trace_equal: bool
    no_clip_throughput_by_seed: SeedValues
    adaptive_throughput_by_seed: SeedValues


def _nearest_rank(values: Sequence[float], proportion: float) -> float:
    if not values:
        raise ValueError("a percentile requires at least one value")
    ordered = sorted(values)
    return ordered[math.ceil(proportion * len(ordered)) - 1]


def _aligned_seeds(*samples: SeedValues) -> tuple[int, ...]:
    if not samples:
        raise ValueError("paired statistics require at least one sample")
    seed_sets = [set(sample) for sample in samples]
    if not seed_sets[0] or any(seeds != seed_sets[0] for seeds in seed_sets[1:]):
        raise ValueError("paired statistics require identical non-empty seed sets")
    return tuple(sorted(seed_sets[0]))


def _bootstrap_seed_statistic(
    samples: Sequence[SeedValues],
    statistic: Callable[[tuple[int, ...]], float],
) -> IntervalEstimate:
    seeds = _aligned_seeds(*samples)
    point = statistic(seeds)
    # Exact enumeration is preferable for expL's registered three seeds: it is deterministic and
    # makes every resampled unit a whole seed cluster, never an individual request.
    if len(seeds) <= 6:
        resamples = itertools.product(seeds, repeat=len(seeds))
    else:
        raise ValueError("exact seed bootstrap supports at most six seed clusters")
    values = [statistic(tuple(resample)) for resample in resamples]
    return IntervalEstimate(
        estimate=point,
        low_95=_nearest_rank(values, 0.025),
        high_95=_nearest_rank(values, 0.975),
        seed_count=len(seeds),
    )


def cluster_mean_interval(values_by_seed: SeedValues) -> IntervalEstimate:
    """Bootstrap a mean by resampling whole seed clusters."""

    return _bootstrap_seed_statistic(
        (values_by_seed,),
        lambda seeds: statistics.fmean(values_by_seed[seed] for seed in seeds),
    )


def paired_difference_interval(
    treatment_by_seed: SeedValues,
    baseline_by_seed: SeedValues,
) -> IntervalEstimate:
    """Return treatment-minus-baseline with a paired seed-cluster interval."""

    return _bootstrap_seed_statistic(
        (treatment_by_seed, baseline_by_seed),
        lambda seeds: statistics.fmean(
            treatment_by_seed[seed] - baseline_by_seed[seed] for seed in seeds
        ),
    )


def paired_ratio_interval(
    numerator_by_seed: SeedValues,
    denominator_by_seed: SeedValues,
) -> IntervalEstimate:
    """Return the mean paired ratio with a paired seed-cluster interval."""

    seeds = _aligned_seeds(numerator_by_seed, denominator_by_seed)
    if any(denominator_by_seed[seed] <= 0.0 for seed in seeds):
        raise ValueError("paired ratios require positive denominators")
    return _bootstrap_seed_statistic(
        (numerator_by_seed, denominator_by_seed),
        lambda resampled: statistics.fmean(
            numerator_by_seed[seed] / denominator_by_seed[seed] for seed in resampled
        ),
    )


def check_low_load_gate(evidence: LowLoadEvidence) -> dict[str, Any]:
    """Enforce the two deterministic §5.3 claims and the paired GPU equivalence interval."""

    deterministic_failures: list[str] = []
    if evidence.clip_applied_count != 0:
        deterministic_failures.append("clip_applied_count is non-zero")
    if evidence.realized_truncation_count != 0:
        deterministic_failures.append("realized_truncation_count is non-zero")
    if not evidence.every_backend_cap_matches_request:
        deterministic_failures.append("a backend cap differs from its requested output")
    if not evidence.every_decision_is_l0:
        deterministic_failures.append("a decision is not L0")
    if not evidence.paired_trace_equal:
        deterministic_failures.append("the no-clip and adaptive traces differ")
    ratio = paired_ratio_interval(
        evidence.adaptive_throughput_by_seed,
        evidence.no_clip_throughput_by_seed,
    )
    if ratio.low_95 < 0.95 or ratio.high_95 > 1.05:
        deterministic_failures.append("paired throughput-ratio interval leaves [0.95, 1.05]")
    if deterministic_failures:
        raise PreRegistrationFailure(
            "low-load zero-utility gate failed: " + "; ".join(deterministic_failures)
        )
    return {"paired_token_throughput_ratio": asdict(ratio)}


def check_depth_eligibility(
    replay: Mapping[str, Any],
    *,
    tighten_hold_s: float,
) -> dict[str, Any]:
    """Reject an empty controller configuration before the formal sweep."""

    eligible = replay.get("eligible_levels")
    if eligible != {"L1": True, "L2": True, "L3": True}:
        raise PreRegistrationFailure(
            "default L1/L2/L3 do not all have non-zero trigger eligibility"
        )
    if not replay.get("l3_reachable", False):
        raise PreRegistrationFailure("L3 is not reachable in the saved expK-B depth trace")
    episode_maxima = replay.get("episode_max_s")
    if not isinstance(episode_maxima, dict) or not any(
        float(value) >= tighten_hold_s for value in episode_maxima.values()
    ):
        raise PreRegistrationFailure(
            "all saved high-load pressure episodes are shorter than tighten hold"
        )
    return dict(replay)


def check_latency_benefit(
    no_clip_wait_by_seed: SeedValues,
    adaptive_wait_by_seed: SeedValues,
    no_clip_interactive_slo_by_seed: SeedValues,
    adaptive_interactive_slo_by_seed: SeedValues,
) -> dict[str, Any]:
    """Require stable high-load improvements in both mean wait and interactive SLO."""

    wait_improvement = paired_difference_interval(no_clip_wait_by_seed, adaptive_wait_by_seed)
    slo_improvement = paired_difference_interval(
        adaptive_interactive_slo_by_seed, no_clip_interactive_slo_by_seed
    )
    if wait_improvement.low_95 <= 0.0 or slo_improvement.low_95 <= 0.0:
        raise PreRegistrationFailure(
            "high-load latency benefit is not stable: a paired 95% interval contains zero"
        )
    return {
        "mean_queue_wait_reduction": asdict(wait_improvement),
        "interactive_slo_increase": asdict(slo_improvement),
    }


def check_benefit_retention(
    no_clip_wait_by_seed: SeedValues,
    fixed_wait_by_seed: SeedValues,
    adaptive_wait_by_seed: SeedValues,
    no_clip_slo_by_seed: SeedValues,
    fixed_slo_by_seed: SeedValues,
    adaptive_slo_by_seed: SeedValues,
) -> dict[str, Any]:
    """Require at least 75% of fixed-512's wait and SLO benefit."""

    seeds = _aligned_seeds(
        no_clip_wait_by_seed,
        fixed_wait_by_seed,
        adaptive_wait_by_seed,
        no_clip_slo_by_seed,
        fixed_slo_by_seed,
        adaptive_slo_by_seed,
    )

    def retention(resampled: tuple[int, ...], *, slo: bool) -> float:
        no_clip = no_clip_slo_by_seed if slo else no_clip_wait_by_seed
        fixed = fixed_slo_by_seed if slo else fixed_wait_by_seed
        adaptive = adaptive_slo_by_seed if slo else adaptive_wait_by_seed
        if slo:
            numerator = statistics.fmean(adaptive[seed] - no_clip[seed] for seed in resampled)
            denominator = statistics.fmean(fixed[seed] - no_clip[seed] for seed in resampled)
        else:
            numerator = statistics.fmean(no_clip[seed] - adaptive[seed] for seed in resampled)
            denominator = statistics.fmean(no_clip[seed] - fixed[seed] for seed in resampled)
        if denominator <= 0.0:
            raise ZeroDivisionError
        return numerator / denominator

    wait_denominator = statistics.fmean(
        no_clip_wait_by_seed[seed] - fixed_wait_by_seed[seed] for seed in seeds
    )
    slo_denominator = statistics.fmean(
        fixed_slo_by_seed[seed] - no_clip_slo_by_seed[seed] for seed in seeds
    )
    if wait_denominator <= 0.0 or slo_denominator <= 0.0:
        raise PreRegistrationFailure(
            "benefit retention is uninterpretable: fixed-512 has a non-positive "
            "wait or SLO denominator"
        )
    wait_point = retention(seeds, slo=False)
    slo_point = retention(seeds, slo=True)
    if wait_point < 0.75 or slo_point < 0.75:
        raise PreRegistrationFailure(
            "adaptive benefit retention is below the frozen 0.75 threshold"
        )

    # Bootstrap samples with a non-positive denominator are explicitly counted as
    # uninterpretable diagnostics rather than silently dropped from a confidence interval.
    wait_bootstrap: list[float] = []
    slo_bootstrap: list[float] = []
    invalid_wait = invalid_slo = 0
    for resampled in itertools.product(seeds, repeat=len(seeds)):
        try:
            wait_bootstrap.append(retention(tuple(resampled), slo=False))
        except ZeroDivisionError:
            invalid_wait += 1
        try:
            slo_bootstrap.append(retention(tuple(resampled), slo=True))
        except ZeroDivisionError:
            invalid_slo += 1
    return {
        "wait_retention": wait_point,
        "slo_retention": slo_point,
        "wait_bootstrap_95": (
            [_nearest_rank(wait_bootstrap, 0.025), _nearest_rank(wait_bootstrap, 0.975)]
            if wait_bootstrap
            else None
        ),
        "slo_bootstrap_95": (
            [_nearest_rank(slo_bootstrap, 0.025), _nearest_rank(slo_bootstrap, 0.975)]
            if slo_bootstrap
            else None
        ),
        "uninterpretable_bootstrap_samples": {
            "wait": invalid_wait,
            "slo": invalid_slo,
        },
    }


def check_utility_recovery(
    fixed_throughput_by_seed: SeedValues,
    adaptive_throughput_by_seed: SeedValues,
    fixed_truncation_by_seed: SeedValues,
    adaptive_truncation_by_seed: SeedValues,
) -> dict[str, Any]:
    """Require stable throughput recovery or stable truncation-rate reduction."""

    throughput_gain = paired_difference_interval(
        adaptive_throughput_by_seed, fixed_throughput_by_seed
    )
    truncation_reduction = paired_difference_interval(
        fixed_truncation_by_seed, adaptive_truncation_by_seed
    )
    if throughput_gain.low_95 <= 0.0 and truncation_reduction.low_95 <= 0.0:
        raise PreRegistrationFailure(
            "adaptive clipping neither stably improves throughput nor stably reduces truncation"
        )
    return {
        "token_throughput_gain": asdict(throughput_gain),
        "realized_truncation_rate_reduction": asdict(truncation_reduction),
    }


def _transition_facts(decisions: Sequence[DecisionFact]) -> list[DecisionFact]:
    return [decision for decision in decisions if decision.new_level != decision.old_level]


def check_thrash(
    decisions: Sequence[DecisionFact],
    *,
    formal_dispatch_count: int,
    tighten_hold_s: float,
    relax_hold_s: float,
) -> dict[str, Any]:
    """Enforce reverse-transition hold times and the frozen 10% switch ceiling."""

    if formal_dispatch_count <= 0:
        raise ValueError("formal_dispatch_count must be positive")
    transitions = _transition_facts(decisions)
    reverse_intervals: list[dict[str, Any]] = []
    implementation_errors: list[str] = []
    cycles: list[float] = []
    for previous, current in itertools.pairwise(transitions):
        if previous.repetition_index != current.repetition_index:
            continue
        previous_direction = 1 if previous.new_level > previous.old_level else -1
        current_direction = 1 if current.new_level > current.old_level else -1
        if previous_direction == current_direction:
            continue
        interval_s = current.decision_time_s - previous.decision_time_s
        required_s = relax_hold_s if previous_direction > 0 else tighten_hold_s
        reverse_intervals.append(
            {
                "from": "tighten" if previous_direction > 0 else "relax",
                "to": "relax" if current_direction < 0 else "tighten",
                "interval_s": interval_s,
                "required_hold_s": required_s,
            }
        )
        if interval_s < required_s:
            implementation_errors.append(
                f"reverse interval {interval_s} is shorter than required hold {required_s}"
            )
    for first, second, third in zip(transitions, transitions[1:], transitions[2:], strict=False):
        if not (
            first.repetition_index == second.repetition_index == third.repetition_index
            and first.new_level > first.old_level
            and second.new_level < second.old_level
            and third.new_level > third.old_level
        ):
            continue
        cycles.append(third.decision_time_s - first.decision_time_s)
    if implementation_errors:
        raise PreRegistrationFailure(
            "control transition hold invariant failed: " + "; ".join(implementation_errors)
        )
    if len(transitions) > formal_dispatch_count * 0.10:
        raise PreRegistrationFailure("adaptive level switches exceed 10% of formal dispatches")
    return {
        "transition_count": len(transitions),
        "formal_dispatch_count": formal_dispatch_count,
        "switch_fraction": len(transitions) / formal_dispatch_count,
        "reverse_intervals": reverse_intervals,
        "tighten_relax_retighten_cycles_s": cycles,
    }


def check_directionality(decisions: Sequence[DecisionFact]) -> dict[str, Any]:
    """Reject a looser recorded pressure/level response when instantaneous depth rises."""

    anomalies: list[dict[str, Any]] = []
    by_repetition: dict[int, list[DecisionFact]] = defaultdict(list)
    for decision in decisions:
        by_repetition[decision.repetition_index].append(decision)
    for repetition, facts in by_repetition.items():
        ordered = sorted(facts, key=lambda fact: fact.decision_time_s)
        for previous, current in itertools.pairwise(ordered):
            if current.q_inst > previous.q_inst and (
                current.pressure < previous.pressure or current.new_level < previous.new_level
            ):
                anomalies.append(
                    {
                        "repetition_index": repetition,
                        "previous_time_s": previous.decision_time_s,
                        "current_time_s": current.decision_time_s,
                    }
                )
    if anomalies:
        raise PreRegistrationFailure("q_inst rose while persisted pressure or level became looser")
    return {"anomaly_count": 0}


def read_decision_facts(path: str | Path) -> tuple[DecisionFact, ...]:
    """Read a decision sidecar strictly; malformed or duplicate identities are fatal."""

    facts: list[DecisionFact] = []
    identities: set[tuple[str, int]] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("decision must be an object")
                expected = {
                    "request_id",
                    "repetition_index",
                    "decision_time_s",
                    "q_inst",
                    "q_bar",
                    "old_level",
                    "new_level",
                    "selected_cap",
                    "trigger_reason",
                }
                if set(raw) != expected:
                    raise ValueError("decision fields differ")
                fact = DecisionFact(
                    request_id=str(raw["request_id"]),
                    repetition_index=int(raw["repetition_index"]),
                    decision_time_s=float(raw["decision_time_s"]),
                    q_inst=int(raw["q_inst"]),
                    q_bar=float(raw["q_bar"]),
                    old_level=AdaptiveClipLevel[str(raw["old_level"])],
                    new_level=AdaptiveClipLevel[str(raw["new_level"])],
                    selected_cap=(
                        None if raw["selected_cap"] is None else int(raw["selected_cap"])
                    ),
                    trigger_reason=AdaptiveClipTrigger(str(raw["trigger_reason"])),
                )
                if (
                    not fact.request_id
                    or fact.repetition_index < 0
                    or fact.q_inst < 0
                    or not math.isfinite(fact.decision_time_s)
                    or not math.isfinite(fact.q_bar)
                    or fact.q_bar < 0.0
                ):
                    raise ValueError("decision values are invalid")
                identity = (fact.request_id, fact.repetition_index)
                if identity in identities:
                    raise ExpLIntegrityError(f"duplicate adaptive decision identity: {identity}")
                identities.add(identity)
                facts.append(fact)
            except ExpLIntegrityError:
                raise
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ExpLIntegrityError(
                    f"invalid adaptive decision at JSONL line {line_number}"
                ) from exc
    return tuple(facts)


def _metrics_mapping(metrics: MetricsSummary) -> dict[str, int | float | None]:
    interactive = metrics.slo_by_class.get(RequestClass.INTERACTIVE)
    batch = metrics.slo_by_class.get(RequestClass.BATCH)
    return {
        "request_count": metrics.request_count,
        "success_count": metrics.success_count,
        "error_count": metrics.error_count,
        "timeout_count": metrics.timeout_count,
        "cancelled_count": metrics.cancelled_count,
        "rejected_count": metrics.rejected_count,
        "success_output_tokens": metrics.success_output_tokens,
        "wall_clock_window_s": metrics.wall_clock_window_s,
        "token_throughput_per_s": metrics.token_throughput_per_s,
        "end_to_end_p50_s": metrics.end_to_end_s.p50,
        "end_to_end_p95_s": metrics.end_to_end_s.p95,
        "end_to_end_p99_s": metrics.end_to_end_s.p99,
        "queue_wait_p50_s": metrics.queue_wait_s.p50,
        "queue_wait_p95_s": metrics.queue_wait_s.p95,
        "queue_wait_p99_s": metrics.queue_wait_s.p99,
        "queue_wait_mean_s": metrics.queue_wait_mean_s,
        "longest_queue_wait_s": metrics.longest_queue_wait_s,
        "clip_applied_count": metrics.clip_applied_count,
        "clip_applied_rate": metrics.clip_applied_rate,
        "realized_truncation_count": metrics.realized_truncation_count,
        "realized_truncation_rate": metrics.realized_truncation_rate,
        "mean_cap_reduction_tokens": metrics.mean_cap_reduction_tokens,
        "slo_overall_rate": metrics.slo_overall.rate,
        "slo_interactive_rate": interactive.rate if interactive is not None else None,
        "slo_batch_rate": batch.rate if batch is not None else None,
    }


def reconcile_point(
    row: Mapping[str, str],
    records: Sequence[RequestRecord],
    config: ExperimentConfig,
) -> dict[str, int | float | None]:
    """Recompute a point from request facts and require byte-derived aggregates to agree."""

    identities = [(record.request_id, record.repetition_index) for record in records]
    if len(set(identities)) != len(identities):
        raise ExpLIntegrityError("duplicate request identity in request JSONL")
    if any(
        record.requested_output_tokens is None or record.backend_max_output_tokens is None
        for record in records
    ):
        raise ExpLIntegrityError("expL request facts lack cap instrumentation")
    if config.workload.realistic_length is None:
        raise ExpLIntegrityError("expL requires the realistic workload")
    if config.workload.realistic_length.force_exact_output_tokens:
        for record in records:
            if record.status is DispatchStatus.SUCCESS and (
                record.output_tokens != record.backend_max_output_tokens
                or record.finish_reason != "length"
            ):
                raise ExpLIntegrityError(
                    "force-exact success disagrees with backend cap or finish_reason"
                )
    metrics = calculate_metrics(records, config.workload)
    recomputed = _metrics_mapping(metrics)
    for field in _RECONCILED_FIELDS:
        if field not in row:
            raise ExpLIntegrityError(f"sweep CSV is missing reconciliation field {field}")
        expected = recomputed[field]
        raw = row[field]
        if expected is None:
            if raw not in ("", None):
                raise ExpLIntegrityError(f"aggregate mismatch for {field}: raw is null")
            continue
        try:
            observed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ExpLIntegrityError(f"aggregate {field} is not numeric") from exc
        if not math.isclose(observed, float(expected), rel_tol=1e-9, abs_tol=1e-9):
            raise ExpLIntegrityError(
                f"aggregate mismatch for {field}: CSV={observed}, raw={expected}"
            )
    return recomputed


def reconcile_decisions(
    records: Sequence[RequestRecord],
    decisions: Sequence[DecisionFact],
) -> None:
    dispatched = {
        (record.request_id, record.repetition_index): record
        for record in records
        if record.dispatch_time_s is not None
    }
    decision_map = {
        (decision.request_id, decision.repetition_index): decision for decision in decisions
    }
    if set(decision_map) != set(dispatched):
        missing = sorted(set(dispatched) - set(decision_map))
        extra = sorted(set(decision_map) - set(dispatched))
        raise ExpLIntegrityError(
            f"adaptive decisions do not match dispatched requests: missing={missing}, extra={extra}"
        )
    for identity, decision in decision_map.items():
        record = dispatched[identity]
        requested = record.requested_output_tokens
        backend = record.backend_max_output_tokens
        if requested is None or backend is None:
            raise ExpLIntegrityError("decision join found uninstrumented request")
        allowed_caps = {requested}
        if decision.selected_cap is not None and decision.selected_cap < requested:
            allowed_caps.add(decision.selected_cap)
        if backend not in allowed_caps:
            raise ExpLIntegrityError("request effective cap disagrees with its adaptive decision")
        if decision.new_level is AdaptiveClipLevel.L0 and decision.selected_cap is not None:
            raise ExpLIntegrityError("L0 decision selected a cap")
        if decision.new_level is not AdaptiveClipLevel.L0 and decision.selected_cap is None:
            raise ExpLIntegrityError("a non-L0 decision omitted its cap")


def load_and_reconcile_decisions(
    result_dir: str | Path,
    label: str,
    records: Sequence[RequestRecord],
) -> tuple[DecisionFact, ...]:
    """Require one adaptive sidecar and join every decision to one dispatched request."""

    sidecar = _one_matching_file(Path(result_dir), label, "-adaptive-cap-decisions.jsonl")
    decisions = read_decision_facts(sidecar)
    reconcile_decisions(records, decisions)
    return decisions


def _duration_summary(values: Sequence[float]) -> dict[str, int | float | None]:
    return {
        "count": len(values),
        "p50_s": _nearest_rank(values, 0.50) if values else None,
        "p95_s": _nearest_rank(values, 0.95) if values else None,
        "min_s": min(values) if values else None,
        "max_s": max(values) if values else None,
    }


def summarize_controller_health(
    records: Sequence[RequestRecord],
    decisions: Sequence[DecisionFact],
    *,
    warmup_requests: int,
    tighten_hold_s: float,
    relax_hold_s: float,
) -> dict[str, Any]:
    """Report residence, transitions, reverse cycles, and controller response latencies."""

    formal_identities = {
        (record.request_id, record.repetition_index)
        for record in records
        if record.sequence_id >= warmup_requests and record.dispatch_time_s is not None
    }
    formal = [
        decision
        for decision in decisions
        if (decision.request_id, decision.repetition_index) in formal_identities
    ]
    by_repetition: dict[int, list[DecisionFact]] = defaultdict(list)
    for decision in formal:
        by_repetition[decision.repetition_index].append(decision)
    level_episodes: dict[AdaptiveClipLevel, list[float]] = defaultdict(list)
    entry_counts: dict[AdaptiveClipLevel, int] = defaultdict(int)
    transition_counts: dict[str, int] = defaultdict(int)
    response_latencies: list[float] = []
    zero_to_relax_latencies: list[float] = []
    reverse_intervals: list[tuple[str, float, float]] = []
    cycles: list[float] = []
    for repetition, facts in by_repetition.items():
        ordered = sorted(facts, key=lambda fact: fact.decision_time_s)
        if not ordered:
            continue
        episode_level = ordered[0].new_level
        episode_start = ordered[0].decision_time_s
        entry_counts[episode_level] += 1
        positive_start: float | None = None
        previous_q_inst = 0
        zero_after_tighten: float | None = None
        transitions: list[DecisionFact] = []
        for decision in ordered:
            if decision.q_inst > 0 and previous_q_inst == 0:
                positive_start = decision.decision_time_s
            elif decision.q_inst == 0:
                positive_start = None
            if decision.pressure == 0.0 and decision.new_level is not AdaptiveClipLevel.L0:
                zero_after_tighten = zero_after_tighten or decision.decision_time_s
            if decision.new_level != decision.old_level:
                transitions.append(decision)
                transition_counts[f"{decision.old_level.name}->{decision.new_level.name}"] += 1
                if decision.new_level > decision.old_level and positive_start is not None:
                    response_latencies.append(decision.decision_time_s - positive_start)
                    positive_start = None
                if decision.new_level < decision.old_level and zero_after_tighten is not None:
                    zero_to_relax_latencies.append(decision.decision_time_s - zero_after_tighten)
                    zero_after_tighten = None
            if decision.new_level != episode_level:
                level_episodes[episode_level].append(decision.decision_time_s - episode_start)
                episode_level = decision.new_level
                episode_start = decision.decision_time_s
                entry_counts[episode_level] += 1
            previous_q_inst = decision.q_inst
        final_dispatch = max(
            record.dispatch_time_s or ordered[-1].decision_time_s
            for record in records
            if record.repetition_index == repetition
            and record.sequence_id >= warmup_requests
            and record.dispatch_time_s is not None
        )
        level_episodes[episode_level].append(max(0.0, final_dispatch - episode_start))
        for previous, current in itertools.pairwise(transitions):
            previous_tighten = previous.new_level > previous.old_level
            current_tighten = current.new_level > current.old_level
            if previous_tighten != current_tighten:
                required = relax_hold_s if previous_tighten else tighten_hold_s
                reverse_intervals.append(
                    (
                        "tighten_to_relax" if previous_tighten else "relax_to_tighten",
                        current.decision_time_s - previous.decision_time_s,
                        required,
                    )
                )
        for first, second, third in zip(
            transitions, transitions[1:], transitions[2:], strict=False
        ):
            if (
                first.new_level > first.old_level
                and second.new_level < second.old_level
                and third.new_level > third.old_level
            ):
                cycles.append(third.decision_time_s - first.decision_time_s)
    total_residence = sum(sum(values) for values in level_episodes.values())
    residence = {
        level.name: {
            "time_s": sum(level_episodes[level]),
            "time_fraction": (
                sum(level_episodes[level]) / total_residence if total_residence > 0.0 else 0.0
            ),
            "entry_count": entry_counts[level],
            "episodes": _duration_summary(level_episodes[level]),
        }
        for level in _LEVELS
    }
    normalized_bins = {
        "below_1x_hold": 0,
        "from_1x_to_2x_hold": 0,
        "above_2x_hold": 0,
    }
    for _, interval, hold in reverse_intervals:
        if interval < hold:
            normalized_bins["below_1x_hold"] += 1
        elif interval <= 2.0 * hold:
            normalized_bins["from_1x_to_2x_hold"] += 1
        else:
            normalized_bins["above_2x_hold"] += 1
    return {
        "level_residence": residence,
        "directed_transition_counts": dict(sorted(transition_counts.items())),
        "reverse_transition_intervals": [
            {"direction": direction, "interval_s": interval, "hold_s": hold}
            for direction, interval, hold in reverse_intervals
        ],
        "reverse_interval_hold_bins": normalized_bins,
        "tighten_relax_retighten_cycles": _duration_summary(cycles),
        "first_positive_depth_to_first_tighten": _duration_summary(response_latencies),
        "pressure_zero_to_relax": _duration_summary(zero_to_relax_latencies),
        "formal_dispatch_count": len(formal),
    }


def replay_expkb_depth_eligibility(
    results: str | Path,
    *,
    warmup_requests: int = 4,
    thresholds: tuple[int, int, int] = (1, 2, 3),
) -> dict[str, Any]:
    """Rebuild §1.2 stable waiting depth from the real expK-B no-clip facts."""

    paths = sorted(Path(results).glob("sweep-*-no-clip-s*.jsonl"))
    if not paths:
        raise ExpLIntegrityError("expK-B no-clip request JSONL files are missing")
    duration_by_depth: dict[int, float] = defaultdict(float)
    episode_durations: dict[int, list[float]] = {threshold: [] for threshold in thresholds}
    peak_by_seed: dict[str, int] = {}
    observation_time = 0.0
    for path in paths:
        records = [
            record
            for record in read_request_records_jsonl(path)
            if record.sequence_id >= warmup_requests
        ]
        seed_peak = 0
        for repetition in sorted({record.repetition_index for record in records}):
            events: dict[float, int] = defaultdict(int)
            repetition_records = [
                record for record in records if record.repetition_index == repetition
            ]
            for record in repetition_records:
                if record.dispatch_time_s is None or record.enqueue_time_s > record.dispatch_time_s:
                    raise ExpLIntegrityError("expK-B depth trace has an invalid waiting interval")
                events[record.enqueue_time_s] += 1
                events[record.dispatch_time_s] -= 1
            depth = 0
            previous: float | None = None
            episode_starts: dict[int, float | None] = {threshold: None for threshold in thresholds}
            for timestamp in sorted(events):
                if previous is not None:
                    duration = timestamp - previous
                    if duration < 0.0:
                        raise ExpLIntegrityError("expK-B depth timestamps are not monotonic")
                    duration_by_depth[depth] += duration
                    observation_time += duration
                    for threshold in thresholds:
                        if depth >= threshold and episode_starts[threshold] is None:
                            episode_starts[threshold] = previous
                        elif depth < threshold and episode_starts[threshold] is not None:
                            episode_durations[threshold].append(
                                previous - float(episode_starts[threshold])
                            )
                            episode_starts[threshold] = None
                depth += events[timestamp]
                if depth < 0:
                    raise ExpLIntegrityError("expK-B depth reconstruction became negative")
                seed_peak = max(seed_peak, depth)
                previous = timestamp
            if depth != 0 or previous is None:
                raise ExpLIntegrityError("expK-B waiting intervals did not close")
            for threshold, start in episode_starts.items():
                if start is not None:
                    episode_durations[threshold].append(previous - start)
        seed_match = re.search(r"-s(\d+)\.jsonl$", path.name)
        peak_by_seed[seed_match.group(1) if seed_match else path.name] = seed_peak
    threshold_time = {
        threshold: sum(
            duration for depth, duration in duration_by_depth.items() if depth >= threshold
        )
        for threshold in thresholds
    }
    return {
        "source_files": [path.name for path in paths],
        "request_scope": f"formal requests with sequence_id >= {warmup_requests}",
        "observation_time_s": observation_time,
        "depth_time_s": {
            str(depth): duration_by_depth[depth] for depth in sorted(duration_by_depth)
        },
        "depth_time_fraction": {
            str(depth): duration_by_depth[depth] / observation_time
            for depth in sorted(duration_by_depth)
        },
        "threshold_time_s": {str(key): value for key, value in threshold_time.items()},
        "threshold_time_fraction": {
            str(key): value / observation_time for key, value in threshold_time.items()
        },
        "episode_count": {str(key): len(value) for key, value in episode_durations.items()},
        "episode_max_s": {
            str(key): max(value) if value else 0.0 for key, value in episode_durations.items()
        },
        "peak_depth_by_seed_suffix": peak_by_seed,
        "eligible_levels": {
            f"L{index}": threshold_time[threshold] > 0.0
            for index, threshold in enumerate(thresholds, start=1)
        },
        "l3_reachable": threshold_time[thresholds[2]] > 0.0,
    }


def _point_identity(point: SweepPoint) -> tuple[str, str]:
    match = _LABEL_PATTERN.fullmatch(point.label)
    if match is None:
        raise ExpLIntegrityError(f"expL label does not encode load and arm: {point.label}")
    return match.group(1), match.group(2)


def _one_matching_file(result_dir: Path, label: str, suffix: str) -> Path:
    matches = tuple(result_dir.glob(f"sweep-*-{label}{suffix}"))
    if len(matches) != 1:
        raise ExpLIntegrityError(
            f"expected one {suffix} fact file for {label}, found {len(matches)}"
        )
    return matches[0]


def _capture_gate(name: str, callback: Callable[[], dict[str, Any]]) -> GateResult:
    try:
        return GateResult(name=name, passed=True, diagnostic="passed", details=callback())
    except PreRegistrationFailure as exc:
        return GateResult(name=name, passed=False, diagnostic=str(exc), details={})


def require_gate_results(gates: Sequence[GateResult | Mapping[str, Any]]) -> None:
    """Raise after an analysis artifact has recorded any failed preregistered gate."""

    failures = [
        gate.name if isinstance(gate, GateResult) else str(gate["name"])
        for gate in gates
        if not (gate.passed if isinstance(gate, GateResult) else bool(gate["passed"]))
    ]
    if failures:
        raise PreRegistrationFailure("failed preregistered expL gates: " + ", ".join(failures))


def _validate_matrix(definition: SweepDefinition) -> None:
    if len(definition.points) != 24 or len({point.label for point in definition.points}) != 24:
        raise ExpLIntegrityError("expL requires exactly 24 unique configured labels")
    counts: dict[str, int] = defaultdict(int)
    for point in definition.points:
        load, _ = _point_identity(point)
        counts[load] += 1
    if counts != {"high": 18, "low": 6}:
        raise ExpLIntegrityError(f"expL configured load counts differ: {dict(counts)}")


def analyze_expl(
    results: str | Path,
    *,
    sweep_config: str | Path,
    expkb_reference: str | Path,
    enforce_gates: bool = True,
) -> dict[str, Any]:
    """Audit and aggregate a complete expL sweep, then evaluate every frozen failure gate."""

    definition = load_sweep_config(sweep_config)
    _validate_matrix(definition)
    result_dir = Path(results)
    with (result_dir / "sweep-results.csv").open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    rows_by_label = {row["label"]: row for row in rows}
    expected_labels = {point.label for point in definition.points}
    if len(rows_by_label) != len(rows) or set(rows_by_label) != expected_labels:
        raise ExpLIntegrityError(
            "sweep CSV labels are duplicate, missing, or outside the frozen matrix"
        )

    point_payloads: list[dict[str, Any]] = []
    point_records: dict[str, tuple[RequestRecord, ...]] = {}
    point_decisions: dict[str, tuple[DecisionFact, ...]] = {}
    grouped_points: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    high_no_clip_records: list[RequestRecord] = []
    for point in definition.points:
        load, arm = _point_identity(point)
        config = config_for_point(definition.base_config, point)
        if not math.isclose(config.workload.request_rate_rps, _EXPECTED_LOADS[load]):
            raise ExpLIntegrityError(f"label/load rate mismatch for {point.label}")
        records = read_request_records_jsonl(_one_matching_file(result_dir, point.label, ".jsonl"))
        formal = tuple(
            record for record in records if record.sequence_id >= config.workload.warmup_requests
        )
        metrics = reconcile_point(rows_by_label[point.label], formal, config)
        decisions: tuple[DecisionFact, ...] = ()
        control: dict[str, Any] | None = None
        if config.admission.adaptive_clip_enabled:
            decisions = load_and_reconcile_decisions(result_dir, point.label, records)
            control = summarize_controller_health(
                records,
                decisions,
                warmup_requests=config.workload.warmup_requests,
                tighten_hold_s=config.admission.adaptive_clip_tighten_hold_s,
                relax_hold_s=config.admission.adaptive_clip_relax_hold_s,
            )
        point_records[point.label] = records
        point_decisions[point.label] = decisions
        if load == "high" and arm == "no-clip":
            high_no_clip_records.extend(formal)
        payload = {
            "label": point.label,
            "load": load,
            "request_rate_rps": config.workload.request_rate_rps,
            "arm": arm,
            "seed": config.workload.random_seed,
            "metrics": metrics,
            "control": control,
        }
        point_payloads.append(payload)
        grouped_points[(load, arm)].append(payload)

    service_fit = fit_linear_service_time(high_no_clip_records)
    grouped_payload: dict[str, Any] = {}
    metric_names = (
        "queue_wait_mean_s",
        "queue_wait_p50_s",
        "queue_wait_p95_s",
        "queue_wait_p99_s",
        "longest_queue_wait_s",
        "end_to_end_p50_s",
        "end_to_end_p95_s",
        "end_to_end_p99_s",
        "slo_interactive_rate",
        "slo_batch_rate",
        "slo_overall_rate",
        "clip_applied_rate",
        "realized_truncation_rate",
        "mean_cap_reduction_tokens",
        "success_output_tokens",
        "token_throughput_per_s",
        "success_count",
        "error_count",
        "timeout_count",
        "cancelled_count",
        "rejected_count",
    )
    for (load, arm), payloads in grouped_points.items():
        aggregates: dict[str, Any] = {}
        for metric_name in metric_names:
            values = {
                int(payload["seed"]): float(payload["metrics"][metric_name])
                for payload in payloads
                if payload["metrics"][metric_name] is not None
            }
            aggregates[metric_name] = asdict(cluster_mean_interval(values)) if values else None
        records = [
            record
            for payload in payloads
            for record in point_records[payload["label"]]
            if record.sequence_id >= definition.base_config.workload.warmup_requests
            and record.status is DispatchStatus.SUCCESS
        ]
        output_lengths = [record.output_tokens for record in records]
        theory = estimate_mg1_wait(
            output_lengths,
            arrival_rate_rps=_EXPECTED_LOADS[load],
            service_fit=service_fit,
            concurrency=definition.base_config.router.max_in_flight,
        )
        grouped_payload[f"{load}/{arm}"] = {
            "seed_count": len(payloads),
            "aggregates": aggregates,
            "completed_output_tokens_total_across_seeds": sum(
                int(payload["metrics"]["success_output_tokens"]) for payload in payloads
            ),
            "mechanism": {
                "mean_service_s": theory.mean_service_s,
                "second_moment_service_s2": theory.second_moment_service_s2,
                "utilization": theory.utilization,
                "mg1_mean_queue_wait_s": theory.mean_queue_wait_s,
                "mg1_is_qualitative_only": True,
            },
        }

    def seed_metric(load: str, arm: str, field: str) -> dict[int, float]:
        return {
            int(payload["seed"]): float(payload["metrics"][field])
            for payload in grouped_points[(load, arm)]
            if payload["metrics"][field] is not None
        }

    low_default_payloads = grouped_points[("low", "adaptive-q-default")]
    low_no_clip_by_seed = {
        int(payload["seed"]): point_records[payload["label"]]
        for payload in grouped_points[("low", "no-clip")]
    }
    paired_trace_equal = True
    for payload in low_default_payloads:
        seed = int(payload["seed"])
        adaptive_records = point_records[payload["label"]]
        baseline_records = low_no_clip_by_seed[seed]
        baseline_starts = {
            repetition: min(
                record.arrival_time_s
                for record in baseline_records
                if record.repetition_index == repetition
            )
            for repetition in {record.repetition_index for record in baseline_records}
        }
        adaptive_starts = {
            repetition: min(
                record.arrival_time_s
                for record in adaptive_records
                if record.repetition_index == repetition
            )
            for repetition in {record.repetition_index for record in adaptive_records}
        }
        baseline_trace = [
            (
                record.request_id,
                record.repetition_index,
                record.sequence_id,
                record.request_class,
                record.input_tokens,
                record.requested_output_tokens,
                record.backend_max_output_tokens,
                record.output_tokens,
                round(record.arrival_time_s - baseline_starts[record.repetition_index], 9),
            )
            for record in baseline_records
        ]
        adaptive_trace = [
            (
                record.request_id,
                record.repetition_index,
                record.sequence_id,
                record.request_class,
                record.input_tokens,
                record.requested_output_tokens,
                record.backend_max_output_tokens,
                record.output_tokens,
                round(record.arrival_time_s - adaptive_starts[record.repetition_index], 9),
            )
            for record in adaptive_records
        ]
        paired_trace_equal &= baseline_trace == adaptive_trace
    low_records = [
        record for payload in low_default_payloads for record in point_records[payload["label"]]
    ]
    low_decisions = [
        decision
        for payload in low_default_payloads
        for decision in point_decisions[payload["label"]]
    ]
    low_evidence = LowLoadEvidence(
        clip_applied_count=sum(record.clip_applied is True for record in low_records),
        realized_truncation_count=sum(
            record.clip_applied is True and record.finish_reason == "length"
            for record in low_records
        ),
        every_backend_cap_matches_request=all(
            record.backend_max_output_tokens == record.requested_output_tokens
            for record in low_records
        ),
        every_decision_is_l0=all(
            decision.new_level is AdaptiveClipLevel.L0 for decision in low_decisions
        ),
        paired_trace_equal=paired_trace_equal,
        no_clip_throughput_by_seed=seed_metric("low", "no-clip", "token_throughput_per_s"),
        adaptive_throughput_by_seed=seed_metric(
            "low", "adaptive-q-default", "token_throughput_per_s"
        ),
    )
    depth_replay = replay_expkb_depth_eligibility(expkb_reference)
    gates = [
        _capture_gate("low_load_zero_utility", lambda: check_low_load_gate(low_evidence)),
        _capture_gate(
            "depth_trigger_eligibility",
            lambda: check_depth_eligibility(depth_replay, tighten_hold_s=4.0),
        ),
        _capture_gate(
            "high_load_latency_benefit",
            lambda: check_latency_benefit(
                seed_metric("high", "no-clip", "queue_wait_mean_s"),
                seed_metric("high", "adaptive-q-default", "queue_wait_mean_s"),
                seed_metric("high", "no-clip", "slo_interactive_rate"),
                seed_metric("high", "adaptive-q-default", "slo_interactive_rate"),
            ),
        ),
        _capture_gate(
            "fixed512_benefit_retention",
            lambda: check_benefit_retention(
                seed_metric("high", "no-clip", "queue_wait_mean_s"),
                seed_metric("high", "fixed-512", "queue_wait_mean_s"),
                seed_metric("high", "adaptive-q-default", "queue_wait_mean_s"),
                seed_metric("high", "no-clip", "slo_interactive_rate"),
                seed_metric("high", "fixed-512", "slo_interactive_rate"),
                seed_metric("high", "adaptive-q-default", "slo_interactive_rate"),
            ),
        ),
        _capture_gate(
            "utility_recovery",
            lambda: check_utility_recovery(
                seed_metric("high", "fixed-512", "token_throughput_per_s"),
                seed_metric("high", "adaptive-q-default", "token_throughput_per_s"),
                seed_metric("high", "fixed-512", "realized_truncation_rate"),
                seed_metric("high", "adaptive-q-default", "realized_truncation_rate"),
            ),
        ),
    ]
    for payload in point_payloads:
        if payload["control"] is None:
            continue
        config = config_for_point(
            definition.base_config,
            next(point for point in definition.points if point.label == payload["label"]),
        )
        decisions = point_decisions[payload["label"]]
        formal_ids = {
            (record.request_id, record.repetition_index)
            for record in point_records[payload["label"]]
            if record.sequence_id >= config.workload.warmup_requests
            and record.dispatch_time_s is not None
        }
        formal_decisions = [
            decision
            for decision in decisions
            if (decision.request_id, decision.repetition_index) in formal_ids
        ]
        formal_dispatch_count = len(formal_ids)
        gates.append(
            _capture_gate(
                f"thrash/{payload['label']}",
                partial(
                    check_thrash,
                    formal_decisions,
                    formal_dispatch_count=formal_dispatch_count,
                    tighten_hold_s=config.admission.adaptive_clip_tighten_hold_s,
                    relax_hold_s=config.admission.adaptive_clip_relax_hold_s,
                ),
            )
        )
        gates.append(
            _capture_gate(
                f"directionality/{payload['label']}",
                lambda decisions=formal_decisions: check_directionality(decisions),
            )
        )
    result = {
        "scope": (
            "External FCFS admission and adaptive output-cap analysis. Seed clusters are the "
            "statistical units. The concurrency-scaled M/G/1 values are directional diagnostics "
            "only and absolute predicted seconds are not acceptance criteria."
        ),
        "baseline_service_fit": asdict(service_fit),
        "depth_replay": depth_replay,
        "points": point_payloads,
        "groups": grouped_payload,
        "gates": [asdict(gate) for gate in gates],
    }
    if enforce_gates:
        require_gate_results(gates)
    return result
