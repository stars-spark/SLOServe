"""Probe Week 7 adaptive clipping across deterministic CPU queue traces."""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sloserve.config import (
    AdmissionControlConfig,
    ArrivalProcess,
    ClipSource,
    LengthModel,
    RealisticLengthConfig,
    WorkloadConfig,
    load_config,
)
from sloserve.router.adaptive_clipping import (
    AdaptiveClipController,
    AdaptiveClipDecision,
    AdaptiveClipLevel,
    AdaptiveClipTrigger,
)
from sloserve.router.clipping import OutputClipper
from sloserve.router.models import RequestEnvelope
from sloserve.workload.generator import generate_requests
from sloserve.workload.length_predictor import LengthPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/raw/week7-highload-depth-probe/probe-current-defaults.json"
HISTORICAL_OUTPUT = PROJECT_ROOT / "results/raw/week7-lowload-depth-probe/probe.json"
EXPKB_DIRECTORY = PROJECT_ROOT / "results/raw/week6-expK-B"
SEEDS = (20250825, 11, 202)
REPETITIONS = 3
MAX_IN_FLIGHT = 4
TOTAL_REQUESTS = 24
WARMUP_REQUESTS = 4
SERVICE_SLOPE_S_PER_TOKEN = 0.01494
SERVICE_INTERCEPT_S = 0.0156
FIT_R_SQUARED = 0.9998
DEFAULT_RPS = 0.08
DEPTH_THRESHOLDS = (1, 2, 3)
LEVELS = tuple(AdaptiveClipLevel)
EVIDENCE_SCOPE = (
    "这是基于 expK-B 拟合的确定性CPU模拟; 不是真机测量。"
    " Deterministic CPU simulation using the expK-B no-clip service-time fit; "
    "this is not a real-machine measurement and must not be treated as one."
)


@dataclass(slots=True)
class ProbeClock:
    """Event-driven clock injected into the pure controller."""

    now_s: float = 0.0

    def __call__(self) -> float:
        return self.now_s


@dataclass(slots=True)
class TraceStatistics:
    """Mutable accumulator for one or more simulated traces."""

    total_time_s: float = 0.0
    depth_time_s: dict[int, float] = field(default_factory=dict)
    level_time_s: list[float] = field(default_factory=lambda: [0.0] * len(LEVELS))
    peak_depth: int = 0
    level_entry_counts: list[int] = field(default_factory=lambda: [0] * len(LEVELS))
    non_l0_entry_count: int = 0
    non_l0_decision_count: int = 0
    output_clipper_condition_met_count: int = 0
    clip_applied_count: int = 0
    tighten_hold_reset_count: int = 0
    depth_episode_durations_s: list[list[float]] = field(
        default_factory=lambda: [[] for _ in DEPTH_THRESHOLDS]
    )
    first_positive_depth_to_first_tighten_s: list[float] = field(default_factory=list)
    tightest_level: AdaptiveClipLevel = AdaptiveClipLevel.L0
    _depth_current_durations_s: list[float] = field(
        default_factory=lambda: [0.0] * len(DEPTH_THRESHOLDS)
    )
    _positive_depth_start_s: float | None = None
    _positive_episode_has_tightened: bool = False

    def add_interval(self, duration_s: float, depth: int, level: AdaptiveClipLevel) -> None:
        if duration_s < 0.0:
            raise ValueError("trace intervals must have non-negative duration")
        if duration_s == 0.0:
            return
        self.total_time_s += duration_s
        self.depth_time_s[depth] = self.depth_time_s.get(depth, 0.0) + duration_s
        self.level_time_s[int(level)] += duration_s
        self.peak_depth = max(self.peak_depth, depth)
        for index, threshold in enumerate(DEPTH_THRESHOLDS):
            if depth >= threshold:
                self._depth_current_durations_s[index] += duration_s
            else:
                self._finish_depth_episode(index)

    def observe_decision(self, decision: AdaptiveClipDecision) -> None:
        """Accumulate transition and hold facts from one controller update."""
        if decision.q_inst > 0 and self._positive_depth_start_s is None:
            self._positive_depth_start_s = decision.decision_time_s
            self._positive_episode_has_tightened = False
        if decision.new_level > decision.old_level:
            self.level_entry_counts[int(decision.new_level)] += 1
            if decision.old_level is AdaptiveClipLevel.L0:
                self.non_l0_entry_count += 1
            if (
                self._positive_depth_start_s is not None
                and not self._positive_episode_has_tightened
            ):
                self.first_positive_depth_to_first_tighten_s.append(
                    decision.decision_time_s - self._positive_depth_start_s
                )
                self._positive_episode_has_tightened = True
        if decision.new_level is not AdaptiveClipLevel.L0:
            self.non_l0_decision_count += 1
        if decision.trigger_reason is AdaptiveClipTrigger.TIGHTEN_HOLD_RESET:
            self.tighten_hold_reset_count += 1
        self.tightest_level = max(self.tightest_level, decision.new_level)

    def observe_depth(self, previous_depth: int, depth: int, now_s: float) -> None:
        """Track starts and ends needed for first-response latency."""
        if previous_depth == 0 and depth > 0 and self._positive_depth_start_s is None:
            self._positive_depth_start_s = now_s
            self._positive_episode_has_tightened = False
        elif previous_depth > 0 and depth == 0:
            self._positive_depth_start_s = None
            self._positive_episode_has_tightened = False

    def finish_episodes(self) -> None:
        """Close episodes that reach the final-dispatch boundary."""
        for index in range(len(DEPTH_THRESHOLDS)):
            self._finish_depth_episode(index)

    def _finish_depth_episode(self, index: int) -> None:
        duration_s = self._depth_current_durations_s[index]
        if duration_s > 0.0:
            self.depth_episode_durations_s[index].append(duration_s)
            self._depth_current_durations_s[index] = 0.0

    def merge(self, other: TraceStatistics) -> None:
        self.total_time_s += other.total_time_s
        for depth, duration_s in other.depth_time_s.items():
            self.depth_time_s[depth] = self.depth_time_s.get(depth, 0.0) + duration_s
        for index in range(len(LEVELS)):
            self.level_time_s[index] += other.level_time_s[index]
            self.level_entry_counts[index] += other.level_entry_counts[index]
        self.peak_depth = max(self.peak_depth, other.peak_depth)
        self.non_l0_entry_count += other.non_l0_entry_count
        self.non_l0_decision_count += other.non_l0_decision_count
        self.output_clipper_condition_met_count += other.output_clipper_condition_met_count
        self.clip_applied_count += other.clip_applied_count
        self.tighten_hold_reset_count += other.tighten_hold_reset_count
        for index in range(len(DEPTH_THRESHOLDS)):
            self.depth_episode_durations_s[index].extend(other.depth_episode_durations_s[index])
        self.first_positive_depth_to_first_tighten_s.extend(
            other.first_positive_depth_to_first_tighten_s
        )
        self.tightest_level = max(self.tightest_level, other.tightest_level)


def _service_time_s(output_tokens: int) -> float:
    return SERVICE_SLOPE_S_PER_TOKEN * output_tokens + SERVICE_INTERCEPT_S


def _workload_for_seed(seed: int, request_rate_rps: float) -> WorkloadConfig:
    base = load_config(PROJECT_ROOT / "configs/base.yaml").workload
    return WorkloadConfig.model_validate(
        {
            **base.model_dump(),
            "arrival_process": ArrivalProcess.POISSON,
            "request_rate_rps": request_rate_rps,
            "total_requests": TOTAL_REQUESTS,
            "warmup_requests": WARMUP_REQUESTS,
            "repetitions": REPETITIONS,
            "interactive_fraction": 0.5,
            "random_seed": seed,
            "length_model": LengthModel.REALISTIC,
            "realistic_length": RealisticLengthConfig(force_exact_output_tokens=True).model_dump(),
        }
    )


def _adaptive_config(*, tighten_hold_s: float | None = None) -> AdmissionControlConfig:
    """Enable the probe while inheriting every controller default from the model."""
    config = AdmissionControlConfig(
        clip_enabled=True,
        clip_source=ClipSource.LEARNED,
        clip_estimator_path=str(PROJECT_ROOT / "results/artifacts/length-predictor.json"),
        adaptive_clip_enabled=True,
    )
    if tighten_hold_s is not None:
        config = config.model_copy(update={"adaptive_clip_tighten_hold_s": tighten_hold_s})
    return config


def _simulate_trace(
    requests: tuple[RequestEnvelope, ...],
    *,
    admission_config: AdmissionControlConfig,
    predictor: LengthPredictor,
) -> TraceStatistics:
    clock = ProbeClock()
    controller = AdaptiveClipController(admission_config, clock=clock)
    clipper = OutputClipper(admission_config)
    waiting: list[RequestEnvelope] = []
    completions: list[tuple[float, int]] = []
    next_arrival = 0
    completion_order = 0
    current_depth = 0
    current_level = AdaptiveClipLevel.L0
    previous_time_s = requests[0].arrival_time_s
    last_dispatch_time_s = previous_time_s
    statistics = TraceStatistics()

    def dispatch_available(now_s: float) -> None:
        nonlocal completion_order, current_depth, current_level, last_dispatch_time_s
        previous_depth = current_depth
        while waiting and len(completions) < MAX_IN_FLIGHT:
            request = waiting.pop(0)
            remaining_free_slots = MAX_IN_FLIGHT - len(completions) - 1
            stable_depth = max(0, len(waiting) - remaining_free_slots)
            clock.now_s = now_s
            decision = controller.update(stable_depth)
            statistics.observe_decision(decision)
            current_level = decision.new_level
            cap = decision.selected_cap
            if cap is not None:
                predicted_tokens = predictor.predict(
                    request.request_class,
                    request.input_tokens,
                    request.prompt_kind,
                    request.advertised_cap_tokens,
                )
                if predicted_tokens > cap and request.max_output_tokens > cap:
                    statistics.output_clipper_condition_met_count += 1
            effective = clipper.apply_adaptive_cap(request, cap)
            statistics.clip_applied_count += int(effective.clip_applied)
            completion_order += 1
            heapq.heappush(
                completions,
                (
                    now_s + _service_time_s(effective.effective_max_output_tokens),
                    completion_order,
                ),
            )
            last_dispatch_time_s = now_s
        current_depth = len(waiting)
        statistics.observe_depth(previous_depth, current_depth, now_s)
        statistics.peak_depth = max(statistics.peak_depth, current_depth)

    while next_arrival < len(requests) or waiting:
        arrival_time_s = (
            requests[next_arrival].arrival_time_s if next_arrival < len(requests) else float("inf")
        )
        completion_time_s = completions[0][0] if completions else float("inf")
        now_s = min(arrival_time_s, completion_time_s)
        statistics.add_interval(now_s - previous_time_s, current_depth, current_level)
        previous_time_s = now_s

        while completions and completions[0][0] == now_s:
            heapq.heappop(completions)
        dispatch_available(now_s)

        while next_arrival < len(requests) and requests[next_arrival].arrival_time_s == now_s:
            waiting.append(requests[next_arrival])
            next_arrival += 1
        dispatch_available(now_s)

    if previous_time_s > last_dispatch_time_s:
        raise RuntimeError("probe integrated beyond the final dispatch")
    statistics.finish_episodes()
    if statistics.output_clipper_condition_met_count != statistics.clip_applied_count:
        raise RuntimeError("OutputClipper condition count disagrees with applied clips")
    return statistics


def _fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else 0.0


def _duration_distribution(durations_s: list[float]) -> dict[str, Any]:
    ordered = sorted(durations_s)

    def nearest_rank(percentile: float) -> float | None:
        if not ordered:
            return None
        return ordered[math.ceil(percentile * len(ordered)) - 1]

    return {
        "count": len(ordered),
        "p50_s": nearest_rank(0.50),
        "p95_s": nearest_rank(0.95),
        "max_s": max(ordered) if ordered else None,
        "durations_s": ordered,
    }


def _summarize(statistics: TraceStatistics) -> dict[str, Any]:
    non_l0_time_s = sum(statistics.level_time_s[1:])
    depth_distribution = {
        str(depth): {
            "time_s": statistics.depth_time_s.get(depth, 0.0),
            "time_fraction": _fraction(
                statistics.depth_time_s.get(depth, 0.0), statistics.total_time_s
            ),
        }
        for depth in range(statistics.peak_depth + 1)
    }
    level_residence = {
        level.name: {
            "time_s": statistics.level_time_s[int(level)],
            "time_fraction": _fraction(
                statistics.level_time_s[int(level)], statistics.total_time_s
            ),
            "entry_count": statistics.level_entry_counts[int(level)],
        }
        for level in LEVELS
    }
    threshold_distributions = {
        f"depth_ge_{threshold}": _duration_distribution(statistics.depth_episode_durations_s[index])
        for index, threshold in enumerate(DEPTH_THRESHOLDS)
    }
    return {
        "total_time_s": statistics.total_time_s,
        "stable_depth_time_weighted_distribution": depth_distribution,
        "peak_stable_waiting_depth": statistics.peak_depth,
        "depth_episode_duration_distributions": threshold_distributions,
        "non_l0_time_s": non_l0_time_s,
        "non_l0_time_fraction": _fraction(non_l0_time_s, statistics.total_time_s),
        "non_l0_entry_count": statistics.non_l0_entry_count,
        "non_l0_decision_count": statistics.non_l0_decision_count,
        "level_residence": level_residence,
        "tightest_level_reached": statistics.tightest_level.name,
        "tighten_hold_reset_count": statistics.tighten_hold_reset_count,
        "first_positive_depth_to_first_tighten_distribution": _duration_distribution(
            statistics.first_positive_depth_to_first_tighten_s
        ),
        "clip_applied_count": statistics.clip_applied_count,
        "output_clipper_condition_met_count": (statistics.output_clipper_condition_met_count),
    }


def _trace_metadata(request_rate_rps: float) -> dict[str, Any]:
    return {
        "arrival_process": "poisson",
        "request_rate_rps": request_rate_rps,
        "length_model": "realistic",
        "force_exact_output_tokens": True,
        "max_in_flight": MAX_IN_FLIGHT,
        "total_requests": TOTAL_REQUESTS,
        "warmup_requests": WARMUP_REQUESTS,
        "repetitions_per_seed": REPETITIONS,
        "seeds": list(SEEDS),
        "statistics_request_scope": "warmup and formal requests on the controller timeline",
        "repetition_method": (
            "Each repetition regenerates and replays the same seeded request trace, "
            "matching run_benchmark semantics."
        ),
        "time_weighting_window": "first enqueue through final dispatch",
        "stable_depth_method": (
            "Integrate the post-fill waiting-list length over each event delta; "
            "a selected request and all immediately fillable slots are excluded."
        ),
    }


def _controller_metadata(config: AdmissionControlConfig) -> dict[str, Any]:
    return {
        "tighten_thresholds": list(config.adaptive_clip_tighten_thresholds),
        "relax_thresholds": list(config.adaptive_clip_relax_thresholds),
        "ewma_tau_s": config.adaptive_clip_ewma_tau_s,
        "tighten_hold_s": config.adaptive_clip_tighten_hold_s,
        "relax_hold_s": config.adaptive_clip_relax_hold_s,
        "caps": list(config.adaptive_clip_caps),
        "clip_source": config.clip_source.value,
        "default_source": (
            "Adaptive thresholds, EWMA tau, holds, and caps are inherited from "
            "AdmissionControlConfig; the established probe explicitly enables learned clipping."
        ),
    }


def _reconstruct_expkb_no_clip() -> dict[str, Any]:
    """Recompute queue-depth episode durations from the real expK-B no-clip facts."""
    paths = sorted(EXPKB_DIRECTORY.glob("sweep-000-no-clip-s*.jsonl"))
    if not paths:
        raise ValueError("expK-B no-clip JSONL files are missing")
    durations = {threshold: [] for threshold in DEPTH_THRESHOLDS}
    sources: list[str] = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        repetition_indices = sorted({int(row["repetition_index"]) for row in rows})
        for repetition_index in repetition_indices:
            repetition_rows = [
                row for row in rows if int(row["repetition_index"]) == repetition_index
            ]
            events: dict[float, int] = {}
            for row in repetition_rows:
                enqueue_time_s = float(row["enqueue_time_s"])
                dispatch_time = row["dispatch_time_s"]
                if dispatch_time is None:
                    raise ValueError("expK-B no-clip row is missing dispatch_time_s")
                dispatch_time_s = float(dispatch_time)
                if enqueue_time_s > dispatch_time_s:
                    raise ValueError("expK-B enqueue_time_s exceeds dispatch_time_s")
                events[enqueue_time_s] = events.get(enqueue_time_s, 0) + 1
                events[dispatch_time_s] = events.get(dispatch_time_s, 0) - 1
            for threshold in DEPTH_THRESHOLDS:
                depth = 0
                previous_time_s: float | None = None
                episode_start_s: float | None = None
                for event_time_s in sorted(events):
                    if previous_time_s is not None and event_time_s > previous_time_s:
                        if depth >= threshold and episode_start_s is None:
                            episode_start_s = previous_time_s
                        elif depth < threshold and episode_start_s is not None:
                            durations[threshold].append(previous_time_s - episode_start_s)
                            episode_start_s = None
                    depth += events[event_time_s]
                    if depth < 0:
                        raise ValueError("expK-B depth reconstruction became negative")
                    previous_time_s = event_time_s
                if depth != 0 or previous_time_s is None:
                    raise ValueError("expK-B depth reconstruction did not close cleanly")
                if episode_start_s is not None:
                    durations[threshold].append(previous_time_s - episode_start_s)
            sources.append(f"{path.name}:repetition-{repetition_index}")
    return {
        "evidence_scope": (
            "This subsection is a recomputation from real expK-B no-clip JSONL, "
            "not an output of the CPU simulation."
        ),
        "request_scope": "all requests, including four warmups per repetition",
        "quantile_method": "nearest rank",
        "sources": sources,
        "depth_episode_duration_distributions": {
            f"depth_ge_{threshold}": _duration_distribution(durations[threshold])
            for threshold in DEPTH_THRESHOLDS
        },
        "reviewer_reference_rounded": {
            "depth_ge_1": {"count": 213, "p50_s": 0.0, "p95_s": 6.1, "max_s": 43.2},
            "depth_ge_2": {"count": 10, "p50_s": 7.6, "p95_s": 42.5, "max_s": 42.5},
            "depth_ge_3": {"count": 14, "p50_s": 4.2, "p95_s": 21.8, "max_s": 21.8},
        },
        "cross_check": (
            "Counts and rounded p95/max agree. Q>=1 p50 is 0.0000557s and rounds "
            "to 0.0s. Q>=2 p50 is 7.5034s by nearest rank (7.5s at one decimal), "
            "rather than the supplied 7.6s; selecting the upper middle observation "
            "gives 7.5598s and explains the 0.1s quantile-convention difference."
        ),
    }


def _historical_baseline(predictor: LengthPredictor) -> dict[str, Any]:
    baseline = json.loads(HISTORICAL_OUTPUT.read_text(encoding="utf-8"))
    aggregate = baseline["aggregate"]
    admission_config = _adaptive_config(tighten_hold_s=0.0)
    reproduced = TraceStatistics()
    for seed in SEEDS:
        requests = generate_requests(_workload_for_seed(seed, DEFAULT_RPS))
        for _ in range(REPETITIONS):
            reproduced.merge(
                _simulate_trace(
                    requests,
                    admission_config=admission_config,
                    predictor=predictor,
                )
            )
    if reproduced.clip_applied_count != aggregate["clip_applied_count"]:
        raise ValueError("zero-hold reproduction does not match historical clip count")
    return {
        "artifact": str(HISTORICAL_OUTPUT.relative_to(PROJECT_ROOT)),
        "controller_semantics": (
            "before tighten-side hold; threshold crossing tightened immediately"
        ),
        "reproduction_tighten_hold_s": 0.0,
        "recorded_clip_applied_count": aggregate["clip_applied_count"],
        "recorded_non_l0_time_fraction": aggregate["non_l0_time_fraction"],
        "reproduced": _summarize(reproduced),
    }


def run_probe(
    request_rate_rps: float = DEFAULT_RPS,
    *,
    tighten_hold_s: float | None = None,
    include_reference_cross_check: bool = False,
) -> dict[str, Any]:
    """Run one deterministic load/hold point and return reproducible facts."""
    admission_config = _adaptive_config(tighten_hold_s=tighten_hold_s)
    predictor = LengthPredictor.load(admission_config.clip_estimator_path or "")
    aggregate = TraceStatistics()
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        requests = generate_requests(_workload_for_seed(seed, request_rate_rps))
        seed_statistics = TraceStatistics()
        for _ in range(REPETITIONS):
            seed_statistics.merge(
                _simulate_trace(
                    requests,
                    admission_config=admission_config,
                    predictor=predictor,
                )
            )
        aggregate.merge(seed_statistics)
        per_seed.append({"seed": seed, **_summarize(seed_statistics)})

    result = {
        "evidence_scope": EVIDENCE_SCOPE,
        "service_time_model": {
            "formula": "S_seconds = 0.01494 * output_tokens + 0.0156",
            "source": "expK-B no-clip fit",
            "r_squared": FIT_R_SQUARED,
        },
        "trace": _trace_metadata(request_rate_rps),
        "controller": _controller_metadata(admission_config),
        "aggregate": _summarize(aggregate),
        "per_seed": per_seed,
    }
    if request_rate_rps == DEFAULT_RPS and tighten_hold_s is None:
        result["historical_before_tighten_hold"] = _historical_baseline(predictor)
    if include_reference_cross_check:
        result["expkb_no_clip_reference_recomputation"] = _reconstruct_expkb_no_clip()
    return result


def run_matrix(
    request_rates_rps: list[float],
    tighten_hold_values_s: list[float],
) -> dict[str, Any]:
    """Run a deterministic load-by-hold matrix for sensitivity analysis."""
    runs = [
        run_probe(request_rate_rps, tighten_hold_s=tighten_hold_s)
        for tighten_hold_s in tighten_hold_values_s
        for request_rate_rps in request_rates_rps
    ]
    return {
        "evidence_scope": EVIDENCE_SCOPE,
        "matrix": {
            "request_rates_rps": request_rates_rps,
            "tighten_hold_values_s": tighten_hold_values_s,
            "all_other_controller_fields": "AdmissionControlConfig defaults",
            "seeds": list(SEEDS),
            "repetitions_per_seed": REPETITIONS,
        },
        "runs": runs,
    }


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("rps must be a finite positive number")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("hold must be a finite non-negative number")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rps",
        action="append",
        type=_positive_float,
        dest="request_rates_rps",
        help="request rate to simulate; repeat to run multiple load points",
    )
    parser.add_argument(
        "--tighten-hold-s",
        action="append",
        type=_non_negative_float,
        dest="tighten_hold_values_s",
        help="tighten hold override; repeat to run a sensitivity matrix",
    )
    parser.add_argument("--include-expkb-reference", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    request_rates_rps = args.request_rates_rps or [DEFAULT_RPS]
    tighten_hold_values_s = args.tighten_hold_values_s
    if len(request_rates_rps) == 1 and not tighten_hold_values_s:
        result = run_probe(
            request_rates_rps[0],
            include_reference_cross_check=args.include_expkb_reference,
        )
    else:
        default_hold_s = _adaptive_config().adaptive_clip_tighten_hold_s
        result = run_matrix(
            request_rates_rps,
            tighten_hold_values_s or [default_hold_s],
        )
        if args.include_expkb_reference:
            result["expkb_no_clip_reference_recomputation"] = _reconstruct_expkb_no_clip()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
