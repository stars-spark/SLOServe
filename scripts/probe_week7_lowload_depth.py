"""Reproduce the Week 7 low-load queue-depth risk probe on CPU."""

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
from sloserve.router.adaptive_clipping import AdaptiveClipController, AdaptiveClipLevel
from sloserve.router.clipping import OutputClipper
from sloserve.router.models import RequestEnvelope
from sloserve.workload.generator import generate_requests
from sloserve.workload.length_predictor import LengthPredictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results/raw/week7-lowload-depth-probe/probe-after-tighten-hold.json"
)
HISTORICAL_OUTPUT = PROJECT_ROOT / "results/raw/week7-lowload-depth-probe/probe.json"
EXPKB_DIRECTORY = PROJECT_ROOT / "results/raw/week6-expK-B"
SEEDS = (20250825, 11, 202)
REPETITIONS = 3
MAX_IN_FLIGHT = 4
SERVICE_SLOPE_S_PER_TOKEN = 0.01494
SERVICE_INTERCEPT_S = 0.0156
FIT_R_SQUARED = 0.9998


@dataclass(slots=True)
class ProbeClock:
    """Event-driven clock injected into the pure controller."""

    now_s: float = 0.0

    def __call__(self) -> float:
        return self.now_s


@dataclass(slots=True)
class TraceStatistics:
    """Mutable accumulator for one simulated trace."""

    total_time_s: float = 0.0
    depth_ge_1_time_s: float = 0.0
    depth_ge_2_time_s: float = 0.0
    depth_ge_3_time_s: float = 0.0
    non_l0_time_s: float = 0.0
    peak_depth: int = 0
    non_l0_entry_count: int = 0
    non_l0_decision_count: int = 0
    non_l0_clip_eligible_count: int = 0
    clip_applied_count: int = 0
    depth_ge_1_episode_durations_s: list[float] = field(default_factory=list)
    non_l0_episode_durations_s: list[float] = field(default_factory=list)
    _depth_ge_1_current_duration_s: float = 0.0
    _non_l0_current_duration_s: float = 0.0

    def add_interval(self, duration_s: float, depth: int, level: AdaptiveClipLevel) -> None:
        if duration_s < 0.0:
            raise ValueError("trace intervals must have non-negative duration")
        if duration_s == 0.0:
            return
        self.total_time_s += duration_s
        self.depth_ge_1_time_s += duration_s if depth >= 1 else 0.0
        self.depth_ge_2_time_s += duration_s if depth >= 2 else 0.0
        self.depth_ge_3_time_s += duration_s if depth >= 3 else 0.0
        self.non_l0_time_s += duration_s if level is not AdaptiveClipLevel.L0 else 0.0
        self.peak_depth = max(self.peak_depth, depth)
        if depth >= 1:
            self._depth_ge_1_current_duration_s += duration_s
        else:
            self._finish_depth_ge_1_episode()
        if level is not AdaptiveClipLevel.L0:
            self._non_l0_current_duration_s += duration_s
        else:
            self._finish_non_l0_episode()

    def finish_episodes(self) -> None:
        """Close any episode that reaches the final-dispatch boundary."""
        self._finish_depth_ge_1_episode()
        self._finish_non_l0_episode()

    def _finish_depth_ge_1_episode(self) -> None:
        if self._depth_ge_1_current_duration_s > 0.0:
            self.depth_ge_1_episode_durations_s.append(self._depth_ge_1_current_duration_s)
            self._depth_ge_1_current_duration_s = 0.0

    def _finish_non_l0_episode(self) -> None:
        if self._non_l0_current_duration_s > 0.0:
            self.non_l0_episode_durations_s.append(self._non_l0_current_duration_s)
            self._non_l0_current_duration_s = 0.0

    def merge(self, other: TraceStatistics) -> None:
        for field_name in (
            "total_time_s",
            "depth_ge_1_time_s",
            "depth_ge_2_time_s",
            "depth_ge_3_time_s",
            "non_l0_time_s",
            "non_l0_entry_count",
            "non_l0_decision_count",
            "non_l0_clip_eligible_count",
            "clip_applied_count",
        ):
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        self.peak_depth = max(self.peak_depth, other.peak_depth)
        self.depth_ge_1_episode_durations_s.extend(other.depth_ge_1_episode_durations_s)
        self.non_l0_episode_durations_s.extend(other.non_l0_episode_durations_s)


def _service_time_s(output_tokens: int) -> float:
    return SERVICE_SLOPE_S_PER_TOKEN * output_tokens + SERVICE_INTERCEPT_S


def _workload_for_seed(seed: int) -> WorkloadConfig:
    base = load_config(PROJECT_ROOT / "configs/base.yaml").workload
    return WorkloadConfig.model_validate(
        {
            **base.model_dump(),
            "arrival_process": ArrivalProcess.POISSON,
            "request_rate_rps": 0.08,
            "total_requests": 24,
            "warmup_requests": 4,
            "repetitions": REPETITIONS,
            "interactive_fraction": 0.5,
            "random_seed": seed,
            "length_model": LengthModel.REALISTIC,
            "realistic_length": RealisticLengthConfig(
                advertised_cap_tokens=2048,
                clamp_max=2048,
            ).model_dump(),
        }
    )


def _adaptive_config() -> AdmissionControlConfig:
    return AdmissionControlConfig(
        clip_enabled=True,
        clip_source=ClipSource.LEARNED,
        clip_estimator_path=str(PROJECT_ROOT / "results/artifacts/length-predictor.json"),
        adaptive_clip_enabled=True,
    )


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
        while waiting and len(completions) < MAX_IN_FLIGHT:
            request = waiting.pop(0)
            remaining_free_slots = MAX_IN_FLIGHT - len(completions) - 1
            stable_depth = max(0, len(waiting) - remaining_free_slots)
            clock.now_s = now_s
            decision = controller.update(stable_depth)
            if (
                current_level is AdaptiveClipLevel.L0
                and decision.new_level is not AdaptiveClipLevel.L0
            ):
                statistics.non_l0_entry_count += 1
            current_level = decision.new_level
            if current_level is not AdaptiveClipLevel.L0:
                statistics.non_l0_decision_count += 1
                cap = decision.selected_cap
                if cap is None:
                    raise RuntimeError("non-L0 decision must select a cap")
                predicted_tokens = predictor.predict(
                    request.request_class,
                    request.input_tokens,
                    request.prompt_kind,
                    request.advertised_cap_tokens,
                )
                if predicted_tokens > cap and request.max_output_tokens > cap:
                    statistics.non_l0_clip_eligible_count += 1
            effective = clipper.apply_adaptive_cap(request, decision.selected_cap)
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


def _expkb_high_load_dominant_episode_distribution() -> dict[str, Any]:
    """Measure the longest formal Q>=1 episode in each expK-B no-clip repetition."""
    dominant_durations_s: list[float] = []
    all_episode_durations_s: list[float] = []
    sources: list[str] = []
    paths = sorted(EXPKB_DIRECTORY.glob("sweep-000-no-clip-s*.jsonl"))
    if not paths:
        raise ValueError("expK-B no-clip JSONL files are missing")
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        repetition_indices = sorted({int(row["repetition_index"]) for row in rows})
        for repetition_index in repetition_indices:
            formal_rows = [
                row
                for row in rows
                if int(row["repetition_index"]) == repetition_index and int(row["sequence_id"]) >= 4
            ]
            events: dict[float, int] = {}
            for row in formal_rows:
                enqueue_time_s = float(row["enqueue_time_s"])
                dispatch_time = row["dispatch_time_s"]
                if dispatch_time is None:
                    raise ValueError("expK-B no-clip row is missing dispatch_time_s")
                dispatch_time_s = float(dispatch_time)
                if enqueue_time_s > dispatch_time_s:
                    raise ValueError("expK-B enqueue_time_s exceeds dispatch_time_s")
                events[enqueue_time_s] = events.get(enqueue_time_s, 0) + 1
                events[dispatch_time_s] = events.get(dispatch_time_s, 0) - 1

            depth = 0
            previous_time_s: float | None = None
            episode_start_s: float | None = None
            episode_durations_s: list[float] = []
            for event_time_s in sorted(events):
                if previous_time_s is not None and event_time_s > previous_time_s:
                    if depth >= 1 and episode_start_s is None:
                        episode_start_s = previous_time_s
                    elif depth == 0 and episode_start_s is not None:
                        episode_durations_s.append(previous_time_s - episode_start_s)
                        episode_start_s = None
                depth += events[event_time_s]
                if depth < 0:
                    raise ValueError("expK-B depth reconstruction became negative")
                previous_time_s = event_time_s
            if depth != 0 or previous_time_s is None:
                raise ValueError("expK-B depth reconstruction did not close cleanly")
            if episode_start_s is not None:
                episode_durations_s.append(previous_time_s - episode_start_s)
            if not episode_durations_s:
                raise ValueError("expK-B repetition has no Q>=1 episode")
            all_episode_durations_s.extend(episode_durations_s)
            dominant_durations_s.append(max(episode_durations_s))
            sources.append(f"{path.name}:repetition-{repetition_index}")

    return {
        "definition": "longest formal-request Q>=1 episode in each no-clip repetition",
        "request_filter": "sequence_id >= 4 (exclude four warmups)",
        "sources": sources,
        "all_q_ge_1_episode_duration_distribution": _duration_distribution(all_episode_durations_s),
        **_duration_distribution(dominant_durations_s),
    }


def _historical_baseline(predictor: LengthPredictor) -> dict[str, Any]:
    baseline = json.loads(HISTORICAL_OUTPUT.read_text(encoding="utf-8"))
    aggregate = baseline["aggregate"]
    admission_config = _adaptive_config().model_copy(update={"adaptive_clip_tighten_hold_s": 0.0})
    reproduced = TraceStatistics()
    for seed in SEEDS:
        requests = generate_requests(_workload_for_seed(seed))
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
        "clip_applied_count": aggregate["clip_applied_count"],
        "non_l0_time_fraction": aggregate["non_l0_time_fraction"],
        "depth_ge_1_time_fraction": aggregate["depth_ge_1_time_fraction"],
        "depth_ge_2_time_fraction": aggregate["depth_ge_2_time_fraction"],
        "depth_ge_1_episode_duration_distribution": _duration_distribution(
            reproduced.depth_ge_1_episode_durations_s
        ),
        "non_l0_episode_duration_distribution": _duration_distribution(
            reproduced.non_l0_episode_durations_s
        ),
    }


def run_probe() -> dict[str, Any]:
    """Run all configured deterministic traces and return reproducible facts."""
    admission_config = _adaptive_config()
    predictor = LengthPredictor.load(admission_config.clip_estimator_path or "")
    aggregate = TraceStatistics()
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        requests = generate_requests(_workload_for_seed(seed))
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
        per_seed.append(
            {
                "seed": seed,
                "peak_stable_waiting_depth": seed_statistics.peak_depth,
                "depth_ge_1_time_fraction": _fraction(
                    seed_statistics.depth_ge_1_time_s,
                    seed_statistics.total_time_s,
                ),
                "depth_ge_2_time_fraction": _fraction(
                    seed_statistics.depth_ge_2_time_s,
                    seed_statistics.total_time_s,
                ),
                "depth_ge_3_time_fraction": _fraction(
                    seed_statistics.depth_ge_3_time_s,
                    seed_statistics.total_time_s,
                ),
                "non_l0_time_fraction": _fraction(
                    seed_statistics.non_l0_time_s,
                    seed_statistics.total_time_s,
                ),
                "non_l0_entry_count": seed_statistics.non_l0_entry_count,
                "non_l0_decision_count": seed_statistics.non_l0_decision_count,
                "non_l0_clip_eligible_count": seed_statistics.non_l0_clip_eligible_count,
                "clip_applied_count": seed_statistics.clip_applied_count,
                "depth_ge_1_episode_duration_distribution": _duration_distribution(
                    seed_statistics.depth_ge_1_episode_durations_s
                ),
                "non_l0_episode_duration_distribution": _duration_distribution(
                    seed_statistics.non_l0_episode_durations_s
                ),
            }
        )

    return {
        "evidence_scope": (
            "Deterministic CPU simulation using the expK-B no-clip service-time fit; "
            "this is not a real-machine measurement and must not be treated as one."
        ),
        "service_time_model": {
            "formula": "S_seconds = 0.01494 * output_tokens + 0.0156",
            "source": "expK-B no-clip fit",
            "r_squared": FIT_R_SQUARED,
        },
        "trace": {
            "arrival_process": "poisson",
            "request_rate_rps": 0.08,
            "length_model": "realistic",
            "max_in_flight": MAX_IN_FLIGHT,
            "total_requests": 24,
            "warmup_requests": 4,
            "repetitions_per_seed": REPETITIONS,
            "seeds": list(SEEDS),
            "time_weighting_window": "first enqueue through final dispatch",
            "stable_depth_method": (
                "Integrate the post-fill waiting-list length over each event delta; "
                "a selected request and all immediately fillable slots are excluded."
            ),
        },
        "controller": {
            "tighten_thresholds": list(admission_config.adaptive_clip_tighten_thresholds),
            "relax_thresholds": list(admission_config.adaptive_clip_relax_thresholds),
            "ewma_tau_s": admission_config.adaptive_clip_ewma_tau_s,
            "tighten_hold_s": admission_config.adaptive_clip_tighten_hold_s,
            "relax_hold_s": admission_config.adaptive_clip_relax_hold_s,
            "caps": list(admission_config.adaptive_clip_caps),
            "clip_source": admission_config.clip_source.value,
        },
        "aggregate": {
            "total_time_s": aggregate.total_time_s,
            "depth_ge_1_time_fraction": _fraction(
                aggregate.depth_ge_1_time_s,
                aggregate.total_time_s,
            ),
            "depth_ge_2_time_fraction": _fraction(
                aggregate.depth_ge_2_time_s,
                aggregate.total_time_s,
            ),
            "depth_ge_3_time_fraction": _fraction(
                aggregate.depth_ge_3_time_s,
                aggregate.total_time_s,
            ),
            "non_l0_time_fraction": _fraction(
                aggregate.non_l0_time_s,
                aggregate.total_time_s,
            ),
            "non_l0_entry_count": aggregate.non_l0_entry_count,
            "non_l0_decision_count": aggregate.non_l0_decision_count,
            "non_l0_clip_eligible_count": aggregate.non_l0_clip_eligible_count,
            "non_l0_has_clip_eligible_request": aggregate.non_l0_clip_eligible_count > 0,
            "clip_applied_count": aggregate.clip_applied_count,
            "depth_ge_1_episode_duration_distribution": _duration_distribution(
                aggregate.depth_ge_1_episode_durations_s
            ),
            "non_l0_episode_duration_distribution": _duration_distribution(
                aggregate.non_l0_episode_durations_s
            ),
        },
        "hold_calibration": {
            "selected_tighten_hold_s": admission_config.adaptive_clip_tighten_hold_s,
            "low_load_depth_ge_1_episode_duration_distribution": _duration_distribution(
                aggregate.depth_ge_1_episode_durations_s
            ),
            "high_load_dominant_congestion_episode_duration_distribution": (
                _expkb_high_load_dominant_episode_distribution()
            ),
        },
        "historical_before_tighten_hold": _historical_baseline(predictor),
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run_probe()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
