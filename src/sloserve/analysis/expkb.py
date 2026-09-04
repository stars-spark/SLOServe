"""Seed aggregation and queueing diagnostics for the expK-B clipping study."""

from __future__ import annotations

import csv
import re
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sloserve.analysis.queueing import estimate_mg1_wait, fit_linear_service_time
from sloserve.metrics import RequestRecord, read_request_records_jsonl
from sloserve.workload.dispatcher import DispatchStatus

_SEED_SUFFIX = re.compile(r"-s\d+$")
_AGGREGATE_FIELDS = (
    "queue_wait_mean_s",
    "queue_wait_p99_s",
    "end_to_end_p99_s",
    "slo_overall_rate",
    "slo_interactive_rate",
    "token_throughput_per_s",
    "clip_applied_rate",
    "realized_truncation_rate",
    "mean_cap_reduction_tokens",
)


def _arm(label: str) -> str:
    return _SEED_SUFFIX.sub("", label)


def _float_or_none(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def _aggregate(values: Sequence[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "n": len(values),
    }


def _read_formal_records(results: Path, label: str, warmup: int) -> tuple[RequestRecord, ...]:
    matches = tuple(results.glob(f"sweep-*-{label}.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one request fact file for {label}, found {len(matches)}")
    return tuple(
        record for record in read_request_records_jsonl(matches[0]) if record.sequence_id >= warmup
    )


def analyze_expkb(results: Path | Sequence[Path], *, warmup_requests: int) -> dict[str, Any]:
    """Return seed aggregates plus M/G/1 diagnostics from a completed expK-B sweep."""
    if warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    result_dirs = (results,) if isinstance(results, Path) else tuple(results)
    if not result_dirs:
        raise ValueError("expK-B analysis requires at least one results directory")

    located_rows: list[tuple[Path, dict[str, str]]] = []
    for result_dir in result_dirs:
        sweep_path = result_dir / "sweep-results.csv"
        with sweep_path.open(encoding="utf-8", newline="") as source:
            located_rows.extend((result_dir, row) for row in csv.DictReader(source))
    if not located_rows:
        raise ValueError("expK-B sweep results are empty")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    records_by_arm: dict[str, list[RequestRecord]] = defaultdict(list)
    seen_labels: set[str] = set()
    for result_dir, row in located_rows:
        label = row["label"]
        if label in seen_labels:
            raise ValueError(f"duplicate expK-B label across result directories: {label}")
        seen_labels.add(label)
        arm = _arm(label)
        grouped[arm].append(row)
        records_by_arm[arm].extend(_read_formal_records(result_dir, label, warmup_requests))

    baseline_records = records_by_arm.get("no-clip")
    if not baseline_records:
        raise ValueError("expK-B analysis requires the no-clip arm")
    service_fit = fit_linear_service_time(baseline_records)
    arrival_rate = float(grouped["no-clip"][0]["request_rate_rps"])
    # The router serves max_in_flight requests at once, so the unscaled single-server model is
    # vacuous here (utilization > 2 on every arm while the measured system is plainly stable).
    # Read the real concurrency from the run rather than assuming it.
    concurrency = int(grouped["no-clip"][0]["max_in_flight"])

    arm_results: dict[str, Any] = {}
    for arm, arm_rows in grouped.items():
        aggregates: dict[str, Any] = {}
        for field in _AGGREGATE_FIELDS:
            values = [
                parsed for row in arm_rows if (parsed := _float_or_none(row.get(field))) is not None
            ]
            aggregates[field] = _aggregate(values)
        output_lengths = [
            record.output_tokens
            for record in records_by_arm[arm]
            if record.status is DispatchStatus.SUCCESS and record.output_tokens > 0
        ]
        theory = estimate_mg1_wait(
            output_lengths,
            arrival_rate_rps=arrival_rate,
            service_fit=service_fit,
            concurrency=concurrency,
        )
        arm_results[arm] = {
            "seed_count": len(arm_rows),
            "aggregates": aggregates,
            "mg1": {
                "concurrency": theory.concurrency,
                "mean_service_s": theory.mean_service_s,
                "second_moment_service_s2": theory.second_moment_service_s2,
                "utilization": theory.utilization,
                "stable": theory.stable,
                "mean_queue_wait_s": theory.mean_queue_wait_s,
            },
        }

    return {
        "scope": (
            "M/G/1 is a qualitative trend baseline only. The real system runs "
            f"max_in_flight={concurrency} under vLLM continuous batching, not one FCFS server, so "
            "fitted service times are divided by that concurrency to keep the model non-vacuous. "
            "That bridge is not M/G/c and understates waiting; compare trends across arms, "
            "not absolute seconds."
        ),
        "arrival_rate_rps": arrival_rate,
        "concurrency": concurrency,
        "baseline_service_fit": {
            "seconds_per_output_token": service_fit.seconds_per_output_token,
            "intercept_s": service_fit.intercept_s,
            "r_squared": service_fit.r_squared,
            "sample_count": service_fit.sample_count,
        },
        "arms": arm_results,
    }
