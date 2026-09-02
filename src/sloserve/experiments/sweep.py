"""Single-GPU evaluation sweeps over the external SLOServe router."""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml
from pydantic import Field

from sloserve.analysis.metrics import MetricsSummary
from sloserve.config import ExperimentConfig, SchedulerPolicyName, StrictModel, load_config
from sloserve.experiments.benchmark import run_benchmark
from sloserve.experiments.correctness import BackendFactory, GpuSamplerFactory
from sloserve.router.models import RequestClass

_CSV_FILENAME = "sweep-results.csv"
_JSON_FILENAME = "sweep-results.json"

SweepValue: TypeAlias = str | int | float | bool | None
SweepRow: TypeAlias = dict[str, SweepValue]

SWEEP_RESULT_COLUMNS = (
    "label",
    "policy",
    "request_rate_rps",
    "interactive_fraction",
    "max_in_flight",
    "aging_threshold_s",
    "cost_weight",
    "slack_weight",
    "waiting_weight",
    "disable_length_estimate",
    "request_count",
    "success_count",
    "error_count",
    "timeout_count",
    "cancelled_count",
    "rejected_count",
    "token_throughput_per_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "ttft_p99_s",
    "tpot_p50_s",
    "tpot_p95_s",
    "tpot_p99_s",
    "end_to_end_p50_s",
    "end_to_end_p95_s",
    "end_to_end_p99_s",
    "queue_wait_p50_s",
    "queue_wait_p95_s",
    "queue_wait_p99_s",
    "longest_queue_wait_s",
    "slo_overall_rate",
    "slo_interactive_rate",
    "slo_batch_rate",
    "slo_attainment_gap",
    "jain_fairness_index",
)


class SweepPoint(StrictModel):
    """A named set of supported overrides applied to one benchmark point."""

    label: str = Field(min_length=1)
    policy: SchedulerPolicyName | None = None
    request_rate_rps: float | None = Field(default=None, gt=0)
    interactive_fraction: float | None = Field(default=None, ge=0, le=1)
    max_in_flight: int | None = Field(default=None, ge=1)
    aging_threshold_s: float | None = Field(default=None, gt=0)
    cost_weight: float | None = Field(default=None, ge=0)
    slack_weight: float | None = Field(default=None, ge=0)
    waiting_weight: float | None = Field(default=None, ge=0)
    disable_length_estimate: bool | None = None


@dataclass(frozen=True, slots=True)
class SweepDefinition:
    """Validated base experiment configuration and ordered sweep points."""

    base_config: ExperimentConfig
    points: tuple[SweepPoint, ...]


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Aggregated rows and the two persisted result paths."""

    rows: tuple[SweepRow, ...]
    csv_path: Path
    json_path: Path


def load_sweep_config(path: str | Path) -> SweepDefinition:
    """Load sweep points plus either an inline or referenced base experiment config."""
    config_path = Path(path)
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read sweep configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in sweep configuration file: {config_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("top-level sweep configuration must be a mapping")

    sweep_raw = dict(raw)
    points_raw = sweep_raw.pop("points", None)
    if not isinstance(points_raw, list) or not points_raw:
        raise ValueError("sweep configuration points must be a non-empty list")

    base_reference = sweep_raw.pop("base_config", None)
    base_overrides = sweep_raw.pop("base_overrides", {})
    if base_reference is None:
        if base_overrides:
            raise ValueError("base_overrides requires base_config")
        base_config = ExperimentConfig.model_validate(sweep_raw)
    else:
        if sweep_raw:
            raise ValueError(f"unsupported fields alongside base_config: {sorted(sweep_raw)}")
        if not isinstance(base_reference, str) or not base_reference:
            raise ValueError("base_config must be a non-empty relative path")
        if not isinstance(base_overrides, dict):
            raise ValueError("base_overrides must be a mapping")
        referenced = load_config(config_path.parent / base_reference)
        merged = _merge_mappings(referenced.model_dump(), base_overrides)
        base_config = ExperimentConfig.model_validate(merged)
    return SweepDefinition(
        base_config=base_config,
        points=tuple(SweepPoint.model_validate(point) for point in points_raw),
    )


def _merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def _config_for_point(base_config: ExperimentConfig, point: SweepPoint) -> ExperimentConfig:
    router_updates: dict[str, object] = {}
    workload_updates: dict[str, object] = {}
    slo_updates: dict[str, object] = {}
    if point.policy is not None:
        router_updates["policy"] = point.policy
    if point.max_in_flight is not None:
        router_updates["max_in_flight"] = point.max_in_flight
    if point.request_rate_rps is not None:
        workload_updates["request_rate_rps"] = point.request_rate_rps
    if point.interactive_fraction is not None:
        workload_updates["interactive_fraction"] = point.interactive_fraction
    for field_name in (
        "aging_threshold_s",
        "cost_weight",
        "slack_weight",
        "waiting_weight",
        "disable_length_estimate",
    ):
        value = getattr(point, field_name)
        if value is not None:
            slo_updates[field_name] = value

    router = base_config.router.model_copy(update=router_updates)
    workload = base_config.workload.model_copy(update=workload_updates)
    slo_aware = base_config.slo_aware.model_copy(update=slo_updates)
    return ExperimentConfig.model_validate(
        {
            **base_config.model_dump(),
            "router": router.model_dump(),
            "workload": workload.model_dump(),
            "slo_aware": slo_aware.model_dump(),
        }
    )


def _metric_row(label: str, config: ExperimentConfig, metrics: MetricsSummary) -> SweepRow:
    interactive = metrics.slo_by_class.get(RequestClass.INTERACTIVE)
    batch = metrics.slo_by_class.get(RequestClass.BATCH)
    return {
        "label": label,
        "policy": config.router.policy.value,
        "request_rate_rps": config.workload.request_rate_rps,
        "interactive_fraction": config.workload.interactive_fraction,
        "max_in_flight": config.router.max_in_flight,
        "aging_threshold_s": config.slo_aware.aging_threshold_s,
        "cost_weight": config.slo_aware.cost_weight,
        "slack_weight": config.slo_aware.slack_weight,
        "waiting_weight": config.slo_aware.waiting_weight,
        "disable_length_estimate": config.slo_aware.disable_length_estimate,
        "request_count": metrics.request_count,
        "success_count": metrics.success_count,
        "error_count": metrics.error_count,
        "timeout_count": metrics.timeout_count,
        "cancelled_count": metrics.cancelled_count,
        "rejected_count": metrics.rejected_count,
        "token_throughput_per_s": metrics.token_throughput_per_s,
        "ttft_p50_s": metrics.ttft_s.p50,
        "ttft_p95_s": metrics.ttft_s.p95,
        "ttft_p99_s": metrics.ttft_s.p99,
        "tpot_p50_s": metrics.tpot_s.p50,
        "tpot_p95_s": metrics.tpot_s.p95,
        "tpot_p99_s": metrics.tpot_s.p99,
        "end_to_end_p50_s": metrics.end_to_end_s.p50,
        "end_to_end_p95_s": metrics.end_to_end_s.p95,
        "end_to_end_p99_s": metrics.end_to_end_s.p99,
        "queue_wait_p50_s": metrics.queue_wait_s.p50,
        "queue_wait_p95_s": metrics.queue_wait_s.p95,
        "queue_wait_p99_s": metrics.queue_wait_s.p99,
        "longest_queue_wait_s": metrics.longest_queue_wait_s,
        "slo_overall_rate": metrics.slo_overall.rate,
        "slo_interactive_rate": interactive.rate if interactive is not None else None,
        "slo_batch_rate": batch.rate if batch is not None else None,
        "slo_attainment_gap": metrics.slo_attainment_gap,
        "jain_fairness_index": metrics.jain_fairness_index,
    }


def _file_stem(index: int, label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", label).strip("-").lower() or "point"
    return f"sweep-{index:03d}-{slug}"


def _write_results(
    rows: Sequence[Mapping[str, SweepValue]], output_directory: Path
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / _CSV_FILENAME
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=SWEEP_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_directory / _JSON_FILENAME
    with json_path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(rows, output, allow_nan=False, ensure_ascii=False, indent=2)
        output.write("\n")
    return csv_path, json_path


async def run_sweep(
    *,
    base_config: ExperimentConfig,
    points: Sequence[SweepPoint],
    make_backend: BackendFactory,
    env_version: str,
    make_gpu_sampler: GpuSamplerFactory | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> SweepResult:
    """Run each point with a fresh backend and persist flat aggregate results."""
    if not env_version:
        raise ValueError("env_version must not be empty")
    if not points:
        raise ValueError("points must not be empty")

    rows: list[SweepRow] = []
    for index, point in enumerate(points):
        point_config = _config_for_point(base_config, point)
        gpu_sampler = make_gpu_sampler() if make_gpu_sampler is not None else None
        async with make_backend() as backend:
            benchmark_result = await run_benchmark(
                config=point_config,
                backend=backend,
                env_version=env_version,
                clock=clock,
                sleep=sleep,
                gpu_sampler=gpu_sampler,
                file_stem=_file_stem(index, point.label),
            )
        rows.append(_metric_row(point.label, point_config, benchmark_result.metrics))

    csv_path, json_path = _write_results(rows, base_config.metrics.output_directory)
    return SweepResult(rows=tuple(rows), csv_path=csv_path, json_path=json_path)
