"""Offline-testable benchmark orchestration for the external admission pipeline."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeAlias

from sloserve.analysis.metrics import AttainmentSummary, MetricsSummary, calculate_metrics
from sloserve.config import ExperimentConfig
from sloserve.metrics.adaptive_cap import (
    AdaptiveCapDecisionRecord,
    write_adaptive_cap_decisions_jsonl,
)
from sloserve.metrics.config_hash import experiment_config_hash
from sloserve.metrics.records import RequestRecord
from sloserve.metrics.serialization import RequestRecordPaths, write_request_records
from sloserve.router.admission import AdmissionQueue, AdmissionResult
from sloserve.router.clipping import OutputClipper
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.router.policies.factory import build_policy
from sloserve.workload.backend import AsyncRequestBackend
from sloserve.workload.generator import generate_requests
from sloserve.workload.http_backend import HttpCallTelemetry

if TYPE_CHECKING:
    from sloserve.experiments.gpu import GpuSampler

_PIPELINE_ONLY_NOTE = "仅管线验证，不作策略性能比较"  # noqa: RUF001
_GPU_NOT_COLLECTED_REASON = "未采集（无采样器，留待 7a-3/7b）"  # noqa: RUF001

JsonScalar: TypeAlias = str | int | float | bool
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class TelemetryBackend(AsyncRequestBackend, Protocol):
    """Backend contract required to join admission results with call telemetry."""

    telemetry: Mapping[str, HttpCallTelemetry]


@dataclass(frozen=True, slots=True)
class GpuSample:
    """One caller-supplied GPU observation; collection is outside this module."""

    timestamp_s: float
    utilization_percent: float
    memory_used_mib: float
    power_w: float | None


@dataclass(frozen=True, slots=True)
class ReportField:
    """A completeness field containing either a value or an explicit reason."""

    value: JsonValue | None
    missing_reason: str | None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.missing_reason is None):
            raise ValueError("report field must contain exactly one of value or missing_reason")


@dataclass(frozen=True, slots=True)
class CompletenessReport:
    """Structured statement of available and unavailable benchmark evidence."""

    scope_note: str
    request_scope: str
    metrics: dict[str, ReportField]
    gpu: dict[str, ReportField]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """In-memory results and paths produced by one benchmark invocation."""

    records: tuple[RequestRecord, ...]
    formal_records: tuple[RequestRecord, ...]
    metrics: MetricsSummary
    record_paths: RequestRecordPaths
    completeness_report: CompletenessReport
    completeness_report_path: Path
    adaptive_decisions: tuple[AdaptiveCapDecisionRecord, ...]
    adaptive_decision_path: Path | None


async def run_benchmark(
    *,
    config: ExperimentConfig,
    backend: TelemetryBackend,
    env_version: str,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    gpu_samples: Sequence[GpuSample] | None = None,
    gpu_sampler: GpuSampler | None = None,
    file_stem: str = "benchmark-requests",
) -> BenchmarkResult:
    """Run every configured repetition and persist raw facts plus completeness metadata."""
    if not env_version:
        raise ValueError("env_version must not be empty")

    config_hash = experiment_config_hash(config)
    all_records: list[RequestRecord] = []
    all_adaptive_decisions: list[AdaptiveCapDecisionRecord] = []
    output_clipper = OutputClipper(config.admission)

    async with AsyncExitStack() as sampler_stack:
        if gpu_sampler is not None:
            await sampler_stack.enter_async_context(gpu_sampler)

        for repetition_index in range(config.workload.repetitions):
            relative_envelopes = generate_requests(config.workload)
            repetition_start_s = clock()
            if config.admission.adaptive_clip_enabled:
                envelopes = tuple(
                    _place_on_clock(envelope, repetition_start_s) for envelope in relative_envelopes
                )
                admission_queue = AdmissionQueue(
                    router_config=config.router,
                    backend_config=config.backend,
                    backend=backend,
                    policy=build_policy(config),
                    clock=clock,
                    sleep=sleep,
                    admission_config=config.admission,
                    adaptive_clipper=output_clipper,
                )
            else:
                envelopes = tuple(
                    output_clipper.apply(_place_on_clock(envelope, repetition_start_s))
                    for envelope in relative_envelopes
                )
                admission_queue = AdmissionQueue(
                    router_config=config.router,
                    backend_config=config.backend,
                    backend=backend,
                    policy=build_policy(config),
                    clock=clock,
                    sleep=sleep,
                )

            async with admission_queue as queue:
                submission_tasks = [
                    asyncio.create_task(
                        _submit_at_arrival(queue, envelope, clock=clock, sleep=sleep),
                        name=(
                            f"sloserve-benchmark-arrival-{repetition_index}-{envelope.sequence_id}"
                        ),
                    )
                    for envelope in envelopes
                ]
                try:
                    admission_results = await asyncio.gather(*submission_tasks)
                except BaseException:
                    for task in submission_tasks:
                        task.cancel()
                    await asyncio.gather(*submission_tasks, return_exceptions=True)
                    raise

                all_adaptive_decisions.extend(
                    AdaptiveCapDecisionRecord.from_admission(
                        decision,
                        repetition_index=repetition_index,
                    )
                    for decision in queue.adaptive_decisions
                )

                # Join before starting the next repetition. Request IDs intentionally repeat,
                # and request-keyed backend telemetry will be overwritten by the next send.
                all_records.extend(
                    _join_repetition(
                        envelopes,
                        admission_results,
                        backend.telemetry,
                        repetition_index=repetition_index,
                        config_hash=config_hash,
                        env_version=env_version,
                        config=config,
                    )
                )

    records = tuple(all_records)
    formal_records = tuple(
        record for record in records if record.sequence_id >= config.workload.warmup_requests
    )
    metrics = calculate_metrics(formal_records, config.workload)
    record_paths = write_request_records(records, config.metrics, file_stem=file_stem)
    report_gpu_samples = gpu_sampler.samples if gpu_sampler is not None else gpu_samples
    completeness_report = build_completeness_report(metrics, gpu_samples=report_gpu_samples)
    completeness_report_path = _write_completeness_report(
        config.metrics.output_directory / f"{file_stem}-completeness.json",
        completeness_report,
    )
    adaptive_decisions = tuple(all_adaptive_decisions)
    adaptive_decision_path = (
        write_adaptive_cap_decisions_jsonl(
            config.metrics.output_directory / f"{file_stem}-adaptive-cap-decisions.jsonl",
            adaptive_decisions,
        )
        if config.admission.adaptive_clip_enabled
        else None
    )
    return BenchmarkResult(
        records=records,
        formal_records=formal_records,
        metrics=metrics,
        record_paths=record_paths,
        completeness_report=completeness_report,
        completeness_report_path=completeness_report_path,
        adaptive_decisions=adaptive_decisions,
        adaptive_decision_path=adaptive_decision_path,
    )


async def _submit_at_arrival(
    queue: AdmissionQueue,
    envelope: RequestEnvelope,
    *,
    clock: Callable[[], float],
    sleep: Callable[[float], Awaitable[None]],
) -> AdmissionResult:
    delay_s = envelope.arrival_time_s - clock()
    if delay_s > 0:
        await sleep(delay_s)
    return await queue.submit(envelope)


def _place_on_clock(envelope: RequestEnvelope, repetition_start_s: float) -> RequestEnvelope:
    """Translate generator-relative times onto the injected monotonic clock."""
    return RequestEnvelope(
        request_id=envelope.request_id,
        sequence_id=envelope.sequence_id,
        request_class=envelope.request_class,
        arrival_time_s=repetition_start_s + envelope.arrival_time_s,
        input_tokens=envelope.input_tokens,
        max_output_tokens=envelope.max_output_tokens,
        deadline_time_s=repetition_start_s + envelope.deadline_time_s,
        advertised_cap_tokens=envelope.advertised_cap_tokens,
        prompt_kind=envelope.prompt_kind,
        backend_max_output_tokens=envelope.backend_max_output_tokens,
        force_exact_output_tokens=envelope.force_exact_output_tokens,
    )


def _join_repetition(
    envelopes: Sequence[RequestEnvelope],
    results: Sequence[AdmissionResult],
    telemetry_by_request: Mapping[str, HttpCallTelemetry],
    *,
    repetition_index: int,
    config_hash: str,
    env_version: str,
    config: ExperimentConfig,
) -> tuple[RequestRecord, ...]:
    results_by_sequence = {result.sequence_id: result for result in results}
    if len(results_by_sequence) != len(envelopes):
        raise RuntimeError("admission results do not map one-to-one to generated requests")

    joined: list[RequestRecord] = []
    for envelope in envelopes:
        try:
            result = results_by_sequence[envelope.sequence_id]
        except KeyError as exc:
            raise RuntimeError(
                f"missing admission result for sequence {envelope.sequence_id}"
            ) from exc
        effective_envelope = result.effective_envelope
        if (
            result.request_id != envelope.request_id
            or result.request_id != effective_envelope.request_id
        ):
            raise RuntimeError("admission result identity does not match generated request")
        if result.sequence_id != effective_envelope.sequence_id:
            raise RuntimeError("effective envelope sequence does not match admission result")
        if result.completion_time_s is None:
            raise RuntimeError(
                f"terminal admission result lacks completion time: {result.request_id}"
            )

        # Never read telemetry for an undispatched request: a repeated request ID may still
        # refer to the preceding repetition, while this request produced no backend call.
        telemetry = (
            telemetry_by_request.get(result.request_id)
            if result.dispatch_time_s is not None
            else None
        )
        joined.append(
            RequestRecord.from_envelope(
                effective_envelope,
                enqueue_time_s=result.enqueue_time_s,
                dispatch_time_s=result.dispatch_time_s,
                first_token_time_s=(
                    telemetry.first_token_time_s if telemetry is not None else None
                ),
                completion_time_s=result.completion_time_s,
                output_tokens=telemetry.output_tokens if telemetry is not None else 0,
                status=result.status,
                error_type=result.error_message
                or (telemetry.error if telemetry is not None else None),
                policy_name=config.router.policy,
                config_hash=config_hash,
                repetition_index=repetition_index,
                env_version=env_version,
                finish_reason=telemetry.finish_reason if telemetry is not None else None,
            )
        )
    return tuple(joined)


def build_completeness_report(
    metrics: MetricsSummary,
    *,
    gpu_samples: Sequence[GpuSample] | None = None,
) -> CompletenessReport:
    """Describe every required result or explain why that observation is absent."""
    latency_groups = {
        "ttft": metrics.ttft_s,
        "tpot": metrics.tpot_s,
        "end_to_end": metrics.end_to_end_s,
        "queue_wait": metrics.queue_wait_s,
    }
    report_metrics: dict[str, ReportField] = {}
    for group_name, percentiles in latency_groups.items():
        for percentile_name in ("p50", "p95", "p99"):
            value = getattr(percentiles, percentile_name)
            report_metrics[f"{group_name}_{percentile_name}_s"] = _optional_field(
                value,
                f"no formal requests eligible for {group_name} {percentile_name}",
            )

    report_metrics.update(
        {
            "formal_request_count": _value_field(metrics.request_count),
            "token_throughput_per_s": _optional_field(
                metrics.token_throughput_per_s,
                "formal request wall-clock window is empty or zero",
            ),
            "success_count": _value_field(metrics.success_count),
            "error_count": _value_field(metrics.error_count),
            "timeout_count": _value_field(metrics.timeout_count),
            "cancelled_count": _value_field(metrics.cancelled_count),
            "rejected_count": _value_field(metrics.rejected_count),
            "clip_applied_count": _value_field(metrics.clip_applied_count),
            "clip_applied_rate": _optional_field(
                metrics.clip_applied_rate,
                "request facts predate clipping instrumentation",
            ),
            "realized_truncation_count": _value_field(metrics.realized_truncation_count),
            "realized_truncation_rate": _optional_field(
                metrics.realized_truncation_rate,
                "request facts predate clipping instrumentation",
            ),
            "mean_cap_reduction_tokens": _optional_field(
                metrics.mean_cap_reduction_tokens,
                "no formal request had its backend output cap reduced",
            ),
            "terminal_count_matches_request_count": _value_field(
                metrics.success_count
                + metrics.error_count
                + metrics.timeout_count
                + metrics.cancelled_count
                + metrics.rejected_count
                == metrics.request_count
            ),
            "longest_queue_wait_s": _optional_field(
                metrics.longest_queue_wait_s,
                "no formal request was dispatched",
            ),
            "queue_wait_mean_s": _optional_field(
                metrics.queue_wait_mean_s,
                "no successful formal request was dispatched",
            ),
            "slo_overall": _attainment_field(metrics.slo_overall),
            "slo_attainment_gap": _value_field(metrics.slo_attainment_gap),
            "jain_fairness_index": _value_field(metrics.jain_fairness_index),
        }
    )
    for request_class in RequestClass:
        summary = metrics.slo_by_class.get(request_class)
        report_metrics[f"slo_{request_class.value}"] = (
            _attainment_field(summary)
            if summary is not None
            else _missing_field("no formal requests of this class")
        )

    gpu_report = _gpu_report(gpu_samples)
    return CompletenessReport(
        scope_note=_PIPELINE_ONLY_NOTE,
        request_scope="formal requests only; warmup records are persisted but excluded",
        metrics=report_metrics,
        gpu=gpu_report,
    )


def _attainment_field(summary: AttainmentSummary) -> ReportField:
    return _value_field(
        {
            "request_count": summary.request_count,
            "met_count": summary.met_count,
            "rate": summary.rate,
        }
    )


def _gpu_report(gpu_samples: Sequence[GpuSample] | None) -> dict[str, ReportField]:
    if not gpu_samples:
        return {
            "utilization_percent": _missing_field(_GPU_NOT_COLLECTED_REASON),
            "memory_used_mib": _missing_field(_GPU_NOT_COLLECTED_REASON),
            "power_w": _missing_field(_GPU_NOT_COLLECTED_REASON),
        }

    utilization = [sample.utilization_percent for sample in gpu_samples]
    memory = [sample.memory_used_mib for sample in gpu_samples]
    power = [sample.power_w for sample in gpu_samples if sample.power_w is not None]
    return {
        "utilization_percent": _value_field(
            {"mean": sum(utilization) / len(utilization), "peak": max(utilization)}
        ),
        "memory_used_mib": _value_field({"peak": max(memory)}),
        "power_w": (
            _value_field({"mean": sum(power) / len(power), "peak": max(power)})
            if power
            else _missing_field("device did not report power")
        ),
    }


def _value_field(value: JsonValue) -> ReportField:
    return ReportField(value=value, missing_reason=None)


def _missing_field(reason: str) -> ReportField:
    return ReportField(value=None, missing_reason=reason)


def _optional_field(value: float | None, missing_reason: str) -> ReportField:
    return _value_field(value) if value is not None else _missing_field(missing_reason)


def _write_completeness_report(path: Path, report: CompletenessReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(asdict(report), output, allow_nan=False, ensure_ascii=False, indent=2)
        output.write("\n")
    return path
