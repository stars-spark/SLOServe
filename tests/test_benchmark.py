"""Pure offline tests for benchmark orchestration and completeness reporting."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from sloserve.config import ExperimentConfig, load_config
from sloserve.experiments.benchmark import (
    BenchmarkResult,
    GpuSample,
    _gpu_report,
    run_benchmark,
)
from sloserve.metrics import read_request_records_csv, read_request_records_jsonl
from sloserve.router.models import RequestEnvelope
from sloserve.workload.dispatcher import DispatchStatus
from sloserve.workload.http_backend import HttpCallTelemetry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicTime:
    """Advance only benchmark arrival sleeps; park queue timeout sleeps until cancellation."""

    def __init__(self) -> None:
        self.now_s = 0.0
        self._arrivals: dict[tuple[int, int], asyncio.Event] = {}

    def clock(self) -> float:
        return self.now_s

    async def sleep(self, delay_s: float) -> None:
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else ""
        prefix = "sloserve-benchmark-arrival-"
        if task_name.startswith(prefix):
            repetition_text, sequence_text = task_name.removeprefix(prefix).split("-")
            self.now_s += delay_s
            await asyncio.sleep(0)
            self.arrival_event(int(repetition_text), int(sequence_text)).set()
            return
        await asyncio.get_running_loop().create_future()

    def arrival_event(self, repetition_index: int, sequence_id: int) -> asyncio.Event:
        return self._arrivals.setdefault((repetition_index, sequence_id), asyncio.Event())


@dataclass(frozen=True, slots=True)
class BackendPlan:
    """Configurable request telemetry and terminal behavior for the local fake."""

    first_token_offset_s: float = 0.0
    output_tokens: int = 2
    status: DispatchStatus = DispatchStatus.SUCCESS

    def __post_init__(self) -> None:
        if self.first_token_offset_s < 0:
            raise ValueError("first_token_offset_s must be non-negative")
        if self.output_tokens < 0:
            raise ValueError("output_tokens must be non-negative")
        if self.status not in {
            DispatchStatus.SUCCESS,
            DispatchStatus.ERROR,
            DispatchStatus.CANCELLED,
        }:
            raise ValueError("fake backend status must be success, error, or cancelled")


class TelemetryFakeBackend:
    """Network-free backend that exposes the same telemetry join surface as HTTP."""

    def __init__(
        self,
        deterministic_time: DeterministicTime,
        plans: dict[str, BackendPlan] | None = None,
    ) -> None:
        self._time = deterministic_time
        self._plans = plans or {}
        self._call_counts: dict[str, int] = {}
        self.telemetry: dict[str, HttpCallTelemetry] = {}

    async def send(self, request: RequestEnvelope) -> None:
        plan = self._plans.get(request.request_id, BackendPlan())
        call_index = self._call_counts.get(request.request_id, 0)
        self._call_counts[request.request_id] = call_index + 1
        telemetry = HttpCallTelemetry()
        self.telemetry[request.request_id] = telemetry

        # Keep the warmup request in flight until the third arrival of the repetition.
        # With max_in_flight=1 and queue_capacity=1 this deterministically rejects it.
        if request.sequence_id == 0:
            await self._time.arrival_event(call_index, 2).wait()

        telemetry.first_token_time_s = self._time.clock() + plan.first_token_offset_s
        telemetry.output_tokens = plan.output_tokens
        if plan.status is DispatchStatus.ERROR:
            telemetry.error = "configured fake error"
            raise RuntimeError(telemetry.error)
        if plan.status is DispatchStatus.CANCELLED:
            raise asyncio.CancelledError


class FakeGpuSampler:
    """Preloaded sampler used to verify benchmark context orchestration."""

    def __init__(self, samples: tuple[GpuSample, ...]) -> None:
        self._samples = samples
        self.entered = False
        self.exited = False

    @property
    def samples(self) -> tuple[GpuSample, ...]:
        return self._samples

    async def __aenter__(self) -> FakeGpuSampler:
        self.entered = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exited = True


def _config(tmp_path: Path) -> ExperimentConfig:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    workload = config.workload.model_copy(
        update={
            "request_rate_rps": 1000.0,
            "total_requests": 2,
            "warmup_requests": 1,
            "repetitions": 2,
        }
    )
    router = config.router.model_copy(update={"max_in_flight": 1, "queue_capacity": 1})
    metrics = config.metrics.model_copy(update={"output_directory": tmp_path})
    return config.model_copy(update={"workload": workload, "router": router, "metrics": metrics})


def _run(config: ExperimentConfig) -> BenchmarkResult:
    deterministic_time = DeterministicTime()
    backend = TelemetryFakeBackend(deterministic_time)
    return asyncio.run(
        run_benchmark(
            config=config,
            backend=backend,
            env_version="offline-test-placeholder",
            clock=deterministic_time.clock,
            sleep=deterministic_time.sleep,
        )
    )


def test_benchmark_repetitions_persistence_metrics_and_completeness(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = _run(config)

    assert len(result.records) == (1 + 2) * 2
    assert [record.repetition_index for record in result.records] == [0, 0, 0, 1, 1, 1]
    assert result.record_paths.jsonl is not None
    assert result.record_paths.csv is not None
    assert read_request_records_jsonl(result.record_paths.jsonl) == result.records
    assert read_request_records_csv(result.record_paths.csv) == result.records

    # Warmups are preserved as facts but sequence 0 is excluded from formal metrics.
    assert len(result.formal_records) == 4
    assert result.metrics.request_count == 4
    assert result.metrics.success_count == 2
    assert result.metrics.rejected_count == 2
    assert all(record.sequence_id >= 1 for record in result.formal_records)
    rejected = [record for record in result.records if record.status is DispatchStatus.REJECTED]
    assert len(rejected) == 2
    assert all(record.dispatch_time_s is None for record in rejected)

    report_raw = json.loads(result.completeness_report_path.read_text(encoding="utf-8"))
    assert report_raw["scope_note"] == "仅管线验证，不作策略性能比较"  # noqa: RUF001
    assert "warmup records are persisted but excluded" in report_raw["request_scope"]
    for section_name in ("metrics", "gpu"):
        for field in report_raw[section_name].values():
            assert (field["value"] is None) != (field["missing_reason"] is None)
    assert report_raw["metrics"]["rejected_count"]["value"] == 2
    assert report_raw["metrics"]["terminal_count_matches_request_count"]["value"] is True
    assert report_raw["gpu"]["utilization_percent"]["value"] is None
    assert "未采集" in report_raw["gpu"]["utilization_percent"]["missing_reason"]


def test_same_seed_and_injected_time_produce_identical_records(tmp_path: Path) -> None:
    config = _config(tmp_path)

    first = _run(config)
    second = _run(config)

    assert first.records == second.records


def test_gpu_report_marks_all_missing_power_explicitly() -> None:
    report = _gpu_report(
        (
            GpuSample(1.0, 45.0, 6136.0, None),
            GpuSample(2.0, 12.0, 6100.0, None),
        )
    )

    assert report["utilization_percent"].value == {"mean": 28.5, "peak": 45.0}
    assert report["memory_used_mib"].value == {"peak": 6136.0}
    assert report["power_w"].value is None
    assert report["power_w"].missing_reason == "device did not report power"


def test_run_benchmark_uses_sampler_samples_over_direct_samples(tmp_path: Path) -> None:
    config = _config(tmp_path)
    deterministic_time = DeterministicTime()
    backend = TelemetryFakeBackend(deterministic_time)
    sampler = FakeGpuSampler(
        (
            GpuSample(10.0, 20.0, 2000.0, None),
            GpuSample(11.0, 60.0, 3000.0, 70.0),
        )
    )

    result = asyncio.run(
        run_benchmark(
            config=config,
            backend=backend,
            env_version="offline-test-placeholder",
            clock=deterministic_time.clock,
            sleep=deterministic_time.sleep,
            gpu_samples=(GpuSample(0.0, 99.0, 9999.0, 999.0),),
            gpu_sampler=sampler,
        )
    )

    assert sampler.entered is True
    assert sampler.exited is True
    assert result.completeness_report.gpu["utilization_percent"].value == {
        "mean": 40.0,
        "peak": 60.0,
    }
    assert result.completeness_report.gpu["memory_used_mib"].value == {"peak": 3000.0}
    assert result.completeness_report.gpu["power_w"].value == {"mean": 70.0, "peak": 70.0}
