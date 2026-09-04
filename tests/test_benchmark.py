"""Pure offline tests for benchmark orchestration and completeness reporting."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

import pytest

from sloserve.config import (
    AdmissionControlConfig,
    ExperimentConfig,
    SchedulerPolicyName,
    TokenRangeConfig,
    load_config,
)
from sloserve.experiments.benchmark import (
    BenchmarkResult,
    GpuSample,
    _gpu_report,
    _place_on_clock,
    run_benchmark,
)
from sloserve.metrics import read_request_records_csv, read_request_records_jsonl
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.dispatcher import DispatchStatus
from sloserve.workload.http_backend import HttpCallTelemetry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DeterministicTime:
    """Advance only benchmark arrival sleeps; park queue timeout sleeps until cancellation."""

    def __init__(self) -> None:
        self.now_s = 0.0
        self.calls = 0
        self._arrivals: dict[tuple[int, int], asyncio.Event] = {}

    def clock(self) -> float:
        self.calls += 1
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
        self.started_request_ids: list[str] = []

    async def send(self, request: RequestEnvelope) -> None:
        self.started_request_ids.append(request.request_id)
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


class GatedLowLoadTime:
    """Release each low-load arrival only after the preceding backend call finishes."""

    def __init__(self) -> None:
        self.now_s = 0.0
        self._gates: dict[int, asyncio.Event] = {}

    def clock(self) -> float:
        return self.now_s

    async def sleep(self, delay_s: float) -> None:
        del delay_s
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else ""
        prefix = "sloserve-benchmark-arrival-"
        if task_name.startswith(prefix):
            _, sequence_text = task_name.removeprefix(prefix).split("-")
            sequence_id = int(sequence_text)
            await self.gate(sequence_id).wait()
            self.now_s = sequence_id / 0.08
            return
        await asyncio.get_running_loop().create_future()

    def gate(self, sequence_id: int) -> asyncio.Event:
        event = self._gates.setdefault(sequence_id, asyncio.Event())
        if sequence_id == 0:
            event.set()
        return event


class GatedLowLoadBackend:
    """Immediate telemetry backend that serializes the fixed low-load trace."""

    def __init__(self, deterministic_time: GatedLowLoadTime) -> None:
        self._time = deterministic_time
        self.telemetry: dict[str, HttpCallTelemetry] = {}
        self.sent: list[RequestEnvelope] = []

    async def send(self, request: RequestEnvelope) -> None:
        self.sent.append(request)
        self.telemetry[request.request_id] = HttpCallTelemetry(
            first_token_time_s=self._time.clock(),
            output_tokens=2,
        )
        asyncio.get_running_loop().call_soon(self._time.gate(request.sequence_id + 1).set)


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


def test_static_priority_benchmark_admits_later_interactive_before_waiting_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    workload = config.workload.model_copy(
        update={"total_requests": 3, "warmup_requests": 0, "repetitions": 1}
    )
    router = config.router.model_copy(
        update={
            "policy": SchedulerPolicyName.STATIC_PRIORITY,
            "max_in_flight": 1,
            "queue_capacity": 3,
        }
    )
    config = config.model_copy(update={"workload": workload, "router": router})
    envelopes = (
        RequestEnvelope(
            request_id="batch-in-flight",
            sequence_id=0,
            request_class=RequestClass.BATCH,
            arrival_time_s=0.0,
            input_tokens=512,
            max_output_tokens=128,
            deadline_time_s=10.0,
        ),
        RequestEnvelope(
            request_id="batch-waiting",
            sequence_id=1,
            request_class=RequestClass.BATCH,
            arrival_time_s=1.0,
            input_tokens=512,
            max_output_tokens=128,
            deadline_time_s=11.0,
        ),
        RequestEnvelope(
            request_id="interactive-waiting",
            sequence_id=2,
            request_class=RequestClass.INTERACTIVE,
            arrival_time_s=2.0,
            input_tokens=64,
            max_output_tokens=32,
            deadline_time_s=12.0,
        ),
    )
    monkeypatch.setattr(
        "sloserve.experiments.benchmark.generate_requests", lambda workload_config: envelopes
    )
    deterministic_time = DeterministicTime()
    backend = TelemetryFakeBackend(deterministic_time)

    result = asyncio.run(
        run_benchmark(
            config=config,
            backend=backend,
            env_version="offline-test-placeholder",
            clock=deterministic_time.clock,
            sleep=deterministic_time.sleep,
        )
    )

    assert [record.status for record in result.records] == [DispatchStatus.SUCCESS] * 3
    assert backend.started_request_ids == [
        "batch-in-flight",
        "interactive-waiting",
        "batch-waiting",
    ]


def test_place_on_clock_preserves_all_scheduling_fields() -> None:
    # Regression: the reclocked envelope reaches the scheduler, so dropping the
    # length-decoupling fields silently turns the advertised/learned length sources
    # back into the true target. Every field except the two shifted times must survive.
    envelope = RequestEnvelope(
        request_id="r0",
        sequence_id=3,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=1.0,
        input_tokens=64,
        max_output_tokens=1500,
        deadline_time_s=11.0,
        advertised_cap_tokens=2048,
        prompt_kind="long",
        backend_max_output_tokens=1024,
    )

    placed = _place_on_clock(envelope, repetition_start_s=100.0)

    assert placed.advertised_cap_tokens == 2048
    assert placed.prompt_kind == "long"
    assert placed.max_output_tokens == 1500
    assert placed.backend_max_output_tokens == 1024
    assert placed.input_tokens == 64
    assert placed.arrival_time_s == 101.0
    assert placed.deadline_time_s == 111.0


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


def _parity_config(output_directory: Path, *, fixed_clip: bool) -> ExperimentConfig:
    config = _config(output_directory)
    interactive = config.workload.interactive.model_copy(
        update={"output_tokens": TokenRangeConfig(minimum=700, maximum=900)}
    )
    batch = config.workload.batch.model_copy(
        update={"output_tokens": TokenRangeConfig(minimum=1200, maximum=1800)}
    )
    workload = config.workload.model_copy(update={"interactive": interactive, "batch": batch})
    admission = config.admission.model_copy(
        update={"clip_enabled": fixed_clip, "clip_max_tokens": 512}
    )
    return config.model_copy(update={"workload": workload, "admission": admission})


@pytest.mark.parametrize(
    ("arm", "fixed_clip"),
    [("no-clip", False), ("fixed-512", True)],
)
def test_disabled_adaptive_path_matches_pre_change_golden_bytes(
    arm: str,
    fixed_clip: bool,
) -> None:
    fixture_directory = PROJECT_ROOT / "tests" / "fixtures" / "parity"
    before = fixture_directory / f"{arm}-before.jsonl"
    after = fixture_directory / f"{arm}-after.jsonl"
    output_directory = Path("/tmp/sloserve-week7-baseline") / f"{arm}-before"
    config = _parity_config(output_directory, fixed_clip=fixed_clip)
    deterministic_time = DeterministicTime()
    backend = TelemetryFakeBackend(deterministic_time)

    result = asyncio.run(
        run_benchmark(
            config=config,
            backend=backend,
            env_version="week7-parity",
            clock=deterministic_time.clock,
            sleep=deterministic_time.sleep,
            file_stem=f"{arm}-generated",
        )
    )

    assert result.record_paths.jsonl is not None
    generated_bytes = result.record_paths.jsonl.read_bytes()
    before_bytes = before.read_bytes()
    after_bytes = after.read_bytes()
    assert generated_bytes == before_bytes == after_bytes
    assert hashlib.sha256(generated_bytes).digest() == hashlib.sha256(before_bytes).digest()
    assert json.loads("[" + generated_bytes.decode().replace("\n", ",").rstrip(",") + "]") == (
        json.loads("[" + before_bytes.decode().replace("\n", ",").rstrip(",") + "]")
    )
    assert result.adaptive_decisions == ()
    assert result.adaptive_decision_path is None
    assert deterministic_time.calls == 44
    assert not (output_directory / f"{arm}-generated-adaptive-cap-decisions.jsonl").exists()

    objects = [json.loads(line) for line in generated_bytes.splitlines()]
    assert [item["request_id"] for item in objects] == [
        "request-000000",
        "request-000001",
        "request-000002",
    ] * 2
    assert [item["backend_max_output_tokens"] for item in objects] == (
        [512] * 6 if fixed_clip else [872, 735, 1453] * 2
    )
    assert [item["status"] for item in objects] == ["success", "success", "rejected"] * 2
    assert [
        (
            item["arrival_time_s"],
            item["enqueue_time_s"],
            item["dispatch_time_s"],
            item["first_token_time_s"],
            item["completion_time_s"],
        )
        for item in objects
    ] == [
        (0.0, 0.0, 0.0, 0.002, 0.002),
        (0.001, 0.002, 0.002, 0.002, 0.002),
        (0.002, 0.002, None, None, 0.002),
        (0.002, 0.002, 0.002, 0.004, 0.004),
        (0.003, 0.004, 0.004, 0.004, 0.004),
        (0.004, 0.004, None, None, 0.004),
    ]


def test_recommended_low_load_trace_has_zero_utility_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    requests = tuple(
        RequestEnvelope(
            request_id=f"low-{index}",
            sequence_id=index,
            request_class=RequestClass.INTERACTIVE,
            arrival_time_s=index / 0.08,
            input_tokens=64,
            max_output_tokens=1800,
            deadline_time_s=index / 0.08 + 120.0,
            advertised_cap_tokens=2048,
            prompt_kind="long",
        )
        for index in range(8)
    )
    monkeypatch.setattr(
        "sloserve.experiments.benchmark.generate_requests",
        lambda workload_config: requests,
    )
    workload = config.workload.model_copy(
        update={
            "request_rate_rps": 0.08,
            "total_requests": len(requests),
            "warmup_requests": 0,
            "repetitions": 1,
        }
    )
    router = config.router.model_copy(update={"max_in_flight": 4})
    metrics = config.metrics.model_copy(update={"output_directory": tmp_path / "adaptive"})
    adaptive_admission = AdmissionControlConfig(
        clip_enabled=True,
        adaptive_clip_enabled=True,
    )
    adaptive_config = config.model_copy(
        update={
            "workload": workload,
            "router": router,
            "metrics": metrics,
            "admission": adaptive_admission,
        }
    )

    adaptive_time = GatedLowLoadTime()
    adaptive_backend = GatedLowLoadBackend(adaptive_time)
    adaptive = asyncio.run(
        run_benchmark(
            config=adaptive_config,
            backend=adaptive_backend,
            env_version="low-load-gate",
            clock=adaptive_time.clock,
            sleep=adaptive_time.sleep,
            file_stem="adaptive",
        )
    )

    no_clip_metrics = config.metrics.model_copy(update={"output_directory": tmp_path / "no-clip"})
    no_clip_config = adaptive_config.model_copy(
        update={
            "metrics": no_clip_metrics,
            "admission": AdmissionControlConfig(),
        }
    )
    no_clip_time = GatedLowLoadTime()
    no_clip_backend = GatedLowLoadBackend(no_clip_time)
    no_clip = asyncio.run(
        run_benchmark(
            config=no_clip_config,
            backend=no_clip_backend,
            env_version="low-load-gate",
            clock=no_clip_time.clock,
            sleep=no_clip_time.sleep,
            file_stem="no-clip",
        )
    )

    assert adaptive.metrics.clip_applied_count == 0
    assert adaptive.metrics.realized_truncation_count == 0
    assert all(
        record.backend_max_output_tokens == record.requested_output_tokens
        for record in adaptive.records
    )
    assert adaptive.adaptive_decision_path is not None
    assert adaptive.adaptive_decision_path.is_file()
    assert len(adaptive.adaptive_decisions) == len(requests)
    assert all(decision.new_level.name == "L0" for decision in adaptive.adaptive_decisions)
    assert no_clip.adaptive_decision_path is None

    def comparable(record: object) -> dict[str, object]:
        values = asdict(record)
        values.pop("config_hash")
        return values

    assert [comparable(record) for record in adaptive.records] == [
        comparable(record) for record in no_clip.records
    ]
    assert [request.effective_max_output_tokens for request in adaptive_backend.sent] == [
        request.max_output_tokens for request in requests
    ]
    assert [request.effective_max_output_tokens for request in no_clip_backend.sent] == [
        request.max_output_tokens for request in requests
    ]


def test_adaptive_sidecar_joins_dispatch_cap_into_request_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    requests = tuple(
        RequestEnvelope(
            request_id=f"burst-{index}",
            sequence_id=index,
            request_class=RequestClass.INTERACTIVE,
            arrival_time_s=float(index),
            input_tokens=64,
            max_output_tokens=1800,
            deadline_time_s=120.0,
            advertised_cap_tokens=2048,
            prompt_kind="long",
        )
        for index in range(3)
    )
    monkeypatch.setattr(
        "sloserve.experiments.benchmark.generate_requests",
        lambda workload_config: requests,
    )
    workload = config.workload.model_copy(
        update={"total_requests": 3, "warmup_requests": 0, "repetitions": 1}
    )
    router = config.router.model_copy(update={"max_in_flight": 1, "queue_capacity": 3})
    metrics = config.metrics.model_copy(update={"output_directory": tmp_path})
    admission = AdmissionControlConfig(clip_enabled=True, adaptive_clip_enabled=True)
    config = config.model_copy(
        update={
            "workload": workload,
            "router": router,
            "metrics": metrics,
            "admission": admission,
        }
    )
    deterministic_time = DeterministicTime()
    backend = TelemetryFakeBackend(deterministic_time)

    result = asyncio.run(
        run_benchmark(
            config=config,
            backend=backend,
            env_version="adaptive-sidecar-test",
            clock=deterministic_time.clock,
            sleep=deterministic_time.sleep,
            file_stem="adaptive-burst",
        )
    )

    assert result.adaptive_decision_path is not None
    sidecar_objects = [
        json.loads(line)
        for line in result.adaptive_decision_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [item["request_id"] for item in sidecar_objects] == backend.started_request_ids
    assert [item["q_inst"] for item in sidecar_objects] == [0, 1, 0]
    assert [item["new_level"] for item in sidecar_objects] == ["L0", "L1", "L1"]
    assert [item["selected_cap"] for item in sidecar_objects] == [None, 1536, 1536]
    assert [record.backend_max_output_tokens for record in result.records] == [1800, 1536, 1536]
    assert [record.requested_output_tokens for record in result.records] == [1800, 1800, 1800]
