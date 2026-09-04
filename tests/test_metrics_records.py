"""Tests for raw request facts, configuration hashes, and result files."""

from __future__ import annotations

import csv
import json
from dataclasses import fields
from pathlib import Path

from sloserve.config import MetricsConfig, SchedulerPolicyName, load_config
from sloserve.metrics import (
    RequestRecord,
    experiment_config_hash,
    read_request_records_csv,
    read_request_records_jsonl,
    write_request_records,
    write_request_records_csv,
    write_request_records_jsonl,
)
from sloserve.router.models import RequestClass, RequestEnvelope
from sloserve.workload.dispatcher import DispatchStatus

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _record(config_hash: str) -> RequestRecord:
    return RequestRecord(
        request_id="request-000001",
        sequence_id=1,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=1.0,
        enqueue_time_s=1.1,
        dispatch_time_s=1.2,
        first_token_time_s=1.5,
        completion_time_s=2.5,
        input_tokens=12,
        output_tokens=3,
        status=DispatchStatus.SUCCESS,
        error_type=None,
        policy_name=SchedulerPolicyName.FCFS,
        config_hash=config_hash,
        repetition_index=0,
        env_version="test-environment-placeholder",
    )


def test_request_record_builds_directly_from_envelope() -> None:
    envelope = RequestEnvelope(
        request_id="request-000007",
        sequence_id=7,
        request_class=RequestClass.BATCH,
        arrival_time_s=3.0,
        input_tokens=512,
        max_output_tokens=64,
        deadline_time_s=13.0,
    )

    record = RequestRecord.from_envelope(
        envelope,
        enqueue_time_s=3.1,
        dispatch_time_s=None,
        first_token_time_s=None,
        completion_time_s=4.0,
        output_tokens=0,
        status=DispatchStatus.TIMEOUT,
        error_type="BackendTimeout",
        policy_name=SchedulerPolicyName.FCFS,
        config_hash="a" * 64,
        repetition_index=2,
        env_version="test-environment-placeholder",
    )

    assert record.request_id == envelope.request_id
    assert record.sequence_id == envelope.sequence_id
    assert record.request_class is envelope.request_class
    assert record.arrival_time_s == envelope.arrival_time_s
    assert record.input_tokens == envelope.input_tokens
    assert record.status is DispatchStatus.TIMEOUT
    assert record.dispatch_time_s is None


def test_jsonl_and_csv_round_trip_every_field(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    original = (_record(experiment_config_hash(config)),)

    jsonl_path = write_request_records_jsonl(tmp_path / "facts.jsonl", original)
    csv_path = write_request_records_csv(tmp_path / "derived.csv", original)

    assert read_request_records_jsonl(jsonl_path) == original
    assert read_request_records_csv(csv_path) == original
    assert csv_path.read_text(encoding="utf-8").splitlines()[0].split(",") == [
        field.name for field in fields(RequestRecord)
    ]


def test_legacy_request_files_remain_readable_after_clipping_fields(tmp_path: Path) -> None:
    original = _record("c" * 64)
    added = {"requested_output_tokens", "backend_max_output_tokens", "finish_reason"}

    current_jsonl = write_request_records_jsonl(tmp_path / "current.jsonl", (original,))
    raw = json.loads(current_jsonl.read_text(encoding="utf-8"))
    for field_name in added:
        raw.pop(field_name)
    legacy_jsonl = tmp_path / "legacy.jsonl"
    legacy_jsonl.write_text(json.dumps(raw) + "\n", encoding="utf-8")

    current_csv = write_request_records_csv(tmp_path / "current.csv", (original,))
    with current_csv.open(encoding="utf-8", newline="") as source:
        csv_row = next(csv.DictReader(source))
    for field_name in added:
        csv_row.pop(field_name)
    legacy_csv = tmp_path / "legacy.csv"
    with legacy_csv.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_row))
        writer.writeheader()
        writer.writerow(csv_row)

    assert read_request_records_jsonl(legacy_jsonl) == (original,)
    assert read_request_records_csv(legacy_csv) == (original,)


def test_undispatched_rejected_record_round_trips_jsonl_and_csv(tmp_path: Path) -> None:
    rejected = RequestRecord(
        request_id="request-rejected",
        sequence_id=2,
        request_class=RequestClass.BATCH,
        arrival_time_s=2.0,
        enqueue_time_s=2.1,
        dispatch_time_s=None,
        first_token_time_s=None,
        completion_time_s=2.2,
        input_tokens=16,
        output_tokens=0,
        status=DispatchStatus.REJECTED,
        error_type="QueueFull",
        policy_name=SchedulerPolicyName.FCFS,
        config_hash="b" * 64,
        repetition_index=1,
        env_version="test-environment-placeholder",
    )

    jsonl_path = write_request_records_jsonl(tmp_path / "rejected.jsonl", (rejected,))
    csv_path = write_request_records_csv(tmp_path / "rejected.csv", (rejected,))

    assert read_request_records_jsonl(jsonl_path) == (rejected,)
    assert read_request_records_csv(csv_path) == (rejected,)
    assert ",," in csv_path.read_text(encoding="utf-8")


def test_metrics_config_controls_output_directory_and_formats(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    metrics_config = MetricsConfig(
        output_directory=tmp_path / "configured-results",
        gpu_sample_interval_s=0.5,
        save_jsonl=True,
        save_csv=False,
    )

    paths = write_request_records(
        (_record(experiment_config_hash(config)),), metrics_config, file_stem="repetition-0"
    )

    assert paths.jsonl == tmp_path / "configured-results" / "repetition-0.jsonl"
    assert paths.jsonl.is_file()
    assert paths.csv is None
    assert not (tmp_path / "configured-results" / "repetition-0.csv").exists()


def test_experiment_config_hash_is_deterministic_and_sensitive_to_changes() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    changed_workload = config.workload.model_copy(
        update={"random_seed": config.workload.random_seed + 1}
    )
    changed_config = config.model_copy(update={"workload": changed_workload})

    first = experiment_config_hash(config)

    assert first == experiment_config_hash(config)
    assert len(first) == 64
    assert first != experiment_config_hash(changed_config)
