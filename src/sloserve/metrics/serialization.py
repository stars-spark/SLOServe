"""JSONL fact-source and derived CSV serialization for request records."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from sloserve.config import MetricsConfig, SchedulerPolicyName
from sloserve.metrics.records import RequestRecord
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus

_FIELD_NAMES = tuple(field.name for field in fields(RequestRecord))


@dataclass(frozen=True, slots=True)
class RequestRecordPaths:
    """Paths produced by the MetricsConfig-aware writer."""

    jsonl: Path | None
    csv: Path | None


def _record_to_mapping(record: RequestRecord) -> dict[str, Any]:
    return {
        "request_id": record.request_id,
        "sequence_id": record.sequence_id,
        "request_class": record.request_class.value,
        "arrival_time_s": record.arrival_time_s,
        "enqueue_time_s": record.enqueue_time_s,
        "dispatch_time_s": record.dispatch_time_s,
        "first_token_time_s": record.first_token_time_s,
        "completion_time_s": record.completion_time_s,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "status": record.status.value,
        "error_type": record.error_type,
        "policy_name": record.policy_name.value,
        "config_hash": record.config_hash,
        "repetition_index": record.repetition_index,
        "env_version": record.env_version,
    }


def _record_from_mapping(data: Mapping[str, Any]) -> RequestRecord:
    if set(data) != set(_FIELD_NAMES):
        missing = sorted(set(_FIELD_NAMES) - set(data))
        extra = sorted(set(data) - set(_FIELD_NAMES))
        raise ValueError(f"request record fields differ: missing={missing}, extra={extra}")

    dispatch_raw = data["dispatch_time_s"]
    first_token_raw = data["first_token_time_s"]
    error_type_raw = data["error_type"]
    return RequestRecord(
        request_id=str(data["request_id"]),
        sequence_id=int(data["sequence_id"]),
        request_class=RequestClass(data["request_class"]),
        arrival_time_s=float(data["arrival_time_s"]),
        enqueue_time_s=float(data["enqueue_time_s"]),
        dispatch_time_s=None if dispatch_raw in (None, "") else float(dispatch_raw),
        first_token_time_s=None if first_token_raw in (None, "") else float(first_token_raw),
        completion_time_s=float(data["completion_time_s"]),
        input_tokens=int(data["input_tokens"]),
        output_tokens=int(data["output_tokens"]),
        status=DispatchStatus(data["status"]),
        error_type=None if error_type_raw in (None, "") else str(error_type_raw),
        policy_name=SchedulerPolicyName(data["policy_name"]),
        config_hash=str(data["config_hash"]),
        repetition_index=int(data["repetition_index"]),
        env_version=str(data["env_version"]),
    )


def write_request_records_jsonl(path: str | Path, records: Sequence[RequestRecord]) -> Path:
    """Write one JSON object per line as the canonical request fact source."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(
                json.dumps(
                    _record_to_mapping(record),
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            output.write("\n")
    return output_path


def read_request_records_jsonl(path: str | Path) -> tuple[RequestRecord, ...]:
    """Read and validate canonical request facts from JSONL."""
    records: list[RequestRecord] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line at {line_number}")
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("record must be a JSON object")
                records.append(_record_from_mapping(raw))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid request record at JSONL line {line_number}") from exc
    return tuple(records)


def write_request_records_csv(path: str | Path, records: Sequence[RequestRecord]) -> Path:
    """Write a derived CSV with columns corresponding to every JSONL field."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=_FIELD_NAMES)
        writer.writeheader()
        for record in records:
            writer.writerow(_record_to_mapping(record))
    return output_path


def read_request_records_csv(path: str | Path) -> tuple[RequestRecord, ...]:
    """Read and validate records from the derived CSV representation."""
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != list(_FIELD_NAMES):
            raise ValueError("CSV header does not match the request record schema")
        try:
            return tuple(_record_from_mapping(row) for row in reader)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid request record in CSV") from exc


def write_request_records(
    records: Sequence[RequestRecord],
    metrics_config: MetricsConfig,
    *,
    file_stem: str,
) -> RequestRecordPaths:
    """Write enabled formats beneath the configured output directory."""
    if not file_stem or Path(file_stem).name != file_stem:
        raise ValueError("file_stem must be a non-empty filename stem")
    output_directory = metrics_config.output_directory
    jsonl_path = output_directory / f"{file_stem}.jsonl" if metrics_config.save_jsonl else None
    csv_path = output_directory / f"{file_stem}.csv" if metrics_config.save_csv else None
    if jsonl_path is not None:
        write_request_records_jsonl(jsonl_path, records)
    if csv_path is not None:
        write_request_records_csv(csv_path, records)
    return RequestRecordPaths(jsonl=jsonl_path, csv=csv_path)
