"""End-to-end fixture for the expK-B aggregate analysis script."""

import csv
from pathlib import Path

from sloserve.analysis.expkb import analyze_expkb
from sloserve.config import SchedulerPolicyName
from sloserve.metrics import RequestRecord, write_request_records_jsonl
from sloserve.router.models import RequestClass
from sloserve.workload.dispatcher import DispatchStatus


def _record(sequence_id: int, output_tokens: int, *, clipped: bool) -> RequestRecord:
    service_time = 0.1 + 0.01 * output_tokens
    target = 100 if clipped else output_tokens
    return RequestRecord(
        request_id=f"r-{sequence_id}",
        sequence_id=sequence_id,
        request_class=RequestClass.INTERACTIVE,
        arrival_time_s=float(sequence_id),
        enqueue_time_s=float(sequence_id),
        dispatch_time_s=float(sequence_id),
        first_token_time_s=float(sequence_id),
        completion_time_s=float(sequence_id) + service_time,
        input_tokens=16,
        output_tokens=output_tokens,
        status=DispatchStatus.SUCCESS,
        error_type=None,
        policy_name=SchedulerPolicyName.FCFS,
        config_hash="a" * 64,
        repetition_index=0,
        env_version="test",
        requested_output_tokens=target,
        backend_max_output_tokens=output_tokens,
        finish_reason="length" if clipped else "stop",
    )


def test_analysis_aggregates_seeds_and_reports_theory_trend(tmp_path: Path) -> None:
    fields = [
        "label",
        "request_rate_rps",
        "max_in_flight",
        "queue_wait_mean_s",
        "queue_wait_p99_s",
        "end_to_end_p99_s",
        "slo_overall_rate",
        "slo_interactive_rate",
        "token_throughput_per_s",
        "clip_applied_rate",
        "realized_truncation_rate",
        "mean_cap_reduction_tokens",
    ]
    remainder = tmp_path / "remainder"
    remainder.mkdir()
    rows_by_dir: dict[Path, list[dict[str, object]]] = {tmp_path: [], remainder: []}
    for index, label in enumerate(("no-clip-s1", "no-clip-s2", "clip512-s1", "clip512-s2")):
        clipped = label.startswith("clip")
        result_dir = remainder if clipped else tmp_path
        rows_by_dir[result_dir].append(
            {
                "label": label,
                "request_rate_rps": 0.1,
                "max_in_flight": 2,
                "queue_wait_mean_s": 0.2 if clipped else 0.5,
                "queue_wait_p99_s": 0.4 if clipped else 1.0,
                "end_to_end_p99_s": 1.0,
                "slo_overall_rate": 1.0,
                "slo_interactive_rate": 1.0,
                "token_throughput_per_s": 10.0,
                "clip_applied_rate": 0.5 if clipped else 0.0,
                "realized_truncation_rate": 0.5 if clipped else 0.0,
                "mean_cap_reduction_tokens": 80.0 if clipped else "",
            }
        )
        records = (
            _record(0, 10, clipped=False),
            _record(1, 20 if clipped else 100, clipped=clipped),
        )
        write_request_records_jsonl(result_dir / f"sweep-{index:03d}-{label}.jsonl", records)
    for result_dir, rows in rows_by_dir.items():
        with (result_dir / "sweep-results.csv").open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    result = analyze_expkb([tmp_path, remainder], warmup_requests=0)

    assert result["arms"]["no-clip"]["seed_count"] == 2
    assert result["arms"]["clip512"]["aggregates"]["queue_wait_mean_s"]["mean"] == 0.2
    assert (
        result["arms"]["clip512"]["mg1"]["second_moment_service_s2"]
        < result["arms"]["no-clip"]["mg1"]["second_moment_service_s2"]
    )
