"""Headless plot regeneration tests using aggregate sweep fixtures only."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sloserve.analysis.plot_results import plot_sweep_results, read_sweep_results_csv
from sloserve.cli import build_parser, main


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for policy, throughput_offset in (("fcfs", 0.0), ("slo_aware", 5.0)):
        for rate in (0.5, 1.0, 2.0):
            rows.append(
                {
                    "label": f"{policy}-{rate}",
                    "policy": policy,
                    "request_rate_rps": rate,
                    "token_throughput_per_s": 20.0 * rate + throughput_offset,
                    "end_to_end_p99_s": 0.4 + rate,
                    "slo_overall_rate": 1.0 - 0.1 * rate,
                    "slo_interactive_rate": 1.0 - 0.05 * rate,
                    "slo_batch_rate": 1.0 - 0.15 * rate,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _assert_pngs(paths: tuple[Path, ...]) -> None:
    assert [path.name for path in paths] == [
        "throughput-latency.png",
        "rate-p99.png",
        "slo-attainment-rate.png",
    ]
    for path in paths:
        assert path.is_file()
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plots_regenerate_from_saved_csv(tmp_path: Path) -> None:
    results_path = tmp_path / "sweep-results.csv"
    _write_csv(results_path, _rows())

    loaded = read_sweep_results_csv(results_path)
    paths = plot_sweep_results(results_path, tmp_path / "figures")

    assert len(loaded) == 6
    assert loaded[0]["request_rate_rps"] == "0.5"
    _assert_pngs(paths)


def test_single_policy_and_missing_per_class_rates_are_supported(tmp_path: Path) -> None:
    row = {
        "label": "single",
        "policy": "fcfs",
        "request_rate_rps": 1.0,
        "token_throughput_per_s": 25.0,
        "end_to_end_p99_s": 0.75,
        "slo_overall_rate": 0.9,
        "slo_interactive_rate": None,
        "slo_batch_rate": "",
    }

    paths = plot_sweep_results([row], tmp_path / "single-policy")

    _assert_pngs(paths)


def test_plot_cli_parser() -> None:
    args = build_parser().parse_args(["plot", "--results", "sweep-results.csv", "--out", "figures"])

    assert args.command == "plot"
    assert args.results == "sweep-results.csv"
    assert args.out == "figures"


def test_plot_cli_writes_all_figures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    results_path = tmp_path / "sweep-results.csv"
    figures_path = tmp_path / "figures"
    _write_csv(results_path, _rows())

    exit_code = main(["plot", "--results", str(results_path), "--out", str(figures_path)])

    assert exit_code == 0
    output = capsys.readouterr().out
    for filename in (
        "throughput-latency.png",
        "rate-p99.png",
        "slo-attainment-rate.png",
    ):
        assert f"figure={figures_path / filename}" in output
