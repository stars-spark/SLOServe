"""Headless figures regenerated solely from aggregated sweep-result rows."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

PlotRow: TypeAlias = Mapping[str, object]


def read_sweep_results_csv(path: str | Path) -> tuple[dict[str, str], ...]:
    """Read the saved aggregate CSV without requiring a dataframe dependency."""
    with Path(path).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("sweep results CSV has no header")
        return tuple(dict(row) for row in reader)


def _pyplot() -> Any:
    """Import matplotlib lazily and force its non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot

    return pyplot


def _number(row: PlotRow, column: str) -> float | None:
    value = row.get(column)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {column}: {value!r}") from exc


def _policies(rows: Sequence[PlotRow]) -> tuple[str, ...]:
    policies = {str(row["policy"]) for row in rows if row.get("policy") not in (None, "")}
    if not policies:
        raise ValueError("sweep results contain no policy values")
    return tuple(sorted(policies))


def _xy_for_policy(
    rows: Sequence[PlotRow], policy: str, x_column: str, y_column: str
) -> tuple[list[float], list[float]]:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        if str(row.get("policy")) != policy:
            continue
        x_value = _number(row, x_column)
        y_value = _number(row, y_column)
        if x_value is not None and y_value is not None:
            pairs.append((x_value, y_value))
    pairs.sort()
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _save_figure(figure: Any, path: Path, pyplot: Any) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    pyplot.close(figure)
    return path


def _plot_throughput_latency(rows: Sequence[PlotRow], path: Path, pyplot: Any) -> Path:
    figure, axis = pyplot.subplots()
    for policy in _policies(rows):
        x_values, y_values = _xy_for_policy(
            rows, policy, "token_throughput_per_s", "end_to_end_p99_s"
        )
        if x_values:
            axis.plot(x_values, y_values, marker="o", label=policy)
    axis.set_xlabel("Output token throughput (tokens/s)")
    axis.set_ylabel("End-to-end P99 latency (s)")
    axis.set_title("Throughput-Latency Trade-off")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save_figure(figure, path, pyplot)


def _plot_rate_p99(rows: Sequence[PlotRow], path: Path, pyplot: Any) -> Path:
    figure, axis = pyplot.subplots()
    for policy in _policies(rows):
        x_values, y_values = _xy_for_policy(rows, policy, "request_rate_rps", "end_to_end_p99_s")
        if x_values:
            axis.plot(x_values, y_values, marker="o", label=policy)
    axis.set_xlabel("Request rate (requests/s)")
    axis.set_ylabel("End-to-end P99 latency (s)")
    axis.set_title("Request Rate vs End-to-End P99 Latency")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save_figure(figure, path, pyplot)


def _plot_slo_attainment(rows: Sequence[PlotRow], path: Path, pyplot: Any) -> Path:
    figure, axis = pyplot.subplots()
    rate_columns = (
        ("slo_overall_rate", "overall"),
        ("slo_interactive_rate", "interactive"),
        ("slo_batch_rate", "batch"),
    )
    for policy in _policies(rows):
        for column, class_label in rate_columns:
            x_values, y_values = _xy_for_policy(rows, policy, "request_rate_rps", column)
            if x_values:
                axis.plot(
                    x_values,
                    y_values,
                    marker="o",
                    label=f"{policy} — {class_label}",
                )
    axis.set_xlabel("Request rate (requests/s)")
    axis.set_ylabel("SLO attainment rate")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("SLO Attainment vs Request Rate")
    axis.legend()
    axis.grid(alpha=0.25)
    return _save_figure(figure, path, pyplot)


def plot_sweep_results(
    results: str | Path | Sequence[PlotRow], output_directory: str | Path
) -> tuple[Path, ...]:
    """Write the three Week 3 figures from a CSV path or equivalent rows."""
    rows = read_sweep_results_csv(results) if isinstance(results, (str, Path)) else tuple(results)
    if not rows:
        raise ValueError("sweep results must contain at least one row")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    pyplot = _pyplot()
    return (
        _plot_throughput_latency(rows, output_path / "throughput-latency.png", pyplot),
        _plot_rate_p99(rows, output_path / "rate-p99.png", pyplot),
        _plot_slo_attainment(rows, output_path / "slo-attainment-rate.png", pyplot),
    )
