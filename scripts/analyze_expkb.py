"""Aggregate expK-B clipping trade-offs and the deliberately limited M/G/1 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from sloserve.analysis.expkb import analyze_expkb


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/raw/week6-expK-B"),
        help="directory containing completed expK-B sweep outputs",
    )
    parser.add_argument(
        "--additional-results",
        type=Path,
        action="append",
        default=[],
        help="additional result directory from a resumed sweep (repeatable)",
    )
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("results/figures/clipping-tradeoff.png"),
    )
    return parser.parse_args()


def _mean(analysis: dict[str, Any], arm: str, field: str) -> float:
    aggregate = analysis["arms"][arm]["aggregates"][field]
    if aggregate is None:
        return 0.0
    return float(aggregate["mean"])


def _plot(analysis: dict[str, Any], path: Path) -> None:
    arms = [
        arm for arm in ("no-clip", "clip1536", "clip1024", "clip512") if arm in analysis["arms"]
    ]
    labels = ["none", "1536", "1024", "512"][: len(arms)]

    def normalized(values: list[float]) -> list[float]:
        return [value / values[0] for value in values]

    # Left panel tells the mechanism -> prediction -> measurement chain from Xu et al.: clipping
    # shrinks the service second moment, Pollaczek-Khinchine turns that into a predicted wait, and
    # the measured wait is what actually happened. Plotting all three shows where the crude
    # single-server bridge over- or under-shoots instead of hiding it behind one curve.
    second_moments = normalized(
        [float(analysis["arms"][arm]["mg1"]["second_moment_service_s2"]) for arm in arms]
    )
    predicted_waits = [analysis["arms"][arm]["mg1"]["mean_queue_wait_s"] for arm in arms]
    measured = normalized([_mean(analysis, arm, "queue_wait_mean_s") for arm in arms])

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(
        labels, second_moments, marker="s", linestyle=":", label="fitted E[S²] (mechanism)"
    )
    if all(wait is not None for wait in predicted_waits):
        axes[0].plot(
            labels,
            normalized([float(wait) for wait in predicted_waits]),
            marker="^",
            linestyle="--",
            label="M/G/1 predicted E[W]",
        )
    axes[0].plot(labels, measured, marker="o", label="measured mean queue wait")
    axes[0].axhline(1.0, color="0.6", linewidth=1, linestyle="--")
    axes[0].set_xlabel("backend max-token cap")
    axes[0].set_ylabel("normalized to no clipping")
    axes[0].set_title("Queueing trend (M/G/1 is qualitative only)")
    axes[0].legend(fontsize=8)

    # Right panel is the trade-off itself. `clip_applied_rate` and `realized_truncation_rate` are
    # identical once force_exact_output_tokens is on -- every capped request runs to the cap -- so
    # plotting both wastes the panel. Show what clipping buys against what it costs instead.
    interactive_slo = [_mean(analysis, arm, "slo_interactive_rate") for arm in arms]
    truncated = [_mean(analysis, arm, "realized_truncation_rate") for arm in arms]
    throughput = [_mean(analysis, arm, "token_throughput_per_s") for arm in arms]

    width = 0.36
    positions = list(range(len(arms)))
    axes[1].bar(
        [position - width / 2 for position in positions],
        interactive_slo,
        width,
        color="#2a7f5f",
        label="interactive SLO attainment",
    )
    axes[1].bar(
        [position + width / 2 for position in positions],
        truncated,
        width,
        color="#b3452c",
        label="requests cut short by the cap",
    )
    axes[1].set_xticks(positions, labels)
    axes[1].set_xlabel("backend max-token cap")
    axes[1].set_ylabel("offered-request rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("What clipping buys, and what it costs")
    axes[1].legend(fontsize=8, loc="upper left")

    cost = axes[1].twinx()
    cost.plot(positions, throughput, marker="o", color="0.35", linewidth=1.2)
    cost.set_ylabel("token throughput (tok/s)", color="0.35")
    cost.tick_params(axis="y", labelcolor="0.35")
    cost.set_ylim(0.0, max(throughput) * 1.25)

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    result_dirs = [args.results, *args.additional_results]
    analysis = analyze_expkb(result_dirs, warmup_requests=args.warmup_requests)
    output = args.results / "clipping-analysis.json"
    output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(analysis, args.figure)
    print(output)
    print(args.figure)


if __name__ == "__main__":
    main()
