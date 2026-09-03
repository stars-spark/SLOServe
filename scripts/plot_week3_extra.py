"""Regenerate the Week 3 report figures that the built-in `sloserve plot` does not cover.

Reads only the committed aggregate CSVs under results/raw/ and writes PNGs to results/figures/.
Headless (Agg). Run from the repo root:  uv run python scripts/plot_week3_extra.py
"""

from __future__ import annotations

import csv
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt

OUT = Path("results/figures")
COLORS = {"fcfs": "#c0453b", "static_priority": "#3b78c0", "slo_aware": "#3ba055"}
NAMES = {"fcfs": "FCFS", "static_priority": "Static Priority", "slo_aware": "SLO-Aware"}


def load(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def saturation_robustness() -> None:
    """expG: interactive SLO and tail TTFT at rps=3.0, mean +/- std over six seeds."""
    rows = load("results/raw/week3-expG-robust/sweep-results.csv")
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by[row["policy"]].append(row)
    order = ["fcfs", "static_priority", "slo_aware"]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(order))
    colors = [COLORS[p] for p in order]
    for axis, column, title, ylabel, ymax in (
        (
            left,
            "slo_interactive_rate",
            "Interactive SLO at saturation (rps=3.0)",
            "Interactive SLO attainment",
            1.05,
        ),
        (right, "ttft_p99_s", "Tail TTFT at saturation (rps=3.0)", "TTFT P99 (s)", None),
    ):
        means = [st.mean([float(r[column]) for r in by[p]]) for p in order]
        stds = [st.pstdev([float(r[column]) for r in by[p]]) for p in order]
        axis.bar(x, means, yerr=stds, capsize=6, color=colors)
        axis.set_xticks(list(x))
        axis.set_xticklabels([NAMES[p] for p in order])
        axis.set_ylabel(ylabel)
        if ymax is not None:
            axis.set_ylim(0, ymax)
        axis.set_title(f"{title}\nmean ± std over 6 seeds")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT / "saturation-robustness.png", dpi=160)
    plt.close(figure)
    print("wrote saturation-robustness.png")


def aging_tradeoff() -> None:
    """expF: interactive SLO vs longest queue wait as the hard-aging threshold grows."""
    rows = load("results/raw/week3-expF-aging/sweep-results.csv")
    labels = [r["label"].replace("aging-", "") for r in rows]
    slo = [float(r["slo_interactive_rate"]) for r in rows]
    wait = [float(r["longest_queue_wait_s"]) for r in rows]
    figure, axis = plt.subplots(figsize=(7, 4.2))
    x = range(len(rows))
    axis.plot(x, slo, "o-", color=COLORS["slo_aware"], label="Interactive SLO")
    axis.set_ylabel("Interactive SLO attainment", color=COLORS["slo_aware"])
    axis.set_ylim(0, 1.05)
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels)
    axis.set_xlabel("Hard-aging threshold")
    twin = axis.twinx()
    twin.plot(x, wait, "s--", color="#c67f0c", label="Longest queue wait")
    twin.set_ylabel("Longest queue wait (s)", color="#c67f0c")
    axis.set_title(
        "Aging threshold trade-off (slo_aware, rps=3.0, single session)\n"
        "higher threshold: SLO ordering preserved longer, worst-case wait grows"
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUT / "aging-tradeoff.png", dpi=160)
    plt.close(figure)
    print("wrote aging-tradeoff.png")


def engine_params() -> None:
    """expD: interactive SLO of FCFS vs SLO-Aware across vLLM engine configs (single seed)."""
    combos = ["mns8", "mns16", "mns32", "chunked"]
    labels = ["max_seqs=8", "max_seqs=16", "max_seqs=32", "chunked prefill"]
    fcfs, slo = [], []
    for name in combos:
        rows = {r["policy"]: r for r in load(f"results/raw/week3-expD-{name}/sweep-results.csv")}
        fcfs.append(float(rows["fcfs"]["slo_interactive_rate"]))
        slo.append(float(rows["slo_aware"]["slo_interactive_rate"]))
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(combos))
    width = 0.38
    axis.bar([i - width / 2 for i in x], fcfs, width, label="FCFS", color=COLORS["fcfs"])
    axis.bar([i + width / 2 for i in x], slo, width, label="SLO-Aware", color=COLORS["slo_aware"])
    axis.set_xticks(list(x))
    axis.set_xticklabels(labels)
    axis.set_ylabel("Interactive SLO attainment")
    axis.set_ylim(0, 1.05)
    axis.set_title(
        "Engine params vs scheduling policy (rps=3.0, single seed)\n"
        "SLO-Aware leads FCFS under every vLLM configuration"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUT / "engine-params.png", dpi=160)
    plt.close(figure)
    print("wrote engine-params.png")


def multilevel_aging() -> None:
    """expI multi-seed: interactive SLO and the batch/wait trade as aging levels K grow."""
    rows = load("results/raw/week4-expI-multiseed/sweep-results.csv")
    by: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by[int(row["label"].split("-")[0][1:])].append(row)
    order = [1, 2, 3, 5]
    labels = [f"K={k}" for k in order]

    def mean_of(column: str) -> list[float]:
        return [st.mean([float(r[column]) for r in by[k]]) for k in order]

    def std_of(column: str) -> list[float]:
        return [st.pstdev([float(r[column]) for r in by[k]]) for k in order]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(order))
    left.bar(
        x,
        mean_of("slo_interactive_rate"),
        yerr=std_of("slo_interactive_rate"),
        capsize=6,
        color=COLORS["slo_aware"],
    )
    left.set_xticks(list(x))
    left.set_xticklabels(labels)
    left.set_ylabel("Interactive SLO attainment")
    left.set_ylim(0, 1.05)
    left.set_title("Interactive SLO vs aging levels (Poisson, rps=3.0)\nmean ± std over 6 seeds")
    left.grid(axis="y", alpha=0.25)

    wait_color = "#c67f0c"
    right.plot(x, mean_of("longest_queue_wait_s"), "s--", color=wait_color, label="Longest wait")
    right.set_ylabel("Longest queue wait (s)", color=wait_color)
    right.set_xticks(list(x))
    right.set_xticklabels(labels)
    right.set_xlabel("Aging levels (K)")
    twin = right.twinx()
    twin.plot(x, mean_of("slo_batch_rate"), "o-", color=COLORS["slo_aware"], label="Batch SLO")
    twin.set_ylabel("Batch SLO attainment", color=COLORS["slo_aware"])
    right.set_title(
        "Secondary metrics vs aging levels\nmore tiers: shorter worst-case wait, higher batch SLO"
    )
    right.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUT / "multilevel-aging.png", dpi=160)
    plt.close(figure)
    print("wrote multilevel-aging.png")


def adaptive_ceiling() -> None:
    """expJ: adaptive ceiling versus fixed thresholds, interactive SLO and worst-case wait."""
    rows = load("results/raw/week5-expJ-adaptive/sweep-results.csv")
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by[row["label"].rsplit("-", 1)[0]].append(row)
    order = ["fix3", "fix10", "fix30", "adp-cap10", "adp-cap30"]
    labels = ["fix 3s", "fix 10s", "fix 30s", "adaptive\ncap 10s", "adaptive\ncap 30s"]
    # Fixed thresholds in blue, the failed adaptive variants in red.
    colors = [COLORS["static_priority"]] * 3 + [COLORS["fcfs"]] * 2

    def mean_of(column: str) -> list[float]:
        return [st.mean([float(r[column]) for r in by[v]]) for v in order]

    def std_of(column: str) -> list[float]:
        return [st.pstdev([float(r[column]) for r in by[v]]) for v in order]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(order))
    for axis, column, title, ylabel, ymax in (
        (left, "slo_interactive_rate", "Interactive SLO", "Interactive SLO attainment", 1.05),
        (right, "longest_queue_wait_s", "Worst-case wait", "Longest queue wait (s)", None),
    ):
        axis.bar(x, mean_of(column), yerr=std_of(column), capsize=6, color=colors)
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels)
        axis.set_ylabel(ylabel)
        if ymax is not None:
            axis.set_ylim(0, ymax)
        axis.set_title(f"{title} (Poisson, rps=3.0)\nmean ± std over 6 seeds")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Adaptive ceiling fails; a well-set fixed threshold (10-30 s) is best and most stable",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(OUT / "adaptive-ceiling.png", dpi=160)
    plt.close(figure)
    print("wrote adaptive-ceiling.png")


def length_source_comparison() -> None:
    """expK-A: interactive SLO and latency for true/advertised/learned length estimates."""
    rows = load("results/raw/week6-expK-A/sweep-results.csv")
    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by[row["label"].rsplit("-", 1)[0]].append(row)
    order = ["true", "adv", "learned"]
    labels = ["true\n(oracle)", "advertised\n(naive)", "learned"]
    colors = [COLORS["static_priority"], COLORS["fcfs"], COLORS["slo_aware"]]

    def mean_of(column: str) -> list[float]:
        return [st.mean([float(r[column]) for r in by[s]]) for s in order]

    def std_of(column: str) -> list[float]:
        return [st.pstdev([float(r[column]) for r in by[s]]) for s in order]

    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(order))
    left.bar(
        x,
        mean_of("slo_interactive_rate"),
        yerr=std_of("slo_interactive_rate"),
        capsize=6,
        color=colors,
    )
    left.set_xticks(list(x))
    left.set_xticklabels(labels)
    left.set_ylabel("Interactive SLO attainment")
    left.set_ylim(0, 1.05)
    left.set_title("Interactive SLO by length estimate\n(realistic workload, rps=1.2, 6 seeds)")
    left.grid(axis="y", alpha=0.25)

    width = 0.38
    p50 = mean_of("end_to_end_p50_s")
    p99 = mean_of("end_to_end_p99_s")
    right.bar([i - width / 2 for i in x], p50, width, label="e2e P50", color=COLORS["slo_aware"])
    right.bar([i + width / 2 for i in x], p99, width, label="e2e P99", color="#c67f0c")
    right.set_xticks(list(x))
    right.set_xticklabels(labels)
    right.set_ylabel("End-to-end latency (s)")
    right.set_title("Latency by length estimate\noracle lowest, coarse learned highest")
    right.legend()
    right.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "True length (oracle) helps SLO-aware scheduling; a coarse learned predictor "
        "underperforms even the naive cap (low MAE != good ordering)",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(OUT / "length-source.png", dpi=160)
    plt.close(figure)
    print("wrote length-source.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saturation_robustness()
    aging_tradeoff()
    engine_params()
    multilevel_aging()
    adaptive_ceiling()
    length_source_comparison()


if __name__ == "__main__":
    main()
