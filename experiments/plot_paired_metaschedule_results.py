#!/usr/bin/env python3
"""Plot paired Relax/MetaSchedule ResNet-18 performance and tuning robustness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "CUDA": "#2A6FBB",
    "SYCL": "#D65A31",
    "Conv2d 1x1": "#C23B4A",
    "Conv2d 3x3": "#E58B2A",
    "Dense/matmul": "#4F8C6D",
    "Pooling": "#7A64A8",
    "Reshape": "#777777",
    "Elementwise/reduction": "#3A8D9D",
}


def number(value: str):
    return None if value in ("", "None") else float(value)


def read_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field in (
                "task_id",
                "weight",
                "cuda_trials",
                "cuda_failed",
                "sycl_trials",
                "sycl_failed",
            ):
                row[field] = int(raw[field])
            for field in ("cuda_best_ms", "sycl_best_ms", "sycl_over_cuda", "weighted_gap_ms"):
                row[field] = number(raw[field])
            rows.append(row)
    return rows


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def performance_figure(root: Path, rows: list[dict[str, object]], output: Path) -> None:
    summary = json.loads((root / "paired_metaschedule_summary.json").read_text(encoding="utf-8"))
    cuda_runs = summary["execution"]["cuda"]["p50_ms_by_run"]
    sycl_runs = summary["execution"]["sycl"]["p50_ms_by_run"]
    paired = [row for row in rows if row["sycl_over_cuda"] is not None]
    paired.sort(key=lambda row: row["sycl_over_cuda"])

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), gridspec_kw={"width_ratios": [0.8, 2.2]})
    ax = axes[0]
    values = [cuda_runs, sycl_runs]
    medians = [np.median(value) for value in values]
    peak = max(max(value) for value in values)
    label_offset = peak * 0.03
    ax.bar([0, 1], medians, width=0.58, color=[COLORS["CUDA"], COLORS["SYCL"]], zorder=2)
    for index, runs in enumerate(values):
        ax.scatter(np.full(len(runs), index), runs, s=18, color="white", edgecolor="black", linewidth=0.6, zorder=3)
        ax.text(index, medians[index] + label_offset, f"{medians[index]:.3f}", ha="center", va="bottom")
    ratio = medians[1] / medians[0]
    ax.set_xticks([0, 1], ["CUDA", "SYCL"])
    ax.set_ylabel("Complete-model p50 (ms)")
    ax.set_ylim(0, peak * 1.16)
    ax.set_title(f"(a) ResNet-18 B1, {ratio:.2f}x", pad=6)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)

    ax = axes[1]
    x = np.arange(len(paired))
    for group in sorted({row["operator_group"] for row in paired}):
        indices = [index for index, row in enumerate(paired) if row["operator_group"] == group]
        ax.scatter(
            indices,
            [paired[index]["sycl_over_cuda"] for index in indices],
            label=group,
            s=28,
            color=COLORS.get(group, "#555555"),
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
    ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Best-record SYCL/CUDA ratio")
    ax.set_xlabel("Structurally matched workload (sorted by ratio)")
    ax.set_xticks(x, [str(row["task_id"]) for row in paired])
    ax.set_title(f"(b) {len(paired)} workloads with finite records on both targets")
    ax.grid(axis="y", which="both", color="#E3E3E3", linewidth=0.55)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout(pad=0.8, w_pad=1.2)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def robustness_figure(rows: list[dict[str, object]], output: Path) -> None:
    totals = {
        "CUDA": (sum(row["cuda_trials"] for row in rows), sum(row["cuda_failed"] for row in rows)),
        "SYCL": (sum(row["sycl_trials"] for row in rows), sum(row["sycl_failed"] for row in rows)),
    }
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.35, 5.25),
        gridspec_kw={"height_ratios": [0.72, 3.28]},
    )

    def simplify_axes(ax: plt.Axes, *, keep_left: bool = True) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(keep_left)
        ax.spines["bottom"].set_color("#6B7280")
        if keep_left:
            ax.spines["left"].set_color("#6B7280")

    ax = axes[0]
    targets = ("CUDA", "SYCL")
    y = np.arange(len(targets))
    failure_rates = [100 * totals[target][1] / totals[target][0] for target in targets]
    bars = ax.barh(
        y,
        failure_rates,
        height=0.48,
        color=[COLORS[target] for target in targets],
        zorder=3,
    )
    for bar, target, rate in zip(bars, targets, failure_rates):
        total, failed = totals[target]
        ax.text(
            rate + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{failed}/{total}  ({rate:.1f}%)",
            va="center",
            ha="left",
            fontsize=7.5,
        )
    ax.set_yticks(y, targets)
    ax.invert_yaxis()
    ax.set_xlim(0, 54)
    ax.set_xticks([0, 25, 50])
    ax.set_xlabel("Unsuccessful records (%)", labelpad=2)
    ax.set_title("(a) Equal-budget aggregate", loc="left", pad=5, fontweight="semibold")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0)
    simplify_axes(ax, keep_left=False)

    ax = axes[1]
    ordered = sorted(
        rows,
        key=lambda row: max(
            row["cuda_failed"] / max(row["cuda_trials"], 1),
            row["sycl_failed"] / max(row["sycl_trials"], 1),
        ),
        reverse=True,
    )
    y = np.arange(len(ordered))
    cuda_rate = np.asarray([100 * row["cuda_failed"] / max(row["cuda_trials"], 1) for row in ordered])
    sycl_rate = np.asarray([100 * row["sycl_failed"] / max(row["sycl_trials"], 1) for row in ordered])
    for index, (cuda_value, sycl_value) in enumerate(zip(cuda_rate, sycl_rate)):
        ax.plot(
            [cuda_value, sycl_value],
            [index, index],
            color="#C7CDD4",
            linewidth=0.8,
            zorder=1,
        )
    ax.scatter(
        cuda_rate,
        y,
        s=28,
        facecolor="white",
        edgecolor=COLORS["CUDA"],
        linewidth=1.15,
        marker="o",
        label="CUDA",
        zorder=4,
    )
    ax.scatter(
        sycl_rate,
        y,
        s=18,
        color=COLORS["SYCL"],
        edgecolor="white",
        linewidth=0.35,
        marker="D",
        label="SYCL",
        zorder=5,
    )
    ax.set_yticks(y, [str(row["task_id"]) for row in ordered])
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Unsuccessful records (%)")
    ax.set_ylabel("Workload ID")
    ax.set_title("(b) Failure concentration by workload", loc="left", pad=5, fontweight="semibold")
    ax.legend(frameon=False, ncol=2, loc="lower right", handletextpad=0.35, columnspacing=1.0)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.55, zorder=0)
    ax.tick_params(axis="y", length=0)
    simplify_axes(ax)
    fig.tight_layout(pad=0.6, h_pad=1.0)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--performance-out", type=Path, required=True)
    parser.add_argument("--robustness-out", type=Path, required=True)
    args = parser.parse_args()
    style()
    rows = read_rows(args.root / "record_analysis" / "paired_metaschedule_workloads.csv")
    performance_figure(args.root, rows, args.performance_out)
    robustness_figure(rows, args.robustness_out)


if __name__ == "__main__":
    main()
