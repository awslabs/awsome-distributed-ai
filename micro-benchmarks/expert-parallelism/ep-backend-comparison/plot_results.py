#!/usr/bin/env python3
"""Render backend box plots from an EP comparison summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Patch


ARMS = ("uccl", "deepep-v1-nvshmem", "deepep-v2-gin-gda")
ARM_STYLES = {
    "uccl": {
        "color": "#0072B2",
        "hatch": "///",
        "label": "UCCL",
        "marker": "o",
    },
    "deepep-v1-nvshmem": {
        "color": "#E69F00",
        "hatch": "\\\\",
        "label": "DeepEP V1 NVSHMEM",
        "marker": "s",
    },
    "deepep-v2-gin-gda": {
        "color": "#009E73",
        "hatch": "...",
        "label": "DeepEP V2 NCCL GIN",
        "marker": "^",
    },
}
DTYPE_ORDER = ("fp8", "bf16")
PROFILE_CONFIG = {
    "decode": {
        "title": "Decode-like: slowest-rank latency (lower is better)",
        "ylabel": "Latency (ms)",
    },
    "prefill": {
        "title": "Prefill-like: slowest-rank latency (lower is better)",
        "ylabel": "Latency (ms)",
    },
}


def load_plot_data(
    path: Path,
) -> tuple[
    int,
    dict[tuple[str, int, str], dict[str, Any]],
    tuple[tuple[int, str], ...],
]:
    summary = json.loads(path.read_text())
    if summary.get("status") != "PASS":
        raise ValueError("the input summary does not have PASS status")

    starts = summary.get("independent_starts_per_cell")
    if not isinstance(starts, int) or starts < 1:
        raise ValueError("independent_starts_per_cell must be a positive integer")

    world_sizes = summary.get("configuration", {}).get("world_sizes_ranks")
    if (
        not isinstance(world_sizes, list)
        or not world_sizes
        or not all(isinstance(world, int) and world > 0 for world in world_sizes)
    ):
        raise ValueError("world_sizes_ranks must be a nonempty list of EP sizes")
    cell_order = tuple(
        (world_size, dtype) for world_size in world_sizes for dtype in DTYPE_ORDER
    )

    cells: dict[tuple[str, int, str], dict[str, Any]] = {}
    for cell in summary.get("cells", []):
        key = (
            cell.get("workload_profile"),
            cell.get("world_size_ranks"),
            cell.get("dispatch_dtype"),
        )
        if key in cells:
            raise ValueError(f"duplicate workload cell: {key}")
        cells[key] = cell

    expected_keys = {
        (profile, world_size, dtype)
        for profile in PROFILE_CONFIG
        for world_size, dtype in cell_order
    }
    if cells.keys() != expected_keys:
        missing = sorted(expected_keys - cells.keys())
        extra = sorted(cells.keys() - expected_keys)
        raise ValueError(f"workload cell mismatch; missing={missing}, extra={extra}")

    for key, cell in cells.items():
        arms = cell.get("arms", {})
        if arms.keys() != set(ARMS):
            missing = sorted(set(ARMS) - arms.keys())
            extra = sorted(arms.keys() - set(ARMS))
            raise ValueError(
                f"backend arm mismatch for {key}; missing={missing}, extra={extra}"
            )
        for arm in ARMS:
            values = arms[arm].get("per_start_primary_values")
            if not isinstance(values, list) or len(values) != starts:
                raise ValueError(
                    f"{key}/{arm} must contain {starts} per-start primary values"
                )
            if not all(
                isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                for value in values
            ):
                raise ValueError(f"{key}/{arm} contains an invalid primary value")
    return starts, cells, cell_order


def render_box_plots(summary_path: Path, output_path: Path) -> None:
    starts, cells, cell_order = load_plot_data(summary_path)
    matplotlib.rcParams.update(
        {
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "font.size": 10,
            "savefig.facecolor": "white",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
        }
    )

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    group_positions = list(range(1, len(cell_order) + 1))
    arm_offsets = (-0.24, 0.0, 0.24)
    # Keep the per-start markers inside their 0.19-wide box for any start count.
    point_spacing = min(0.025, 0.17 / max(starts - 1, 1))
    point_offsets = tuple(
        (index - (starts - 1) / 2) * point_spacing for index in range(starts)
    )

    for axis, profile in zip(axes, PROFILE_CONFIG, strict=True):
        all_values: list[float] = []
        for arm, arm_offset in zip(ARMS, arm_offsets, strict=True):
            values_by_cell = [
                cells[(profile, world_size, dtype)]["arms"][arm][
                    "per_start_primary_values"
                ]
                for world_size, dtype in cell_order
            ]
            all_values.extend(value for values in values_by_cell for value in values)
            positions = [position + arm_offset for position in group_positions]
            style = ARM_STYLES[arm]
            box_plot = axis.boxplot(
                values_by_cell,
                positions=positions,
                widths=0.19,
                whis=(0, 100),
                showfliers=False,
                patch_artist=True,
                manage_ticks=False,
                boxprops={
                    "facecolor": style["color"],
                    "edgecolor": "#222222",
                    "hatch": style["hatch"],
                    "linewidth": 1.0,
                    "alpha": 0.55,
                },
                whiskerprops={"color": "#333333", "linewidth": 1.0},
                capprops={"color": "#333333", "linewidth": 1.0},
                medianprops={"color": "#111111", "linewidth": 1.8},
            )
            for median in box_plot["medians"]:
                median.set_zorder(4)
            for position, values in zip(positions, values_by_cell, strict=True):
                axis.scatter(
                    [position + offset for offset in point_offsets],
                    values,
                    color=style["color"],
                    edgecolor="#111111",
                    linewidth=0.6,
                    marker=style["marker"],
                    s=29,
                    zorder=5,
                )

        config = PROFILE_CONFIG[profile]
        axis.set_title(config["title"], loc="left", fontweight="bold", pad=10)
        axis.set_ylabel(config["ylabel"])
        axis.set_ylim(0, max(all_values) * 1.12)
        axis.set_xlim(0.5, len(cell_order) + 0.5)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[-1].set_xticks(
        group_positions,
        [f"{world_size} ranks\n{dtype.upper()}" for world_size, dtype in cell_order],
    )
    axes[-1].set_xlabel("Expert-parallel size and dispatch dtype", labelpad=9)

    legend_handles = [
        Patch(
            facecolor=ARM_STYLES[arm]["color"],
            edgecolor="#222222",
            hatch=ARM_STYLES[arm]["hatch"],
            alpha=0.55,
            label=ARM_STYLES[arm]["label"],
        )
        for arm in ARMS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        frameon=False,
        ncol=len(ARMS),
    )
    figure.suptitle(
        "B200 expert-parallel backend comparison",
        fontsize=15,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        (
            f"Each arm has {starts} independent process starts. "
            "Box: Q1 to Q3; line: median; whiskers: minimum to maximum; "
            "markers: per-start medians."
        ),
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.12, hspace=0.34)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=160,
        metadata={"Software": "matplotlib via plot_results.py"},
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="machine-readable summary JSON")
    parser.add_argument("--output", type=Path, required=True, help="output image path")
    args = parser.parse_args()
    render_box_plots(args.summary, args.output)


if __name__ == "__main__":
    main()
