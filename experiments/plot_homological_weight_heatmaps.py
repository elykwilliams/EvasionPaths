#!/usr/bin/env python3
"""Plot 2x4 heatmaps of homological motion weights across (n, r) grid cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


WEIGHT_ORDER = [
    "max_speed",
    "lambda_shrink",
    "mu_curvature",
    "eta_cohesion",
    "repulsion_strength",
    "repulsion_power",
    "d_safe_manual",
    "auto_d_safe",
]


def parse_combo_key(key: str) -> Tuple[int, float]:
    # Keys are stored like "n=10,r=0.240000".
    left, right = key.split(",")
    n = int(left.split("=")[1])
    r = float(right.split("=")[1])
    return n, r


def candidate_score(combo_metrics: Dict[str, float], *, failure_penalty: float, worst_case_weight: float) -> float:
    return (
        float(combo_metrics["mean_tau"])
        + failure_penalty * float(combo_metrics["failure_rate"])
        + worst_case_weight * float(combo_metrics["worst_tau"])
    )


def build_best_per_combo_map(
    trials: List[Dict],
    *,
    failure_penalty: float,
    worst_case_weight: float,
) -> Dict[Tuple[int, float], Dict]:
    best: Dict[Tuple[int, float], Dict] = {}
    for trial in trials:
        weights = trial["weights"]
        for combo_key, combo_metrics in trial["by_combo"].items():
            combo = parse_combo_key(combo_key)
            score = candidate_score(
                combo_metrics,
                failure_penalty=failure_penalty,
                worst_case_weight=worst_case_weight,
            )
            prev = best.get(combo)
            if prev is None or score < prev["score"]:
                best[combo] = {
                    "score": score,
                    "weights": weights,
                    "trial": int(trial.get("trial", -1)),
                    "combo_metrics": combo_metrics,
                }
    return best


def plot_heatmaps(
    best_map: Dict[Tuple[int, float], Dict],
    *,
    output_path: Path,
    title: str,
) -> None:
    n_values = sorted({n for (n, _r) in best_map})
    r_values = sorted({r for (_n, r) in best_map})

    fig, axes = plt.subplots(4, 2, figsize=(15, 19), constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for idx, weight in enumerate(WEIGHT_ORDER):
        ax = axes[idx // 2, idx % 2]
        grid = np.zeros((len(n_values), len(r_values)), dtype=float)

        for i, n in enumerate(n_values):
            for j, r in enumerate(r_values):
                record = best_map[(n, r)]
                value = record["weights"][weight]
                grid[i, j] = float(value)

        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(weight)
        ax.set_xticks(np.arange(len(r_values)))
        ax.set_yticks(np.arange(len(n_values)))
        ax.set_xticklabels([f"{r:.2f}" for r in r_values], rotation=0)
        ax.set_yticklabels([str(n) for n in n_values])
        ax.set_xlabel("r (sensing radius)")
        ax.set_ylabel("n (mobile sensors)")

        # Numeric annotation helps read exact values from each cell.
        grid_span = float(np.ptp(grid))
        for i in range(len(n_values)):
            for j in range(len(r_values)):
                v = grid[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:.3f}" if weight != "auto_d_safe" else f"{int(round(v))}",
                    ha="center",
                    va="center",
                    color="white" if (v - grid.min()) / (grid_span + 1e-12) > 0.45 else "black",
                    fontsize=8,
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 8 weight heatmaps over (n, r) grid.")
    parser.add_argument(
        "--trials-json",
        type=str,
        default="experiments/output/homological_weight_search/default_grid_t5_trials30_seed19/all_trials_full.json",
        help="Path to all_trials_full.json produced by homological_weight_search.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/output/homological_weight_search/weight_heatmaps_best_per_combo.png",
    )
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--worst-case-weight", type=float, default=0.25)
    parser.add_argument(
        "--title",
        type=str,
        default="HomologicalDynamicsMotion weights by (n, r) cell (best trial per cell)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trials_path = Path(args.trials_json)
    output_path = Path(args.output)

    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    best_map = build_best_per_combo_map(
        trials,
        failure_penalty=args.failure_penalty,
        worst_case_weight=args.worst_case_weight,
    )
    plot_heatmaps(best_map, output_path=output_path, title=args.title)
    print(f"Wrote heatmap figure: {output_path}")


if __name__ == "__main__":
    main()
