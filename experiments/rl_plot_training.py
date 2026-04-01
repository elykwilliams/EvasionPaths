#!/usr/bin/env python3
"""Generate training/evaluation plots for RL unit-square experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot RL training/eval diagnostics.")
    parser.add_argument("--run-dir", type=str, required=True, help="Training run directory.")
    parser.add_argument("--outdir", type=str, default="", help="Output plots directory (default: <run-dir>/plots).")
    parser.add_argument(
        "--reeb-dir",
        type=str,
        default="",
        help="Optional rl_reeb_from_model_* directory for per-sim policy-term plots.",
    )
    parser.add_argument("--rolling-window", type=int, default=50, help="Rolling window for event smoothing.")
    parser.add_argument("--train-bin-size", type=int, default=100, help="Bin width for train-event aggregation.")
    return parser


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    w = max(1, int(window))
    if w <= 1:
        return values
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(values, kernel, mode="same")


def _mean_and_ci95(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if samples.ndim == 1:
        samples = samples[:, None]
    mean = np.mean(samples, axis=1)
    if samples.shape[1] <= 1:
        return mean, mean, mean
    sem = np.std(samples, axis=1, ddof=1) / np.sqrt(samples.shape[1])
    delta = 1.96 * sem
    return mean, mean - delta, mean + delta


def _bin_series_with_ci(x: np.ndarray, y: np.ndarray, bin_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if x.size == 0 or y.size == 0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty
    width = max(1, int(bin_size))
    bin_ids = np.floor(x / float(width)).astype(int)
    centers = []
    means = []
    lowers = []
    uppers = []
    for bin_id in sorted(set(bin_ids.tolist())):
        mask = bin_ids == bin_id
        values = y[mask]
        if values.size == 0:
            continue
        center = (bin_id + 0.5) * float(width)
        mean = float(np.mean(values))
        if values.size > 1:
            sem = float(np.std(values, ddof=1) / np.sqrt(values.size))
            delta = 1.96 * sem
        else:
            delta = 0.0
        centers.append(center)
        means.append(mean)
        lowers.append(mean - delta)
        uppers.append(mean + delta)
    return (
        np.asarray(centers, dtype=float),
        np.asarray(means, dtype=float),
        np.asarray(lowers, dtype=float),
        np.asarray(uppers, dtype=float),
    )


def _plot_eval_npz(run_dir: Path, outdir: Path) -> Optional[Path]:
    npz_path = run_dir / "eval" / "evaluations.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    timesteps = np.asarray(data["timesteps"], dtype=float)
    results = np.asarray(data["results"], dtype=float)  # [num_eval, num_episodes]
    ep_lengths = np.asarray(data["ep_lengths"], dtype=float)
    if timesteps.size == 0:
        return None

    mean_reward, reward_lo, reward_hi = _mean_and_ci95(results)
    mean_len, len_lo, len_hi = _mean_and_ci95(ep_lengths)
    successes = np.asarray(data["successes"], dtype=float) if "successes" in data.files else None
    true_cycle_area_norm = (
        np.asarray(data["true_cycle_area_norm"], dtype=float) if "true_cycle_area_norm" in data.files else None
    )
    largest_true_cycle_area_norm = (
        np.asarray(data["largest_true_cycle_area_norm"], dtype=float)
        if "largest_true_cycle_area_norm" in data.files
        else None
    )
    if successes is not None and successes.size:
        success_rate = np.mean(successes, axis=1)
        if successes.shape[1] > 1:
            success_sem = np.sqrt(np.clip(success_rate * (1.0 - success_rate), 0.0, 1.0) / successes.shape[1])
            success_delta = 1.96 * success_sem
        else:
            success_delta = np.zeros_like(success_rate)
        success_lo = np.clip(success_rate - success_delta, 0.0, 1.0)
        success_hi = np.clip(success_rate + success_delta, 0.0, 1.0)
    else:
        success_rate = success_lo = success_hi = None

    has_area = true_cycle_area_norm is not None and true_cycle_area_norm.size > 0
    nrows = 4 if (success_rate is not None and has_area) else (3 if (success_rate is not None or has_area) else 2)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 11 if nrows == 4 else (9 if nrows == 3 else 7)), sharex=True)
    if nrows == 2:
        axes = np.asarray(axes)
    axes[0].plot(timesteps, mean_reward, color="#0b7285", lw=2, label="mean_reward")
    axes[0].fill_between(timesteps, reward_lo, reward_hi, color="#0b7285", alpha=0.2, label="95% CI")
    axes[0].set_ylabel("Eval Reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(timesteps, mean_len, color="#c92a2a", lw=2, label="mean_ep_length")
    axes[1].fill_between(timesteps, len_lo, len_hi, color="#c92a2a", alpha=0.2, label="95% CI")
    axes[1].set_ylabel("Eval Final Time")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    next_axis = 2
    if success_rate is not None:
        axes[next_axis].plot(timesteps, success_rate, color="#2f9e44", lw=2, label="success_rate")
        axes[next_axis].fill_between(timesteps, success_lo, success_hi, color="#2f9e44", alpha=0.2, label="95% CI")
        axes[next_axis].set_ylabel("Success Rate")
        axes[next_axis].set_ylim(-0.02, 1.02)
        axes[next_axis].grid(alpha=0.25)
        axes[next_axis].legend(loc="best")
        next_axis += 1

    if has_area:
        area_mean, area_lo, area_hi = _mean_and_ci95(true_cycle_area_norm)
        axes[next_axis].plot(timesteps, area_mean, color="#5f3dc4", lw=2, label="remaining_true_area_norm")
        axes[next_axis].fill_between(timesteps, area_lo, area_hi, color="#5f3dc4", alpha=0.2, label="95% CI")
        if largest_true_cycle_area_norm is not None and largest_true_cycle_area_norm.size > 0:
            largest_mean, _, _ = _mean_and_ci95(largest_true_cycle_area_norm)
            axes[next_axis].plot(
                timesteps,
                largest_mean,
                color="#e67700",
                lw=1.8,
                linestyle="--",
                label="largest_true_area_norm",
            )
        axes[next_axis].set_ylabel("Remaining Area")
        axes[next_axis].grid(alpha=0.25)
        axes[next_axis].legend(loc="best")

    axes[-1].set_xlabel("Timesteps")

    fig.suptitle("Evaluation Summary")
    fig.tight_layout()
    out_path = outdir / "eval_summary.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_event_metrics(run_dir: Path, outdir: Path, rolling_window: int, train_bin_size: int) -> Optional[Path]:
    train_path = run_dir / "train_events.jsonl"
    rows = _load_jsonl(train_path)
    if not rows:
        return None

    step = np.asarray([float(r.get("step_index", i + 1)) for i, r in enumerate(rows)], dtype=float)
    closed = np.asarray([float(r.get("true_cycles_closed", 0.0)) for r in rows], dtype=float)
    added = np.asarray([float(r.get("true_cycles_added", 0.0)) for r in rows], dtype=float)
    depth = np.asarray([float(r.get("trace_max_recursion_depth", 0.0)) for r in rows], dtype=float)
    split = np.asarray([float(r.get("trace_split_count", 0.0)) for r in rows], dtype=float)
    recur = np.asarray([1.0 if bool(r.get("trace_recursion_limit_hit", False)) else 0.0 for r in rows], dtype=float)
    reward = np.asarray([float(r.get("reward", 0.0)) for r in rows], dtype=float)

    reward_x, reward_mean, reward_lo, reward_hi = _bin_series_with_ci(step, reward, train_bin_size)
    closed_x, closed_mean, closed_lo, closed_hi = _bin_series_with_ci(step, closed, train_bin_size)
    added_x, added_mean, added_lo, added_hi = _bin_series_with_ci(step, added, train_bin_size)
    depth_x, depth_mean, depth_lo, depth_hi = _bin_series_with_ci(step, depth, train_bin_size)
    split_x, split_mean, split_lo, split_hi = _bin_series_with_ci(step, split, train_bin_size)
    recur_x, recur_mean, recur_lo, recur_hi = _bin_series_with_ci(step, recur, train_bin_size)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(reward_x, reward_mean, color="#2f9e44", lw=2, label=f"reward (bin={train_bin_size})")
    axes[0].fill_between(reward_x, reward_lo, reward_hi, color="#2f9e44", alpha=0.2, label="95% CI")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(closed_x, closed_mean, color="#0b7285", lw=2, label="true_cycles_closed")
    axes[1].fill_between(closed_x, closed_lo, closed_hi, color="#0b7285", alpha=0.2)
    axes[1].plot(added_x, added_mean, color="#c92a2a", lw=2, label="true_cycles_added")
    axes[1].fill_between(added_x, added_lo, added_hi, color="#c92a2a", alpha=0.2)
    axes[1].set_ylabel("Cycle Events")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(depth_x, depth_mean, color="#5f3dc4", lw=2, label="trace_depth")
    axes[2].fill_between(depth_x, depth_lo, depth_hi, color="#5f3dc4", alpha=0.2)
    axes[2].plot(split_x, split_mean, color="#e67700", lw=2, label="trace_splits")
    axes[2].fill_between(split_x, split_lo, split_hi, color="#e67700", alpha=0.2)
    axes[2].plot(recur_x, recur_mean, color="#d6336c", lw=2, label="recursion_limit_hit_rate")
    axes[2].fill_between(recur_x, recur_lo, recur_hi, color="#d6336c", alpha=0.2)
    axes[2].set_ylabel("Trace Diagnostics")
    axes[2].set_xlabel("Train Step Index")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    fig.suptitle("Training Event Summary")
    fig.tight_layout()
    out_path = outdir / "train_event_summary.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _plot_attention_metrics(run_dir: Path, outdir: Path, rolling_window: int) -> Optional[Path]:
    attention_path = run_dir / "attention_train.jsonl"
    rows = _load_jsonl(attention_path)
    if not rows:
        return None

    steps = []
    entropy = []
    true_mass = []
    for row in rows:
        steps.append(float(row.get("global_step", len(steps) + 1)))
        layer_payload = row.get("attention_layers", {})
        ent_vals = []
        mass_vals = []
        for layer in layer_payload.values():
            for head in layer.get("heads", []):
                ent_vals.append(float(head.get("entropy", 0.0)))
                mass_vals.append(float(head.get("mass_on_true_cycle_edges", 0.0)))
        entropy.append(float(np.mean(ent_vals)) if ent_vals else 0.0)
        true_mass.append(float(np.mean(mass_vals)) if mass_vals else 0.0)

    x = np.asarray(steps, dtype=float)
    y1 = _rolling_mean(np.asarray(entropy, dtype=float), rolling_window)
    y2 = _rolling_mean(np.asarray(true_mass, dtype=float), rolling_window)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x, y1, color="#5f3dc4", lw=2)
    axes[0].set_ylabel("Attention Entropy")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Attention Diagnostics")

    axes[1].plot(x, y2, color="#0b7285", lw=2)
    axes[1].set_ylabel("Mass on True-Cycle Edges")
    axes[1].set_xlabel("Global Step")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    out_path = outdir / "attention_diagnostics.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _extract_series_from_timeline(timeline: Dict, *, key: str) -> Tuple[np.ndarray, np.ndarray]:
    frames = timeline.get("frames", [])
    xs = []
    ys = []
    for frame in frames:
        terms = frame.get("policy_terms", {})
        raw = terms.get("raw", {})
        if key in raw:
            xs.append(float(frame.get("time", len(xs))))
            ys.append(float(raw[key]))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _plot_reeb_policy_terms(reeb_dir: Path, outdir: Path) -> Optional[Path]:
    if not reeb_dir.exists():
        return None
    sim_dirs = sorted(path for path in reeb_dir.iterdir() if path.is_dir() and path.name.startswith("sim_"))
    if not sim_dirs:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
    ax_a, ax_p, ax_da, ax_dp = axes.flatten()
    plotted = False
    for sim_dir in sim_dirs:
        timeline_path = sim_dir / "timeline.json"
        if not timeline_path.exists():
            continue
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        t_a, y_a = _extract_series_from_timeline(timeline, key="area_residual_norm")
        t_p, y_p = _extract_series_from_timeline(timeline, key="perimeter_residual_norm")
        t_da, y_da = _extract_series_from_timeline(timeline, key="area_progress_norm")
        t_dp, y_dp = _extract_series_from_timeline(timeline, key="perimeter_progress_norm")
        label = sim_dir.name
        if t_a.size:
            ax_a.plot(t_a, y_a, lw=1.5, label=label)
            plotted = True
        if t_p.size:
            ax_p.plot(t_p, y_p, lw=1.5, label=label)
        if t_da.size:
            ax_da.plot(t_da, y_da, lw=1.5, label=label)
        if t_dp.size:
            ax_dp.plot(t_dp, y_dp, lw=1.5, label=label)

    if not plotted:
        plt.close(fig)
        return None

    ax_a.set_title("Area Residual (raw)")
    ax_p.set_title("Perimeter Residual (raw)")
    ax_da.set_title("Area Progress Delta (raw)")
    ax_dp.set_title("Perimeter Progress Delta (raw)")
    for ax in (ax_a, ax_p, ax_da, ax_dp):
        ax.set_xlabel("Simulation Time")
        ax.grid(alpha=0.25)
    ax_a.set_ylabel("Value")
    ax_da.set_ylabel("Value")
    if len(sim_dirs) <= 8:
        ax_dp.legend(loc="best", fontsize=8)

    fig.suptitle("Saved Sim Policy Geometry Terms")
    fig.tight_layout()
    out_path = outdir / "reeb_policy_terms.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> None:
    args = _make_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else (run_dir / "plots")
    outdir.mkdir(parents=True, exist_ok=True)

    produced: List[Path] = []
    eval_plot = _plot_eval_npz(run_dir, outdir)
    if eval_plot is not None:
        produced.append(eval_plot)
    events_plot = _plot_event_metrics(run_dir, outdir, args.rolling_window, args.train_bin_size)
    if events_plot is not None:
        produced.append(events_plot)
    attention_plot = _plot_attention_metrics(run_dir, outdir, args.rolling_window)
    if attention_plot is not None:
        produced.append(attention_plot)

    reeb_dir = Path(args.reeb_dir).resolve() if args.reeb_dir else None
    if reeb_dir is None:
        candidates = sorted(run_dir.glob("rl_reeb_from_model_*"))
        reeb_dir = candidates[-1] if candidates else None
    if reeb_dir is not None:
        reeb_plot = _plot_reeb_policy_terms(reeb_dir, outdir)
        if reeb_plot is not None:
            produced.append(reeb_plot)

    summary = {
        "run_dir": str(run_dir),
        "outdir": str(outdir),
        "plots": [str(path) for path in produced],
    }
    (outdir / "plot_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if produced:
        print("Generated plots:")
        for path in produced:
            print(f"- {path}")
    else:
        print("No plots were generated (missing source logs/artifacts).")
    print(f"Plot manifest: {outdir / 'plot_manifest.json'}")


if __name__ == "__main__":
    main()
