#!/usr/bin/env python3
"""Run Bunimovich-stadium Vicsek experiments with detection-time and polarization summaries."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_csv_floats(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated float.")
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated floats.") from exc


def noise_slug(value: float) -> str:
    return f"{float(value):0.6f}".replace("-", "m").replace(".", "p")


def width_slug(value: float) -> str:
    return f"{float(value):0.1f}".replace("-", "m").replace(".", "p")


def stadium_area(width: float, radius: float) -> float:
    return math.pi * float(radius) ** 2 + 4.0 * float(width) * float(radius)


def sensing_radius_from_k(*, width: float, stadium_radius: float, n_mobile: int, k_parameter: float) -> float:
    area = stadium_area(width, stadium_radius)
    return math.sqrt(area / (k_parameter * n_mobile * math.pi))


def seed_for(*, width: float, noise_scale: float, replicate: int, base_seed: int) -> int:
    w_int = int(round(float(width) * 1_000_000))
    n_int = int(round(float(noise_scale) * 1_000_000))
    seed = (
        base_seed * 1_315_423_911
        + w_int * 97_531
        + n_int * 104_729
        + int(replicate) * 2_654_435_761
    ) % (2**32 - 1)
    return int(seed)


def polarization(sensor_network) -> float:
    velocities = np.array([np.asarray(sensor.vel, dtype=float) for sensor in sensor_network.mobile_sensors], dtype=float)
    if velocities.size == 0:
        return 0.0
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    headings = velocities / np.maximum(speeds, 1e-12)
    return float(np.linalg.norm(np.sum(headings, axis=0)) / len(headings))


@dataclass
class RunResult:
    noise_label: str
    noise_scale: float
    width: float
    replicate: int
    seed: int
    sensing_radius: float
    interaction_radius: float
    detection_time: float
    capped: bool
    detected: bool
    status: str
    error: str
    mean_polarization: float
    final_polarization: float
    polarization_curve: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bunimovich-stadium Vicsek experiments.")
    parser.add_argument("--num-sensors", type=int, default=20, help="Number of mobile sensors.")
    parser.add_argument("--k", type=float, default=0.3, help="Coverage parameter in the sensing-radius rule.")
    parser.add_argument("--dt", type=float, default=0.01, help="Top-level simulation timestep.")
    parser.add_argument("--sensor-velocity", type=float, default=1.0, help="Initial speed magnitude for each sensor.")
    parser.add_argument("--n-sims", type=int, default=100, help="Number of simulations per (width, noise) condition.")
    parser.add_argument("--domain-radius", type=float, default=1.0, help="Radius of the stadium endcaps.")
    parser.add_argument(
        "--widths",
        type=parse_csv_floats,
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        help="Comma-separated list of Bunimovich half-widths.",
    )
    parser.add_argument(
        "--noise-scales",
        type=parse_csv_floats,
        default=[0.0, math.pi / 48.0, math.pi / 24.0, math.pi / 12.0],
        help="Comma-separated Vicsek angular noise scales in radians.",
    )
    parser.add_argument(
        "--interaction-radius-scale",
        type=float,
        default=1.0,
        help="Set Vicsek interaction radius to this multiple of the sensing radius.",
    )
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup motion steps before detection begins.")
    parser.add_argument(
        "--polarization-horizon",
        type=float,
        default=15.0,
        help="Duration over which polarization curves are recorded from detection-phase start.",
    )
    parser.add_argument(
        "--max-detection-time",
        type=float,
        default=15.0,
        help="Cap detection time at this value; runs that reach the cap are recorded as failures at the cap.",
    )
    parser.add_argument("--output-dir", type=str, default="./output", help="Root output directory.")
    parser.add_argument("--run-name", type=str, default="", help="Optional output directory name.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Joblib worker count.")
    parser.add_argument("--base-seed", type=int, default=0, help="Base seed for reproducible replicates.")
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument(
        "--square-init",
        dest="square_init",
        action="store_true",
        help="Initialize mobile sensors from the centered student-style square box.",
    )
    init_group.add_argument(
        "--full-stadium-init",
        dest="square_init",
        action="store_false",
        help="Initialize mobile sensors from the full stadium interior.",
    )
    parser.set_defaults(square_init=True)
    return parser.parse_args()


def simulate_condition(width: float, noise_scale: float, replicate: int, args: argparse.Namespace, time_grid: np.ndarray) -> RunResult:
    from boundary_geometry import BunimovichStadium
    from motion_model import Viscek
    from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors
    from time_stepping import EvasionPathSimulation

    seed = seed_for(width=width, noise_scale=noise_scale, replicate=replicate, base_seed=args.base_seed)
    np.random.seed(seed)

    sensing_radius = sensing_radius_from_k(
        width=float(width),
        stadium_radius=float(args.domain_radius),
        n_mobile=int(args.num_sensors),
        k_parameter=float(args.k),
    )
    interaction_radius = float(args.interaction_radius_scale) * sensing_radius

    domain = BunimovichStadium(
        w=float(width),
        r=float(args.domain_radius),
        L=2.0 * (float(width) + float(args.domain_radius)),
        square_init=bool(args.square_init),
        square_init_length=min(float(width), float(args.domain_radius)),
    )
    motion_model = Viscek(
        large_dt=float(args.dt),
        radius=float(interaction_radius),
        noise_scale=float(noise_scale),
    )
    fence = generate_fence_sensors(domain, float(sensing_radius))
    mobile = generate_mobile_sensors(domain, int(args.num_sensors), float(sensing_radius), float(args.sensor_velocity))
    sensor_network = SensorNetwork(mobile, motion_model, fence, float(sensing_radius), domain)

    for _ in range(max(0, int(args.warmup_steps))):
        sensor_network.move(float(args.dt))
        sensor_network.update()

    pol_times = [0.0]
    pol_values = [polarization(sensor_network)]
    detection_time = float(args.max_detection_time)
    capped = False
    detected = False
    status = "ok"
    error = ""

    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            sim = EvasionPathSimulation(sensor_network, float(args.dt), end_time=0)

            while sim.cycle_label.has_intruder() and sim.time < float(args.max_detection_time):
                sim.do_timestep()
                pol_times.append(float(sim.time))
                pol_values.append(polarization(sensor_network))

            if sim.time >= float(args.max_detection_time):
                capped = True
                detected = False
                status = "capped"
                detection_time = float(args.max_detection_time)
            elif sim.cycle_label.has_intruder():
                capped = True
                detected = False
                status = "capped"
                detection_time = float(args.max_detection_time)
            else:
                detected = True
                detection_time = min(float(sim.time), float(args.max_detection_time))

        while pol_times[-1] < float(args.polarization_horizon):
            step = min(float(args.dt), float(args.polarization_horizon) - pol_times[-1])
            sensor_network.move(step)
            sensor_network.update()
            pol_times.append(pol_times[-1] + step)
            pol_values.append(polarization(sensor_network))

    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
        detection_time = float("nan")

    curve = np.full_like(time_grid, np.nan, dtype=float)
    if len(pol_times) >= 2 and np.all(np.diff(pol_times) >= -1e-12):
        unique_times, unique_idx = np.unique(np.asarray(pol_times, dtype=float), return_index=True)
        unique_values = np.asarray(pol_values, dtype=float)[unique_idx]
        if len(unique_times) == 1:
            curve[:] = unique_values[0]
        else:
            curve = np.interp(time_grid, unique_times, unique_values)

    mean_pol = float(np.nanmean(curve)) if np.any(np.isfinite(curve)) else float("nan")
    final_pol = float(curve[-1]) if np.isfinite(curve[-1]) else float("nan")

    return RunResult(
        noise_label=noise_slug(noise_scale),
        noise_scale=float(noise_scale),
        width=float(width),
        replicate=int(replicate),
        seed=int(seed),
        sensing_radius=float(sensing_radius),
        interaction_radius=float(interaction_radius),
        detection_time=float(detection_time),
        capped=bool(capped),
        detected=bool(detected),
        status=status,
        error=error,
        mean_polarization=mean_pol,
        final_polarization=final_pol,
        polarization_curve=curve,
    )


def ci_bounds(values: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(valid))
    if len(valid) == 1:
        return mean, mean
    delta = 1.96 * float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
    return mean - delta, mean + delta


def raw_results_dataframe(results: Iterable[RunResult]):
    import pandas as pd

    rows = []
    for result in results:
        row = asdict(result)
        row.pop("polarization_curve", None)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_conditions(raw_df):
    import pandas as pd

    rows = []
    grouped = raw_df.groupby(["noise_label", "noise_scale", "width"], sort=True)
    for (noise_label, noise_scale, width), group in grouped:
        detection = pd.to_numeric(group["detection_time"], errors="coerce").to_numpy(dtype=float)
        mean_det = float(np.nanmean(detection)) if np.isfinite(detection).any() else float("nan")
        det_lo, det_hi = ci_bounds(detection)

        mean_pol = pd.to_numeric(group["mean_polarization"], errors="coerce").to_numpy(dtype=float)
        mean_pol_avg = float(np.nanmean(mean_pol)) if np.isfinite(mean_pol).any() else float("nan")
        pol_lo, pol_hi = ci_bounds(mean_pol)

        rows.append(
            {
                "noise_label": noise_label,
                "noise_scale": float(noise_scale),
                "width": float(width),
                "runs": int(len(group)),
                "error_runs": int((group["status"] == "error").sum()),
                "capped_runs": int(group["capped"].sum()),
                "detected_runs": int(group["detected"].sum()),
                "mean_detection_time": mean_det,
                "detection_ci_low": det_lo,
                "detection_ci_high": det_hi,
                "mean_polarization": mean_pol_avg,
                "polarization_ci_low": pol_lo,
                "polarization_ci_high": pol_hi,
            }
        )
    return pd.DataFrame(rows).sort_values(["noise_scale", "width"]).reset_index(drop=True)


def polarization_curve_dataframe(results: list[RunResult], time_grid: np.ndarray):
    import pandas as pd

    rows = []
    by_condition: dict[tuple[str, float], list[np.ndarray]] = {}
    for result in results:
        key = (result.noise_label, result.width)
        if result.status != "error" and len(result.polarization_curve) == len(time_grid):
            by_condition.setdefault(key, []).append(np.asarray(result.polarization_curve, dtype=float))

    for (noise_label, width), curves in sorted(by_condition.items(), key=lambda item: (item[0][0], item[0][1])):
        curve_stack = np.vstack(curves)
        mean = np.nanmean(curve_stack, axis=0)
        lower = np.full_like(mean, np.nan, dtype=float)
        upper = np.full_like(mean, np.nan, dtype=float)
        counts = np.sum(np.isfinite(curve_stack), axis=0)

        for idx in range(curve_stack.shape[1]):
            lo, hi = ci_bounds(curve_stack[:, idx])
            lower[idx] = lo
            upper[idx] = hi

        for t, m, lo, hi, n in zip(time_grid, mean, lower, upper, counts):
            rows.append(
                {
                    "noise_label": noise_label,
                    "width": float(width),
                    "time": float(t),
                    "mean_polarization": float(m),
                    "polarization_ci_low": float(lo),
                    "polarization_ci_high": float(hi),
                    "n_runs": int(n),
                }
            )
    return pd.DataFrame(rows)


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_cross_noise_plot(summary_df, *, value_col: str, low_col: str, high_col: str, ylabel: str, title: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)

    for noise_scale, group in summary_df.groupby("noise_scale", sort=True):
        group = group.sort_values("width")
        label = f"noise={float(noise_scale):0.3f}"
        x = group["width"].to_numpy(dtype=float)
        y = group[value_col].to_numpy(dtype=float)
        lo = group[low_col].to_numpy(dtype=float)
        hi = group[high_col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.6, label=label)
        ax.fill_between(x, lo, hi, alpha=0.20)

    ax.set_title(title)
    ax.set_xlabel("Width")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_single_noise_summary_plot(summary_df, *, noise_scale: float, value_col: str, low_col: str, high_col: str, ylabel: str, title: str, output_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)

    group = summary_df.loc[np.isclose(summary_df["noise_scale"], float(noise_scale))].sort_values("width")
    x = group["width"].to_numpy(dtype=float)
    y = group[value_col].to_numpy(dtype=float)
    lo = group[low_col].to_numpy(dtype=float)
    hi = group[high_col].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=1.7, color="#1f77b4")
    ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.22)
    ax.set_title(title)
    ax.set_xlabel("Width")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_polarization_curve_plot(curve_df, *, width: float, noise_label: str, noise_scale: float, output_path: Path) -> None:
    plt = ensure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)

    group = curve_df.loc[(curve_df["noise_label"] == noise_label) & np.isclose(curve_df["width"], float(width))].sort_values("time")
    x = group["time"].to_numpy(dtype=float)
    y = group["mean_polarization"].to_numpy(dtype=float)
    lo = group["polarization_ci_low"].to_numpy(dtype=float)
    hi = group["polarization_ci_high"].to_numpy(dtype=float)
    ax.plot(x, y, linewidth=1.7, color="#1f77b4")
    ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.22)
    ax.set_title(f"Mean polarization, width={float(width):0.1f}, noise={float(noise_scale):0.3f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Polarization")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_outputs(*, args: argparse.Namespace, run_dir: Path, time_grid: np.ndarray, results: list[RunResult]) -> None:
    raw_df = raw_results_dataframe(results)
    summary_df = summarize_conditions(raw_df)
    curve_df = polarization_curve_dataframe(results, time_grid)

    raw_df.to_csv(run_dir / "raw_runs.csv", index=False)
    summary_df.to_csv(run_dir / "condition_summary.csv", index=False)
    curve_df.to_csv(run_dir / "mean_polarization_curves.csv", index=False)

    save_cross_noise_plot(
        summary_df,
        value_col="mean_detection_time",
        low_col="detection_ci_low",
        high_col="detection_ci_high",
        ylabel="Detection time",
        title="Detection time by width and Vicsek noise",
        output_path=run_dir / "detection_time_by_width_and_noise.png",
    )
    save_cross_noise_plot(
        summary_df,
        value_col="mean_polarization",
        low_col="polarization_ci_low",
        high_col="polarization_ci_high",
        ylabel="Mean polarization",
        title="Mean polarization by width and Vicsek noise",
        output_path=run_dir / "polarization_by_width_and_noise.png",
    )

    for noise_scale in args.noise_scales:
        label = noise_slug(noise_scale)
        noise_dir = run_dir / f"noise_{label}"
        curves_dir = noise_dir / "polarization_curves"
        noise_dir.mkdir(parents=True, exist_ok=True)
        curves_dir.mkdir(parents=True, exist_ok=True)

        raw_df.loc[np.isclose(raw_df["noise_scale"], float(noise_scale))].sort_values(["width", "replicate"]).to_csv(
            noise_dir / "raw_runs.csv", index=False
        )
        curve_df.loc[curve_df["noise_label"] == label].sort_values(["width", "time"]).to_csv(
            noise_dir / "mean_polarization_curves.csv", index=False
        )

        save_single_noise_summary_plot(
            summary_df,
            noise_scale=float(noise_scale),
            value_col="mean_detection_time",
            low_col="detection_ci_low",
            high_col="detection_ci_high",
            ylabel="Detection time",
            title=f"Detection time vs width, noise={float(noise_scale):0.3f}",
            output_path=noise_dir / "detection_time_curve.png",
        )
        save_single_noise_summary_plot(
            summary_df,
            noise_scale=float(noise_scale),
            value_col="mean_polarization",
            low_col="polarization_ci_low",
            high_col="polarization_ci_high",
            ylabel="Mean polarization",
            title=f"Mean polarization vs width, noise={float(noise_scale):0.3f}",
            output_path=noise_dir / "mean_polarization_summary.png",
        )

        for width in args.widths:
            save_polarization_curve_plot(
                curve_df,
                width=float(width),
                noise_label=label,
                noise_scale=float(noise_scale),
                output_path=curves_dir / f"width_{width_slug(width)}_mean_polarization.png",
            )

    metadata = {
        "script": "experiments/bunimovich_vicsek_experiment.py",
        "timestamp": datetime.now().isoformat(),
        "n_sims": int(args.n_sims),
        "num_sensors": int(args.num_sensors),
        "k": float(args.k),
        "dt": float(args.dt),
        "sensor_velocity": float(args.sensor_velocity),
        "domain_radius": float(args.domain_radius),
        "widths": [float(v) for v in args.widths],
        "noise_scales": [float(v) for v in args.noise_scales],
        "interaction_radius_scale": float(args.interaction_radius_scale),
        "warmup_steps": int(args.warmup_steps),
        "polarization_horizon": float(args.polarization_horizon),
        "max_detection_time": float(args.max_detection_time),
        "square_init": bool(args.square_init),
        "base_seed": int(args.base_seed),
        "n_jobs": int(args.n_jobs),
        "time_grid": [float(t) for t in time_grid],
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(args: argparse.Namespace, run_dir: Path) -> None:
    time_grid = np.arange(0.0, float(args.polarization_horizon) + 0.5 * float(args.dt), float(args.dt), dtype=float)
    all_results: list[RunResult] = []
    conditions = [(float(noise), float(width)) for noise in args.noise_scales for width in args.widths]

    outer_bar = tqdm(total=len(conditions), desc="Conditions", unit="cond")
    try:
        for noise_scale, width in conditions:
            sim_desc = f"sims @ noise={noise_scale:0.3f}, width={width:0.1f}"
            parallel = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")
            tasks = (
                delayed(simulate_condition)(width, noise_scale, replicate, args, time_grid)
                for replicate in range(int(args.n_sims))
            )
            with tqdm(total=args.n_sims, desc=sim_desc, unit="sim", leave=False) as sim_bar:
                for result in parallel(tasks):
                    all_results.append(result)
                    sim_bar.update(1)
            outer_bar.update(1)
    finally:
        outer_bar.close()

    write_outputs(args=args, run_dir=run_dir, time_grid=time_grid, results=all_results)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_run_name = f"bunimovich_vicsek_{int(args.n_sims)}_{timestamp}"
    run_name = args.run_name or default_run_name
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    run_experiment(args, run_dir)


if __name__ == "__main__":
    main()
