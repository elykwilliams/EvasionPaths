#!/usr/bin/env python3
"""Benchmark motion models on a domain-specific (n, r) grid with paired analyses."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_common import (
    DOMAIN_DISPLAY,
    MODEL_DISPLAY,
    build_domain,
    build_motion_model,
    clone_sensors,
    combo_key,
    domain_metadata,
    generate_connected_initial_condition,
    load_best_params_by_combo,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_strs,
    replicate_seed,
)
from sensor_network import SensorNetwork
from time_stepping import EvasionPathSimulation


COMPARISON_DISPLAY = {
    "homological_minus_billiard": "Homological - Billiard",
    "homological_minus_brownian_low": "Homological - Brownian (Low)",
    "homological_minus_brownian_med": "Homological - Brownian (Medium)",
    "homological_minus_brownian_high": "Homological - Brownian (High)",
    "homological_minus_vicsek_low": "Homological - Vicsek (Low)",
    "homological_minus_vicsek_med": "Homological - Vicsek (Medium)",
    "homological_minus_vicsek_high": "Homological - Vicsek (High)",
    "sequential_homological_minus_billiard": "Sequential Homological - Billiard",
    "sequential_homological_minus_brownian_high": "Sequential Homological - Brownian (High)",
}


def run_simulation_once(
    *,
    sensor_network: SensorNetwork,
    dt: float,
    t_cap: float,
    max_wall_seconds: float,
    suppress_output: bool,
) -> Tuple[float, bool, float]:
    def _timeout_handler(_signum, _frame):
        raise TimeoutError("Simulation exceeded wall-clock timeout.")

    def _run() -> Tuple[float, bool, float]:
        sim = EvasionPathSimulation(sensor_network, dt, end_time=t_cap)
        while sim.cycle_label.has_intruder():
            sim.do_timestep()
            if 0 < sim.Tend < sim.time:
                break
        t_val = float(sim.time)
        failed = bool(sim.cycle_label.has_intruder()) or t_val >= (t_cap - 1e-12)
        tau = float(t_cap if failed else min(t_val, t_cap))
        detection_time = float("inf") if failed else t_val
        return tau, failed, detection_time

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        if max_wall_seconds > 0:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, float(max_wall_seconds))

        if not suppress_output:
            return _run()

        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            return _run()
    except TimeoutError:
        return float(t_cap), True, float("inf")
    finally:
        if max_wall_seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)


def bootstrap_mean_ci(values: np.ndarray, n_samples: int, seed: int, alpha: float = 0.05) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_samples, len(values)), replace=True)
    means = np.mean(draws, axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def annotate_heatmap(ax, grid: np.ndarray, fmt: str = ".2f") -> None:
    finite = grid[np.isfinite(grid)]
    min_val = float(np.min(finite)) if finite.size else 0.0
    span = float(np.ptp(finite)) if finite.size else 0.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = float(grid[i, j])
            if not np.isfinite(value):
                text = "--"
                color = "black"
            else:
                text = format(value, fmt)
                normalized = (value - min_val) / (span + 1e-12)
                color = "white" if normalized > 0.45 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)


def make_model_heatmaps(
    *,
    agg_df: pd.DataFrame,
    domain_name: str,
    model_name: str,
    n_values: Sequence[int],
    r_values: Sequence[float],
    t_cap: float,
    output_path: Path,
) -> None:
    model_df = agg_df[(agg_df["domain"] == domain_name) & (agg_df["model"] == model_name)].copy()
    model_df["n"] = model_df["n"].astype(int)
    model_df["r"] = model_df["r"].astype(float)

    mean_grid = (
        model_df.pivot(index="n", columns="r", values="mean_tau")
        .reindex(index=list(n_values), columns=list(r_values))
        .to_numpy(dtype=float)
    )
    std_grid = (
        model_df.pivot(index="n", columns="r", values="std_tau")
        .reindex(index=list(n_values), columns=list(r_values))
        .to_numpy(dtype=float)
    )
    fail_grid = (
        model_df.pivot(index="n", columns="r", values="fail_rate")
        .reindex(index=list(n_values), columns=list(r_values))
        .to_numpy(dtype=float)
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        f"{MODEL_DISPLAY.get(model_name, model_name)} on {DOMAIN_DISPLAY.get(domain_name, domain_name)}",
        fontsize=13,
    )

    im0 = axes[0].imshow(mean_grid, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=float(t_cap))
    axes[0].set_title("Mean capped detection time")
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("n")
    axes[0].set_xticks(np.arange(len(r_values)))
    axes[0].set_yticks(np.arange(len(n_values)))
    axes[0].set_xticklabels([f"{r:.2f}" for r in r_values])
    axes[0].set_yticklabels([str(n) for n in n_values])
    annotate_heatmap(axes[0], mean_grid, fmt=".2f")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    std_max = max(1e-12, float(np.nanmax(std_grid)))
    im1 = axes[1].imshow(std_grid, origin="lower", aspect="auto", cmap="magma", vmin=0.0, vmax=std_max)
    axes[1].set_title("Std capped detection time")
    axes[1].set_xlabel("r")
    axes[1].set_ylabel("n")
    axes[1].set_xticks(np.arange(len(r_values)))
    axes[1].set_yticks(np.arange(len(n_values)))
    axes[1].set_xticklabels([f"{r:.2f}" for r in r_values])
    axes[1].set_yticklabels([str(n) for n in n_values])
    annotate_heatmap(axes[1], std_grid, fmt=".2f")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(fail_grid, origin="lower", aspect="auto", cmap="cividis", vmin=0.0, vmax=1.0)
    axes[2].set_title("Failure rate")
    axes[2].set_xlabel("r")
    axes[2].set_ylabel("n")
    axes[2].set_xticks(np.arange(len(r_values)))
    axes[2].set_yticks(np.arange(len(n_values)))
    axes[2].set_xticklabels([f"{r:.2f}" for r in r_values])
    axes[2].set_yticklabels([str(n) for n in n_values])
    annotate_heatmap(axes[2], fail_grid, fmt=".2f")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_pairwise_delta_heatmap(
    *,
    paired_summary: pd.DataFrame,
    domain_name: str,
    comparison: str,
    n_values: Sequence[int],
    r_values: Sequence[float],
    t_cap: float,
    output_path: Path,
) -> None:
    comp_df = paired_summary[
        (paired_summary["domain"] == domain_name) & (paired_summary["comparison"] == comparison)
    ].copy()
    comp_df["n"] = comp_df["n"].astype(int)
    comp_df["r"] = comp_df["r"].astype(float)
    delta_grid = (
        comp_df.pivot(index="n", columns="r", values="mean_delta_tau")
        .reindex(index=list(n_values), columns=list(r_values))
        .to_numpy(dtype=float)
    )
    vmax = max(1e-6, float(np.nanmax(np.abs(delta_grid))))
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.0), constrained_layout=True)
    im = ax.imshow(delta_grid, origin="lower", aspect="auto", cmap="coolwarm_r", vmin=-vmax, vmax=vmax)
    ax.set_title(COMPARISON_DISPLAY.get(comparison, comparison))
    ax.set_xlabel("r")
    ax.set_ylabel("n")
    ax.set_xticks(np.arange(len(r_values)))
    ax.set_yticks(np.arange(len(n_values)))
    ax.set_xticklabels([f"{r:.2f}" for r in r_values])
    ax.set_yticklabels([str(n) for n in n_values])
    annotate_heatmap(ax, delta_grid, fmt=".2f")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean capped-time delta")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_aggregate_stats(raw_df: pd.DataFrame) -> pd.DataFrame:
    def _std(series: pd.Series) -> float:
        return float(np.std(np.asarray(series, dtype=float), ddof=0))

    def _mean_success(group: pd.DataFrame) -> float:
        success = group.loc[~group["failed"], "tau"].to_numpy(dtype=float)
        return float(np.mean(success)) if len(success) else float("nan")

    def _std_success(group: pd.DataFrame) -> float:
        success = group.loc[~group["failed"], "tau"].to_numpy(dtype=float)
        return float(np.std(success, ddof=0)) if len(success) else float("nan")

    rows: List[Dict] = []
    grouped = raw_df.groupby(["domain", "domain_display", "model", "model_display", "n", "r"], sort=True)
    for (domain, domain_display, model, model_display, n_val, r_val), group in grouped:
        rows.append(
            {
                "domain": domain,
                "domain_display": domain_display,
                "model": model,
                "model_display": model_display,
                "n": int(n_val),
                "r": float(r_val),
                "mean_tau": float(group["tau"].mean()),
                "std_tau": _std(group["tau"]),
                "fail_rate": float(group["failed"].mean()),
                "fail_count": int(group["failed"].sum()),
                "runs": int(len(group)),
                "mean_success_tau": _mean_success(group),
                "std_success_tau": _std_success(group),
                "success_count": int((~group["failed"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "model", "n", "r"]).reset_index(drop=True)


def compute_model_summary(agg_df: pd.DataFrame) -> pd.DataFrame:
    def _safe_nanmean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if len(finite) else float("nan")

    best_df = agg_df.copy()
    best_df["rank_mean_tau"] = best_df.groupby(["domain", "n", "r"])["mean_tau"].rank(method="min")
    best_counts = (
        best_df[best_df["rank_mean_tau"] == 1.0]
        .groupby(["domain", "model"])
        .size()
        .to_dict()
    )

    rows: List[Dict] = []
    grouped = agg_df.groupby(["domain", "domain_display", "model", "model_display"], sort=True)
    for (domain, domain_display, model, model_display), group in grouped:
        rows.append(
            {
                "domain": domain,
                "domain_display": domain_display,
                "model": model,
                "model_display": model_display,
                "grid_mean_tau": float(group["mean_tau"].mean()),
                "grid_std_of_means": float(np.std(group["mean_tau"].to_numpy(dtype=float), ddof=0)),
                "grid_fail_rate": float(group["fail_rate"].mean()),
                "grid_mean_success_tau": _safe_nanmean(group["mean_success_tau"].to_numpy(dtype=float)),
                "best_cell_count": int(best_counts.get((domain, model), 0)),
                "n_cells": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "grid_mean_tau", "model"]).reset_index(drop=True)


def compute_paired_artifacts(
    *,
    raw_df: pd.DataFrame,
    domain_name: str,
    bootstrap_samples: int,
    seed: int,
    home_model: str | None = None,
    comparison_models: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_models = set(raw_df["model"].astype(str).tolist())
    if home_model is None:
        if "sequential_homological" in available_models:
            home_model = "sequential_homological"
        elif "homological" in available_models:
            home_model = "homological"
        else:
            home_model = ""
    if not home_model or home_model not in available_models:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if comparison_models is None:
        comparison_models = [m for m in sorted(available_models) if m != home_model]

    comparisons = [
        (home_model, other_model, f"{home_model}_minus_{other_model}")
        for other_model in comparison_models
        if other_model in available_models
    ]

    paired_rows: List[Dict] = []
    summary_rows: List[Dict] = []
    overall_rows: List[Dict] = []

    for domain in [domain_name]:
        domain_df = raw_df[raw_df["domain"] == domain]
        for home_model, other_model, comparison_name in comparisons:
            home = domain_df[domain_df["model"] == home_model].rename(
                columns={
                    "tau": "tau_home",
                    "failed": "failed_home",
                    "detection_time": "detection_time_home",
                    "error": "error_home",
                }
            )
            away = domain_df[domain_df["model"] == other_model].rename(
                columns={
                    "tau": "tau_other",
                    "failed": "failed_other",
                    "detection_time": "detection_time_other",
                    "error": "error_other",
                }
            )
            join_cols = ["domain", "n", "r", "run_idx", "seed_init", "seed_run"]
            merged = home.merge(away, on=join_cols, how="inner")
            if merged.empty:
                continue
            merged["comparison"] = comparison_name
            merged["delta_tau"] = merged["tau_home"] - merged["tau_other"]
            merged["delta_success_detection_time"] = np.where(
                (~merged["failed_home"]) & (~merged["failed_other"]),
                merged["detection_time_home"] - merged["detection_time_other"],
                np.nan,
            )
            paired_rows.extend(merged.to_dict(orient="records"))

            for (n_val, r_val), group in merged.groupby(["n", "r"], sort=True):
                delta = group["delta_tau"].to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_mean_ci(
                    delta,
                    n_samples=bootstrap_samples,
                    seed=int(seed + 37 * int(n_val) + 100_003 * int(round(float(r_val) * 1_000))),
                )
                success_delta = group["delta_success_detection_time"].to_numpy(dtype=float)
                success_delta = success_delta[np.isfinite(success_delta)]
                summary_rows.append(
                    {
                        "domain": domain,
                        "domain_display": DOMAIN_DISPLAY.get(domain, domain),
                        "comparison": comparison_name,
                        "n": int(n_val),
                        "r": float(r_val),
                        "mean_delta_tau": float(np.mean(delta)),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "mean_home_tau": float(np.mean(group["tau_home"])),
                        "mean_other_tau": float(np.mean(group["tau_other"])),
                        "home_fail_rate": float(np.mean(group["failed_home"])),
                        "other_fail_rate": float(np.mean(group["failed_other"])),
                        "mean_delta_success_detection_time": float(np.mean(success_delta)) if len(success_delta) else float("nan"),
                        "paired_success_count": int(len(success_delta)),
                        "runs": int(len(group)),
                    }
                )

            all_delta = merged["delta_tau"].to_numpy(dtype=float)
            domain_code = sum((idx + 1) * ord(ch) for idx, ch in enumerate(domain))
            ci_low, ci_high = bootstrap_mean_ci(
                all_delta,
                n_samples=bootstrap_samples,
                seed=int(seed + 10_000 * len(overall_rows) + domain_code),
            )
            success_delta = merged["delta_success_detection_time"].to_numpy(dtype=float)
            success_delta = success_delta[np.isfinite(success_delta)]
            overall_rows.append(
                {
                    "domain": domain,
                    "domain_display": DOMAIN_DISPLAY.get(domain, domain),
                    "comparison": comparison_name,
                    "mean_delta_tau": float(np.mean(all_delta)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "home_fail_rate": float(np.mean(merged["failed_home"])),
                    "other_fail_rate": float(np.mean(merged["failed_other"])),
                    "mean_delta_success_detection_time": float(np.mean(success_delta)) if len(success_delta) else float("nan"),
                    "paired_success_count": int(len(success_delta)),
                    "runs": int(len(merged)),
                }
            )

    paired_df = pd.DataFrame(paired_rows)
    summary_df = pd.DataFrame(summary_rows)
    overall_df = pd.DataFrame(overall_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["domain", "comparison", "n", "r"]).reset_index(drop=True)
    if not overall_df.empty:
        overall_df = overall_df.sort_values(["domain", "comparison"]).reset_index(drop=True)
    return paired_df, summary_df, overall_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark motion models on a domain-specific (n, r) grid.")
    parser.add_argument(
        "--domain",
        type=str,
        default="square",
        choices=["square", "circle", "rectangle_2to1_area1", "stadium_w0p6", "stadium_w1p2"],
    )
    parser.add_argument("--n-values", type=str, default="8,10,12,14,16")
    parser.add_argument("--r-values", type=str, default="0.20,0.24,0.28,0.32")
    parser.add_argument("--runs-per-combo", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--t-cap", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-wall-seconds-per-run",
        type=float,
        default=20.0,
        help="Wall-clock guard per simulation run; when exceeded, run is marked as capped failure.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="billiard,brownian_low,brownian_med,brownian_high,vicsek_low,vicsek_med,vicsek_high,homological",
        help="Comma-separated model ids.",
    )
    parser.add_argument("--homological-trials-json", type=str, default="")
    parser.add_argument("--sequential-homological-trials-json", type=str, default="")
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--worst-case-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="experiments/output/motion_model_grid_benchmark")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--show-sim-output", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    domain_name = args.domain
    n_values = parse_csv_ints(args.n_values)
    r_values = parse_csv_floats(args.r_values)
    models = parse_csv_strs(args.models)

    unknown = [m for m in models if m not in MODEL_DISPLAY]
    if unknown:
        raise ValueError(f"Unknown model ids: {unknown}. Allowed: {list(MODEL_DISPLAY)}")

    tuned_json_by_model = {
        "homological": Path(args.homological_trials_json) if args.homological_trials_json else None,
        "sequential_homological": Path(args.sequential_homological_trials_json)
        if args.sequential_homological_trials_json
        else None,
    }
    tuned_params_by_model: Dict[str, Dict[str, Dict]] = {}
    for model_name in ["homological", "sequential_homological"]:
        if model_name not in models:
            continue
        path = tuned_json_by_model[model_name]
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"{model_name} benchmark requested but tuned-parameter archive is missing. "
                f"Provide --{model_name}-trials-json."
            )
        tuned_params_by_model[model_name] = load_best_params_by_combo(
            path,
            failure_penalty=float(args.failure_penalty),
            worst_case_weight=float(args.worst_case_weight),
        )
        missing = [combo_key(n, r) for n in n_values for r in r_values if combo_key(n, r) not in tuned_params_by_model[model_name]]
        if missing:
            raise ValueError(
                f"Missing tuned {model_name} parameters for some (n,r) combinations. "
                f"First missing key: {missing[0]}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{domain_name}_grid_benchmark_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {domain_name}")
    print(f"Models: {models}")
    print(f"n-values: {n_values}")
    print(f"r-values: {r_values}")
    print(f"runs per (n,r): {args.runs_per_combo}")

    domain = build_domain(domain_name)
    init_cache: Dict[Tuple[int, float, int], object] = {}
    infeasible_inits: List[Dict[str, float | int]] = []
    for n_val in n_values:
        for r_val in r_values:
            for rep in range(args.runs_per_combo):
                seed_init = replicate_seed(args.seed, domain_name, n_val, r_val, rep)
                init_result = generate_connected_initial_condition(
                    domain_name=domain_name,
                    domain=domain,
                    n_sensors=n_val,
                    radius=r_val,
                    sensor_velocity=float(args.velocity),
                    seed=seed_init,
                    max_retries=200,
                )
                init_cache[(n_val, r_val, rep)] = init_result
                if not init_result.feasible:
                    infeasible_inits.append(
                        {
                            "n": int(n_val),
                            "r": float(r_val),
                            "run_idx": int(rep),
                            "seed_init": int(seed_init),
                        }
                    )

    feasible_replicates = sum(1 for result in init_cache.values() if getattr(result, "feasible", False))
    total_runs = len(models) * feasible_replicates
    done = 0
    rows: List[Dict] = []

    for model_name in models:
        tuned_params = tuned_params_by_model.get(model_name, {})
        for n_val in n_values:
            for r_val in r_values:
                for rep in range(args.runs_per_combo):
                    init_result = init_cache[(n_val, r_val, rep)]
                    if not init_result.feasible or init_result.initial_condition is None:
                        continue
                    init = init_result.initial_condition
                    seed_init = int(init_result.seed_used if init_result.seed_used is not None else replicate_seed(args.seed, domain_name, n_val, r_val, rep))
                    seed_run = replicate_seed(args.seed + 101, domain_name, n_val, r_val, rep)
                    np.random.seed(seed_run)

                    tau = float(args.t_cap)
                    failed = True
                    detection_time = float("inf")
                    error = ""
                    try:
                        motion_model = build_motion_model(
                            model_name,
                            domain_name=domain_name,
                            n_sensors=n_val,
                            radius=r_val,
                            dt=float(args.dt),
                            sensor_velocity=float(args.velocity),
                            tuned_params_by_combo=tuned_params,
                        )
                        fence, mobile = clone_sensors(init, r_val)
                        sensor_network = SensorNetwork(mobile, motion_model, fence, r_val, domain)
                        tau, failed, detection_time = run_simulation_once(
                            sensor_network=sensor_network,
                            dt=float(args.dt),
                            t_cap=float(args.t_cap),
                            max_wall_seconds=float(args.max_wall_seconds_per_run),
                            suppress_output=not bool(args.show_sim_output),
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"

                    rows.append(
                        {
                            "domain": domain_name,
                            "domain_display": DOMAIN_DISPLAY.get(domain_name, domain_name),
                            "model": model_name,
                            "model_display": MODEL_DISPLAY.get(model_name, model_name),
                            "n": int(n_val),
                            "r": float(r_val),
                            "run_idx": int(rep),
                            "seed_init": int(seed_init),
                            "init_retries_used": int(init_result.retries_used),
                            "seed_run": int(seed_run),
                            "tau": float(tau),
                            "failed": bool(failed),
                            "detection_time": float(detection_time),
                            "error": error,
                        }
                    )
                    done += 1
                    if done % 25 == 0 or done == total_runs:
                        print(f"Completed {done}/{total_runs} runs")

    raw_df = pd.DataFrame(rows)
    raw_csv_path = run_dir / "raw_runs.csv"
    raw_df.to_csv(raw_csv_path, index=False)

    agg_df = compute_aggregate_stats(raw_df)
    agg_csv_path = run_dir / "aggregate_stats.csv"
    agg_df.to_csv(agg_csv_path, index=False)

    model_summary_df = compute_model_summary(agg_df)
    model_summary_path = run_dir / "model_summary.csv"
    model_summary_df.to_csv(model_summary_path, index=False)

    paired_df, paired_summary_df, paired_overall_df = compute_paired_artifacts(
        raw_df=raw_df,
        domain_name=domain_name,
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
        home_model="sequential_homological" if "sequential_homological" in models else None,
        comparison_models=[model for model in models if model != "sequential_homological"]
        if "sequential_homological" in models
        else None,
    )
    paired_csv_path = run_dir / "paired_deltas.csv"
    paired_summary_path = run_dir / "paired_summary.csv"
    paired_overall_path = run_dir / "paired_overall_summary.csv"
    paired_df.to_csv(paired_csv_path, index=False)
    paired_summary_df.to_csv(paired_summary_path, index=False)
    paired_overall_df.to_csv(paired_overall_path, index=False)

    heatmap_dir = run_dir / "heatmaps"
    for model_name in models:
        make_model_heatmaps(
            agg_df=agg_df,
            domain_name=domain_name,
            model_name=model_name,
            n_values=n_values,
            r_values=r_values,
            t_cap=float(args.t_cap),
            output_path=heatmap_dir / f"{domain_name}_{model_name}_mean_std_fail_heatmaps.png",
        )

    if not paired_summary_df.empty:
        for comparison in paired_summary_df["comparison"].drop_duplicates().tolist():
            make_pairwise_delta_heatmap(
                paired_summary=paired_summary_df,
                domain_name=domain_name,
                comparison=comparison,
                n_values=n_values,
                r_values=r_values,
                t_cap=float(args.t_cap),
                output_path=heatmap_dir / f"{domain_name}_{comparison}_delta_heatmap.png",
            )

    manifest = {
        "script": "experiments/motion_model_grid_benchmark.py",
        "domain": domain_name,
        "domain_metadata": domain_metadata(domain_name),
        "models": models,
        "raw_runs_csv": str(raw_csv_path),
        "aggregate_csv": str(agg_csv_path),
        "model_summary_csv": str(model_summary_path),
        "paired_deltas_csv": str(paired_csv_path),
        "paired_summary_csv": str(paired_summary_path),
        "paired_overall_summary_csv": str(paired_overall_path),
        "heatmap_dir": str(heatmap_dir),
    }
    (run_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    metadata = {
        "domain": domain_name,
        "domain_metadata": domain_metadata(domain_name),
        "n_values": n_values,
        "r_values": r_values,
        "runs_per_combo": int(args.runs_per_combo),
        "dt": float(args.dt),
        "velocity": float(args.velocity),
        "t_cap": float(args.t_cap),
        "seed": int(args.seed),
        "max_wall_seconds_per_run": float(args.max_wall_seconds_per_run),
        "models": models,
        "failure_penalty": float(args.failure_penalty),
        "worst_case_weight": float(args.worst_case_weight),
        "bootstrap_samples": int(args.bootstrap_samples),
        "tuned_json_by_model": {k: (str(v) if v is not None else "") for k, v in tuned_json_by_model.items()},
        "n_total_runs": int(total_runs),
        "n_failed_runs": int(raw_df["failed"].sum()),
        "n_error_runs": int((raw_df["error"].astype(str).str.len() > 0).sum()),
        "n_infeasible_initializations": int(len(infeasible_inits)),
        "infeasible_initializations": infeasible_inits,
        "artifact_manifest": str(run_dir / "artifact_manifest.json"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print("\nBenchmark complete.")
    print(f"Raw runs: {raw_csv_path}")
    print(f"Aggregate stats: {agg_csv_path}")
    print(f"Model summary: {model_summary_path}")
    print(f"Paired summary: {paired_summary_path}")
    print(f"Heatmaps: {heatmap_dir}")
    print(f"Error runs: {metadata['n_error_runs']} / {metadata['n_total_runs']}")


if __name__ == "__main__":
    main()
