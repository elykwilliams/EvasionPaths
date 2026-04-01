#!/usr/bin/env python3
"""Rerun only Homological Motion, merge with existing baseline benchmarks, and refresh paper assets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from benchmark_common import DOMAIN_DISPLAY, MODEL_DISPLAY, parse_csv_floats, parse_csv_ints, parse_csv_strs
from motion_model_grid_benchmark import (
    compute_aggregate_stats,
    compute_model_summary,
    compute_paired_artifacts,
    make_model_heatmaps,
    make_pairwise_delta_heatmap,
)


DEFAULT_DOMAINS = ["square", "circle", "stadium_w0p6", "stadium_w1p2"]
DEFAULT_TUNING_SEEDS = [7, 19]
DEFAULT_MODELS = [
    "billiard",
    "brownian_low",
    "brownian_med",
    "brownian_high",
    "vicsek_low",
    "vicsek_med",
    "vicsek_high",
    "homological",
]


def run_command(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_rederive_homological")
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(cwd), check=True, env=env)


def combine_trials(src_paths: List[Path], dst_path: Path) -> None:
    combined: List[Dict] = []
    trial_offset = 0
    for src in src_paths:
        trials = json.loads(src.read_text(encoding="utf-8"))
        for trial in trials:
            item = dict(trial)
            item["trial"] = int(item.get("trial", 0)) + trial_offset
            combined.append(item)
        trial_offset += len(trials)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun Homological Motion only and merge with existing baselines.")
    parser.add_argument("--python", type=str, default="/Users/marco/miniconda3/envs/EvasionPaths/bin/python")
    parser.add_argument("--domains", type=str, default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--tuning-seeds", type=str, default=",".join(str(x) for x in DEFAULT_TUNING_SEEDS))
    parser.add_argument("--search-trials", type=int, default=30)
    parser.add_argument("--runs-per-combo", type=int, default=10)
    parser.add_argument("--n-values", type=str, default="8,10,12,14,16")
    parser.add_argument("--r-values", type=str, default="0.20,0.24,0.28,0.32")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--t-cap", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--worst-case-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-wall-seconds-per-run", type=float, default=20.0)
    parser.add_argument(
        "--baseline-root",
        type=str,
        default="experiments/output/cross_domain_motion_benchmark_8model",
    )
    parser.add_argument(
        "--search-output-dir",
        type=str,
        default="experiments/output/homological_rederived_search",
    )
    parser.add_argument(
        "--homological-output-dir",
        type=str,
        default="experiments/output/homological_rederived_only",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=str,
        default="experiments/output/cross_domain_motion_benchmark_8model_rederived_homological",
    )
    parser.add_argument("--paper-dir", type=str, default="latex/homologicalDynamics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    domains = parse_csv_strs(args.domains)
    n_values = parse_csv_ints(args.n_values)
    r_values = parse_csv_floats(args.r_values)
    tuning_seeds = [int(x.strip()) for x in args.tuning_seeds.split(",") if x.strip()]
    baseline_root = repo_root / args.baseline_root
    search_root = repo_root / args.search_output_dir
    homological_root = repo_root / args.homological_output_dir
    merged_root = repo_root / args.merged_output_dir
    merged_root.mkdir(parents=True, exist_ok=True)

    merged_run_dirs: Dict[str, Path] = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for domain in domains:
        baseline_run_dir = baseline_root / f"{domain}_eight_model_grid"
        if not baseline_run_dir.exists():
            raise FileNotFoundError(f"Missing baseline run directory: {baseline_run_dir}")

        trial_paths: List[Path] = []
        for seed in tuning_seeds:
            run_name = f"homological_{domain}_seed{seed}_{timestamp}"
            run_command(
                [
                    args.python,
                    "experiments/motion_model_param_search.py",
                    "--model",
                    "homological",
                    "--domain",
                    domain,
                    "--seed",
                    str(seed),
                    "--trials",
                    str(args.search_trials),
                    "--n-values",
                    args.n_values,
                    "--r-values",
                    args.r_values,
                    "--dt",
                    str(args.dt),
                    "--velocity",
                    str(args.velocity),
                    "--t-cap",
                    str(args.t_cap),
                    "--failure-penalty",
                    str(args.failure_penalty),
                    "--worst-case-weight",
                    str(args.worst_case_weight),
                    "--max-wall-seconds-per-run",
                    str(args.max_wall_seconds_per_run),
                    "--output-dir",
                    args.search_output_dir,
                    "--run-name",
                    run_name,
                ],
                cwd=repo_root,
            )
            trial_paths.append(search_root / run_name / "all_trials_full.json")

        combined_dir = search_root / f"homological_{domain}_combined_{timestamp}"
        combined_path = combined_dir / "all_trials_full_combined.json"
        combine_trials(trial_paths, combined_path)

        homo_run_name = f"{domain}_homological_only_{timestamp}"
        run_command(
            [
                args.python,
                "experiments/motion_model_grid_benchmark.py",
                "--domain",
                domain,
                "--models",
                "homological",
                "--runs-per-combo",
                str(args.runs_per_combo),
                "--n-values",
                args.n_values,
                "--r-values",
                args.r_values,
                "--dt",
                str(args.dt),
                "--velocity",
                str(args.velocity),
                "--t-cap",
                str(args.t_cap),
                "--seed",
                str(args.seed),
                "--failure-penalty",
                str(args.failure_penalty),
                "--worst-case-weight",
                str(args.worst_case_weight),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                "--max-wall-seconds-per-run",
                str(args.max_wall_seconds_per_run),
                "--homological-trials-json",
                str(combined_path),
                "--output-dir",
                args.homological_output_dir,
                "--run-name",
                homo_run_name,
            ],
            cwd=repo_root,
        )
        homological_run_dir = homological_root / homo_run_name

        baseline_raw = pd.read_csv(baseline_run_dir / "raw_runs.csv")
        new_homological_raw = pd.read_csv(homological_run_dir / "raw_runs.csv")
        merged_raw = pd.concat(
            [
                baseline_raw[baseline_raw["model"] != "homological"],
                new_homological_raw,
            ],
            ignore_index=True,
        ).sort_values(["domain", "model", "n", "r", "run_idx"]).reset_index(drop=True)

        merged_run_dir = merged_root / f"{domain}_eight_model_grid"
        if merged_run_dir.exists():
            shutil.rmtree(merged_run_dir)
        merged_run_dir.mkdir(parents=True, exist_ok=True)
        merged_run_dirs[domain] = merged_run_dir

        merged_raw.to_csv(merged_run_dir / "raw_runs.csv", index=False)

        agg_df = compute_aggregate_stats(merged_raw)
        agg_df.to_csv(merged_run_dir / "aggregate_stats.csv", index=False)

        model_summary_df = compute_model_summary(agg_df)
        model_summary_df.to_csv(merged_run_dir / "model_summary.csv", index=False)

        paired_df, paired_summary_df, paired_overall_df = compute_paired_artifacts(
            raw_df=merged_raw,
            domain_name=domain,
            bootstrap_samples=int(args.bootstrap_samples),
            seed=int(args.seed),
        )
        paired_df.to_csv(merged_run_dir / "paired_deltas.csv", index=False)
        paired_summary_df.to_csv(merged_run_dir / "paired_summary.csv", index=False)
        paired_overall_df.to_csv(merged_run_dir / "paired_overall_summary.csv", index=False)

        heatmap_dir = merged_run_dir / "heatmaps"
        for model_name in DEFAULT_MODELS:
            make_model_heatmaps(
                agg_df=agg_df,
                domain_name=domain,
                model_name=model_name,
                n_values=n_values,
                r_values=r_values,
                t_cap=float(args.t_cap),
                output_path=heatmap_dir / f"{domain}_{model_name}_mean_std_fail_heatmaps.png",
            )
        if not paired_summary_df.empty:
            for comparison in paired_summary_df["comparison"].drop_duplicates().tolist():
                make_pairwise_delta_heatmap(
                    paired_summary=paired_summary_df,
                    domain_name=domain,
                    comparison=comparison,
                    n_values=n_values,
                    r_values=r_values,
                    t_cap=float(args.t_cap),
                    output_path=heatmap_dir / f"{domain}_{comparison}_delta_heatmap.png",
                )

        manifest = {
            "script": "experiments/rederive_homological_and_merge.py",
            "timestamp": timestamp,
            "domain": domain,
            "domain_display": DOMAIN_DISPLAY.get(domain, domain),
            "models": [MODEL_DISPLAY[m] for m in DEFAULT_MODELS],
            "baseline_run_dir": str(baseline_run_dir),
            "homological_run_dir": str(homological_run_dir),
            "combined_trials_json": str(combined_path),
            "seed": int(args.seed),
            "runs_per_combo": int(args.runs_per_combo),
            "n_values": n_values,
            "r_values": r_values,
            "dt": float(args.dt),
            "velocity": float(args.velocity),
            "t_cap": float(args.t_cap),
            "failure_penalty": float(args.failure_penalty),
            "worst_case_weight": float(args.worst_case_weight),
            "bootstrap_samples": int(args.bootstrap_samples),
            "max_wall_seconds_per_run": float(args.max_wall_seconds_per_run),
        }
        (merged_run_dir / "metadata.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    asset_cmd = [
        args.python,
        "experiments/generate_homological_paper_assets.py",
        "--paper-dir",
        args.paper_dir,
    ]
    for domain in domains:
        asset_cmd.extend(["--run", f"{domain}={merged_run_dirs[domain]}"])
    run_command(asset_cmd, cwd=repo_root)
    print("Homological refresh complete.")


if __name__ == "__main__":
    main()
