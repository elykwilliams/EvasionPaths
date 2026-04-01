#!/usr/bin/env python3
"""Tune/evaluate Sequential Homological Motion and merge it into existing benchmark artifacts."""

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

from benchmark_common import DOMAIN_DISPLAY, domain_metadata, parse_csv_floats, parse_csv_ints, parse_csv_strs
from motion_model_grid_benchmark import (
    compute_aggregate_stats,
    compute_model_summary,
    compute_paired_artifacts,
    make_model_heatmaps,
    make_pairwise_delta_heatmap,
)


ORIGINAL_DOMAINS = ["square", "circle", "stadium_w0p6", "stadium_w1p2"]
SPARSE_DOMAINS = ["square", "circle", "rectangle_2to1_area1"]
TUNING_SEEDS = [7, 19]
SPARSE_MODELS = ["sequential_homological", "billiard", "brownian_high"]


def run_command(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_sequential_extension")
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


def write_run_metadata(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequential Homological benchmark extension.")
    parser.add_argument("--python", type=str, default="/Users/marco/miniconda3/envs/EvasionPaths/bin/python")
    parser.add_argument("--original-domains", type=str, default=",".join(ORIGINAL_DOMAINS))
    parser.add_argument("--sparse-domains", type=str, default=",".join(SPARSE_DOMAINS))
    parser.add_argument("--tuning-seeds", type=str, default=",".join(str(x) for x in TUNING_SEEDS))
    parser.add_argument("--search-trials", type=int, default=30)
    parser.add_argument("--runs-per-combo", type=int, default=10)
    parser.add_argument("--original-n-values", type=str, default="8,10,12,14,16")
    parser.add_argument("--original-r-values", type=str, default="0.20,0.24,0.28,0.32")
    parser.add_argument("--sparse-n-values", type=str, default="6,8,10,12")
    parser.add_argument("--sparse-r-values", type=str, default="0.10,0.12,0.14,0.16")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--original-t-cap", type=float, default=5.0)
    parser.add_argument("--sparse-t-cap", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--worst-case-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-wall-seconds-per-run", type=float, default=20.0)
    parser.add_argument(
        "--baseline-root",
        type=str,
        default="experiments/output/cross_domain_motion_benchmark_8model_rederived_homological",
    )
    parser.add_argument(
        "--original-search-output-dir",
        type=str,
        default="experiments/output/sequential_homological_original_search",
    )
    parser.add_argument(
        "--original-sequential-output-dir",
        type=str,
        default="experiments/output/sequential_homological_original_only",
    )
    parser.add_argument(
        "--merged-output-dir",
        type=str,
        default="experiments/output/cross_domain_motion_benchmark_with_sequential",
    )
    parser.add_argument(
        "--sparse-search-output-dir",
        type=str,
        default="experiments/output/sequential_homological_sparse_search",
    )
    parser.add_argument(
        "--sparse-benchmark-output-dir",
        type=str,
        default="experiments/output/sequential_homological_sparse_benchmark",
    )
    parser.add_argument("--paper-dir", type=str, default="latex/homologicalDynamics")
    parser.add_argument("--skip-paper-assets", action="store_true", default=False)
    return parser.parse_args()


def refresh_original_grid(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    domains: List[str],
    tuning_seeds: List[int],
    timestamp: str,
) -> Dict[str, Path]:
    search_root = repo_root / args.original_search_output_dir
    sequential_root = repo_root / args.original_sequential_output_dir
    merged_root = repo_root / args.merged_output_dir
    baseline_root = repo_root / args.baseline_root
    merged_root.mkdir(parents=True, exist_ok=True)

    original_n_values = parse_csv_ints(args.original_n_values)
    original_r_values = parse_csv_floats(args.original_r_values)
    merged_run_dirs: Dict[str, Path] = {}

    for domain in domains:
        baseline_run_dir = baseline_root / f"{domain}_eight_model_grid"
        if not baseline_run_dir.exists():
            raise FileNotFoundError(f"Missing baseline run directory: {baseline_run_dir}")

        trial_paths: List[Path] = []
        for seed in tuning_seeds:
            run_name = f"sequential_{domain}_seed{seed}_{timestamp}"
            run_command(
                [
                    args.python,
                    "experiments/motion_model_param_search.py",
                    "--model",
                    "sequential_homological",
                    "--domain",
                    domain,
                    "--seed",
                    str(seed),
                    "--trials",
                    str(args.search_trials),
                    "--n-values",
                    args.original_n_values,
                    "--r-values",
                    args.original_r_values,
                    "--dt",
                    str(args.dt),
                    "--velocity",
                    str(args.velocity),
                    "--t-cap",
                    str(args.original_t_cap),
                    "--failure-penalty",
                    str(args.failure_penalty),
                    "--worst-case-weight",
                    str(args.worst_case_weight),
                    "--max-wall-seconds-per-run",
                    str(args.max_wall_seconds_per_run),
                    "--output-dir",
                    args.original_search_output_dir,
                    "--run-name",
                    run_name,
                ],
                cwd=repo_root,
            )
            trial_paths.append(search_root / run_name / "all_trials_full.json")

        combined_dir = search_root / f"sequential_{domain}_combined_{timestamp}"
        combined_path = combined_dir / "all_trials_full_combined.json"
        combine_trials(trial_paths, combined_path)

        seq_run_name = f"{domain}_sequential_only_{timestamp}"
        run_command(
            [
                args.python,
                "experiments/motion_model_grid_benchmark.py",
                "--domain",
                domain,
                "--models",
                "sequential_homological",
                "--runs-per-combo",
                str(args.runs_per_combo),
                "--n-values",
                args.original_n_values,
                "--r-values",
                args.original_r_values,
                "--dt",
                str(args.dt),
                "--velocity",
                str(args.velocity),
                "--t-cap",
                str(args.original_t_cap),
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
                "--sequential-homological-trials-json",
                str(combined_path),
                "--output-dir",
                args.original_sequential_output_dir,
                "--run-name",
                seq_run_name,
            ],
            cwd=repo_root,
        )
        sequential_run_dir = sequential_root / seq_run_name

        baseline_raw = pd.read_csv(baseline_run_dir / "raw_runs.csv")
        seq_raw = pd.read_csv(sequential_run_dir / "raw_runs.csv")
        merged_raw = pd.concat([baseline_raw, seq_raw], ignore_index=True)
        merged_raw = merged_raw.sort_values(["domain", "model", "n", "r", "run_idx"]).reset_index(drop=True)

        merged_run_dir = merged_root / f"{domain}_nine_model_grid"
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
            home_model="sequential_homological",
            comparison_models=[m for m in sorted(merged_raw["model"].astype(str).unique()) if m != "sequential_homological"],
        )
        paired_df.to_csv(merged_run_dir / "paired_deltas.csv", index=False)
        paired_summary_df.to_csv(merged_run_dir / "paired_summary.csv", index=False)
        paired_overall_df.to_csv(merged_run_dir / "paired_overall_summary.csv", index=False)

        heatmap_dir = merged_run_dir / "heatmaps"
        for model_name in sorted(merged_raw["model"].astype(str).unique()):
            make_model_heatmaps(
                agg_df=agg_df,
                domain_name=domain,
                model_name=model_name,
                n_values=original_n_values,
                r_values=original_r_values,
                t_cap=float(args.original_t_cap),
                output_path=heatmap_dir / f"{domain}_{model_name}_mean_std_fail_heatmaps.png",
            )
        if not paired_summary_df.empty:
            for comparison in paired_summary_df["comparison"].drop_duplicates().tolist():
                make_pairwise_delta_heatmap(
                    paired_summary=paired_summary_df,
                    domain_name=domain,
                    comparison=comparison,
                    n_values=original_n_values,
                    r_values=original_r_values,
                    t_cap=float(args.original_t_cap),
                    output_path=heatmap_dir / f"{domain}_{comparison}_delta_heatmap.png",
                )

        metadata = {
            "script": "experiments/run_sequential_homological_extension.py",
            "mode": "original_grid_merge",
            "domain": domain,
            "domain_metadata": domain_metadata(domain),
            "baseline_run_dir": str(baseline_run_dir),
            "sequential_run_dir": str(sequential_run_dir),
            "combined_trials_json": str(combined_path),
            "n_values": original_n_values,
            "r_values": original_r_values,
            "t_cap": float(args.original_t_cap),
            "models": sorted(merged_raw["model"].astype(str).unique().tolist()),
        }
        write_run_metadata(merged_run_dir / "metadata.json", metadata)

    return merged_run_dirs


def run_sparse_suite(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    domains: List[str],
    tuning_seeds: List[int],
    timestamp: str,
) -> Dict[str, Path]:
    search_root = repo_root / args.sparse_search_output_dir
    bench_root = repo_root / args.sparse_benchmark_output_dir
    bench_root.mkdir(parents=True, exist_ok=True)
    sparse_n_values = parse_csv_ints(args.sparse_n_values)
    sparse_r_values = parse_csv_floats(args.sparse_r_values)
    sparse_run_dirs: Dict[str, Path] = {}

    for domain in domains:
        trial_paths: List[Path] = []
        for seed in tuning_seeds:
            run_name = f"sequential_sparse_{domain}_seed{seed}_{timestamp}"
            run_command(
                [
                    args.python,
                    "experiments/motion_model_param_search.py",
                    "--model",
                    "sequential_homological",
                    "--domain",
                    domain,
                    "--seed",
                    str(seed),
                    "--trials",
                    str(args.search_trials),
                    "--n-values",
                    args.sparse_n_values,
                    "--r-values",
                    args.sparse_r_values,
                    "--dt",
                    str(args.dt),
                    "--velocity",
                    str(args.velocity),
                    "--t-cap",
                    str(args.sparse_t_cap),
                    "--failure-penalty",
                    str(args.failure_penalty),
                    "--worst-case-weight",
                    str(args.worst_case_weight),
                    "--max-wall-seconds-per-run",
                    str(args.max_wall_seconds_per_run),
                    "--output-dir",
                    args.sparse_search_output_dir,
                    "--run-name",
                    run_name,
                ],
                cwd=repo_root,
            )
            trial_paths.append(search_root / run_name / "all_trials_full.json")

        combined_dir = search_root / f"sequential_sparse_{domain}_combined_{timestamp}"
        combined_path = combined_dir / "all_trials_full_combined.json"
        combine_trials(trial_paths, combined_path)

        run_name = f"{domain}_sparse_grid_{timestamp}"
        run_command(
            [
                args.python,
                "experiments/motion_model_grid_benchmark.py",
                "--domain",
                domain,
                "--models",
                ",".join(SPARSE_MODELS),
                "--runs-per-combo",
                str(args.runs_per_combo),
                "--n-values",
                args.sparse_n_values,
                "--r-values",
                args.sparse_r_values,
                "--dt",
                str(args.dt),
                "--velocity",
                str(args.velocity),
                "--t-cap",
                str(args.sparse_t_cap),
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
                "--sequential-homological-trials-json",
                str(combined_path),
                "--output-dir",
                args.sparse_benchmark_output_dir,
                "--run-name",
                run_name,
            ],
            cwd=repo_root,
        )
        sparse_run_dir = bench_root / run_name
        sparse_run_dirs[domain] = sparse_run_dir

        metadata_path = sparse_run_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "mode": "sparse_width_regime",
                "inradius_rule": "N_mobile * r > R_in(domain)",
                "sparse_grid": {
                    "n_values": sparse_n_values,
                    "r_values": sparse_r_values,
                },
            }
        )
        write_run_metadata(metadata_path, metadata)

    return sparse_run_dirs


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    original_domains = parse_csv_strs(args.original_domains)
    sparse_domains = parse_csv_strs(args.sparse_domains)
    tuning_seeds = [int(x.strip()) for x in args.tuning_seeds.split(",") if x.strip()]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    merged_run_dirs = refresh_original_grid(
        repo_root=repo_root,
        args=args,
        domains=original_domains,
        tuning_seeds=tuning_seeds,
        timestamp=timestamp,
    )
    sparse_run_dirs = run_sparse_suite(
        repo_root=repo_root,
        args=args,
        domains=sparse_domains,
        tuning_seeds=tuning_seeds,
        timestamp=timestamp,
    )

    if not args.skip_paper_assets:
        asset_cmd = [
            args.python,
            "experiments/generate_homological_paper_assets.py",
            "--paper-dir",
            args.paper_dir,
        ]
        for domain in original_domains:
            asset_cmd.extend(["--run", f"{domain}={merged_run_dirs[domain]}"])
        run_command(asset_cmd, cwd=repo_root)

    manifest = {
        "script": "experiments/run_sequential_homological_extension.py",
        "original_domains": original_domains,
        "sparse_domains": sparse_domains,
        "tuning_seeds": tuning_seeds,
        "original_merged_run_dirs": {domain: str(path) for domain, path in merged_run_dirs.items()},
        "sparse_run_dirs": {domain: str(path) for domain, path in sparse_run_dirs.items()},
        "paper_assets_refreshed": not bool(args.skip_paper_assets),
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
