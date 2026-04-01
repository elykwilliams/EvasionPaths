#!/usr/bin/env python3
"""Run the cross-domain homological-tuning benchmark pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


DEFAULT_DOMAINS = ["square", "circle", "stadium_w0p6", "stadium_w1p2"]
DEFAULT_SEARCH_MODELS = ["homological"]
DEFAULT_TUNING_SEEDS = [7, 19]


def run_command(cmd: List[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl_cross_domain_pipeline")
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
    dst_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-domain 8-model homological benchmark pipeline.")
    parser.add_argument("--python", type=str, default="/Users/marco/miniconda3/envs/EvasionPaths/bin/python")
    parser.add_argument("--domains", type=str, default=",".join(DEFAULT_DOMAINS))
    parser.add_argument("--search-models", type=str, default=",".join(DEFAULT_SEARCH_MODELS))
    parser.add_argument(
        "--benchmark-models",
        type=str,
        default="billiard,brownian_low,brownian_med,brownian_high,vicsek_low,vicsek_med,vicsek_high,homological",
    )
    parser.add_argument("--tuning-seeds", type=str, default=",".join(str(x) for x in DEFAULT_TUNING_SEEDS))
    parser.add_argument("--search-trials", type=int, default=30)
    parser.add_argument("--runs-per-combo", type=int, default=10)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--t-cap", type=float, default=5.0)
    parser.add_argument("--max-wall-seconds-per-run", type=float, default=20.0)
    parser.add_argument("--search-output-dir", type=str, default="experiments/output/cross_domain_param_search")
    parser.add_argument("--benchmark-output-dir", type=str, default="experiments/output/cross_domain_motion_benchmark")
    parser.add_argument("--paper-dir", type=str, default="latex/homologicalDynamics")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    search_models = [x.strip() for x in args.search_models.split(",") if x.strip()]
    benchmark_models = [x.strip() for x in args.benchmark_models.split(",") if x.strip()]
    tuning_seeds = [int(x.strip()) for x in args.tuning_seeds.split(",") if x.strip()]

    combined_jsons: Dict[str, Dict[str, Path]] = {}
    benchmark_run_dirs: Dict[str, Path] = {}

    for domain in domains:
        combined_jsons[domain] = {}
        for model in search_models:
            trial_paths: List[Path] = []
            for seed in tuning_seeds:
                run_name = f"{model}_{domain}_seed{seed}"
                run_command(
                    [
                        args.python,
                        "experiments/motion_model_param_search.py",
                        "--model",
                        model,
                        "--domain",
                        domain,
                        "--seed",
                        str(seed),
                        "--trials",
                        str(args.search_trials),
                        "--dt",
                        str(args.dt),
                        "--velocity",
                        str(args.velocity),
                        "--t-cap",
                        str(args.t_cap),
                        "--max-wall-seconds-per-run",
                        str(args.max_wall_seconds_per_run),
                        "--output-dir",
                        args.search_output_dir,
                        "--run-name",
                        run_name,
                    ],
                    cwd=repo_root,
                )
                trial_paths.append(repo_root / args.search_output_dir / run_name / "all_trials_full.json")

            combined_dir = repo_root / args.search_output_dir / f"{model}_{domain}_combined"
            combined_dir.mkdir(parents=True, exist_ok=True)
            combined_path = combined_dir / "all_trials_full_combined.json"
            combine_trials(trial_paths, combined_path)
            combined_jsons[domain][model] = combined_path

        benchmark_run_name = f"{domain}_eight_model_grid"
        run_command(
            [
                args.python,
                "experiments/motion_model_grid_benchmark.py",
                "--domain",
                domain,
                "--models",
                ",".join(benchmark_models),
                "--runs-per-combo",
                str(args.runs_per_combo),
                "--dt",
                str(args.dt),
                "--velocity",
                str(args.velocity),
                "--t-cap",
                str(args.t_cap),
                "--max-wall-seconds-per-run",
                str(args.max_wall_seconds_per_run),
                "--homological-trials-json",
                str(combined_jsons[domain]["homological"]),
                "--output-dir",
                args.benchmark_output_dir,
                "--run-name",
                benchmark_run_name,
            ],
            cwd=repo_root,
        )
        benchmark_run_dirs[domain] = repo_root / args.benchmark_output_dir / benchmark_run_name

    asset_cmd = [
        args.python,
        "experiments/generate_homological_paper_assets.py",
        "--paper-dir",
        args.paper_dir,
    ]
    for domain in domains:
        asset_cmd.extend(["--run", f"{domain}={benchmark_run_dirs[domain]}"])
    run_command(asset_cmd, cwd=repo_root)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
