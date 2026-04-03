#!/usr/bin/env python3
"""Search homological motion-model parameters that minimize capped detection time over an (n, r) grid."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from benchmark_common import (
    build_best_params_archive,
    build_domain,
    build_motion_model,
    canonical_model_name,
    combo_key,
    default_params_archive_path,
    domain_metadata,
    generate_connected_initial_condition,
    parse_csv_floats,
    parse_csv_ints,
)
from sensor_network import Sensor, SensorNetwork
from time_stepping import EvasionPathSimulation


@dataclass
class RunResult:
    n_sensors: int
    sensing_radius: float
    seed: int
    detection_time: float
    tau: float
    failed: bool
    error: str = ""


def run_one(
    *,
    model_name: str,
    params: Dict[str, float | bool],
    domain_name: str,
    n_sensors: int,
    radius: float,
    dt: float,
    sensor_velocity: float,
    seed: int,
    t_cap: float,
    max_wall_seconds: float,
    suppress_output: bool,
) -> RunResult:
    np.random.seed(seed)
    domain = build_domain(domain_name)
    init_result = generate_connected_initial_condition(
        domain_name=domain_name,
        domain=domain,
        n_sensors=n_sensors,
        radius=radius,
        sensor_velocity=sensor_velocity,
        seed=seed,
        max_retries=200,
    )
    if not init_result.feasible or init_result.initial_condition is None:
        return RunResult(
            n_sensors=n_sensors,
            sensing_radius=radius,
            seed=seed,
            detection_time=float("inf"),
            tau=float(t_cap),
            failed=True,
            error="InitializationInfeasible: unable to sample a connected initialization.",
        )
    motion_model = build_motion_model(
        model_name,
        domain_name=domain_name,
        n_sensors=n_sensors,
        radius=radius,
        dt=dt,
        sensor_velocity=sensor_velocity,
        tuned_params_by_combo={combo_key(n_sensors, radius): {"params": params}},
    )
    fence = [
        Sensor(np.array(pos, dtype=float), np.zeros(2, dtype=float), radius, boundary_sensor=True)
        for pos in init_result.initial_condition.fence_positions
    ]
    mobile_sensors = [
        Sensor(np.array(pos, dtype=float), np.array(vel, dtype=float), radius, boundary_sensor=False)
        for pos, vel in zip(
            init_result.initial_condition.mobile_positions,
            init_result.initial_condition.mobile_velocities,
        )
    ]
    sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, radius, domain)
    def _timeout_handler(_signum, _frame):
        raise TimeoutError("Simulation exceeded wall-clock timeout.")

    def _run() -> RunResult:
        sim = EvasionPathSimulation(sensor_network, dt, end_time=t_cap)
        while sim.cycle_label.has_intruder():
            sim.do_timestep()
            if 0 < sim.Tend < sim.time:
                break
        time_reached = float(sim.time)
        unresolved = bool(sim.cycle_label.has_intruder())
        failed = unresolved or time_reached >= (t_cap - 1e-12)
        tau = float(t_cap if failed else min(time_reached, t_cap))
        detection_time = float("inf") if failed else time_reached
        return RunResult(
            n_sensors=n_sensors,
            sensing_radius=radius,
            seed=seed,
            detection_time=detection_time,
            tau=tau,
            failed=failed,
        )

    old_handler = signal.getsignal(signal.SIGALRM)
    try:
        if max_wall_seconds > 0:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, float(max_wall_seconds))

        if suppress_output:
            sink = io.StringIO()
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                return _run()
        return _run()
    except TimeoutError:
        return RunResult(
            n_sensors=n_sensors,
            sensing_radius=radius,
            seed=seed,
            detection_time=float("inf"),
            tau=float(t_cap),
            failed=True,
            error="TimeoutError: Simulation exceeded wall-clock timeout.",
        )
    except Exception as exc:
        return RunResult(
            n_sensors=n_sensors,
            sensing_radius=radius,
            seed=seed,
            detection_time=float("inf"),
            tau=float(t_cap),
            failed=True,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if max_wall_seconds > 0:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)


def evaluate_candidate(
    *,
    model_name: str,
    params: Dict[str, float | bool],
    domain_name: str,
    n_values: Sequence[int],
    r_values: Sequence[float],
    seeds: Sequence[int],
    dt: float,
    sensor_velocity: float,
    t_cap: float,
    failure_penalty: float,
    worst_case_weight: float,
    max_wall_seconds_per_run: float,
    suppress_output: bool,
) -> Dict:
    run_results: List[RunResult] = []
    for n_sensors in n_values:
        for radius in r_values:
            for seed in seeds:
                run_results.append(
                    run_one(
                        model_name=model_name,
                        params=params,
                        domain_name=domain_name,
                        n_sensors=n_sensors,
                        radius=radius,
                        dt=dt,
                        sensor_velocity=sensor_velocity,
                        seed=seed,
                        t_cap=t_cap,
                        max_wall_seconds=max_wall_seconds_per_run,
                        suppress_output=suppress_output,
                    )
                )

    taus = np.asarray([r.tau for r in run_results], dtype=float)
    failed = np.asarray([1.0 if r.failed else 0.0 for r in run_results], dtype=float)
    mean_tau = float(np.mean(taus)) if len(taus) else float("inf")
    worst_tau = float(np.max(taus)) if len(taus) else float("inf")
    failure_rate = float(np.mean(failed)) if len(failed) else 1.0
    score = mean_tau + failure_penalty * failure_rate + worst_case_weight * worst_tau

    by_combo: Dict[str, Dict[str, float]] = {}
    for n_sensors in n_values:
        for radius in r_values:
            combo_runs = [r for r in run_results if r.n_sensors == n_sensors and abs(r.sensing_radius - radius) < 1e-12]
            combo_taus = np.asarray([r.tau for r in combo_runs], dtype=float)
            combo_failed = np.asarray([1.0 if r.failed else 0.0 for r in combo_runs], dtype=float)
            by_combo[combo_key(n_sensors, radius)] = {
                "mean_tau": float(np.mean(combo_taus)) if len(combo_taus) else float("inf"),
                "worst_tau": float(np.max(combo_taus)) if len(combo_taus) else float("inf"),
                "failure_rate": float(np.mean(combo_failed)) if len(combo_failed) else 1.0,
            }

    return {
        "score": score,
        "mean_tau": mean_tau,
        "worst_tau": worst_tau,
        "failure_rate": failure_rate,
        "params": params,
        "runs": [r.__dict__ for r in run_results],
        "by_combo": by_combo,
    }


def sample_candidate(
    *,
    model_name: str,
    rng: np.random.Generator,
    sensor_velocity: float,
) -> Dict[str, float | bool]:
    if model_name in {"homological", "sequential_homological"}:
        auto_d_safe = bool(rng.integers(0, 2))
        params = {
            "max_speed": float(rng.uniform(0.2, 2.0)),
            "lambda_shrink": float(10 ** rng.uniform(-1.0, 0.7)),
            "mu_curvature": float(rng.uniform(0.0, 2.0)),
            "eta_cohesion": float(rng.uniform(0.0, 2.0)),
            "repulsion_strength": float(10 ** rng.uniform(-3.0, 0.0)),
            "repulsion_power": float(rng.uniform(1.0, 4.0)),
            "auto_d_safe": auto_d_safe,
            "d_safe_manual": float(rng.uniform(0.05, 1.0)),
        }
        if model_name == "sequential_homological":
            params["overlap_threshold"] = float(rng.uniform(0.1, 0.7))
        return params
    raise ValueError(f"Unsupported tunable model: {model_name}")


def run_random_search(
    *,
    model_name: str,
    domain_name: str,
    trials: int,
    seed: int,
    n_values: Sequence[int],
    r_values: Sequence[float],
    eval_seeds: Sequence[int],
    dt: float,
    sensor_velocity: float,
    t_cap: float,
    failure_penalty: float,
    worst_case_weight: float,
    max_wall_seconds_per_run: float,
    suppress_output: bool,
) -> Tuple[Dict, List[Dict]]:
    rng = np.random.default_rng(seed)
    history: List[Dict] = []
    best: Dict | None = None

    for trial_idx in range(trials):
        params = sample_candidate(model_name=model_name, rng=rng, sensor_velocity=sensor_velocity)
        metrics = evaluate_candidate(
            model_name=model_name,
            params=params,
            domain_name=domain_name,
            n_values=n_values,
            r_values=r_values,
            seeds=eval_seeds,
            dt=dt,
            sensor_velocity=sensor_velocity,
            t_cap=t_cap,
            failure_penalty=failure_penalty,
            worst_case_weight=worst_case_weight,
            max_wall_seconds_per_run=max_wall_seconds_per_run,
            suppress_output=suppress_output,
        )
        metrics["trial"] = trial_idx
        history.append(metrics)
        if best is None or float(metrics["score"]) < float(best["score"]):
            best = metrics
            print(
                f"[{model_name}][{domain_name}] [trial {trial_idx:04d}] NEW BEST "
                f"score={best['score']:.4f} fail={best['failure_rate']:.3f} mean_tau={best['mean_tau']:.4f}"
            )
        else:
            print(
                f"[{model_name}][{domain_name}] [trial {trial_idx:04d}] "
                f"score={metrics['score']:.4f} fail={metrics['failure_rate']:.3f} mean_tau={metrics['mean_tau']:.4f}"
            )

    assert best is not None
    return best, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random search over homological-family motion-model parameters on an (n, r) grid.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["homological", "sequential_homological", "sequential_homological_motion"],
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="square",
        choices=["square", "circle", "rectangle_2to1_area1", "stadium_w0p6", "stadium_w1p2"],
    )
    parser.add_argument("--n-values", type=str, default="8,10,12,14,16")
    parser.add_argument("--r-values", type=str, default="0.20,0.24,0.28,0.32")
    parser.add_argument("--eval-seeds", type=str, default="6,13,29")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--t-cap", type=float, default=5.0)
    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--worst-case-weight", type=float, default=0.25)
    parser.add_argument("--max-wall-seconds-per-run", type=float, default=20.0)
    parser.add_argument("--show-sim-output", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="experiments/output/motion_model_param_search")
    parser.add_argument("--params-dir", type=str, default="output/params")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--write-full-history", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    model_name = canonical_model_name(args.model)
    n_values = parse_csv_ints(args.n_values)
    r_values = parse_csv_floats(args.r_values)
    eval_seeds = parse_csv_ints(args.eval_seeds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{model_name}_{args.domain}_grid_t{int(args.t_cap)}_trials{args.trials}_seed{args.seed}_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best, history = run_random_search(
        model_name=model_name,
        domain_name=args.domain,
        trials=int(args.trials),
        seed=int(args.seed),
        n_values=n_values,
        r_values=r_values,
        eval_seeds=eval_seeds,
        dt=float(args.dt),
        sensor_velocity=float(args.velocity),
        t_cap=float(args.t_cap),
        failure_penalty=float(args.failure_penalty),
        worst_case_weight=float(args.worst_case_weight),
        max_wall_seconds_per_run=float(args.max_wall_seconds_per_run),
        suppress_output=not bool(args.show_sim_output),
    )

    config = {
        "model": model_name,
        "domain": args.domain,
        "domain_metadata": domain_metadata(args.domain),
        "n_values": n_values,
        "r_values": r_values,
        "eval_seeds": eval_seeds,
        "trials": int(args.trials),
        "seed": int(args.seed),
        "dt": float(args.dt),
        "velocity": float(args.velocity),
        "t_cap": float(args.t_cap),
        "failure_penalty": float(args.failure_penalty),
        "worst_case_weight": float(args.worst_case_weight),
        "max_wall_seconds_per_run": float(args.max_wall_seconds_per_run),
        "show_sim_output": bool(args.show_sim_output),
    }
    (run_dir / "search_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "best_result.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

    params_archive = build_best_params_archive(
        model_name=model_name,
        domain_name=args.domain,
        dt=float(args.dt),
        sensor_velocity=float(args.velocity),
        t_cap=float(args.t_cap),
        failure_penalty=float(args.failure_penalty),
        worst_case_weight=float(args.worst_case_weight),
        n_values=n_values,
        r_values=r_values,
        history=history,
        search_config=config,
        source_run_dir=str(run_dir),
    )
    run_params_path = run_dir / "best_params_by_combo.json"
    run_params_path.write_text(json.dumps(params_archive, indent=2, sort_keys=True), encoding="utf-8")
    params_dir = repo_root / args.params_dir
    params_dir.mkdir(parents=True, exist_ok=True)
    shared_params_path = default_params_archive_path(
        repo_root,
        model_name=model_name,
        domain_name=args.domain,
        dt=float(args.dt),
        sensor_velocity=float(args.velocity),
        t_cap=float(args.t_cap),
    )
    shared_params_path = params_dir / shared_params_path.name
    shared_params_path.write_text(json.dumps(params_archive, indent=2, sort_keys=True), encoding="utf-8")

    history_summary = [
        {
            "trial": int(item["trial"]),
            "score": float(item["score"]),
            "mean_tau": float(item["mean_tau"]),
            "worst_tau": float(item["worst_tau"]),
            "failure_rate": float(item["failure_rate"]),
            "params": item["params"],
        }
        for item in history
    ]
    (run_dir / "trial_summary.json").write_text(json.dumps(history_summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.write_full_history:
        (run_dir / "all_trials_full.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Search complete: {run_dir}")
    print(f"Best score: {best['score']:.4f}")
    print(f"Per-combo params: {run_params_path}")
    print(f"Shared params archive: {shared_params_path}")


if __name__ == "__main__":
    main()
