#!/usr/bin/env python3
"""Search HomologicalDynamicsMotion weights that minimize detection time over an (n, r) grid."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from boundary_geometry import RectangularDomain
from motion_model import HomologicalDynamicsMotion
from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors
from time_stepping import EvasionPathSimulation


@dataclass
class CandidateWeights:
    max_speed: float
    lambda_shrink: float
    mu_curvature: float
    eta_cohesion: float
    repulsion_strength: float
    repulsion_power: float
    auto_d_safe: bool
    d_safe_manual: float


@dataclass
class RunResult:
    n_sensors: int
    sensing_radius: float
    seed: int
    detection_time: float
    tau: float
    failed: bool
    error: str = ""


def parse_csv_ints(raw: str) -> List[int]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(v) for v in values]


def parse_csv_floats(raw: str) -> List[float]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(v) for v in values]


def build_simulation(
    *,
    n_sensors: int,
    radius: float,
    dt: float,
    sensor_velocity: float,
    seed: int,
    end_time: float,
    weights: CandidateWeights,
) -> EvasionPathSimulation:
    np.random.seed(seed)
    domain = RectangularDomain()
    motion_model = HomologicalDynamicsMotion(
        sensing_radius=radius,
        max_speed=weights.max_speed,
        lambda_shrink=weights.lambda_shrink,
        mu_curvature=weights.mu_curvature,
        eta_cohesion=weights.eta_cohesion,
        repulsion_strength=weights.repulsion_strength,
        repulsion_power=weights.repulsion_power,
        auto_d_safe=weights.auto_d_safe,
        d_safe_manual=weights.d_safe_manual,
    )
    fence = generate_fence_sensors(domain, radius)
    mobile_sensors = generate_mobile_sensors(domain, n_sensors, radius, sensor_velocity)
    sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, radius, domain)
    return EvasionPathSimulation(sensor_network, dt, end_time=end_time)


def run_one(
    *,
    n_sensors: int,
    radius: float,
    dt: float,
    sensor_velocity: float,
    seed: int,
    t_cap: float,
    weights: CandidateWeights,
) -> RunResult:
    try:
        sim = build_simulation(
            n_sensors=n_sensors,
            radius=radius,
            dt=dt,
            sensor_velocity=sensor_velocity,
            seed=seed,
            end_time=t_cap,
            weights=weights,
        )
        time_reached = float(sim.run())
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
    except Exception as exc:  # Keep long search robust to occasional integrator/topology failures.
        return RunResult(
            n_sensors=n_sensors,
            sensing_radius=radius,
            seed=seed,
            detection_time=float("inf"),
            tau=float(t_cap),
            failed=True,
            error=f"{type(exc).__name__}: {exc}",
        )


def evaluate_candidate(
    *,
    weights: CandidateWeights,
    n_values: Sequence[int],
    r_values: Sequence[float],
    seeds: Sequence[int],
    dt: float,
    sensor_velocity: float,
    t_cap: float,
    failure_penalty: float,
    worst_case_weight: float,
) -> Dict:
    run_results: List[RunResult] = []
    for n_sensors in n_values:
        for radius in r_values:
            for seed in seeds:
                result = run_one(
                    n_sensors=n_sensors,
                    radius=radius,
                    dt=dt,
                    sensor_velocity=sensor_velocity,
                    seed=seed,
                    t_cap=t_cap,
                    weights=weights,
                )
                run_results.append(result)

    taus = np.asarray([r.tau for r in run_results], dtype=float)
    failed = np.asarray([1.0 if r.failed else 0.0 for r in run_results], dtype=float)
    mean_tau = float(np.mean(taus)) if len(taus) else float("inf")
    worst_tau = float(np.max(taus)) if len(taus) else float("inf")
    failure_rate = float(np.mean(failed)) if len(failed) else 1.0
    score = mean_tau + failure_penalty * failure_rate + worst_case_weight * worst_tau

    by_combo = {}
    for n_sensors in n_values:
        for radius in r_values:
            combo_runs = [r for r in run_results if r.n_sensors == n_sensors and abs(r.sensing_radius - radius) < 1e-12]
            combo_taus = np.asarray([r.tau for r in combo_runs], dtype=float)
            combo_failed = np.asarray([1.0 if r.failed else 0.0 for r in combo_runs], dtype=float)
            by_combo[f"n={n_sensors},r={radius:.6f}"] = {
                "mean_tau": float(np.mean(combo_taus)) if len(combo_taus) else float("inf"),
                "worst_tau": float(np.max(combo_taus)) if len(combo_taus) else float("inf"),
                "failure_rate": float(np.mean(combo_failed)) if len(combo_failed) else 1.0,
            }

    return {
        "score": score,
        "mean_tau": mean_tau,
        "worst_tau": worst_tau,
        "failure_rate": failure_rate,
        "weights": asdict(weights),
        "runs": [asdict(r) for r in run_results],
        "by_combo": by_combo,
    }


def sample_random_candidate(rng: np.random.Generator) -> CandidateWeights:
    auto_d_safe = bool(rng.integers(0, 2))
    return CandidateWeights(
        max_speed=float(rng.uniform(0.2, 2.0)),
        lambda_shrink=float(10 ** rng.uniform(-1.0, 0.7)),
        mu_curvature=float(rng.uniform(0.0, 2.0)),
        eta_cohesion=float(rng.uniform(0.0, 2.0)),
        repulsion_strength=float(10 ** rng.uniform(-3.0, 0.0)),
        repulsion_power=float(rng.uniform(1.0, 4.0)),
        auto_d_safe=auto_d_safe,
        d_safe_manual=float(rng.uniform(0.05, 1.0)),
    )


def run_random_search(
    *,
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
) -> Tuple[Dict, List[Dict]]:
    rng = np.random.default_rng(seed)
    history: List[Dict] = []
    best: Dict | None = None

    for trial_idx in range(trials):
        weights = sample_random_candidate(rng)
        metrics = evaluate_candidate(
            weights=weights,
            n_values=n_values,
            r_values=r_values,
            seeds=eval_seeds,
            dt=dt,
            sensor_velocity=sensor_velocity,
            t_cap=t_cap,
            failure_penalty=failure_penalty,
            worst_case_weight=worst_case_weight,
        )
        metrics["trial"] = trial_idx
        history.append(metrics)

        if best is None or metrics["score"] < best["score"]:
            best = metrics
            print(
                f"[trial {trial_idx:04d}] NEW BEST score={best['score']:.4f} "
                f"fail={best['failure_rate']:.3f} mean_tau={best['mean_tau']:.4f}"
            )
        else:
            print(
                f"[trial {trial_idx:04d}] score={metrics['score']:.4f} "
                f"fail={metrics['failure_rate']:.3f} mean_tau={metrics['mean_tau']:.4f}"
            )

    assert best is not None
    return best, history


def run_optuna_search(
    *,
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
) -> Tuple[Dict, List[Dict]]:
    import optuna

    trial_history: List[Dict] = []

    def objective(trial: "optuna.trial.Trial") -> float:
        auto_d_safe = trial.suggest_categorical("auto_d_safe", [True, False])
        weights = CandidateWeights(
            max_speed=trial.suggest_float("max_speed", 0.2, 2.0),
            lambda_shrink=trial.suggest_float("lambda_shrink", 0.1, 5.0, log=True),
            mu_curvature=trial.suggest_float("mu_curvature", 0.0, 2.0),
            eta_cohesion=trial.suggest_float("eta_cohesion", 0.0, 2.0),
            repulsion_strength=trial.suggest_float("repulsion_strength", 1e-3, 1.0, log=True),
            repulsion_power=trial.suggest_float("repulsion_power", 1.0, 4.0),
            auto_d_safe=auto_d_safe,
            d_safe_manual=trial.suggest_float("d_safe_manual", 0.05, 1.0),
        )
        metrics = evaluate_candidate(
            weights=weights,
            n_values=n_values,
            r_values=r_values,
            seeds=eval_seeds,
            dt=dt,
            sensor_velocity=sensor_velocity,
            t_cap=t_cap,
            failure_penalty=failure_penalty,
            worst_case_weight=worst_case_weight,
        )
        metrics["trial"] = trial.number
        trial_history.append(metrics)
        trial.set_user_attr("mean_tau", metrics["mean_tau"])
        trial.set_user_attr("failure_rate", metrics["failure_rate"])
        trial.set_user_attr("weights", metrics["weights"])
        print(
            f"[trial {trial.number:04d}] score={metrics['score']:.4f} "
            f"fail={metrics['failure_rate']:.3f} mean_tau={metrics['mean_tau']:.4f}"
        )
        return float(metrics["score"])

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=trials)

    best_trial_number = int(study.best_trial.number)
    best = min((m for m in trial_history if int(m["trial"]) == best_trial_number), key=lambda m: m["score"])
    return best, trial_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search HomologicalDynamicsMotion weights for minimum detection time.")
    parser.add_argument("--n-values", type=str, default="8,10,12,14,16")
    parser.add_argument("--r-values", type=str, default="0.20,0.24,0.28,0.32")
    parser.add_argument("--seed-values", type=str, default="6,13,29")

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--t-cap", type=float, default=5.0)

    parser.add_argument("--method", type=str, choices=["auto", "optuna", "random"], default="auto")
    parser.add_argument("--trials", type=int, default=80)
    parser.add_argument("--search-seed", type=int, default=7)

    parser.add_argument("--failure-penalty", type=float, default=10.0)
    parser.add_argument("--worst-case-weight", type=float, default=0.25)

    parser.add_argument("--output-dir", type=str, default="experiments/output/homological_weight_search")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--save-all-runs", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_values = parse_csv_ints(args.n_values)
    r_values = parse_csv_floats(args.r_values)
    eval_seeds = parse_csv_ints(args.seed_values)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"weight_search_{timestamp}"
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Grid n-values: {n_values}")
    print(f"Grid r-values: {r_values}")
    print(f"Eval seeds: {eval_seeds}")
    print(f"Trials: {args.trials}, method={args.method}, t_cap={args.t_cap}")

    method = args.method
    if method == "auto":
        try:
            import optuna  # noqa: F401

            method = "optuna"
        except Exception:
            method = "random"

    if method == "optuna":
        best, history = run_optuna_search(
            trials=args.trials,
            seed=args.search_seed,
            n_values=n_values,
            r_values=r_values,
            eval_seeds=eval_seeds,
            dt=args.dt,
            sensor_velocity=args.velocity,
            t_cap=args.t_cap,
            failure_penalty=args.failure_penalty,
            worst_case_weight=args.worst_case_weight,
        )
    else:
        best, history = run_random_search(
            trials=args.trials,
            seed=args.search_seed,
            n_values=n_values,
            r_values=r_values,
            eval_seeds=eval_seeds,
            dt=args.dt,
            sensor_velocity=args.velocity,
            t_cap=args.t_cap,
            failure_penalty=args.failure_penalty,
            worst_case_weight=args.worst_case_weight,
        )

    config = {
        "method_requested": args.method,
        "method_used": method,
        "trials": args.trials,
        "n_values": n_values,
        "r_values": r_values,
        "seed_values": eval_seeds,
        "dt": args.dt,
        "velocity": args.velocity,
        "t_cap": args.t_cap,
        "failure_penalty": args.failure_penalty,
        "worst_case_weight": args.worst_case_weight,
        "search_seed": args.search_seed,
    }

    (run_dir / "search_config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "best_result.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")

    history_summary = [
        {
            "trial": int(item.get("trial", idx)),
            "score": float(item["score"]),
            "mean_tau": float(item["mean_tau"]),
            "worst_tau": float(item["worst_tau"]),
            "failure_rate": float(item["failure_rate"]),
            **item["weights"],
        }
        for idx, item in enumerate(history)
    ]
    (run_dir / "trial_summary.json").write_text(json.dumps(history_summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.save_all_runs:
        (run_dir / "all_trials_full.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")

    print("\nBest candidate:")
    print(json.dumps(best["weights"], indent=2, sort_keys=True))
    print(
        f"score={best['score']:.4f} mean_tau={best['mean_tau']:.4f} "
        f"worst_tau={best['worst_tau']:.4f} failure_rate={best['failure_rate']:.3f}"
    )
    print(f"Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
