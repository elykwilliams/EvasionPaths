# ******************************************************************************
#  Copyright (c) 2023, Kyle Williams - All Rights Reserved.
#  You may use, distribute and modify this code under the terms of the BSD-3
#  license. You should have received a copy of the BSD-3 license with this file.
#  If not, visit: https://opensource.org/licenses/BSD-3-Clause
# ******************************************************************************
from __future__ import annotations

import argparse
import contextlib
import io
import os
import math
import signal
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from joblib import Parallel, delayed
from tqdm import tqdm

def handler(*_):
    from utilities import TimedOutExc

    raise TimedOutExc

def parse_widths(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one width value.")
    try:
        return [float(value) for value in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Widths must be comma-separated floats.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bunimovich stadium detection-time experiments.")
    parser.add_argument("--num-sensors", type=int, default=20, help="Number of mobile sensors.")
    parser.add_argument("--k", type=float, default=0.3, help="Coverage parameter in the sensing-radius formula.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep.")
    parser.add_argument("--sensor-velocity", type=float, default=1.0, help="Initial speed magnitude for each mobile sensor.")
    parser.add_argument(
        "--n-sims",
        "--n-runs",
        dest="n_sims",
        type=int,
        default=2,
        help="Number of simulations to run for each width configuration.",
    )
    parser.add_argument("--max-time", type=int, default=10, help="Reserved timeout parameter; currently not enforced.")
    parser.add_argument("--domain-radius", type=float, default=1.0, help="Radius of the stadium endcaps.")
    parser.add_argument(
        "--widths",
        type=parse_widths,
        default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        help="Comma-separated list of stadium half-widths.",
    )
    parser.add_argument("--warmup-steps", type=int, default=500, help="Number of billiard-only warmup steps before detection.")
    parser.add_argument(
        "--square-init",
        action="store_true",
        help="Initialize mobile sensors from the student-style centered square box instead of the full stadium.",
    )
    parser.add_argument("--output-dir", type=str, default="./output", help="Root output directory.")
    parser.add_argument("--filename-base", type=str, default="data", help="Base filename for CSV and plot artifacts.")
    parser.add_argument("--run-name", type=str, default="", help="Optional run directory name. Defaults to a timestamped name.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Joblib parallel worker count.")
    return parser.parse_args()


def simulate(domain_width: float, args: argparse.Namespace) -> float:
    from boundary_geometry import BunimovichStadium
    from motion_model import BilliardMotion
    from sensor_network import SensorNetwork, generate_fence_sensors, generate_mobile_sensors
    from time_stepping import EvasionPathSimulation
    from utilities import EvasionPathError

    try:
        gamma = domain_width / args.domain_radius
        sensing_radius = math.sqrt((math.pi + 4 * gamma) / (args.k * args.num_sensors * math.pi))
        domain = BunimovichStadium(
            w=domain_width,
            r=args.domain_radius,
            L=2.0 * (domain_width + args.domain_radius),
            square_init=args.square_init,
            square_init_length=min(domain_width, args.domain_radius),
        )
        motion_model = BilliardMotion()
        fence = generate_fence_sensors(domain, sensing_radius)
        mobile_sensors = generate_mobile_sensors(domain, args.num_sensors, sensing_radius, args.sensor_velocity)
        sensor_network = SensorNetwork(mobile_sensors, motion_model, fence, sensing_radius, domain)
        for _ in range(args.warmup_steps):
            sensor_network.move(args.dt)
            sensor_network.update()
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            simulation = EvasionPathSimulation(sensor_network, args.dt)

            # signal.signal(signal.SIGALRM, handler)
            # signal.alarm(args.max_time)
            data = simulation.run()

    # Catch internal errors
    except EvasionPathError as e:
        data = str(e)

    # Catch all other errors
    except Exception as e:
        data = str(e)
    # Reset sigalarm
    finally:
        # signal.alarm(0) #max time of simulation
        return data

def width_label(domain_width: float) -> str:
    return f"{domain_width:.1f}"


def build_results_table(results_by_width: dict[float, list], domain_widths: list[float]):
    import numpy as np
    import pandas as pd

    columns = {width_label(width): list(results_by_width[width]) for width in domain_widths}
    df = pd.DataFrame(columns)
    df.index = np.arange(1, len(df) + 1)
    df.index.name = "run"
    return df


def numeric_results(results: list):
    import numpy as np
    import pandas as pd

    numeric = pd.to_numeric(pd.Series(results, dtype="object"), errors="coerce").to_numpy(dtype=float)
    return numeric[~np.isnan(numeric)]


def save_summary_plot(results_by_width: dict[float, list], domain_widths: list[float], output_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    xs = np.array(domain_widths, dtype=float)
    means = []
    lower = []
    upper = []

    for width in domain_widths:
        valid = numeric_results(results_by_width[width])
        if len(valid) == 0:
            means.append(np.nan)
            lower.append(np.nan)
            upper.append(np.nan)
            continue

        mean = float(np.mean(valid))
        if len(valid) > 1:
            sem = float(np.std(valid, ddof=1) / np.sqrt(len(valid)))
            delta = 1.96 * sem
        else:
            delta = 0.0

        means.append(mean)
        lower.append(mean - delta)
        upper.append(mean + delta)

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(xs, means, marker="o", linewidth=1.6, color="#1f77b4")
    ax.fill_between(xs, lower, upper, color="#1f77b4", alpha=0.22)
    ax.set_title("Average detection time with 95% confidence intervals")
    ax.set_xlabel("Width")
    ax.set_ylabel("Detection time")
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_experiment(args: argparse.Namespace, run_dir: str) -> None:
    results_by_width = {}
    for domain_width in tqdm(args.widths, desc="Widths", unit="width"):
        times = []
        parallel = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")
        tasks = (delayed(simulate)(domain_width, args) for _ in range(args.n_sims))
        sim_desc = f"Sims @ width={domain_width:.1f}"
        with tqdm(total=args.n_sims, desc=sim_desc, unit="sim", leave=False) as sim_bar:
            for result in parallel(tasks):
                times.append(result)
                sim_bar.update(1)
        results_by_width[domain_width] = times

    csv_path = os.path.join(run_dir, f"{args.filename_base}.csv")
    plot_path = os.path.join(run_dir, f"data_plot_{args.n_sims}.png")

    results_df = build_results_table(results_by_width, args.widths)
    results_df.to_csv(csv_path)
    save_summary_plot(results_by_width, args.widths, plot_path)



if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now()
    prefix = "bunimovich_carolyn_sq" if args.square_init else "bunimovich_carolyn"
    default_run_name = f"{prefix}_{args.n_sims}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    run_name = args.run_name or default_run_name
    run_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_dir, exist_ok=False)

    run_experiment(args, run_dir)
