from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive setup UI + post-run Reeb playback for Homological Motion.")
    parser.add_argument("--num-mobile", type=int, default=12)
    parser.add_argument("--radius", type=float, default=0.22)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sensor-velocity", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--clear-streak", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--lambda-shrink", type=float, default=1.0)
    parser.add_argument("--mu-curvature", type=float, default=0.5)
    parser.add_argument("--eta-cohesion", type=float, default=0.2)
    parser.add_argument("--repulsion-strength", type=float, default=0.1)
    parser.add_argument("--repulsion-power", type=float, default=2.0)
    parser.add_argument("--d-safe-manual", type=float, default=0.2)
    parser.add_argument("--auto-d-safe", action="store_true", default=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parents[1] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from motion_model import HomologicalDynamicsMotion
    from _motion_reeb_setup_ui import launch_motion_reeb_setup_ui

    launch_motion_reeb_setup_ui(
        title="Homological Motion: Setup + Reeb Playback",
        model_key="homological_motion",
        model_label="Homological Motion",
        motion_factory=HomologicalDynamicsMotion,
        num_mobile=args.num_mobile,
        sensing_radius=args.radius,
        dt=args.dt,
        sensor_velocity=args.sensor_velocity,
        seed=args.seed,
        max_steps=args.max_steps,
        clear_streak=args.clear_streak,
        host=args.host,
        port=args.port,
        motion_kwargs={
            "max_speed": args.max_speed,
            "lambda_shrink": args.lambda_shrink,
            "mu_curvature": args.mu_curvature,
            "eta_cohesion": args.eta_cohesion,
            "repulsion_strength": args.repulsion_strength,
            "repulsion_power": args.repulsion_power,
            "d_safe_manual": args.d_safe_manual,
            "auto_d_safe": args.auto_d_safe,
        },
    )

    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
