#!/usr/bin/env python3
"""Train PPO on fixed unit-square RL environment (v1 contract)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO on the fixed unit-square evasion RL task.")
    parser.add_argument("--total-timesteps", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--model-kind",
        type=str,
        default="baseline_gat",
        choices=("baseline_gat", "dart_gat", "boundary_cycle", "homological_gat", "structured_velocity"),
    )
    parser.add_argument("--max-speed-scale", type=float, default=None)
    parser.add_argument("--num-mobile-sensors", type=int, default=None)
    parser.add_argument("--fence-sensing-radius", type=float, default=None)
    parser.add_argument("--fence-offset-ratio", type=float, default=None)
    parser.add_argument("--use-weighted-alpha", action="store_true")
    parser.add_argument("--outdir", type=str, default="experiments/output/rl_unit_square")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-max-steps", type=int, default=800)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--disable-attention-log", action="store_true")
    parser.add_argument("--attention-topk", type=int, default=40)
    parser.add_argument("--attention-log-every", type=int, default=10)
    parser.add_argument("--attention-log-full", action="store_true")
    parser.add_argument("--attention-log-path", type=str, default="")
    parser.add_argument("--enable-event-log", action="store_true")
    parser.add_argument("--true-cycle-closed-reward", type=float, default=None)
    parser.add_argument("--true-cycle-added-penalty", type=float, default=None)
    parser.add_argument("--time-penalty", type=float, default=None)
    parser.add_argument("--control-effort-penalty", type=float, default=None)
    parser.add_argument("--clear-bonus", type=float, default=None)
    parser.add_argument("--success-time-bonus-weight", type=float, default=None)
    parser.add_argument("--one-hole-linger-weight", type=float, default=None)
    parser.add_argument("--one-hole-area-scale-alpha", type=float, default=None)
    parser.add_argument("--one-hole-perimeter-scale-alpha", type=float, default=None)
    parser.add_argument("--area-progress-weight", type=float, default=None)
    parser.add_argument("--perimeter-progress-weight", type=float, default=None)
    parser.add_argument("--largest-area-progress-weight", type=float, default=None)
    parser.add_argument("--largest-perimeter-progress-weight", type=float, default=None)
    parser.add_argument("--area-regress-weight", type=float, default=None)
    parser.add_argument("--perimeter-regress-weight", type=float, default=None)
    parser.add_argument("--largest-area-regress-weight", type=float, default=None)
    parser.add_argument("--largest-perimeter-regress-weight", type=float, default=None)
    parser.add_argument("--area-residual-weight", type=float, default=None)
    parser.add_argument("--perimeter-residual-weight", type=float, default=None)
    parser.add_argument("--largest-area-residual-weight", type=float, default=None)
    parser.add_argument("--largest-perimeter-residual-weight", type=float, default=None)
    parser.add_argument("--neighbor-min-ratio", type=float, default=None)
    parser.add_argument("--neighbor-max-ratio", type=float, default=None)
    parser.add_argument("--neighbor-close-weight", type=float, default=None)
    parser.add_argument("--neighbor-far-weight", type=float, default=None)
    parser.add_argument("--mobile-overlap-weight", type=float, default=None)
    parser.add_argument("--hard-close-distance-ratio", type=float, default=None)
    parser.add_argument("--hard-close-mobile-weight", type=float, default=None)
    parser.add_argument("--hard-close-fence-weight", type=float, default=None)
    parser.add_argument("--fence-close-weight", type=float, default=None)
    parser.add_argument("--fence-far-weight", type=float, default=None)
    parser.add_argument("--merge-hazard-weight", type=float, default=None)
    parser.add_argument("--interface-edge-loss-weight", type=float, default=None)
    parser.add_argument("--interface-edge-stretch-weight", type=float, default=None)
    parser.add_argument("--disable-phase-reward-schedule", action="store_true")
    parser.add_argument("--enable-final-hole-compression-schedule", action="store_true")
    parser.add_argument("--enable-policy-probes", action="store_true")
    parser.add_argument("--policy-probe-freq", type=int, default=2000)
    parser.add_argument("--policy-probe-seeds", type=str, default="5000,5001,5002")
    parser.add_argument("--policy-probe-max-steps", type=int, default=200)
    parser.add_argument("--disable-initial-policy-probe", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser


def _coerce_batch_size(n_steps: int, requested_batch_size: int) -> int:
    n_steps = max(2, int(n_steps))
    requested_batch_size = max(2, int(requested_batch_size))
    cap = min(n_steps, requested_batch_size)
    for candidate in range(cap, 1, -1):
        if n_steps % candidate == 0:
            return candidate
    return 2


def _resolve_device(device_name: str):
    import torch

    requested = str(device_name).strip().lower()
    if requested == "auto":
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_built():
            raise SystemExit("PyTorch in this environment was not built with MPS support.")
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested, but Apple Metal is not available in this environment.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested, but no CUDA device is available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise SystemExit(f"Unsupported device '{device_name}'. Use one of: auto, mps, cuda, cpu.")


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    try:
        import torch
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "stable-baselines3 is required for training. Install with: pip install stable-baselines3"
        ) from exc

    from rl_eval_tqdm_callback import TqdmEvalCallback
    from rl_attention_logging import AttentionLogConfig, create_attention_logging_callback
    from rl_env import PhaseRewardMultipliers, PhaseRewardSchedule
    from rl_policy_probe import PolicyProbeConfig, create_policy_probe_callback
    from rl_progress_ppo import ProgressPPO
    from rl_boundary_cycle_policy import BoundaryCycleActorCriticPolicy, BoundaryCycleStructuredExtractor
    from rl_gat_baseline import BaselineGraphAttentionExtractor
    from rl_gat_policy import DartAwareActorCriticPolicy, GraphAttentionExtractor
    from rl_homological_gat_policy import (
        HomologicalGATActorCriticPolicy,
        HomologicalGraphAttentionExtractor,
        HomologicalTokenStructuredExtractor,
    )
    from rl_structured_velocity_policy import (
        StructuredVelocityActorCriticPolicy,
        StructuredVelocityCycleExtractor,
        StructuredVelocityGraphExtractor,
    )
    from rl_unit_square_experiment import UnitSquareRLConfig, make_eval_env, make_training_env, make_unit_square_env

    resolved_device = _resolve_device(args.device)
    print(
        "Torch device selection:",
        {
            "requested": args.device,
            "resolved": str(resolved_device),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
            "cuda_available": bool(torch.cuda.is_available()),
        },
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_unitsquare_{timestamp}"
    run_dir = Path(args.outdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    tb_dir = run_dir / "tensorboard"
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    config = UnitSquareRLConfig()
    rc = config.reward_config
    config = replace(
        config,
        num_mobile_sensors=(
            config.num_mobile_sensors
            if args.num_mobile_sensors is None
            else int(args.num_mobile_sensors)
        ),
        max_speed_scale=(
            config.max_speed_scale
            if args.max_speed_scale is None
            else float(args.max_speed_scale)
        ),
        fence_sensing_radius=(
            config.fence_sensing_radius
            if args.fence_sensing_radius is None
            else float(args.fence_sensing_radius)
        ),
        fence_offset_ratio=(
            config.fence_offset_ratio
            if args.fence_offset_ratio is None
            else float(args.fence_offset_ratio)
        ),
        use_weighted_alpha=bool(args.use_weighted_alpha),
        reward_config=replace(
            rc,
            true_cycle_closed_reward=(
                rc.true_cycle_closed_reward
                if args.true_cycle_closed_reward is None
                else float(args.true_cycle_closed_reward)
            ),
            true_cycle_added_penalty=(
                rc.true_cycle_added_penalty
                if args.true_cycle_added_penalty is None
                else float(args.true_cycle_added_penalty)
            ),
            time_penalty=(
                rc.time_penalty
                if args.time_penalty is None
                else float(args.time_penalty)
            ),
            control_effort_penalty=(
                rc.control_effort_penalty
                if args.control_effort_penalty is None
                else float(args.control_effort_penalty)
            ),
            clear_bonus=(
                rc.clear_bonus
                if args.clear_bonus is None
                else float(args.clear_bonus)
            ),
            success_time_bonus_weight=(
                rc.success_time_bonus_weight
                if args.success_time_bonus_weight is None
                else float(args.success_time_bonus_weight)
            ),
            one_hole_linger_penalty_weight=(
                rc.one_hole_linger_penalty_weight
                if args.one_hole_linger_weight is None
                else float(args.one_hole_linger_weight)
            ),
            one_hole_area_scale_alpha=(
                rc.one_hole_area_scale_alpha
                if args.one_hole_area_scale_alpha is None
                else float(args.one_hole_area_scale_alpha)
            ),
            one_hole_perimeter_scale_alpha=(
                rc.one_hole_perimeter_scale_alpha
                if args.one_hole_perimeter_scale_alpha is None
                else float(args.one_hole_perimeter_scale_alpha)
            ),
            area_progress_reward_weight=(
                rc.area_progress_reward_weight
                if args.area_progress_weight is None
                else float(args.area_progress_weight)
            ),
            perimeter_progress_reward_weight=(
                rc.perimeter_progress_reward_weight
                if args.perimeter_progress_weight is None
                else float(args.perimeter_progress_weight)
            ),
            largest_area_progress_reward_weight=(
                rc.largest_area_progress_reward_weight
                if args.largest_area_progress_weight is None
                else float(args.largest_area_progress_weight)
            ),
            largest_perimeter_progress_reward_weight=(
                rc.largest_perimeter_progress_reward_weight
                if args.largest_perimeter_progress_weight is None
                else float(args.largest_perimeter_progress_weight)
            ),
            area_regress_penalty_weight=(
                rc.area_regress_penalty_weight
                if args.area_regress_weight is None
                else float(args.area_regress_weight)
            ),
            perimeter_regress_penalty_weight=(
                rc.perimeter_regress_penalty_weight
                if args.perimeter_regress_weight is None
                else float(args.perimeter_regress_weight)
            ),
            largest_area_regress_penalty_weight=(
                rc.largest_area_regress_penalty_weight
                if args.largest_area_regress_weight is None
                else float(args.largest_area_regress_weight)
            ),
            largest_perimeter_regress_penalty_weight=(
                rc.largest_perimeter_regress_penalty_weight
                if args.largest_perimeter_regress_weight is None
                else float(args.largest_perimeter_regress_weight)
            ),
            area_residual_penalty_weight=(
                rc.area_residual_penalty_weight
                if args.area_residual_weight is None
                else float(args.area_residual_weight)
            ),
            perimeter_residual_penalty_weight=(
                rc.perimeter_residual_penalty_weight
                if args.perimeter_residual_weight is None
                else float(args.perimeter_residual_weight)
            ),
            largest_area_residual_penalty_weight=(
                rc.largest_area_residual_penalty_weight
                if args.largest_area_residual_weight is None
                else float(args.largest_area_residual_weight)
            ),
            largest_perimeter_residual_penalty_weight=(
                rc.largest_perimeter_residual_penalty_weight
                if args.largest_perimeter_residual_weight is None
                else float(args.largest_perimeter_residual_weight)
            ),
            neighbor_min_distance_ratio=(
                rc.neighbor_min_distance_ratio
                if args.neighbor_min_ratio is None
                else float(args.neighbor_min_ratio)
            ),
            neighbor_max_distance_ratio=(
                rc.neighbor_max_distance_ratio
                if args.neighbor_max_ratio is None
                else float(args.neighbor_max_ratio)
            ),
            neighbor_close_penalty_weight=(
                rc.neighbor_close_penalty_weight
                if args.neighbor_close_weight is None
                else float(args.neighbor_close_weight)
            ),
            neighbor_far_penalty_weight=(
                rc.neighbor_far_penalty_weight
                if args.neighbor_far_weight is None
                else float(args.neighbor_far_weight)
            ),
            mobile_overlap_penalty_weight=(
                rc.mobile_overlap_penalty_weight
                if args.mobile_overlap_weight is None
                else float(args.mobile_overlap_weight)
            ),
            hard_close_distance_ratio=(
                rc.hard_close_distance_ratio
                if args.hard_close_distance_ratio is None
                else float(args.hard_close_distance_ratio)
            ),
            hard_close_mobile_penalty_weight=(
                rc.hard_close_mobile_penalty_weight
                if args.hard_close_mobile_weight is None
                else float(args.hard_close_mobile_weight)
            ),
            hard_close_fence_penalty_weight=(
                rc.hard_close_fence_penalty_weight
                if args.hard_close_fence_weight is None
                else float(args.hard_close_fence_weight)
            ),
            fence_close_penalty_weight=(
                rc.fence_close_penalty_weight
                if args.fence_close_weight is None
                else float(args.fence_close_weight)
            ),
            fence_far_penalty_weight=(
                rc.fence_far_penalty_weight
                if args.fence_far_weight is None
                else float(args.fence_far_weight)
            ),
            merge_hazard_penalty_weight=(
                rc.merge_hazard_penalty_weight
                if args.merge_hazard_weight is None
                else float(args.merge_hazard_weight)
            ),
            interface_edge_loss_penalty_weight=(
                rc.interface_edge_loss_penalty_weight
                if args.interface_edge_loss_weight is None
                else float(args.interface_edge_loss_weight)
            ),
            interface_edge_stretch_penalty_weight=(
                rc.interface_edge_stretch_penalty_weight
                if args.interface_edge_stretch_weight is None
                else float(args.interface_edge_stretch_weight)
            ),
        ),
    )
    if args.enable_final_hole_compression_schedule:
        config = replace(
            config,
            phase_reward_schedule=PhaseRewardSchedule(
                simplify=PhaseRewardMultipliers(
                    merge_hazard_penalty_weight=0.0,
                    area_progress_reward_weight=0.2,
                    perimeter_progress_reward_weight=0.0,
                    largest_area_progress_reward_weight=0.0,
                    largest_perimeter_progress_reward_weight=0.0,
                    area_regress_penalty_weight=0.2,
                    perimeter_regress_penalty_weight=0.0,
                    largest_area_regress_penalty_weight=0.0,
                    largest_perimeter_regress_penalty_weight=0.0,
                    area_residual_penalty_weight=0.1,
                    perimeter_residual_penalty_weight=0.0,
                    largest_area_residual_penalty_weight=0.0,
                    largest_perimeter_residual_penalty_weight=0.0,
                ),
                consolidate=PhaseRewardMultipliers(
                    merge_hazard_penalty_weight=0.0,
                    area_progress_reward_weight=0.35,
                    perimeter_progress_reward_weight=0.0,
                    largest_area_progress_reward_weight=0.0,
                    largest_perimeter_progress_reward_weight=0.0,
                    area_regress_penalty_weight=0.35,
                    perimeter_regress_penalty_weight=0.0,
                    largest_area_regress_penalty_weight=0.0,
                    largest_perimeter_regress_penalty_weight=0.0,
                    area_residual_penalty_weight=0.15,
                    perimeter_residual_penalty_weight=0.0,
                    largest_area_residual_penalty_weight=0.0,
                    largest_perimeter_residual_penalty_weight=0.0,
                ),
                compress=PhaseRewardMultipliers(
                    true_cycle_added_penalty=0.0,
                    merge_hazard_penalty_weight=1.0,
                ),
            ),
        )
    elif args.disable_phase_reward_schedule:
        config = replace(config, phase_reward_schedule=PhaseRewardSchedule())

    if args.model_kind in {"boundary_cycle", "homological_gat", "structured_velocity"}:
        config = replace(config, state_mode="cycle_graph")

    attention_config = AttentionLogConfig(
        enabled=not args.disable_attention_log,
        topk=max(1, args.attention_topk),
        every_n_steps_train=max(1, args.attention_log_every),
        every_n_steps_eval=config.attention_log_config.every_n_steps_eval,
        log_full_attention=bool(args.attention_log_full),
        path=args.attention_log_path or str(run_dir / "attention_train.jsonl"),
    )

    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        payload = config.as_serializable()
        payload["attention_log_config"] = {
            "enabled": attention_config.enabled,
            "topk": attention_config.topk,
            "every_n_steps_train": attention_config.every_n_steps_train,
            "every_n_steps_eval": attention_config.every_n_steps_eval,
            "log_full_attention": attention_config.log_full_attention,
            "path": attention_config.path,
        }
        payload["model_kind"] = str(args.model_kind)
        json.dump(payload, handle, indent=2, sort_keys=True)

    train_env = Monitor(
        make_training_env(
            config,
            event_log_path=(str(run_dir / "train_events.jsonl") if args.enable_event_log else None),
            enable_event_logging=bool(args.enable_event_log),
        )
    )
    eval_env = Monitor(
        make_eval_env(
            config,
            event_log_path=(str(run_dir / "eval_events_during_train.jsonl") if args.enable_event_log else None),
            enable_event_logging=bool(args.enable_event_log),
            max_steps_override=(None if int(args.eval_max_steps) <= 0 else int(args.eval_max_steps)),
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(models_dir),
        name_prefix="ppo_unitsquare",
    )
    eval_callback = TqdmEvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=max(1, args.eval_freq),
        n_eval_episodes=max(1, args.eval_episodes),
        deterministic=True,
        render=False,
    )
    attention_callback = create_attention_logging_callback(attention_config)
    probe_seeds = tuple(
        int(token.strip())
        for token in args.policy_probe_seeds.split(",")
        if token.strip()
    )
    probe_callback = create_policy_probe_callback(
        PolicyProbeConfig(
            enabled=bool(args.enable_policy_probes),
            freq=max(1, int(args.policy_probe_freq)),
            seeds=probe_seeds or tuple(config.eval_seeds[:3]),
            max_steps=max(1, int(args.policy_probe_max_steps)),
            outdir=str(run_dir / "policy_probes"),
            run_initial_probe=not bool(args.disable_initial_policy_probe),
        ),
        env_builder_factory=lambda seeds: make_unit_square_env(
            config,
            seeds=seeds,
            event_log_path=None,
            enable_event_logging=False,
        ),
    )
    n_steps = max(2, args.n_steps)
    batch_size = _coerce_batch_size(n_steps, args.batch_size)
    print(f"PPO rollout config: n_steps={n_steps}, batch_size={batch_size}")

    if args.model_kind == "baseline_gat":
        policy = "MultiInputPolicy"
        policy_kwargs = {
            "features_extractor_class": BaselineGraphAttentionExtractor,
            "features_extractor_kwargs": {
                "hidden_dim": 128,
                "num_layers": 3,
                "num_heads": 2,
                "dropout": 0.0,
            },
            "net_arch": {
                "pi": [256, 128],
                "vf": [256, 128],
            },
        }
    elif args.model_kind == "dart_gat":
        policy = DartAwareActorCriticPolicy
        policy_kwargs = {
            "features_extractor_class": GraphAttentionExtractor,
            "features_extractor_kwargs": {
                "hidden_dim": 128,
                "num_layers": 3,
                "num_heads": 2,
                "dropout": 0.0,
            },
            "net_arch": {
                "pi": [256, 128],
                "vf": [256, 128],
            },
        }
    elif args.model_kind == "boundary_cycle":
        policy = BoundaryCycleActorCriticPolicy
        policy_kwargs = {
            "features_extractor_class": BoundaryCycleStructuredExtractor,
            "features_extractor_kwargs": {},
            "net_arch": {
                "pi": [256, 128],
                "vf": [256, 128],
            },
        }
    elif args.model_kind == "homological_gat":
        policy = HomologicalGATActorCriticPolicy
        policy_kwargs = {
            "features_extractor_class": HomologicalGraphAttentionExtractor,
            "features_extractor_kwargs": {
                "token_hidden_dim": 16,
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "dropout": 0.0,
            },
            "log_std_init": -1.0,
            "net_arch": {
                "pi": [128, 64],
                "vf": [128, 64],
            },
        }
    elif args.model_kind == "structured_velocity":
        policy = StructuredVelocityActorCriticPolicy
        policy_kwargs = {
            "features_extractor_class": StructuredVelocityGraphExtractor,
            "features_extractor_kwargs": {
                "cycle_width": 8,
                "graph_hidden_dim": 32,
                "num_layers": 2,
                "num_heads": 2,
                "dropout": 0.0,
            },
            "log_std_init": -1.0,
            "net_arch": {
                "pi": [16],
                "vf": [16],
            },
        }
    else:
        raise SystemExit(f"Unsupported model kind: {args.model_kind}")

    print(f"Model kind: {args.model_kind}")

    model = ProgressPPO(
        policy=policy,
        env=train_env,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(tb_dir),
        device=resolved_device,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=float(args.learning_rate),
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        show_epoch_progress=False,
    )

    init_path = models_dir / "ppo_unitsquare_init"
    model.save(str(init_path))

    model.learn(
        total_timesteps=max(1, args.total_timesteps),
        callback=[checkpoint_callback, eval_callback, attention_callback, probe_callback],
        tb_log_name=run_name,
        progress_bar=bool(args.progress_bar),
    )

    final_path = models_dir / "ppo_unitsquare_final"
    model.save(str(final_path))
    best_path = models_dir / "best" / "best_model.zip"

    print(f"Training complete. Run dir: {run_dir}")
    print(f"Initial model: {init_path}.zip")
    print(f"Final model: {final_path}.zip")
    print(f"Best eval model: {best_path if best_path.exists() else 'not produced'}")


if __name__ == "__main__":
    main()
