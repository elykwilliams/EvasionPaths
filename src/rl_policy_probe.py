from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PHASE_COLORS = {
    "simplify": "#f4a261",
    "consolidate": "#2a9d8f",
    "compress": "#457b9d",
}


@dataclass(frozen=True)
class PolicyProbeConfig:
    enabled: bool = False
    freq: int = 2000
    seeds: Sequence[int] = (5000, 5001, 5002)
    max_steps: int = 200
    outdir: str = ""
    run_initial_probe: bool = True


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _phase_segments(rows: List[Dict]) -> List[tuple[float, float, str]]:
    if not rows:
        return []
    segments: List[tuple[float, float, str]] = []
    times = [_safe_float(r.get("time", 0.0)) for r in rows]
    phases = [str(r.get("phase", "simplify")) for r in rows]
    start_idx = 0
    for idx in range(1, len(rows)):
        if phases[idx] != phases[start_idx]:
            segments.append((times[start_idx], times[idx], phases[start_idx]))
            start_idx = idx
    segments.append((times[start_idx], times[-1], phases[start_idx]))
    return segments


def _plot_timeseries(
    x: np.ndarray,
    series: Dict[str, np.ndarray],
    title: str,
    ylabel: str,
    path: Path,
    phase_segments: List[tuple[float, float, str]],
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ymin = None
    ymax = None
    for name, values in series.items():
        ax.plot(x, values, lw=1.8, label=name)
        cur_min = float(np.min(values)) if values.size else 0.0
        cur_max = float(np.max(values)) if values.size else 0.0
        ymin = cur_min if ymin is None else min(ymin, cur_min)
        ymax = cur_max if ymax is None else max(ymax, cur_max)
    if ymin is None or ymax is None:
        ymin, ymax = 0.0, 1.0
    yrange = max(1e-9, ymax - ymin)
    for left, right, phase in phase_segments:
        color = PHASE_COLORS.get(phase, "#cccccc")
        ax.axvspan(left, right, color=color, alpha=0.08, zorder=0)
    # top strip to make phase changes obvious even when lines overlap
    strip_bottom = ymax + 0.02 * yrange
    strip_height = 0.04 * yrange
    for left, right, phase in phase_segments:
        color = PHASE_COLORS.get(phase, "#cccccc")
        ax.axvspan(left, right, ymin=1.0 - 0.05, ymax=1.0, color=color, alpha=0.8, zorder=3)
    ax.set_ylim(ymin - 0.02 * yrange, ymax + 0.08 * yrange)
    ax.set_title(title)
    ax.set_xlabel("sim time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_probe_plots(seed_dir: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    x = np.asarray([_safe_float(r.get("time", 0.0)) for r in rows], dtype=float)
    phase_segments = _phase_segments(rows)

    reward_keys = [
        "reward_total",
        "true_cycles_closed",
        "true_cycles_added",
        "merge_hazard_count",
        "interface_edge_loss",
        "interface_edge_stretch",
        "elapsed",
        "effort",
        "neighbor_close_violation",
        "neighbor_far_violation",
        "hard_close_mobile_violation",
        "fence_close_violation",
        "fence_far_violation",
        "hard_close_fence_violation",
        "area_progress_norm",
        "perimeter_progress_norm",
        "largest_area_progress_norm",
        "largest_perimeter_progress_norm",
        "area_regress_norm",
        "perimeter_regress_norm",
        "largest_area_regress_norm",
        "largest_perimeter_regress_norm",
        "area_residual_norm",
        "perimeter_residual_norm",
        "largest_area_residual_norm",
        "largest_perimeter_residual_norm",
        "clear_indicator",
        "timeout_indicator",
    ]
    reward_series = {
        key: np.asarray([_safe_float(r.get("reward_terms", {}).get(key, r.get(key, 0.0))) for r in rows], dtype=float)
        for key in reward_keys
    }
    _plot_timeseries(
        x,
        reward_series,
        "Reward Decomposition",
        "reward contribution",
        seed_dir / "reward_terms.png",
        phase_segments,
    )

    topo_series = {
        "true_cycle_count": np.asarray([_safe_float(r.get("true_cycle_count", 0.0)) for r in rows], dtype=float),
        "true_cycle_area": np.asarray([_safe_float(r.get("true_cycle_area", 0.0)) for r in rows], dtype=float),
        "largest_true_cycle_area": np.asarray([_safe_float(r.get("largest_true_cycle_area", 0.0)) for r in rows], dtype=float),
        "true_cycle_perimeter": np.asarray([_safe_float(r.get("true_cycle_perimeter", 0.0)) for r in rows], dtype=float),
        "largest_true_cycle_perimeter": np.asarray([_safe_float(r.get("largest_true_cycle_perimeter", 0.0)) for r in rows], dtype=float),
    }
    _plot_timeseries(
        x,
        topo_series,
        "Topology Metrics",
        "metric value",
        seed_dir / "topology_metrics.png",
        phase_segments,
    )

    action_series = {
        "mean_action_norm": np.asarray([_safe_float(r.get("mean_action_norm", 0.0)) for r in rows], dtype=float),
        "max_action_norm": np.asarray([_safe_float(r.get("max_action_norm", 0.0)) for r in rows], dtype=float),
        "neighbor_nearest_mean_distance": np.asarray([_safe_float(r.get("neighbor_nearest_mean_distance", 0.0)) for r in rows], dtype=float),
        "fence_nearest_mean_distance": np.asarray([_safe_float(r.get("fence_nearest_mean_distance", 0.0)) for r in rows], dtype=float),
    }
    _plot_timeseries(
        x,
        action_series,
        "Action and Spacing Metrics",
        "metric value",
        seed_dir / "action_metrics.png",
        phase_segments,
    )


def create_policy_probe_callback(
    config: PolicyProbeConfig,
    *,
    env_builder_factory,
):
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for policy probes.") from exc

    class _PolicyProbeCallback(BaseCallback):
        def __init__(self, cfg: PolicyProbeConfig):
            super().__init__()
            self.cfg = cfg
            self._last_probe_step = None
            self._outdir = Path(cfg.outdir)

        def _run_probe(self, step_tag: int) -> None:
            step_dir = self._outdir / f"step_{step_tag:06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            manifest = {"step": int(step_tag), "seeds": []}

            for seed in self.cfg.seeds:
                env = env_builder_factory((int(seed),))
                obs, _ = env.reset(seed=int(seed))
                done = False
                truncated = False
                episode_rows: List[Dict] = []
                step_count = 0
                ep_return = 0.0

                while not (done or truncated):
                    if self.cfg.max_steps > 0 and step_count >= int(self.cfg.max_steps):
                        break

                    action, _ = self.model.predict(obs, deterministic=True)
                    action_arr = np.asarray(action, dtype=float)
                    action_norms = np.linalg.norm(action_arr, axis=1) if action_arr.size else np.asarray([], dtype=float)
                    obs, reward, done, truncated, info = env.step(action)
                    ep_return += float(reward)
                    episode_rows.append(
                        {
                            "step": int(step_count),
                            "time": _safe_float(info.get("time", 0.0)),
                            "phase": str(info.get("phase", "simplify")),
                            "phase_one_hot": list(info.get("phase_one_hot", [1.0, 0.0, 0.0])),
                            "elapsed": _safe_float(info.get("elapsed", 0.0)),
                            "reward_total": float(reward),
                            "true_cycle_count": _safe_float(info.get("true_cycle_count", 0.0)),
                            "true_cycle_area": _safe_float(info.get("true_cycle_area", 0.0)),
                            "true_cycle_perimeter": _safe_float(info.get("true_cycle_perimeter", 0.0)),
                            "largest_true_cycle_area": _safe_float(info.get("largest_true_cycle_area", 0.0)),
                            "largest_true_cycle_perimeter": _safe_float(info.get("largest_true_cycle_perimeter", 0.0)),
                            "neighbor_nearest_mean_distance": _safe_float(info.get("neighbor_nearest_mean_distance", 0.0)),
                            "fence_nearest_mean_distance": _safe_float(info.get("fence_nearest_mean_distance", 0.0)),
                            "mean_action_norm": float(np.mean(action_norms)) if action_norms.size else 0.0,
                            "max_action_norm": float(np.max(action_norms)) if action_norms.size else 0.0,
                            "reward_terms": dict(info.get("reward_terms", {})),
                        }
                    )
                    step_count += 1

                seed_dir = step_dir / f"seed_{int(seed)}"
                seed_dir.mkdir(parents=True, exist_ok=True)
                (seed_dir / "timeseries.json").write_text(json.dumps(episode_rows, indent=2), encoding="utf-8")
                summary = {
                    "seed": int(seed),
                    "steps": int(step_count),
                    "return": float(ep_return),
                    "done": bool(done),
                    "truncated": bool(truncated),
                    "final_time": _safe_float(episode_rows[-1]["time"], 0.0) if episode_rows else 0.0,
                    "final_true_cycle_count": _safe_float(episode_rows[-1]["true_cycle_count"], 0.0) if episode_rows else 0.0,
                    "final_phase": str(episode_rows[-1]["phase"]) if episode_rows else "simplify",
                    "phase_transitions": [
                        {
                            "step": int(row["step"]),
                            "time": _safe_float(row["time"], 0.0),
                            "phase": str(row["phase"]),
                        }
                        for idx, row in enumerate(episode_rows)
                        if idx == 0 or row["phase"] != episode_rows[idx - 1]["phase"]
                    ],
                }
                (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                _save_probe_plots(seed_dir, episode_rows)
                manifest["seeds"].append(summary)
                env.close()

            (step_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        def _on_training_start(self) -> None:
            if not self.cfg.enabled:
                return
            self._outdir.mkdir(parents=True, exist_ok=True)
            if self.cfg.run_initial_probe:
                self._run_probe(0)
                self._last_probe_step = 0

        def _on_step(self) -> bool:
            if not self.cfg.enabled:
                return True
            if self.cfg.freq <= 0:
                return True
            if self.num_timesteps % int(self.cfg.freq) != 0:
                return True
            if self._last_probe_step == int(self.num_timesteps):
                return True
            self._run_probe(int(self.num_timesteps))
            self._last_probe_step = int(self.num_timesteps)
            return True

    return _PolicyProbeCallback(config)
