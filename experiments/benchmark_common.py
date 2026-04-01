#!/usr/bin/env python3
"""Shared helpers for motion-model searches and grid benchmarks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from boundary_geometry import BunimovichStadium, CircularDomain, Domain, RectangularDomain
from motion_model import (
    BilliardMotion,
    BrownianMotion,
    HomologicalDynamicsMotion,
    SequentialHomologicalMotion,
    Viscek,
)
from sensor_network import Sensor, generate_fence_sensors, generate_mobile_sensors
from topology import generate_topology


MODEL_DISPLAY = {
    "billiard": "Billiard",
    "brownian_low": "Brownian (Low)",
    "brownian_med": "Brownian (Medium)",
    "brownian_high": "Brownian (High)",
    "vicsek_low": "Vicsek (Low)",
    "vicsek_med": "Vicsek (Medium)",
    "vicsek_high": "Vicsek (High)",
    "homological": "Homological Motion",
    "sequential_homological": "Sequential Homological Motion",
}

DOMAIN_DISPLAY = {
    "square": "Unit Square",
    "circle": "Area-Matched Circle",
    "rectangle_2to1_area1": "Area-Matched Rectangle (2:1)",
    "stadium_w0p6": "Area-Matched Stadium ($w/r=0.6$)",
    "stadium_w1p2": "Area-Matched Stadium ($w/r=1.2$)",
}


@dataclass
class InitialCondition:
    fence_positions: np.ndarray
    mobile_positions: np.ndarray
    mobile_velocities: np.ndarray


@dataclass
class ConnectedInitResult:
    initial_condition: InitialCondition | None
    seed_used: int | None
    retries_used: int
    feasible: bool


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


def parse_csv_strs(raw: str) -> List[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("Expected at least one string value.")
    return values


def combo_key(n_sensors: int, radius: float) -> str:
    return f"n={int(n_sensors)},r={float(radius):.6f}"


def score_combo(combo_metrics: Dict[str, float], failure_penalty: float, worst_case_weight: float) -> float:
    return (
        float(combo_metrics["mean_tau"])
        + failure_penalty * float(combo_metrics["failure_rate"])
        + worst_case_weight * float(combo_metrics["worst_tau"])
    )


def domain_seed(domain_name: str) -> int:
    value = 0
    for idx, ch in enumerate(domain_name):
        value = (value + (idx + 1) * ord(ch)) % (2**32 - 1)
    return int(value)


def replicate_seed(base_seed: int, domain_name: str, n_sensors: int, radius: float, replicate_idx: int) -> int:
    r_int = int(round(float(radius) * 1_000_000))
    d_int = domain_seed(domain_name)
    seed = (
        base_seed * 1_315_423_911
        + d_int * 97_919
        + n_sensors * 2_654_435_761
        + r_int * 97_531
        + replicate_idx * 104_729
    ) % (2**32 - 1)
    return int(seed)


def stadium_parameters(aspect_ratio: float) -> Tuple[float, float]:
    radius = 1.0 / math.sqrt(math.pi + 4.0 * float(aspect_ratio))
    width = float(aspect_ratio) * radius
    return width, radius


def build_domain(domain_name: str) -> Domain:
    if domain_name == "square":
        return RectangularDomain()
    if domain_name == "circle":
        return CircularDomain(radius=1.0 / math.sqrt(math.pi))
    if domain_name == "rectangle_2to1_area1":
        width = math.sqrt(2.0)
        height = 1.0 / math.sqrt(2.0)
        return RectangularDomain(x_min=0.0, x_max=width, y_min=0.0, y_max=height)
    if domain_name == "stadium_w0p6":
        width, radius = stadium_parameters(0.6)
        return BunimovichStadium(w=width, r=radius, L=2.0 * (width + radius))
    if domain_name == "stadium_w1p2":
        width, radius = stadium_parameters(1.2)
        return BunimovichStadium(w=width, r=radius, L=2.0 * (width + radius))
    raise ValueError(f"Unsupported domain name: {domain_name}")


def domain_metadata(domain_name: str) -> Dict[str, float | str]:
    if domain_name == "square":
        return {
            "domain": domain_name,
            "display_name": DOMAIN_DISPLAY[domain_name],
            "area": 1.0,
            "inradius": 0.5,
        }
    if domain_name == "circle":
        radius = 1.0 / math.sqrt(math.pi)
        return {
            "domain": domain_name,
            "display_name": DOMAIN_DISPLAY[domain_name],
            "radius": radius,
            "area": 1.0,
            "inradius": radius,
        }
    if domain_name == "rectangle_2to1_area1":
        width = math.sqrt(2.0)
        height = 1.0 / math.sqrt(2.0)
        return {
            "domain": domain_name,
            "display_name": DOMAIN_DISPLAY[domain_name],
            "width": width,
            "height": height,
            "aspect_ratio": 2.0,
            "area": 1.0,
            "inradius": 0.5 * height,
        }
    if domain_name == "stadium_w0p6":
        width, radius = stadium_parameters(0.6)
        return {
            "domain": domain_name,
            "display_name": DOMAIN_DISPLAY[domain_name],
            "width": width,
            "radius": radius,
            "aspect_ratio_wr": 0.6,
            "area": 1.0,
        }
    if domain_name == "stadium_w1p2":
        width, radius = stadium_parameters(1.2)
        return {
            "domain": domain_name,
            "display_name": DOMAIN_DISPLAY[domain_name],
            "width": width,
            "radius": radius,
            "aspect_ratio_wr": 1.2,
            "area": 1.0,
        }
    raise ValueError(f"Unsupported domain name: {domain_name}")


def domain_area(domain_name: str) -> float:
    return float(domain_metadata(domain_name)["area"])


def hex_spacing_scale(area: float, n_sensors: int) -> float:
    return math.sqrt((2.0 * float(area)) / (math.sqrt(3.0) * float(n_sensors)))


def brownian_length_scale(area: float, n_sensors: int, radius: float) -> float:
    return min(float(radius), hex_spacing_scale(float(area), int(n_sensors)))


def brownian_sigma_for_level(
    *,
    level: str,
    area: float,
    n_sensors: int,
    radius: float,
    dt: float,
) -> float:
    level_to_fraction = {
        "low": 0.10,
        "med": 0.20,
        "high": 0.35,
    }
    if level not in level_to_fraction:
        raise ValueError(f"Unsupported Brownian level: {level}")
    ell_b = brownian_length_scale(float(area), int(n_sensors), float(radius))
    return float(level_to_fraction[level] * ell_b / math.sqrt(float(dt)))


def vicsek_occupancy(area: float, n_sensors: int, radius: float) -> float:
    return float(n_sensors) * math.pi * float(radius) ** 2 / float(area)


def vicsek_noise_for_level(
    *,
    level: str,
    area: float,
    n_sensors: int,
    radius: float,
) -> float:
    m = vicsek_occupancy(float(area), int(n_sensors), float(radius))
    scale = 1.0 + math.sqrt(max(m, 0.0))
    if level == "low":
        return float((math.pi / 48.0) * scale)
    if level == "med":
        return float((math.pi / 24.0) * scale)
    if level == "high":
        return float(min(math.pi / 4.0, (math.pi / 12.0) * scale))
    raise ValueError(f"Unsupported Vicsek level: {level}")


def generate_initial_condition(
    *,
    domain_name: str,
    domain: Domain,
    n_sensors: int,
    radius: float,
    sensor_velocity: float,
    seed: int,
) -> InitialCondition:
    np.random.seed(seed)
    fence = generate_fence_sensors(domain, radius)
    mobile = generate_mobile_sensors(domain, n_sensors, radius, sensor_velocity)
    return InitialCondition(
        fence_positions=np.asarray([np.array(sensor.pos, dtype=float) for sensor in fence], dtype=float),
        mobile_positions=np.asarray([np.array(sensor.pos, dtype=float) for sensor in mobile], dtype=float),
        mobile_velocities=np.asarray([np.array(sensor.vel, dtype=float) for sensor in mobile], dtype=float),
    )


def _is_face_connected_initial_condition(
    *,
    init: InitialCondition,
    radius: float,
    domain: Domain,
) -> bool:
    fence_positions = np.asarray(init.fence_positions, dtype=float)
    mobile_positions = np.asarray(init.mobile_positions, dtype=float)
    points = np.vstack([fence_positions, mobile_positions])
    interior_point = None
    if fence_positions.size:
        interior_point = np.mean(fence_positions, axis=0)
    topology = generate_topology(
        points,
        radius,
        fence_node_count=int(len(fence_positions)),
        interior_point=interior_point,
    )
    return bool(topology.is_face_connected())


def generate_connected_initial_condition(
    *,
    domain_name: str,
    domain: Domain,
    n_sensors: int,
    radius: float,
    sensor_velocity: float,
    seed: int,
    max_retries: int = 200,
) -> ConnectedInitResult:
    for retry in range(max_retries + 1):
        retry_seed = int((seed + retry * 104_729) % (2**32 - 1))
        init = generate_initial_condition(
            domain_name=domain_name,
            domain=domain,
            n_sensors=n_sensors,
            radius=radius,
            sensor_velocity=sensor_velocity,
            seed=retry_seed,
        )
        if _is_face_connected_initial_condition(init=init, radius=radius, domain=domain):
            return ConnectedInitResult(
                initial_condition=init,
                seed_used=retry_seed,
                retries_used=retry,
                feasible=True,
            )
    return ConnectedInitResult(
        initial_condition=None,
        seed_used=None,
        retries_used=max_retries,
        feasible=False,
    )


def clone_sensors(init: InitialCondition, radius: float) -> Tuple[List[Sensor], List[Sensor]]:
    fence = [Sensor(np.array(pos, dtype=float), np.zeros(2, dtype=float), radius, boundary_sensor=True) for pos in init.fence_positions]
    mobile = [
        Sensor(np.array(pos, dtype=float), np.array(vel, dtype=float), radius, boundary_sensor=False)
        for pos, vel in zip(init.mobile_positions, init.mobile_velocities)
    ]
    return fence, mobile


def load_best_params_by_combo(
    trials_json: Path,
    *,
    failure_penalty: float,
    worst_case_weight: float,
) -> Dict[str, Dict]:
    trials = json.loads(trials_json.read_text(encoding="utf-8"))
    best_by_combo: Dict[str, Dict] = {}
    for trial in trials:
        params = dict(trial.get("params", trial.get("weights", {})))
        for key, combo_metrics in trial["by_combo"].items():
            score = score_combo(combo_metrics, failure_penalty, worst_case_weight)
            prev = best_by_combo.get(key)
            if prev is None or score < float(prev["score"]):
                best_by_combo[key] = {"params": params, "score": score, "trial": int(trial.get("trial", -1))}
    return best_by_combo


def build_motion_model(
    model_name: str,
    *,
    domain_name: str,
    n_sensors: int,
    radius: float,
    dt: float,
    sensor_velocity: float,
    tuned_params_by_combo: Dict[str, Dict] | None = None,
):
    tuned_params_by_combo = tuned_params_by_combo or {}
    if model_name == "billiard":
        return BilliardMotion()

    if model_name in {"homological", "sequential_homological"}:
        key = combo_key(n_sensors, radius)
        tuned = tuned_params_by_combo.get(key, {})
        params = tuned.get("params", {})

    area = domain_area(domain_name)

    if model_name.startswith("brownian_"):
        level = model_name.split("_", 1)[1]
        sigma = brownian_sigma_for_level(
            level=level,
            area=area,
            n_sensors=n_sensors,
            radius=radius,
            dt=dt,
        )
        return BrownianMotion(sigma=sigma, large_dt=float(dt))

    if model_name.startswith("vicsek_"):
        level = model_name.split("_", 1)[1]
        noise_scale = vicsek_noise_for_level(
            level=level,
            area=area,
            n_sensors=n_sensors,
            radius=radius,
        )
        return Viscek(large_dt=float(dt), radius=float(radius), noise_scale=noise_scale)

    if model_name == "homological":
        if not params:
            raise KeyError(f"No tuned homological parameters available for {combo_key(n_sensors, radius)}")
        return HomologicalDynamicsMotion(
            sensing_radius=float(radius),
            max_speed=float(params["max_speed"]),
            lambda_shrink=float(params["lambda_shrink"]),
            mu_curvature=float(params["mu_curvature"]),
            eta_cohesion=float(params["eta_cohesion"]),
            repulsion_strength=float(params["repulsion_strength"]),
            repulsion_power=float(params["repulsion_power"]),
            auto_d_safe=bool(params["auto_d_safe"]),
            d_safe_manual=float(params["d_safe_manual"]),
        )

    if model_name == "sequential_homological":
        if not params:
            raise KeyError(
                f"No tuned sequential_homological parameters available for {combo_key(n_sensors, radius)}"
            )
        return SequentialHomologicalMotion(
            sensing_radius=float(radius),
            max_speed=float(params["max_speed"]),
            lambda_shrink=float(params["lambda_shrink"]),
            mu_curvature=float(params["mu_curvature"]),
            eta_cohesion=float(params["eta_cohesion"]),
            repulsion_strength=float(params["repulsion_strength"]),
            repulsion_power=float(params["repulsion_power"]),
            auto_d_safe=bool(params["auto_d_safe"]),
            d_safe_manual=float(params["d_safe_manual"]),
            overlap_threshold=float(params["overlap_threshold"]),
        )

    raise ValueError(f"Unsupported model name: {model_name}")
