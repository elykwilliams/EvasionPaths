#!/usr/bin/env python3
"""Publish selected experiment simulations into the static web viewer catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_SITE_ROOT = Path.home() / "projects" / "evasion-paths-experiments"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from web_export import publish_experiment_bundle, rebuild_catalog_from_manifests


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish selected simulations into web/ for static hosting.")
    parser.add_argument("--source-dir", type=str, required=True, help="Artifact directory containing sim_* bundles.")
    parser.add_argument("--category", type=str, required=True, help="Experiment category, e.g. rl or homological.")
    parser.add_argument(
        "--motion-model",
        type=str,
        required=True,
        help="Motion-model family label shown in the catalog, e.g. RL or Homological Motion.",
    )
    parser.add_argument(
        "--display-name",
        type=str,
        required=True,
        help="Experiment display name, e.g. structured_velocity_weighted_fence025_offset06_10k_phasearea.",
    )
    parser.add_argument("--sim-indices", type=str, default="", help="Comma-separated source sim indices to publish.")
    parser.add_argument("--site-root", type=str, default=str(DEFAULT_SITE_ROOT))
    parser.add_argument("--experiment-id", type=str, default="", help="Optional stable catalog id override.")
    parser.add_argument("--source-run", type=str, default="", help="Optional source run label override.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint label override.")
    return parser


def _parse_sim_indices(raw: str) -> list[int] | None:
    text = raw.strip()
    if not text:
        return None
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    args = _make_parser().parse_args()
    published = publish_experiment_bundle(
        source_dir=Path(args.source_dir),
        web_root=Path(args.site_root),
        category=args.category,
        motion_model=args.motion_model,
        display_name=args.display_name,
        selected_indices=_parse_sim_indices(args.sim_indices),
        experiment_id=(args.experiment_id.strip() or None),
        source_run=args.source_run.strip(),
        checkpoint=args.checkpoint.strip(),
    )
    rebuild_catalog_from_manifests(Path(args.site_root))
    print(f"Published hosted experiment bundle to: {published}")


if __name__ == "__main__":
    main()
