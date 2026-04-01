#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Run-level plots from a training run directory.
RUN_NAME="${RUN_NAME:-ppo_uq_10k_lr1e4_vmax02}"
ROLLING_WINDOW="${ROLLING_WINDOW:-50}"

python "${REPO_ROOT}/experiments/rl_plot_training.py" \
  --run-dir "${REPO_ROOT}/experiments/output/rl_unit_square/${RUN_NAME}" \
  --rolling-window "${ROLLING_WINDOW}"
