#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Evaluate a specific saved model.
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/experiments/output/rl_unit_square/ppo_uq_10k_lr1e4_vmax02/models/ppo_unitsquare_final.zip}"
EPISODES="${EPISODES:-50}"
RUN_NAME="${RUN_NAME:-eval_uq_final}"

python "${REPO_ROOT}/experiments/rl_eval_unit_square.py" \
  --model "${MODEL_PATH}" \
  --episodes "${EPISODES}" \
  --run-name "${RUN_NAME}"
