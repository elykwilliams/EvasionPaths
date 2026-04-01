#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Point this to the training run directory name under experiments/output/rl_unit_square.
RUN_NAME="${RUN_NAME:-50k_baseline_final_hole_mild}"
CHECKPOINT="${CHECKPOINT:-best}"   # best|final|init|step_2000
NUM_SIMS="${NUM_SIMS:-1}"
NUM_SAVE="${NUM_SAVE:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
PRIMARY_RUN_DIR="${REPO_ROOT}/experiments/output/rl_unit_square/${RUN_NAME}"
LEGACY_RUN_DIR="${REPO_ROOT}/scripts/experiments/output/rl_unit_square/${RUN_NAME}"
if [[ -d "${PRIMARY_RUN_DIR}" ]]; then
  RUN_DIR="${PRIMARY_RUN_DIR}"
elif [[ -d "${LEGACY_RUN_DIR}" ]]; then
  RUN_DIR="${LEGACY_RUN_DIR}"
  echo "Using legacy run dir: ${RUN_DIR}"
else
  echo "Run dir not found in either location:"
  echo "  ${PRIMARY_RUN_DIR}"
  echo "  ${LEGACY_RUN_DIR}"
  exit 1
fi

python "${REPO_ROOT}/experiments/rl_reeb_from_model.py" \
  --run-dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --num-sims "${NUM_SIMS}" \
  --num-save "${NUM_SAVE}" \
  --max-steps "${MAX_STEPS}" \
  --open
