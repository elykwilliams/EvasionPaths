#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Example run names:
# 50k_baseline_final_hole_mild
# 50klargehole
# 50klargehole_merges
# boundary_cycle_50k
# homological_gat_50k
# homological_gat_joint_actor
RUN_NAME="${RUN_NAME:-homological_gat_50k}"
CHECKPOINT="${CHECKPOINT:-best}"
NUM_SIMS="${NUM_SIMS:-12}"
MAX_STEPS="${MAX_STEPS:-800}"
OUTDIR="${OUTDIR:-}"
OPEN_FLAG="${OPEN_FLAG:---open}"
STOCHASTIC_FLAG="${STOCHASTIC_FLAG:-}"

CMD=(
  python
  "${REPO_ROOT}/experiments/analyze_sims.py"
  --run-dir "${REPO_ROOT}/experiments/output/rl_unit_square/${RUN_NAME}"
  --checkpoint "${CHECKPOINT}"
  --num-sims "${NUM_SIMS}"
  --max-steps "${MAX_STEPS}"
)

if [[ -n "${OUTDIR}" ]]; then
  CMD+=(--outdir "${OUTDIR}")
fi

if [[ -n "${STOCHASTIC_FLAG}" ]]; then
  CMD+=("${STOCHASTIC_FLAG}")
fi

if [[ -n "${OPEN_FLAG}" ]]; then
  CMD+=("${OPEN_FLAG}")
fi

"${CMD[@]}"
