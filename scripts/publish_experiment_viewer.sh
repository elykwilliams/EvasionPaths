#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SOURCE_DIR="${SOURCE_DIR:-}"
CATEGORY="${CATEGORY:-rl}"
MOTION_MODEL="${MOTION_MODEL:-RL}"
DISPLAY_NAME="${DISPLAY_NAME:-}"
SIM_INDICES="${SIM_INDICES:-}"
DEFAULT_SITE_ROOT="${HOME}/projects/evasion-paths-experiments"
SITE_ROOT="${SITE_ROOT:-${DEFAULT_SITE_ROOT}}"
EXPERIMENT_ID="${EXPERIMENT_ID:-}"
SOURCE_RUN="${SOURCE_RUN:-}"
CHECKPOINT="${CHECKPOINT:-}"

if [[ -z "${SOURCE_DIR}" ]]; then
  echo "SOURCE_DIR is required."
  exit 1
fi

if [[ -z "${DISPLAY_NAME}" ]]; then
  echo "DISPLAY_NAME is required."
  exit 1
fi

python "${REPO_ROOT}/experiments/publish_experiment_viewer.py" \
  --source-dir "${SOURCE_DIR}" \
  --category "${CATEGORY}" \
  --motion-model "${MOTION_MODEL}" \
  --display-name "${DISPLAY_NAME}" \
  --sim-indices "${SIM_INDICES}" \
  --site-root "${SITE_ROOT}" \
  --experiment-id "${EXPERIMENT_ID}" \
  --source-run "${SOURCE_RUN}" \
  --checkpoint "${CHECKPOINT}"
