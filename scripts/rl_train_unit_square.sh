#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Edit these for each run.
RUN_NAME="${RUN_NAME:-homological_gat_50k}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
N_STEPS="${N_STEPS:-512}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DEVICE="${DEVICE:-mps}"
MODEL_KIND="${MODEL_KIND:-homological_gat}"
MAX_SPEED_SCALE="${MAX_SPEED_SCALE:-0.2}"
FENCE_SENSING_RADIUS="${FENCE_SENSING_RADIUS:-}"
FENCE_OFFSET_RATIO="${FENCE_OFFSET_RATIO:-}"
USE_WEIGHTED_ALPHA="${USE_WEIGHTED_ALPHA:-0}"
SEED="${SEED:-1000}"
EVAL_FREQ="${EVAL_FREQ:-2000}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-800}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-2000}"
ENABLE_EVENT_LOG="${ENABLE_EVENT_LOG:-0}"
PLOT_AFTER_TRAIN="${PLOT_AFTER_TRAIN:-0}"
DISABLE_PHASE_REWARD_SCHEDULE="${DISABLE_PHASE_REWARD_SCHEDULE:-0}"
ENABLE_FINAL_HOLE_COMPRESSION_SCHEDULE="${ENABLE_FINAL_HOLE_COMPRESSION_SCHEDULE:-1}"

# Reward weights (set to 0.0 for ablation).
# Current default keeps only area-based geometry shaping active.
AREA_PROGRESS_WEIGHT="${AREA_PROGRESS_WEIGHT:-1.5}"
PERIMETER_PROGRESS_WEIGHT="${PERIMETER_PROGRESS_WEIGHT:-0.0}"
LARGEST_AREA_PROGRESS_WEIGHT="${LARGEST_AREA_PROGRESS_WEIGHT:-0.0}"
LARGEST_PERIMETER_PROGRESS_WEIGHT="${LARGEST_PERIMETER_PROGRESS_WEIGHT:-0.0}"
AREA_REGRESS_WEIGHT="${AREA_REGRESS_WEIGHT:-1.0}"
PERIMETER_REGRESS_WEIGHT="${PERIMETER_REGRESS_WEIGHT:-0.0}"
LARGEST_AREA_REGRESS_WEIGHT="${LARGEST_AREA_REGRESS_WEIGHT:-0.0}"
LARGEST_PERIMETER_REGRESS_WEIGHT="${LARGEST_PERIMETER_REGRESS_WEIGHT:-0.0}"
AREA_RESIDUAL_WEIGHT="${AREA_RESIDUAL_WEIGHT:-0.25}"
PERIMETER_RESIDUAL_WEIGHT="${PERIMETER_RESIDUAL_WEIGHT:-0.0}"
LARGEST_AREA_RESIDUAL_WEIGHT="${LARGEST_AREA_RESIDUAL_WEIGHT:-0.0}"
LARGEST_PERIMETER_RESIDUAL_WEIGHT="${LARGEST_PERIMETER_RESIDUAL_WEIGHT:-0.0}"

# Neighbor-distance soft band controls.
NEIGHBOR_MIN_RATIO="${NEIGHBOR_MIN_RATIO:-0.8660254037844386}"
NEIGHBOR_MAX_RATIO="${NEIGHBOR_MAX_RATIO:-3.0}"
NEIGHBOR_CLOSE_WEIGHT="${NEIGHBOR_CLOSE_WEIGHT:-0.0}"
NEIGHBOR_FAR_WEIGHT="${NEIGHBOR_FAR_WEIGHT:-0.0}"
MOBILE_OVERLAP_WEIGHT="${MOBILE_OVERLAP_WEIGHT:-0.02}"
HARD_CLOSE_DISTANCE_RATIO="${HARD_CLOSE_DISTANCE_RATIO:-0.5}"
HARD_CLOSE_MOBILE_WEIGHT="${HARD_CLOSE_MOBILE_WEIGHT:-0.0}"
HARD_CLOSE_FENCE_WEIGHT="${HARD_CLOSE_FENCE_WEIGHT:-0.0}"
FENCE_CLOSE_WEIGHT="${FENCE_CLOSE_WEIGHT:-0.0}"
FENCE_FAR_WEIGHT="${FENCE_FAR_WEIGHT:-0.0}"
TRUE_CYCLE_CLOSED_REWARD="${TRUE_CYCLE_CLOSED_REWARD:-3.0}"
TRUE_CYCLE_ADDED_PENALTY="${TRUE_CYCLE_ADDED_PENALTY:-4.0}"
CLEAR_BONUS="${CLEAR_BONUS:-12.0}"
TIME_PENALTY="${TIME_PENALTY:-0.03}"
MERGE_HAZARD_WEIGHT="${MERGE_HAZARD_WEIGHT:-5.0}"
INTERFACE_EDGE_LOSS_WEIGHT="${INTERFACE_EDGE_LOSS_WEIGHT:-0.0}"
INTERFACE_EDGE_STRETCH_WEIGHT="${INTERFACE_EDGE_STRETCH_WEIGHT:-0.0}"
SUCCESS_TIME_BONUS_WEIGHT="${SUCCESS_TIME_BONUS_WEIGHT:-10.0}"
ONE_HOLE_LINGER_WEIGHT="${ONE_HOLE_LINGER_WEIGHT:-0.0}"
ONE_HOLE_AREA_SCALE_ALPHA="${ONE_HOLE_AREA_SCALE_ALPHA:-1.5}"
ONE_HOLE_PERIMETER_SCALE_ALPHA="${ONE_HOLE_PERIMETER_SCALE_ALPHA:-0.0}"
POLICY_PROBE_ARGS="${POLICY_PROBE_ARGS:-}"

# Available model kinds:
#   baseline_gat
#   dart_gat
#   boundary_cycle
#   homological_gat

python "${REPO_ROOT}/experiments/rl_train_unit_square.py" \
  --run-name "${RUN_NAME}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --device "${DEVICE}" \
  --model-kind "${MODEL_KIND}" \
  --max-speed-scale "${MAX_SPEED_SCALE}" \
  $( [[ -n "${FENCE_SENSING_RADIUS}" ]] && printf -- '--fence-sensing-radius %s' "${FENCE_SENSING_RADIUS}" ) \
  $( [[ -n "${FENCE_OFFSET_RATIO}" ]] && printf -- '--fence-offset-ratio %s' "${FENCE_OFFSET_RATIO}" ) \
  $( [[ "${USE_WEIGHTED_ALPHA}" == "1" ]] && printf '%s' "--use-weighted-alpha" ) \
  --seed "${SEED}" \
  --area-progress-weight "${AREA_PROGRESS_WEIGHT}" \
  --perimeter-progress-weight "${PERIMETER_PROGRESS_WEIGHT}" \
  --largest-area-progress-weight "${LARGEST_AREA_PROGRESS_WEIGHT}" \
  --largest-perimeter-progress-weight "${LARGEST_PERIMETER_PROGRESS_WEIGHT}" \
  --area-regress-weight "${AREA_REGRESS_WEIGHT}" \
  --perimeter-regress-weight "${PERIMETER_REGRESS_WEIGHT}" \
  --largest-area-regress-weight "${LARGEST_AREA_REGRESS_WEIGHT}" \
  --largest-perimeter-regress-weight "${LARGEST_PERIMETER_REGRESS_WEIGHT}" \
  --area-residual-weight "${AREA_RESIDUAL_WEIGHT}" \
  --perimeter-residual-weight "${PERIMETER_RESIDUAL_WEIGHT}" \
  --largest-area-residual-weight "${LARGEST_AREA_RESIDUAL_WEIGHT}" \
  --largest-perimeter-residual-weight "${LARGEST_PERIMETER_RESIDUAL_WEIGHT}" \
  --neighbor-min-ratio "${NEIGHBOR_MIN_RATIO}" \
  --neighbor-max-ratio "${NEIGHBOR_MAX_RATIO}" \
  --neighbor-close-weight "${NEIGHBOR_CLOSE_WEIGHT}" \
  --neighbor-far-weight "${NEIGHBOR_FAR_WEIGHT}" \
  --mobile-overlap-weight "${MOBILE_OVERLAP_WEIGHT}" \
  --hard-close-distance-ratio "${HARD_CLOSE_DISTANCE_RATIO}" \
  --hard-close-mobile-weight "${HARD_CLOSE_MOBILE_WEIGHT}" \
  --hard-close-fence-weight "${HARD_CLOSE_FENCE_WEIGHT}" \
  --fence-close-weight "${FENCE_CLOSE_WEIGHT}" \
  --fence-far-weight "${FENCE_FAR_WEIGHT}" \
  --merge-hazard-weight "${MERGE_HAZARD_WEIGHT}" \
  --interface-edge-loss-weight "${INTERFACE_EDGE_LOSS_WEIGHT}" \
  --interface-edge-stretch-weight "${INTERFACE_EDGE_STRETCH_WEIGHT}" \
  $( [[ "${ENABLE_FINAL_HOLE_COMPRESSION_SCHEDULE}" == "1" ]] && printf '%s' "--enable-final-hole-compression-schedule" ) \
  $( [[ "${DISABLE_PHASE_REWARD_SCHEDULE}" == "1" && "${ENABLE_FINAL_HOLE_COMPRESSION_SCHEDULE}" != "1" ]] && printf '%s' "--disable-phase-reward-schedule" ) \
  --true-cycle-closed-reward "${TRUE_CYCLE_CLOSED_REWARD}" \
  --true-cycle-added-penalty "${TRUE_CYCLE_ADDED_PENALTY}" \
  --clear-bonus "${CLEAR_BONUS}" \
  --success-time-bonus-weight "${SUCCESS_TIME_BONUS_WEIGHT}" \
  --one-hole-linger-weight "${ONE_HOLE_LINGER_WEIGHT}" \
  --one-hole-area-scale-alpha "${ONE_HOLE_AREA_SCALE_ALPHA}" \
  --one-hole-perimeter-scale-alpha "${ONE_HOLE_PERIMETER_SCALE_ALPHA}" \
  --time-penalty "${TIME_PENALTY}" \
  --eval-freq "${EVAL_FREQ}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --eval-max-steps "${EVAL_MAX_STEPS}" \
  --checkpoint-freq "${CHECKPOINT_FREQ}" \
  $( [[ "${ENABLE_EVENT_LOG}" == "1" ]] && printf '%s' "--enable-event-log" ) \
  --progress-bar \
  --disable-attention-log \
  ${POLICY_PROBE_ARGS}

if [[ "${PLOT_AFTER_TRAIN}" == "1" ]]; then
  python "${REPO_ROOT}/experiments/rl_plot_training.py" \
    --run-dir "${REPO_ROOT}/experiments/output/rl_unit_square/${RUN_NAME}" \
    --train-bin-size 100
fi
