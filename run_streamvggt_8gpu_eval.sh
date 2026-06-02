#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SHARD_COUNT="${SHARD_COUNT:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_LIST}"
if (( ${#GPU_IDS[@]} != SHARD_COUNT )); then
  echo "[run_streamvggt_8gpu_eval] GPU_LIST has ${#GPU_IDS[@]} entries but SHARD_COUNT=${SHARD_COUNT}" >&2
  exit 2
fi

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/manip_long_train_64gpu}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_DIR}/eval_streamvggt_8gpu_full_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

export RUN_DIR
export TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-${RUN_DIR}/args.json}"
export STREAMVGGT_REPO="${STREAMVGGT_REPO:-/cpfs/user/guowenqi/StreamVGGT}"
export STREAMVGGT_WEIGHTS="${STREAMVGGT_WEIGHTS:-/cpfs/user/guowenqi/StreamVGGT/checkpoints.pth}"
export FORWARD_MODE="${FORWARD_MODE:-stream}"
export SPLIT="${SPLIT:-val}"
export MAX_SCENES_EVAL="${MAX_SCENES_EVAL:-0}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PER_SCENE_CSV="${PER_SCENE_CSV:-1}"
export SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
export PRINT_EVERY="${PRINT_EVERY:-20}"
export EVAL_STRATEGY="${EVAL_STRATEGY:-manip_track}"
export EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES:-64}"
export EVAL_WRIST_CAMERA_NAME="${EVAL_WRIST_CAMERA_NAME:-realsense_left}"
# Empty string means all 6 Long5 surround cameras for manip_track.
export EVAL_SURROUND_CAMERA_NAME="${EVAL_SURROUND_CAMERA_NAME-}"
export IMAGE_SIZE="${IMAGE_SIZE:-518}"
export DEPTH_ALIGN="${DEPTH_ALIGN:-pi3_scale_shift}"
export SECONDARY_DEPTH_ALIGN="${SECONDARY_DEPTH_ALIGN:-}"
export CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
export POINTCLOUD_METRICS="${POINTCLOUD_METRICS:-1}"
export POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-100000}"
export POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
export POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD:-0.1}"
export POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS:-30}"

cat <<EOF
[run_streamvggt_8gpu_eval]
  output_root : ${OUTPUT_ROOT}
  shard_count : ${SHARD_COUNT}
  gpu_list    : ${GPU_LIST}
  split       : ${SPLIT}
  strategy    : ${EVAL_STRATEGY}
  frames      : ${EVAL_NUM_FRAMES}
  surround    : ${EVAL_SURROUND_CAMERA_NAME:-all_6}
  pointcloud  : ${POINTCLOUD_METRICS}
EOF

declare -a PIDS=()
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPU_IDS[$shard_idx]}"
  shard_name="$(printf 'shard_%02d_of_%02d' "${shard_idx}" "${SHARD_COUNT}")"
  shard_dir="${OUTPUT_ROOT}/${shard_name}"
  log_path="${LOG_DIR}/${shard_name}.log"
  mkdir -p "${shard_dir}"
  echo "[run_streamvggt_8gpu_eval] launch ${shard_name} on GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export OUTPUT_DIR="${shard_dir}"
    bash "${SCRIPT_DIR}/eval_streamvggt.sh" \
      --eval_shard_count "${SHARD_COUNT}" \
      --eval_shard_index "${shard_idx}"
  ) >"${log_path}" 2>&1 &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "[run_streamvggt_8gpu_eval] at least one shard failed; see ${LOG_DIR}" >&2
  exit "${status}"
fi

"${CONDA_BIN:-/cpfs/user/guowenqi/miniconda3/condabin/conda}" run --no-capture-output -n "${CONDA_ENV:-lingbot-map}" \
  python "${SCRIPT_DIR}/aggregate_streamvggt_shards.py" "${OUTPUT_ROOT}"

echo "[run_streamvggt_8gpu_eval] done: ${OUTPUT_ROOT}/combined_metrics.json"
