#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

STREAMVGGT_REPO="${STREAMVGGT_REPO:-/cpfs/user/guowenqi/StreamVGGT}"
export PYTHONPATH="${SCRIPT_DIR}:${STREAMVGGT_REPO}/src:${PYTHONPATH:-}"

CONDA_BIN="${CONDA_BIN:-/cpfs/user/guowenqi/miniconda3/condabin/conda}"
CONDA_ENV="${CONDA_ENV:-lingbot-map}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python)
fi

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/manip_long_train_64gpu}"
TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-${RUN_DIR}/args.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval_streamvggt}"
STREAMVGGT_WEIGHTS="${STREAMVGGT_WEIGHTS:-/cpfs/user/guowenqi/StreamVGGT/checkpoints.pth}"
FORWARD_MODE="${FORWARD_MODE:-stream}"

SPLIT="${SPLIT:-val}"
MAX_SCENES_EVAL="${MAX_SCENES_EVAL:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
PER_SCENE_CSV="${PER_SCENE_CSV:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
PRINT_EVERY="${PRINT_EVERY:-5}"

EVAL_STRATEGY="${EVAL_STRATEGY:-manip_track}"
EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES:-64}"
EVAL_WRIST_CAMERA_NAME="${EVAL_WRIST_CAMERA_NAME:-realsense_left}"
EVAL_SURROUND_CAMERA_NAME="${EVAL_SURROUND_CAMERA_NAME-surround_cam_0}"
EVAL_SEED="${EVAL_SEED:-42}"

IMAGE_SIZE="${IMAGE_SIZE:-518}"
DEPTH_ALIGN="${DEPTH_ALIGN:-pi3_scale_shift}"
SECONDARY_DEPTH_ALIGN="${SECONDARY_DEPTH_ALIGN:-}"
CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
POINTCLOUD_METRICS="${POINTCLOUD_METRICS:-1}"
POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-100000}"
POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD:-0.1}"
POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS:-30}"

ARGS=(
  --train_args_json "${TRAIN_ARGS_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --streamvggt_repo "${STREAMVGGT_REPO}"
  --model_weights "${STREAMVGGT_WEIGHTS}"
  --forward_mode "${FORWARD_MODE}"
  --split "${SPLIT}"
  --max_scenes_eval "${MAX_SCENES_EVAL}"
  --num_workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --print_every "${PRINT_EVERY}"
  --eval_strategy "${EVAL_STRATEGY}"
  --eval_num_frames "${EVAL_NUM_FRAMES}"
  --eval_wrist_camera_name "${EVAL_WRIST_CAMERA_NAME}"
  --eval_surround_camera_name "${EVAL_SURROUND_CAMERA_NAME}"
  --eval_seed "${EVAL_SEED}"
  --image_size "${IMAGE_SIZE}"
  --depth_align "${DEPTH_ALIGN}"
  --secondary_depth_align "${SECONDARY_DEPTH_ALIGN}"
  --camera_align "${CAMERA_ALIGN}"
  --pointcloud_max_points "${POINTCLOUD_MAX_POINTS}"
  --pointcloud_align "${POINTCLOUD_ALIGN}"
  --pointcloud_icp_threshold "${POINTCLOUD_ICP_THRESHOLD}"
  --pointcloud_icp_max_iterations "${POINTCLOUD_ICP_MAX_ITERATIONS}"
)

if [[ "${PER_SCENE_CSV}" != "1" ]]; then
  ARGS+=(--no_per_scene_csv)
fi
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  ARGS+=(--save_predictions)
fi
if [[ "${POINTCLOUD_METRICS}" != "1" ]]; then
  ARGS+=(--no_pointcloud_metrics)
fi

cat <<EOF

========================================
 StreamVGGT Manip Validation Evaluation
========================================
[eval_streamvggt]
  repo          : ${STREAMVGGT_REPO}
  model_weights : ${STREAMVGGT_WEIGHTS}
  forward_mode  : ${FORWARD_MODE}
  train_args    : ${TRAIN_ARGS_JSON}
  split         : ${SPLIT}
  device        : ${DEVICE}
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  output_dir    : ${OUTPUT_DIR}
  eval_strategy : ${EVAL_STRATEGY}
  eval_n_frames : ${EVAL_NUM_FRAMES}
  image_size    : ${IMAGE_SIZE}
  geometry_norm: none (metric alignment only)
  camera_align : ${CAMERA_ALIGN}
  depth_align   : ${DEPTH_ALIGN} secondary=${SECONDARY_DEPTH_ALIGN}
  pointcloud    : ${POINTCLOUD_METRICS} source=streamvggt_depth+pose align=${POINTCLOUD_ALIGN}
========================================

EOF

"${PYTHON_CMD[@]}" eval_streamvggt.py "${ARGS[@]}" "$@"
