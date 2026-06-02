#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_BIN="${CONDA_BIN:-/cpfs/user/guowenqi/miniconda3/condabin/conda}"
CONDA_ENV="${CONDA_ENV:-ttt3r}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python)
fi

TTT3R_REPO="${TTT3R_REPO:-/cpfs/user/guowenqi/TTT3R}"
TTT3R_CHECKPOINT="${TTT3R_CHECKPOINT:-${TTT3R_REPO}/src/cut3r_512_dpt_4_64.pth}"
MODEL_UPDATE_TYPE="${MODEL_UPDATE_TYPE:-ttt3r}"
RESET_INTERVAL="${RESET_INTERVAL:-1000000}"

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/manip_long_train_64gpu}"
TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-${RUN_DIR}/args.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval_ttt3r}"

SPLIT="${SPLIT:-val}"
MAX_SCENES_EVAL="${MAX_SCENES_EVAL:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
PER_SCENE_CSV="${PER_SCENE_CSV:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
PRINT_EVERY="${PRINT_EVERY:-5}"

EVAL_STRATEGY="${EVAL_STRATEGY:-manip_track}"
EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES:-64}"
EVAL_WRIST_CAMERA_NAME="${EVAL_WRIST_CAMERA_NAME:-realsense_left}"
EVAL_SURROUND_CAMERA_NAME="${EVAL_SURROUND_CAMERA_NAME:-surround_cam_0}"
EVAL_SEED="${EVAL_SEED:-42}"

# Keep geometry in native predicted/GT frames. Alignment is metric-time only,
# mirroring eval.sh's DEPTH_ALIGN/CAMERA_ALIGN/POINTCLOUD_ALIGN behavior.
GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION:-none}"
CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
DEPTH_ALIGN="${DEPTH_ALIGN:-pi3_scale_shift}"
SECONDARY_DEPTH_ALIGN="${SECONDARY_DEPTH_ALIGN:-}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
POINTCLOUD_METRICS="${POINTCLOUD_METRICS:-1}"
POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-100000}"
POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD:-0.1}"
POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS:-30}"
FOCAL_MODE="${FOCAL_MODE:-weiszfeld}"

export PYTHONPATH="${SCRIPT_DIR}:${TTT3R_REPO}:${TTT3R_REPO}/src:${PYTHONPATH:-}"

ARGS=(
  --train_args_json "${TRAIN_ARGS_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --ttt3r_repo "${TTT3R_REPO}"
  --ttt3r_checkpoint "${TTT3R_CHECKPOINT}"
  --model_update_type "${MODEL_UPDATE_TYPE}"
  --reset_interval "${RESET_INTERVAL}"
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
  --geometry_normalization "${GEOMETRY_NORMALIZATION}"
  --camera_align "${CAMERA_ALIGN}"
  --depth_align "${DEPTH_ALIGN}"
  --secondary_depth_align "${SECONDARY_DEPTH_ALIGN}"
  --pointcloud_max_points "${POINTCLOUD_MAX_POINTS}"
  --pointcloud_align "${POINTCLOUD_ALIGN}"
  --pointcloud_icp_threshold "${POINTCLOUD_ICP_THRESHOLD}"
  --pointcloud_icp_max_iterations "${POINTCLOUD_ICP_MAX_ITERATIONS}"
  --focal_mode "${FOCAL_MODE}"
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
 TTT3R on LingBot-MAP Manip Evaluation
========================================
[eval_ttt3r]
  ttt3r_repo    : ${TTT3R_REPO}
  ttt3r_ckpt    : ${TTT3R_CHECKPOINT}
  update_type   : ${MODEL_UPDATE_TYPE}
  train_args    : ${TRAIN_ARGS_JSON}
  split         : ${SPLIT}
  device        : ${DEVICE}
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  output_dir    : ${OUTPUT_DIR}
  eval_strategy : ${EVAL_STRATEGY}
  eval_n_frames : ${EVAL_NUM_FRAMES}
  wrist_camera  : ${EVAL_WRIST_CAMERA_NAME}
  surround_cam  : ${EVAL_SURROUND_CAMERA_NAME:-<all_6_surround>}
  image_size    : ${IMAGE_SIZE}
  geometry_norm : ${GEOMETRY_NORMALIZATION} (metric-time alignment only)
  camera_align  : ${CAMERA_ALIGN}
  depth_align   : ${DEPTH_ALIGN} secondary=${SECONDARY_DEPTH_ALIGN}
  pointcloud    : ${POINTCLOUD_METRICS} source=pts3d+c2w align=${POINTCLOUD_ALIGN}
========================================

EOF

"${PYTHON_CMD[@]}" eval_ttt3r.py "${ARGS[@]}" "$@"
