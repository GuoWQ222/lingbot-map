#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_BIN="${CONDA_BIN:-/cpfs/user/guowenqi/miniconda3/condabin/conda}"
CONDA_ENV="${CONDA_ENV:-lingbot-map}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python)
fi

SCAL3R_REPO="${SCAL3R_REPO:-/cpfs/user/guowenqi/Scal3R}"
SCAL3R_PYTHON="${SCAL3R_PYTHON:-${CONDA_BIN} run --no-capture-output -n scal3r python}"
SCAL3R_CONFIG="${SCAL3R_CONFIG:-configs/models/scal3r.yaml}"
SCAL3R_CHECKPOINT="${SCAL3R_CHECKPOINT:-}"
SCAL3R_DEVICE="${SCAL3R_DEVICE:-${DEVICE:-cuda}}"
SCAL3R_PREPROCESS_WORKERS="${SCAL3R_PREPROCESS_WORKERS:-8}"
SCAL3R_BLOCK_SIZE="${SCAL3R_BLOCK_SIZE:-60}"
SCAL3R_OVERLAP_SIZE="${SCAL3R_OVERLAP_SIZE:-30}"
SCAL3R_USE_LOOP="${SCAL3R_USE_LOOP:-1}"
SCAL3R_USE_XYZ_ALIGN="${SCAL3R_USE_XYZ_ALIGN:-0}"
SCAL3R_PGO_WORKERS="${SCAL3R_PGO_WORKERS:-8}"
SCAL3R_SAVE_XYZ="${SCAL3R_SAVE_XYZ:-0}"
SCAL3R_TEST_USE_AMP="${SCAL3R_TEST_USE_AMP:-0}"
FORCE_RERUN_SCAL3R="${FORCE_RERUN_SCAL3R:-0}"
KEEP_SCAL3R_INPUTS="${KEEP_SCAL3R_INPUTS:-0}"

RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/runs/manip_long_train_64gpu}"
TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-${RUN_DIR}/args.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/eval_scal3r}"

SPLIT="${SPLIT:-val}"
MAX_SCENES_EVAL="${MAX_SCENES_EVAL:-0}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
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

IMAGE_SIZE="${IMAGE_SIZE:-518}"
GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION:-none}"
CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
DEPTH_ALIGN="${DEPTH_ALIGN:-pi3_scale_shift}"
SECONDARY_DEPTH_ALIGN="${SECONDARY_DEPTH_ALIGN:-}"
POINTCLOUD_METRICS="${POINTCLOUD_METRICS:-1}"
POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-100000}"
POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD:-0.1}"
POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS:-30}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

ARGS=(
  --train_args_json "${TRAIN_ARGS_JSON}"
  --output_dir "${OUTPUT_DIR}"
  --scal3r_repo "${SCAL3R_REPO}"
  --scal3r_python "${SCAL3R_PYTHON}"
  --scal3r_config "${SCAL3R_CONFIG}"
  --scal3r_device "${SCAL3R_DEVICE}"
  --scal3r_preprocess_workers "${SCAL3R_PREPROCESS_WORKERS}"
  --scal3r_block_size "${SCAL3R_BLOCK_SIZE}"
  --scal3r_overlap_size "${SCAL3R_OVERLAP_SIZE}"
  --scal3r_use_loop "${SCAL3R_USE_LOOP}"
  --scal3r_use_xyz_align "${SCAL3R_USE_XYZ_ALIGN}"
  --scal3r_pgo_workers "${SCAL3R_PGO_WORKERS}"
  --scal3r_save_xyz "${SCAL3R_SAVE_XYZ}"
  --split "${SPLIT}"
  --max_scenes_eval "${MAX_SCENES_EVAL}"
  --eval_shard_count "${EVAL_SHARD_COUNT}"
  --eval_shard_index "${EVAL_SHARD_INDEX}"
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
)

if [[ -n "${SCAL3R_CHECKPOINT}" ]]; then
  ARGS+=(--scal3r_checkpoint "${SCAL3R_CHECKPOINT}")
fi
if [[ "${SCAL3R_TEST_USE_AMP}" == "1" ]]; then
  ARGS+=(--scal3r_test_use_amp)
fi
if [[ "${FORCE_RERUN_SCAL3R}" == "1" ]]; then
  ARGS+=(--force_rerun_scal3r)
fi
if [[ "${KEEP_SCAL3R_INPUTS}" == "1" ]]; then
  ARGS+=(--keep_scal3r_inputs)
fi
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
 Scal3R on LingBot-MAP Manip Evaluation
========================================
[eval_scal3r]
  scal3r_repo   : ${SCAL3R_REPO}
  scal3r_python : ${SCAL3R_PYTHON}
  train_args    : ${TRAIN_ARGS_JSON}
  split         : ${SPLIT}
  shard         : ${EVAL_SHARD_INDEX}/${EVAL_SHARD_COUNT}
  device        : ${DEVICE}
  scal3r_device : ${SCAL3R_DEVICE}
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  output_dir    : ${OUTPUT_DIR}
  eval_strategy : ${EVAL_STRATEGY}
  eval_n_frames : ${EVAL_NUM_FRAMES}
  image_size    : ${IMAGE_SIZE}
  geometry_norm : ${GEOMETRY_NORMALIZATION}
  camera_align  : ${CAMERA_ALIGN}
  depth_align   : ${DEPTH_ALIGN} secondary=${SECONDARY_DEPTH_ALIGN}
  pointcloud    : ${POINTCLOUD_METRICS} source=depth+c2w align=${POINTCLOUD_ALIGN}
========================================

EOF

"${PYTHON_CMD[@]}" eval_scal3r.py "${ARGS[@]}" "$@"
