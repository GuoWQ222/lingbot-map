#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/scripts/train:${REPO_DIR}/scripts/eval:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONDA_BIN="${CONDA_BIN:-/cpfs/user/guowenqi/miniconda3/condabin/conda}"
CONDA_ENV="${CONDA_ENV:-lingbot-map}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python)
fi

LOGER_REPO="${LOGER_REPO:-/cpfs/user/guowenqi/LoGeR}"
LOGER_CKPT_NAME="${LOGER_CKPT_NAME:-LoGeR}"
LOGER_CHECKPOINT="${LOGER_CHECKPOINT:-${LOGER_REPO}/ckpts/${LOGER_CKPT_NAME}/latest.pt}"
LOGER_CONFIG="${LOGER_CONFIG:-${LOGER_REPO}/ckpts/${LOGER_CKPT_NAME}/original_config.yaml}"
LOGER_STRICT_LOAD="${LOGER_STRICT_LOAD:-0}"
LOGER_WINDOW_SIZE="${LOGER_WINDOW_SIZE:-32}"
LOGER_OVERLAP_SIZE="${LOGER_OVERLAP_SIZE:-3}"
LOGER_RESET_EVERY="${LOGER_RESET_EVERY:-0}"
LOGER_NUM_ITERATIONS="${LOGER_NUM_ITERATIONS:-}"
LOGER_SIM3="${LOGER_SIM3:-0}"
LOGER_SE3="${LOGER_SE3:-auto}"
LOGER_SIM3_SCALE_MODE="${LOGER_SIM3_SCALE_MODE:-median}"
LOGER_NO_TTT="${LOGER_NO_TTT:-0}"
LOGER_NO_SWA="${LOGER_NO_SWA:-0}"
LOGER_PI3X="${LOGER_PI3X:-0}"
LOGER_PI3X_METRIC="${LOGER_PI3X_METRIC:-1}"
LOGER_DISABLE_COMPILE="${LOGER_DISABLE_COMPILE:-1}"
LOGER_INPUT_PREPROCESS="${LOGER_INPUT_PREPROCESS:-native}"
LOGER_NATIVE_WIDTH="${LOGER_NATIVE_WIDTH:-504}"
LOGER_NATIVE_HEIGHT="${LOGER_NATIVE_HEIGHT:-280}"

TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-${REPO_DIR}/outputs/runs/manip_long_train_64gpu/args.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/eval/loger/${LOGER_CKPT_NAME}}"
SPLIT="${SPLIT:-val}"
MAX_SCENES_EVAL="${MAX_SCENES_EVAL:-0}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
PER_SCENE_CSV="${PER_SCENE_CSV:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
PRINT_EVERY="${PRINT_EVERY:-5}"
EVAL_STRATEGY="${EVAL_STRATEGY:-left_moving_tracks}"
EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES:-64}"
EVAL_WRIST_CAMERA_NAME="${EVAL_WRIST_CAMERA_NAME:-realsense_left}"
EVAL_SURROUND_CAMERA_NAME="${EVAL_SURROUND_CAMERA_NAME:-surround_cam_moving}"
EVAL_SEED="${EVAL_SEED:-42}"

# Match lingbot-map/eval.sh: keep GT/predictions in native frames and align
# only inside metric computation.
GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION:-none}"
CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
DEPTH_ALIGN="${DEPTH_ALIGN:-pi3_scale_shift}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
POINTCLOUD_METRICS="${POINTCLOUD_METRICS:-1}"
POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-100000}"
POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD:-0.1}"
POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS:-30}"
POINTCLOUD_ICP_BACKEND="${POINTCLOUD_ICP_BACKEND:-open3d}"
POINTCLOUD_KDTREE_WORKERS="${POINTCLOUD_KDTREE_WORKERS:-1}"
POINTCLOUD_WORKERS="${POINTCLOUD_WORKERS:-1}"
RECOVER_FOCAL="${RECOVER_FOCAL:-1}"
FOCAL_MASK_THRESHOLD="${FOCAL_MASK_THRESHOLD:-0.1}"
FOCAL_DOWNSAMPLE_H="${FOCAL_DOWNSAMPLE_H:-64}"
FOCAL_DOWNSAMPLE_W="${FOCAL_DOWNSAMPLE_W:-64}"

ARGS=(
  --train_args_json "${TRAIN_ARGS_JSON}"
  --loger_repo "${LOGER_REPO}"
  --loger_checkpoint "${LOGER_CHECKPOINT}"
  --loger_config "${LOGER_CONFIG}"
  --loger_window_size "${LOGER_WINDOW_SIZE}"
  --loger_overlap_size "${LOGER_OVERLAP_SIZE}"
  --loger_reset_every "${LOGER_RESET_EVERY}"
  --loger_sim3_scale_mode "${LOGER_SIM3_SCALE_MODE}"
  --loger_input_preprocess "${LOGER_INPUT_PREPROCESS}"
  --loger_native_width "${LOGER_NATIVE_WIDTH}"
  --loger_native_height "${LOGER_NATIVE_HEIGHT}"
  --output_dir "${OUTPUT_DIR}"
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
  --geometry_normalization "${GEOMETRY_NORMALIZATION}"
  --camera_align "${CAMERA_ALIGN}"
  --depth_align "${DEPTH_ALIGN}"
  --image_size "${IMAGE_SIZE}"
  --pointcloud_max_points "${POINTCLOUD_MAX_POINTS}"
  --pointcloud_align "${POINTCLOUD_ALIGN}"
  --pointcloud_icp_threshold "${POINTCLOUD_ICP_THRESHOLD}"
  --pointcloud_icp_max_iterations "${POINTCLOUD_ICP_MAX_ITERATIONS}"
  --pointcloud_icp_backend "${POINTCLOUD_ICP_BACKEND}"
  --pointcloud_kdtree_workers "${POINTCLOUD_KDTREE_WORKERS}"
  --pointcloud_workers "${POINTCLOUD_WORKERS}"
  --focal_mask_threshold "${FOCAL_MASK_THRESHOLD}"
  --focal_downsample_size "${FOCAL_DOWNSAMPLE_H}" "${FOCAL_DOWNSAMPLE_W}"
)

if [[ "${LOGER_STRICT_LOAD}" == "1" ]]; then
  ARGS+=(--loger_strict_load)
fi
if [[ -n "${LOGER_NUM_ITERATIONS}" ]]; then
  ARGS+=(--loger_num_iterations "${LOGER_NUM_ITERATIONS}")
fi
if [[ "${LOGER_SIM3}" == "1" ]]; then
  ARGS+=(--loger_sim3)
fi
if [[ "${LOGER_SE3}" == "1" ]]; then
  ARGS+=(--loger_se3)
elif [[ "${LOGER_SE3}" == "0" ]]; then
  ARGS+=(--no-loger_se3)
fi
if [[ "${LOGER_NO_TTT}" == "1" ]]; then
  ARGS+=(--loger_no_ttt)
fi
if [[ "${LOGER_NO_SWA}" == "1" ]]; then
  ARGS+=(--loger_no_swa)
fi
if [[ "${LOGER_PI3X}" == "1" ]]; then
  ARGS+=(--loger_pi3x)
fi
if [[ "${LOGER_PI3X_METRIC}" == "0" ]]; then
  ARGS+=(--no-loger_pi3x_metric)
fi
if [[ "${LOGER_DISABLE_COMPILE}" == "0" ]]; then
  ARGS+=(--no-loger_disable_compile)
fi
if [[ "${POINTCLOUD_METRICS}" == "1" ]]; then
  ARGS+=(--pointcloud_metrics)
elif [[ "${POINTCLOUD_METRICS}" == "0" ]]; then
  ARGS+=(--no_pointcloud_metrics)
fi
if [[ "${PER_SCENE_CSV}" == "1" ]]; then
  ARGS+=(--per_scene_csv)
fi
if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  ARGS+=(--save_predictions)
fi
if [[ "${RECOVER_FOCAL}" == "0" ]]; then
  ARGS+=(--no-recover_focal)
fi

cat <<EOF

========================================
 LoGeR on LingBot-MAP Manip Evaluation
========================================
[eval_loger]
  loger_repo    : ${LOGER_REPO}
  checkpoint    : ${LOGER_CHECKPOINT}
  config        : ${LOGER_CONFIG}
  train_args    : ${TRAIN_ARGS_JSON}
  split         : ${SPLIT}
  shard         : ${EVAL_SHARD_INDEX}/${EVAL_SHARD_COUNT}
  device        : ${DEVICE}
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  output_dir    : ${OUTPUT_DIR}
  eval_strategy : ${EVAL_STRATEGY}
  eval_n_frames : ${EVAL_NUM_FRAMES}
  wrist_camera  : ${EVAL_WRIST_CAMERA_NAME}
  surround_cam  : ${EVAL_SURROUND_CAMERA_NAME:-<all_6_surround>}
  geometry_norm : ${GEOMETRY_NORMALIZATION}
  camera_align  : ${CAMERA_ALIGN}
  depth_align   : ${DEPTH_ALIGN}
  image_size    : ${IMAGE_SIZE}
  loger_native  : ${LOGER_NATIVE_WIDTH}x${LOGER_NATIVE_HEIGHT}
  pointcloud    : ${POINTCLOUD_METRICS} max_points=${POINTCLOUD_MAX_POINTS} align=${POINTCLOUD_ALIGN} backend=${POINTCLOUD_ICP_BACKEND} kdtree_workers=${POINTCLOUD_KDTREE_WORKERS} workers=${POINTCLOUD_WORKERS}
  loger_forward : window=${LOGER_WINDOW_SIZE} overlap=${LOGER_OVERLAP_SIZE} reset_every=${LOGER_RESET_EVERY} se3=${LOGER_SE3} sim3=${LOGER_SIM3} disable_compile=${LOGER_DISABLE_COMPILE}
  input_preproc : ${LOGER_INPUT_PREPROCESS}
  recover_focal : ${RECOVER_FOCAL} threshold=${FOCAL_MASK_THRESHOLD} downsample=${FOCAL_DOWNSAMPLE_H}x${FOCAL_DOWNSAMPLE_W}
========================================

EOF

"${PYTHON_CMD[@]}" "${REPO_DIR}/scripts/eval/eval_loger.py" "${ARGS[@]}" "$@"
