#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
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

PI3_REPO="${PI3_REPO:-/cpfs/user/guowenqi/Pi3}"
PI3_CHECKPOINT="${PI3_CHECKPOINT:-${PI3_REPO}/ckpts/model.safetensors}"
if [[ ! -f "${PI3_CHECKPOINT}" ]]; then
  PI3_CHECKPOINT=""
fi
PI3_PRETRAINED="${PI3_PRETRAINED:-yyfz233/Pi3}"

TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-${SCRIPT_DIR}/runs/manip_long_train_64gpu/args.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/eval/pi3}"
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
DEPTH_ALIGN="${DEPTH_ALIGN:-pi3_scale_shift}"
CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION:-none}"
IMAGE_SIZE="${IMAGE_SIZE:-518}"
POINTCLOUD_METRICS="${POINTCLOUD_METRICS:-1}"
POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-100000}"
POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD:-0.1}"
POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS:-30}"
RECOVER_FOCAL="${RECOVER_FOCAL:-1}"
FOCAL_MASK_THRESHOLD="${FOCAL_MASK_THRESHOLD:-0.1}"
FOCAL_DOWNSAMPLE_H="${FOCAL_DOWNSAMPLE_H:-64}"
FOCAL_DOWNSAMPLE_W="${FOCAL_DOWNSAMPLE_W:-64}"

ARGS=(
  --train_args_json "${TRAIN_ARGS_JSON}"
  --pi3_repo "${PI3_REPO}"
  --pi3_pretrained_model_name_or_path "${PI3_PRETRAINED}"
  --output_dir "${OUTPUT_DIR}"
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
  --depth_align "${DEPTH_ALIGN}"
  --camera_align "${CAMERA_ALIGN}"
  --geometry_normalization "${GEOMETRY_NORMALIZATION}"
  --image_size "${IMAGE_SIZE}"
  --pointcloud_max_points "${POINTCLOUD_MAX_POINTS}"
  --pointcloud_align "${POINTCLOUD_ALIGN}"
  --pointcloud_icp_threshold "${POINTCLOUD_ICP_THRESHOLD}"
  --pointcloud_icp_max_iterations "${POINTCLOUD_ICP_MAX_ITERATIONS}"
  --focal_mask_threshold "${FOCAL_MASK_THRESHOLD}"
  --focal_downsample_size "${FOCAL_DOWNSAMPLE_H}" "${FOCAL_DOWNSAMPLE_W}"
)

if [[ -n "${PI3_CHECKPOINT}" ]]; then
  ARGS+=(--pi3_checkpoint "${PI3_CHECKPOINT}")
fi
if [[ "${POINTCLOUD_METRICS}" == "1" ]]; then
  ARGS+=(--pointcloud_metrics)
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
 Pi3 on LingBot-MAP Manip Evaluation
========================================
[eval_pi3]
  pi3_repo      : ${PI3_REPO}
  pi3_checkpoint: ${PI3_CHECKPOINT:-<from_pretrained:${PI3_PRETRAINED}>}
  train_args    : ${TRAIN_ARGS_JSON}
  split         : ${SPLIT}
  device        : ${DEVICE}
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  output_dir    : ${OUTPUT_DIR}
  eval_strategy : ${EVAL_STRATEGY}
  eval_n_frames : ${EVAL_NUM_FRAMES}
  wrist_camera  : ${EVAL_WRIST_CAMERA_NAME}
  surround_cam  : ${EVAL_SURROUND_CAMERA_NAME:-<all_6_surround>}
  depth_align   : ${DEPTH_ALIGN}
  camera_align  : ${CAMERA_ALIGN}
  geom_norm     : ${GEOMETRY_NORMALIZATION}
  image_size    : ${IMAGE_SIZE} (0 = use train_args_json default)
  pointcloud    : ${POINTCLOUD_METRICS} max_points=${POINTCLOUD_MAX_POINTS} align=${POINTCLOUD_ALIGN}
  recover_focal : ${RECOVER_FOCAL} threshold=${FOCAL_MASK_THRESHOLD} downsample=${FOCAL_DOWNSAMPLE_H}x${FOCAL_DOWNSAMPLE_W}
========================================

EOF

"${PYTHON_CMD[@]}" eval_pi3.py "${ARGS[@]}"
