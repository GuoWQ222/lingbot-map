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

EVAL_BACKEND="${EVAL_BACKEND:-${MODEL_BACKEND:-lingbot}}"
case "${EVAL_BACKEND}" in
  lingbot|lingbot-map)
    ;;
  ttt3r|TTT3R)
    exec bash "${SCRIPT_DIR}/eval_ttt3r.sh" "$@"
    ;;
  *)
    echo "[eval.sh] unknown EVAL_BACKEND=${EVAL_BACKEND} (expected lingbot or ttt3r)" >&2
    exit 2
    ;;
esac

# Batch mode defaults to the 64-GPU run requested here. To evaluate one
# checkpoint, pass CHECKPOINT=/path/to/checkpoint.pt bash eval.sh.
CHECKPOINT="${CHECKPOINT:-}"
CHECKPOINTS="${CHECKPOINTS:-}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/runs/manip_long_train_64gpu}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-checkpoint_*.pt}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-${CHECKPOINT_DIR}/eval}"
EVAL_SUMMARY_CSV="${EVAL_SUMMARY_CSV:-${EVAL_OUTPUT_ROOT}/summary.csv}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
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
GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION:-none}"
CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
DEPTH_ALIGN="${DEPTH_ALIGN:-pi3_scale_shift}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
EVAL_DEPTH_FRAMES_CHUNK_SIZE="${EVAL_DEPTH_FRAMES_CHUNK_SIZE:-0}"
POINTCLOUD_METRICS="${POINTCLOUD_METRICS:-1}"
POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS:-100000}"
POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD:-0.1}"
POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS:-30}"

declare -a CHECKPOINT_LIST=()
if [[ -n "${CHECKPOINT}" ]]; then
  CHECKPOINT_LIST=("${CHECKPOINT}")
elif [[ -n "${CHECKPOINTS}" ]]; then
  # Space-separated explicit checkpoint list. Paths with spaces are not supported.
  read -r -a CHECKPOINT_LIST <<< "${CHECKPOINTS}"
else
  shopt -s nullglob
  CHECKPOINT_LIST=("${CHECKPOINT_DIR}"/${CHECKPOINT_GLOB})
  shopt -u nullglob
fi

if (( ${#CHECKPOINT_LIST[@]} == 0 )); then
  echo "[eval.sh] no checkpoints matched: ${CHECKPOINT_DIR}/${CHECKPOINT_GLOB}" >&2
  exit 2
fi

IFS=$'\n' CHECKPOINT_LIST=($(printf '%s\n' "${CHECKPOINT_LIST[@]}" | sort -V))
unset IFS

set -m
EVAL_PID=""
cleanup() {
  trap - INT TERM EXIT
  if [[ -n "${EVAL_PID}" ]] && kill -0 "${EVAL_PID}" 2>/dev/null; then
    echo
    echo "[eval.sh] caught signal, terminating eval (pgid=${EVAL_PID})..."
    kill -TERM "-${EVAL_PID}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "${EVAL_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "-${EVAL_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup INT TERM

build_args() {
  local ckpt="$1"
  local out_dir="$2"
  ARGS=(
    --checkpoint "${ckpt}"
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
    --geometry_normalization "${GEOMETRY_NORMALIZATION}"
    --camera_align "${CAMERA_ALIGN}"
    --depth_align "${DEPTH_ALIGN}"
    --image_size "${IMAGE_SIZE}"
    --depth_frames_chunk_size "${EVAL_DEPTH_FRAMES_CHUNK_SIZE}"
    --pointcloud_max_points "${POINTCLOUD_MAX_POINTS}"
    --pointcloud_align "${POINTCLOUD_ALIGN}"
    --pointcloud_icp_threshold "${POINTCLOUD_ICP_THRESHOLD}"
    --pointcloud_icp_max_iterations "${POINTCLOUD_ICP_MAX_ITERATIONS}"
  )
  if [[ "${POINTCLOUD_METRICS}" == "1" ]]; then
    ARGS+=(--pointcloud_metrics)
  fi
  if [[ -n "${TRAIN_ARGS_JSON}" ]]; then
    ARGS+=(--train_args_json "${TRAIN_ARGS_JSON}")
  fi
  if [[ -n "${out_dir}" ]]; then
    ARGS+=(--output_dir "${out_dir}")
  fi
  if [[ "${PER_SCENE_CSV}" == "1" ]]; then
    ARGS+=(--per_scene_csv)
  fi
  if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
    ARGS+=(--save_predictions)
  fi
}

write_summary_csv() {
  local summary_csv="$1"
  shift
  if (( $# == 0 )); then
    return 0
  fi
  mkdir -p "$(dirname "${summary_csv}")"
  "${PYTHON_CMD[@]}" - "${summary_csv}" "$@" <<'PY_SUMMARY'
import csv
import json
import math
import re
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
metric_paths = [Path(p) for p in sys.argv[2:]]

base_fields = ["checkpoint", "checkpoint_name", "step", "split", "eval_strategy", "eval_num_frames", "geometry_normalization", "depth_align", "camera_align", "pointcloud_align"]
rows = []
all_fields = set(base_fields)

def scalar_to_cell(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return json.dumps(value, sort_keys=True)

def flatten(prefix, value, row):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten(child_prefix, child, row)
    else:
        row[prefix] = scalar_to_cell(value)
        all_fields.add(prefix)

for metrics_path in metric_paths:
    if not metrics_path.is_file():
        continue
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ckpt = str(data.get("checkpoint", ""))
    ckpt_name = Path(ckpt).name if ckpt else metrics_path.parent.name
    step_match = re.search(r"checkpoint_step_(\d+)", ckpt_name)
    epoch_match = re.search(r"checkpoint_epoch_(\d+)", ckpt_name)
    row = {
        "checkpoint": ckpt,
        "checkpoint_name": ckpt_name,
        "step": int(step_match.group(1)) if step_match else (f"epoch_{epoch_match.group(1)}" if epoch_match else ""),
        "split": data.get("split", ""),
        "eval_strategy": data.get("eval_strategy", ""),
        "eval_num_frames": data.get("eval_num_frames", ""),
        "geometry_normalization": data.get("geometry_normalization", ""),
        "depth_align": data.get("depth_align", ""),
        "camera_align": data.get("camera_align", ""),
        "pointcloud_align": data.get("pointcloud_align", ""),
    }
    modes = data.get("modes", {})
    if isinstance(modes, dict):
        for mode_name, mode_metrics in modes.items():
            flatten(str(mode_name), mode_metrics, row)
    rows.append(row)

extra_fields = sorted(all_fields.difference(base_fields))
fieldnames = base_fields + extra_fields
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
print(f"[eval.sh] wrote {summary_csv} ({len(rows)} rows)")
PY_SUMMARY
}

declare -a METRICS_FILES=()
TOTAL=${#CHECKPOINT_LIST[@]}
for idx in "${!CHECKPOINT_LIST[@]}"; do
  CKPT="${CHECKPOINT_LIST[$idx]}"
  if [[ ! -f "${CKPT}" ]]; then
    echo "[eval.sh] checkpoint not found: ${CKPT}" >&2
    exit 2
  fi

  CKPT_PARENT="$(dirname "${CKPT}")"
  CKPT_STEM="$(basename "${CKPT}")"
  CKPT_STEM="${CKPT_STEM%.pt}"

  if [[ -n "${OUTPUT_DIR}" ]]; then
    CKPT_OUTPUT_DIR="${OUTPUT_DIR}"
  elif [[ -z "${CHECKPOINT}" && -z "${CHECKPOINTS}" ]]; then
    CKPT_OUTPUT_DIR="${EVAL_OUTPUT_ROOT}/${CKPT_STEM}"
  else
    CKPT_OUTPUT_DIR="${CKPT_PARENT}/eval/${CKPT_STEM}"
  fi
  METRICS_PATH="${CKPT_OUTPUT_DIR}/metrics.json"
  METRICS_FILES+=("${METRICS_PATH}")

  cat <<EOF

========================================
 LingBot-MAP Manip Evaluation $((idx + 1))/${TOTAL}
========================================
[eval]
  checkpoint    : ${CKPT}
  split         : ${SPLIT}
  device        : ${DEVICE}
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  output_dir    : ${CKPT_OUTPUT_DIR}
  train_args    : ${TRAIN_ARGS_JSON:-<auto: <ckpt_parent>/args.json>}
  per_scene_csv : ${PER_SCENE_CSV}
  save_preds    : ${SAVE_PREDICTIONS}
  eval_strategy : ${EVAL_STRATEGY}
  eval_n_frames : ${EVAL_NUM_FRAMES} (linspace over each clip's full trajectory)
  wrist_camera  : ${EVAL_WRIST_CAMERA_NAME}
  surround_cam  : ${EVAL_SURROUND_CAMERA_NAME:-<all_6_surround>}
  geometry_norm: ${GEOMETRY_NORMALIZATION}
  camera_align : ${CAMERA_ALIGN}
  depth_align   : ${DEPTH_ALIGN} (Pi3-style sequence alignment)
  image_size    : ${IMAGE_SIZE} (0 = use train_args_json default)
  depth_chunk   : ${EVAL_DEPTH_FRAMES_CHUNK_SIZE} (0 = use train_args_json default)
  pointcloud   : ${POINTCLOUD_METRICS} max_points=${POINTCLOUD_MAX_POINTS} align=${POINTCLOUD_ALIGN} icp_threshold=${POINTCLOUD_ICP_THRESHOLD} icp_iters=${POINTCLOUD_ICP_MAX_ITERATIONS}
========================================

EOF

  if [[ "${SKIP_EXISTING}" == "1" && -f "${METRICS_PATH}" ]]; then
    echo "[eval.sh] skip existing metrics: ${METRICS_PATH}"
    continue
  fi

  build_args "${CKPT}" "${CKPT_OUTPUT_DIR}"
  "${PYTHON_CMD[@]}" eval.py "${ARGS[@]}" "$@" &
  EVAL_PID=$!
  wait "${EVAL_PID}"
  EXIT_CODE=$?
  EVAL_PID=""
  if (( EXIT_CODE != 0 )); then
    trap - INT TERM
    exit "${EXIT_CODE}"
  fi
done

trap - INT TERM
write_summary_csv "${EVAL_SUMMARY_CSV}" "${METRICS_FILES[@]}"
