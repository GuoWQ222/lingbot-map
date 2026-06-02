#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Exact full-eval launcher: same sampling/metrics as the official eval.sh run,
# but schedules checkpoints in parallel across GPUs. On the 49GB 4090 nodes here,
# two eval processes per GPU fit comfortably (~12.7GB each in the current config).
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRIPT_DIR}/runs/manip_long_train_64gpu}"
CHECKPOINT_GLOB="${CHECKPOINT_GLOB:-checkpoint_*.pt}"
TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON:-${CHECKPOINT_DIR}/args.json}"
EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES:-64}"
EVAL_SURROUND_CAMERA_NAME="${EVAL_SURROUND_CAMERA_NAME:-}"  # empty = all 6 surround cams
EVAL_DEPTH_FRAMES_CHUNK_SIZE="${EVAL_DEPTH_FRAMES_CHUNK_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PER_SCENE_CSV="${PER_SCENE_CSV:-1}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
PRINT_EVERY="${PRINT_EVERY:-100}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
EVAL_STRATEGY="${EVAL_STRATEGY:-manip_track}"
DEPTH_ALIGN="${DEPTH_ALIGN:-median}"
GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION:-none}"
CAMERA_ALIGN="${CAMERA_ALIGN:-sim3}"
POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN:-pi3_icp}"
SPLIT="${SPLIT:-val}"
MAX_SCENES_EVAL="${MAX_SCENES_EVAL:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-0}"
CONDA_BIN="${CONDA_BIN:-/cpfs/user/guowenqi/miniconda3/condabin/conda}"
CONDA_ENV="${CONDA_ENV:-lingbot-map}"

CAM_TAG="${EVAL_SURROUND_CAMERA_NAME:-all6}"
CHUNK_TAG="c${EVAL_DEPTH_FRAMES_CHUNK_SIZE}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${CHECKPOINT_DIR}/eval_full_n${EVAL_NUM_FRAMES}_${CAM_TAG}_${CHUNK_TAG}}"
SUMMARY_CSV="${SUMMARY_CSV:-${OUTPUT_ROOT}/summary.csv}"

CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-${CUDA_VISIBLE_DEVICES:-0}}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICE_LIST}"
if (( ${#GPU_IDS[@]} == 0 )); then
  GPU_IDS=(0)
fi
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-$(( ${#GPU_IDS[@]} * PROCS_PER_GPU ))}"
if (( MAX_PARALLEL < 1 )); then
  MAX_PARALLEL=1
fi

declare -a CHECKPOINT_LIST=()
if [[ -n "${CHECKPOINTS:-}" ]]; then
  read -r -a CHECKPOINT_LIST <<< "${CHECKPOINTS}"
else
  shopt -s nullglob
  CHECKPOINT_LIST=("${CHECKPOINT_DIR}"/${CHECKPOINT_GLOB})
  shopt -u nullglob
fi
if [[ -n "${EXTRA_CHECKPOINTS:-}" ]]; then
  read -r -a EXTRA_LIST <<< "${EXTRA_CHECKPOINTS}"
  CHECKPOINT_LIST+=("${EXTRA_LIST[@]}")
fi
if (( ${#CHECKPOINT_LIST[@]} == 0 )); then
  echo "[eval_full_parallel.sh] no checkpoints matched" >&2
  exit 2
fi

IFS=$'\n' CHECKPOINT_LIST=($(printf '%s\n' "${CHECKPOINT_LIST[@]}" | sort -V))
unset IFS

mkdir -p "${OUTPUT_ROOT}/logs"

sanitize_name() {
  local path="$1"
  local stem
  stem="$(basename "${path}")"
  stem="${stem%.pt}"
  printf '%s' "${stem//[^A-Za-z0-9_.-]/_}"
}

declare -a ACTIVE_PIDS=()
declare -a METRICS_FILES=()

wait_active() {
  local status=0
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  ACTIVE_PIDS=()
  return "${status}"
}

for idx in "${!CHECKPOINT_LIST[@]}"; do
  ckpt="${CHECKPOINT_LIST[$idx]}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[eval_full_parallel.sh] checkpoint not found: ${ckpt}" >&2
    exit 2
  fi
  name="$(sanitize_name "${ckpt}")"
  out_dir="${OUTPUT_ROOT}/${name}"
  log_path="${OUTPUT_ROOT}/logs/${name}.log"
  METRICS_FILES+=("${out_dir}/metrics.json")

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/metrics.json" ]]; then
    echo "[eval_full_parallel.sh] skip existing ${name}"
    continue
  fi

  gpu="${GPU_IDS[$(( idx % ${#GPU_IDS[@]} ))]}"
  cat <<EOF
[eval_full_parallel.sh] launch ${name}
  gpu        : ${gpu}
  checkpoint : ${ckpt}
  output     : ${out_dir}
  log        : ${log_path}
  depth_chunk: ${EVAL_DEPTH_FRAMES_CHUNK_SIZE}
EOF

  CUDA_VISIBLE_DEVICES="${gpu}" \
  TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON}" \
  CHECKPOINT="${ckpt}" \
  OUTPUT_DIR="${out_dir}" \
  EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES}" \
  EVAL_SURROUND_CAMERA_NAME="${EVAL_SURROUND_CAMERA_NAME}" \
  EVAL_DEPTH_FRAMES_CHUNK_SIZE="${EVAL_DEPTH_FRAMES_CHUNK_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  PER_SCENE_CSV="${PER_SCENE_CSV}" \
  SAVE_PREDICTIONS="${SAVE_PREDICTIONS}" \
  PRINT_EVERY="${PRINT_EVERY}" \
  SKIP_EXISTING=0 \
  EVAL_STRATEGY="${EVAL_STRATEGY}" \
  DEPTH_ALIGN="${DEPTH_ALIGN}" \
  GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION}" \
  CAMERA_ALIGN="${CAMERA_ALIGN}" \
  POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN}" \
  SPLIT="${SPLIT}" \
  MAX_SCENES_EVAL="${MAX_SCENES_EVAL}" \
  IMAGE_SIZE="${IMAGE_SIZE}" \
  bash "${SCRIPT_DIR}/eval.sh" >"${log_path}" 2>&1 &
  ACTIVE_PIDS+=("$!")

  if (( ${#ACTIVE_PIDS[@]} >= MAX_PARALLEL )); then
    wait_active
  fi
done

if (( ${#ACTIVE_PIDS[@]} > 0 )); then
  wait_active
fi

"${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python - "${SUMMARY_CSV}" "${METRICS_FILES[@]}" <<'PY_SUMMARY'
import csv
import json
import math
import re
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
metric_paths = [Path(p) for p in sys.argv[2:]]
base_fields = ["checkpoint", "checkpoint_name", "step", "split", "eval_strategy", "eval_num_frames", "depth_frames_chunk_size"]
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
        print(f"[eval_full_parallel.sh][warn] missing metrics: {metrics_path}")
        continue
    data = json.loads(metrics_path.read_text())
    ckpt = str(data.get("checkpoint", ""))
    ckpt_name = Path(ckpt).name if ckpt else metrics_path.parent.name
    step_match = re.search(r"checkpoint_step_(\d+)", ckpt_name)
    row = {
        "checkpoint": ckpt,
        "checkpoint_name": ckpt_name,
        "step": int(step_match.group(1)) if step_match else "",
        "split": data.get("split", ""),
        "eval_strategy": data.get("eval_strategy", ""),
        "eval_num_frames": data.get("eval_num_frames", ""),
        "depth_frames_chunk_size": data.get("depth_frames_chunk_size", ""),
    }
    modes = data.get("modes", {})
    if isinstance(modes, dict):
        for mode_name, mode_metrics in modes.items():
            flatten(str(mode_name), mode_metrics, row)
    rows.append(row)

summary_csv.parent.mkdir(parents=True, exist_ok=True)
fieldnames = base_fields + sorted(all_fields.difference(base_fields))
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"[eval_full_parallel.sh] wrote {summary_csv} ({len(rows)} rows)")
PY_SUMMARY

echo "[eval_full_parallel.sh] done"
