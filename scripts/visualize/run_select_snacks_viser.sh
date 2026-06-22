#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_BIN="/cpfs/user/guowenqi/miniconda3/condabin/conda"
ENV_NAME="lingbot-map"

IMAGE_FOLDER="/oss-guowenqi/Select_snacks_and_weigh_them/set39-1_collector1_20250712/0000002/observation/cam_left_wrist/color_image/rgb"
CHECKPOINT="${REPO_DIR}/outputs/runs/manip_long_train_64gpu/checkpoint_step_00100000.pt"

PORT="${PORT:-8080}"
STRIDE="${STRIDE:-7}"
CONF_THRESHOLD="${CONF_THRESHOLD:-1.5}"
DOWNSAMPLE_FACTOR="${DOWNSAMPLE_FACTOR:-1}"
POINT_SIZE="${POINT_SIZE:-0.00001}"

exec "${CONDA_BIN}" run -n "${ENV_NAME}" python "${REPO_DIR}/demo.py" \
  --image_folder "${IMAGE_FOLDER}" \
  --model_path "${CHECKPOINT}" \
  --image_size 280 \
  --model_image_size 280 \
  --mode streaming \
  --stride "${STRIDE}" \
  --use_sdpa \
  --offload_to_cpu \
  --port "${PORT}" \
  --conf_threshold "${CONF_THRESHOLD}" \
  --downsample_factor "${DOWNSAMPLE_FACTOR}" \
  --point_size "${POINT_SIZE}"
