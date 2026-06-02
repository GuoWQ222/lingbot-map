#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Single-node 4-GPU speed test.
# Default preset keeps the main 64-GPU training shape, but limits steps.
# Use SPEED_PRESET=light for a smaller sanity test.
SPEED_PRESET="${SPEED_PRESET:-full}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

export NNODES=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NODE_RANK=0
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_IGNORE_DISABLED_P2P="${NCCL_IGNORE_DISABLED_P2P:-1}"

export MAX_STEPS="${MAX_STEPS:-5}"
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-5}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0}"
export LOG_EVERY="${LOG_EVERY:-1}"
export PRINT_INPUT_EVERY="${PRINT_INPUT_EVERY:-1}"
export SAVE_EVERY="${SAVE_EVERY:-0}"
export VAL_EVERY="${VAL_EVERY:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/speed_4gpu_${SPEED_PRESET}_${STAMP}}"

case "${SPEED_PRESET}" in
  full)
    # Same expensive shape as the 64-GPU run, just on one node.
    ;;
  light)
    export MAX_SAMPLE_FRAMES="${MAX_SAMPLE_FRAMES:-32}"
    export CLIP_LEN="${CLIP_LEN:-32}"
    export IMAGE_SIZE="${IMAGE_SIZE:-224}"
    export NUM_SCALE_FRAMES="${NUM_SCALE_FRAMES:-4}"
    export DEPTH_FRAMES_CHUNK_SIZE="${DEPTH_FRAMES_CHUNK_SIZE:-1}"
    ;;
  *)
    echo "[error] SPEED_PRESET must be 'full' or 'light', got: ${SPEED_PRESET}" >&2
    exit 2
    ;;
esac

LOG_FILE="${OUTPUT_DIR}/speed_test.log"
mkdir -p "${OUTPUT_DIR}"

cat <<EOF
========================================
 LingBot-MAP single-node 4-GPU speed test
========================================
  preset       : ${SPEED_PRESET}
  gpus         : ${CUDA_VISIBLE_DEVICES}
  max_steps    : ${MAX_STEPS}
  log_every    : ${LOG_EVERY}
  output_dir   : ${OUTPUT_DIR}
  log_file     : ${LOG_FILE}
========================================
EOF

exec bash ./train_64gpu.sh 2>&1 | tee "${LOG_FILE}"
