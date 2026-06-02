#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Single-node 8-GPU DDP launch. This script intentionally reuses
# train_64gpu.sh so the training/data defaults stay in one place.
export NNODES="${NNODES:-1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29508}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Single-node rendezvous should stay local. NCCL can still use GPU P2P.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/manip_long_train_8gpu}"

cat <<EOF
========================================
 LingBot-MAP single-node 8-GPU wrapper
========================================
  nnodes          : ${NNODES}
  nproc_per_node  : ${NPROC_PER_NODE}
  node_rank       : ${NODE_RANK}
  master          : ${MASTER_ADDR}:${MASTER_PORT}
  cuda_devices    : ${CUDA_VISIBLE_DEVICES}
  output_dir      : ${OUTPUT_DIR}
========================================
EOF

exec bash "${SCRIPT_DIR}/train_64gpu.sh" "$@"
