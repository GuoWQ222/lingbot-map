#!/usr/bin/env bash
set -euo pipefail

START_SHARD="${START_SHARD:?set START_SHARD to this node first global shard index}"
NGPU="${NGPU:?set NGPU to the number of GPUs on this node}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/cpfs/user/guowenqi/miniconda3/envs/lingbot-map/bin/python}"
CHECKPOINT="${CHECKPOINT:-${REPO_DIR}/runs/manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/checkpoint_step_00100000.pt}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/runs/manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/eval/checkpoint_step_00100000_full_val_moving_allpoints_36shard}"
TOTAL_SHARDS="${TOTAL_SHARDS:-36}"

cd "${REPO_DIR}"

mkdir -p "${OUT_ROOT}/logs"

echo "[eval-36shard] repo=${REPO_DIR}"
echo "[eval-36shard] checkpoint=${CHECKPOINT}"
echo "[eval-36shard] output=${OUT_ROOT}"
echo "[eval-36shard] start_shard=${START_SHARD} ngpu=${NGPU} total_shards=${TOTAL_SHARDS}"

for ((local_gpu=0; local_gpu<NGPU; local_gpu++)); do
  shard=$((START_SHARD + local_gpu))
  if (( shard >= TOTAL_SHARDS )); then
    echo "[eval-36shard] skip local_gpu=${local_gpu}, shard=${shard} >= total_shards=${TOTAL_SHARDS}"
    continue
  fi

  shard_out="${OUT_ROOT}/shard_${shard}"
  log_path="${OUT_ROOT}/logs/shard_${shard}.log"
  mkdir -p "${shard_out}"

  echo "[eval-36shard] launch shard=${shard}/${TOTAL_SHARDS} on local_gpu=${local_gpu}, log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${local_gpu}"
    export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    "${PYTHON_BIN}" "${REPO_DIR}/scripts/eval/eval.py" \
      --checkpoint "${CHECKPOINT}" \
      --output_dir "${shard_out}" \
      --split val \
      --eval_strategy manip_track \
      --eval_surround_camera_name surround_cam_moving \
      --eval_num_frames 64 \
      --eval_shard_count "${TOTAL_SHARDS}" \
      --eval_shard_index "${shard}" \
      --num_workers 1 \
      --print_every 1 \
      --per_scene_csv \
      --pointcloud_metrics \
      --pointcloud_align pi3_icp \
      --pointcloud_icp_backend open3d \
      --pointcloud_max_points 0 \
      --pointcloud_workers 0 \
      --pointcloud_kdtree_workers -1
  ) > "${log_path}" 2>&1 &
done

wait
echo "[eval-36shard] node done: shards ${START_SHARD}..$((START_SHARD + NGPU - 1))"
