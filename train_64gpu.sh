#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 16 nodes x 4 RTX 4090 by default. Override NNODES/NPROC_PER_NODE/NODE_RANK
# when your DLC job exposes different names.
NNODES="${NNODES:-${DLC_WORKER_NUM:-${PAI_WORKER_NUM:-16}}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NODE_RANK="${NODE_RANK:-${DLC_WORKER_INDEX:-${PAI_WORKER_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-${PAI_MASTER_ADDR:-}}}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ -z "${MASTER_ADDR}" ]]; then
  if [[ "${NNODES}" == "1" ]]; then
    MASTER_ADDR="127.0.0.1"
  else
    cat >&2 <<'EOF'
[error] MASTER_ADDR is empty. Set it to rank-0 node's intranet IP/hostname.
        In DLC/PAI this is often injected by the job scheduler; otherwise pass
        MASTER_ADDR=<node0-ip> NODE_RANK=<0..15> on every node.
EOF
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NCCL/RDMA defaults for PAI Lingjun-style multi-node jobs. Tune the interface
# names in the job environment if your cluster uses a specific NIC name.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0,veth}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

CONDA_BIN="${CONDA_BIN:-/cpfs/user/guowenqi/miniconda3/condabin/conda}"
CONDA_ENV="${CONDA_ENV:-lingbot-map}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=("${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV}" python)
fi

DATA_ROOT_LONG3="${DATA_ROOT_LONG3:-${DATA_ROOT_100K:-/oss-guowenqi/Manip_long3/data}}"
DATA_ROOT_LONG4="${DATA_ROOT_LONG4:-${DATA_ROOT_20K:-/oss-guowenqi/Manip_long4/data}}"
DATA_ROOT_LONG5="${DATA_ROOT_LONG5:-/oss-guowenqi/Manip_long5/data}"
OSS_URI_LONG3="${OSS_URI_LONG3:-oss://pjlab-bjpai-sim/guowenqi/Manip_long3/data}"
OSS_URI_LONG4="${OSS_URI_LONG4:-oss://pjlab-bjpai-sim/guowenqi/Manip_long4/data}"
OSS_URI_LONG5="${OSS_URI_LONG5:-oss://pjlab-bjpai-sim/guowenqi/Manip_long5/data}"
OSSUTIL_BIN="${OSSUTIL_BIN:-/cpfs/user/guowenqi/ossutil/ossutil}"
OSSUTIL_CONFIG="${OSSUTIL_CONFIG:-/cpfs/user/guowenqi/ossutil/.ossutilconfig}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/lingbot-map.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/manip_long_train_64gpu}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_DIR}/manip_trajectory_manifest.txt}"
WRITE_MANIFEST="${WRITE_MANIFEST:-}"

EPOCHS="${EPOCHS:-150}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_SCENES="${MAX_SCENES:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.02}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-5e-5}"
MIN_LR="${MIN_LR:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
CLIP_LEN="${CLIP_LEN:-64}"
SAMPLES_PER_SCENE="${SAMPLES_PER_SCENE:-1}"
SEQUENCE_MODE="${SEQUENCE_MODE:-manip_4d_mixed}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-random_interval}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
RANDOM_STRIDE_MIN="${RANDOM_STRIDE_MIN:-10}"
RANDOM_STRIDE_MAX="${RANDOM_STRIDE_MAX:-60}"
RANDOM_INTERVAL_START="${RANDOM_INTERVAL_START:-first}"
MAX_SAMPLE_FRAMES="${MAX_SAMPLE_FRAMES:-64}"
MIN_SAMPLE_FRAMES="${MIN_SAMPLE_FRAMES:-16}"
WRIST_CAMERA_PREFIX="${WRIST_CAMERA_PREFIX:-realsense}"
STATIC_CAMERA_PREFIX="${STATIC_CAMERA_PREFIX:-surround}"
M_STRIDE_MIN="${M_STRIDE_MIN:-8}"
M_STRIDE_MAX="${M_STRIDE_MAX:-24}"
S_VIEWS_MIN="${S_VIEWS_MIN:-4}"
S_VIEWS_MAX="${S_VIEWS_MAX:-6}"
M_NUM_VIEWS="${M_NUM_VIEWS:-4}"
M_NUM_TIMES="${M_NUM_TIMES:-0}"
M_VIEWS_MIN="${M_VIEWS_MIN:-3}"
M_VIEWS_MAX="${M_VIEWS_MAX:-6}"
MODE_WEIGHTS_INITIAL="${MODE_WEIGHTS_INITIAL:-S=0.0,W=1.0,M=0.0}"
MODE_WEIGHTS_FINAL="${MODE_WEIGHTS_FINAL:-S=0.0,W=1.0,M=0.0}"
MODE_WARMUP_START="${MODE_WARMUP_START:-2000}"
MODE_WARMUP_END="${MODE_WARMUP_END:-8000}"
T_STRIDE_MIN="${T_STRIDE_MIN:-15}"
T_STRIDE_MAX="${T_STRIDE_MAX:-60}"
LONG5_ROOT_MARKER="${LONG5_ROOT_MARKER:-Manip_long5}"
COLOR_JITTER_STRENGTH="${COLOR_JITTER_STRENGTH:-0.2}"
COLOR_JITTER_PROB="${COLOR_JITTER_PROB:-0.5}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
DEPTH_SCALE="${DEPTH_SCALE:-0}"
USE_MASK="${USE_MASK:-1}"
NUM_SCALE_FRAMES="${NUM_SCALE_FRAMES:-8}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-1}"
KV_CACHE_SLIDING_WINDOW="${KV_CACHE_SLIDING_WINDOW:-64}"
DEPTH_FRAMES_CHUNK_SIZE="${DEPTH_FRAMES_CHUNK_SIZE:-2}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
LOG_EVERY="${LOG_EVERY:-10}"
PRINT_INPUT_EVERY="${PRINT_INPUT_EVERY:-0}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
VAL_EVERY="${VAL_EVERY:-10000}"
RESUME="${RESUME:-}"

MIXTURE_CURRICULUM="${MIXTURE_CURRICULUM:-1}"
MIXTURE_P_MANIP_START="${MIXTURE_P_MANIP_START:-0.40}"
MIXTURE_P_MANIP_END="${MIXTURE_P_MANIP_END:-0.90}"
MIXTURE_WARMUP_START="${MIXTURE_WARMUP_START:-2000}"
MIXTURE_WARMUP_END="${MIXTURE_WARMUP_END:-15000}"

DL3DV_ROOT="${DL3DV_ROOT:-/cpfs/shared/landmark/renkerui/data/dl3dv}"
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/shared/smartbot/renkerui/data/scannetppv2}"
TARTANAIR_ROOT="${TARTANAIR_ROOT:-/cpfs/shared/landmark/renkerui/data/tartanair}"
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/shared/smartbot/renkerui/data/dynamic_replica}"
MAPFREE_ROOT="${MAPFREE_ROOT:-/cpfs/shared/landmark/renkerui/data/mapfree}"

ARGS=(
  --data_roots "${DATA_ROOT_LONG3}" "${DATA_ROOT_LONG4}" "${DATA_ROOT_LONG5}"
  --oss_uri_roots "${OSS_URI_LONG3},${OSS_URI_LONG4},${OSS_URI_LONG5}"
  --ossutil_bin "${OSSUTIL_BIN}"
  --ossutil_config "${OSSUTIL_CONFIG}"
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --scene_manifest "${SCENE_MANIFEST}"
  --epochs "${EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --max_scenes "${MAX_SCENES}"
  --val_fraction "${VAL_FRACTION}"
  --batch_size "${BATCH_SIZE}"
  --accum_steps "${ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --lr "${LR}"
  --min_lr "${MIN_LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --warmup_ratio "${WARMUP_RATIO}"
  --warmup_steps "${WARMUP_STEPS}"
  --clip_len "${CLIP_LEN}"
  --samples_per_scene "${SAMPLES_PER_SCENE}"
  --sequence_mode "${SEQUENCE_MODE}"
  --sample_strategy "${SAMPLE_STRATEGY}"
  --frame_stride "${FRAME_STRIDE}"
  --random_stride_min "${RANDOM_STRIDE_MIN}"
  --random_stride_max "${RANDOM_STRIDE_MAX}"
  --random_interval_start "${RANDOM_INTERVAL_START}"
  --max_sample_frames "${MAX_SAMPLE_FRAMES}"
  --min_sample_frames "${MIN_SAMPLE_FRAMES}"
  --wrist_camera_prefix "${WRIST_CAMERA_PREFIX}"
  --static_camera_prefix "${STATIC_CAMERA_PREFIX}"
  --m_stride_min "${M_STRIDE_MIN}"
  --m_stride_max "${M_STRIDE_MAX}"
  --s_views_min "${S_VIEWS_MIN}"
  --s_views_max "${S_VIEWS_MAX}"
  --m_num_views "${M_NUM_VIEWS}"
  --m_num_times "${M_NUM_TIMES}"
  --m_views_min "${M_VIEWS_MIN}"
  --m_views_max "${M_VIEWS_MAX}"
  --mode_weights_initial "${MODE_WEIGHTS_INITIAL}"
  --mode_weights_final "${MODE_WEIGHTS_FINAL}"
  --mode_warmup_start "${MODE_WARMUP_START}"
  --mode_warmup_end "${MODE_WARMUP_END}"
  --t_stride_min "${T_STRIDE_MIN}"
  --t_stride_max "${T_STRIDE_MAX}"
  --long5_root_marker "${LONG5_ROOT_MARKER}"
  --color_jitter_strength "${COLOR_JITTER_STRENGTH}"
  --color_jitter_prob "${COLOR_JITTER_PROB}"
  --image_size "${IMAGE_SIZE}"
  --depth_scale "${DEPTH_SCALE}"
  --num_scale_frames "${NUM_SCALE_FRAMES}"
  --num_frame_per_block "${NUM_FRAME_PER_BLOCK}"
  --kv_cache_sliding_window "${KV_CACHE_SLIDING_WINDOW}"
  --depth_frames_chunk_size "${DEPTH_FRAMES_CHUNK_SIZE}"
  --limit_train_batches "${LIMIT_TRAIN_BATCHES}"
  --limit_val_batches "${LIMIT_VAL_BATCHES}"
  --log_every "${LOG_EVERY}"
  --print_input_every "${PRINT_INPUT_EVERY}"
  --save_every "${SAVE_EVERY}"
  --val_every "${VAL_EVERY}"
  --mixture_p_manip_start "${MIXTURE_P_MANIP_START}"
  --mixture_p_manip_end "${MIXTURE_P_MANIP_END}"
  --mixture_warmup_start "${MIXTURE_WARMUP_START}"
  --mixture_warmup_end "${MIXTURE_WARMUP_END}"
  --dl3dv_root "${DL3DV_ROOT}"
  --scannetpp_root "${SCANNETPP_ROOT}"
  --tartanair_root "${TARTANAIR_ROOT}"
  --dynamic_replica_root "${DYNAMIC_REPLICA_ROOT}"
  --mapfree_root "${MAPFREE_ROOT}"
  --freeze_dino_patch_embed
  --canonicalize_first_frame
  --tensorboard
)

if [[ -n "${WRITE_MANIFEST}" ]]; then ARGS+=(--write_manifest "${WRITE_MANIFEST}"); fi
if [[ -n "${RESUME}" ]]; then ARGS+=(--resume "${RESUME}"); fi
if [[ "${USE_MASK}" == "1" ]]; then ARGS+=(--use_mask); fi
if [[ "${MIXTURE_CURRICULUM}" == "1" ]]; then ARGS+=(--mixture_curriculum); else ARGS+=(--no-mixture_curriculum); fi

cat <<EOF
========================================
 LingBot-MAP 64-GPU DDP Launch
========================================
  nnodes          : ${NNODES}
  nproc_per_node  : ${NPROC_PER_NODE}
  node_rank       : ${NODE_RANK}
  master          : ${MASTER_ADDR}:${MASTER_PORT}
  cuda_devices    : ${CUDA_VISIBLE_DEVICES}
  global_batch    : $(( BATCH_SIZE * ACCUM_STEPS * NNODES * NPROC_PER_NODE ))
  output_dir      : ${OUTPUT_DIR}
========================================
EOF

exec "${PYTHON_CMD[@]}" -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${SCRIPT_DIR}/train_multinode.py" \
  "${ARGS[@]}" \
  "$@"
