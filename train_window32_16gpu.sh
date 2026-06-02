#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 2 nodes x 8 GPUs by default. Run this script on both nodes with the same
# MASTER_ADDR/MASTER_PORT and NODE_RANK=0 or 1.
NNODES="${NNODES:-${DLC_WORKER_NUM:-${PAI_WORKER_NUM:-2}}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-${DLC_WORKER_INDEX:-${PAI_WORKER_INDEX:-${PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX:-${RANK:-0}}}}}"
MASTER_ADDR="${MASTER_ADDR:-${DLC_MASTER_ADDR:-${PAI_MASTER_ADDR:-}}}"
MASTER_PORT="${WINDOW32_MASTER_PORT:-29532}"

if [[ -z "${MASTER_ADDR}" ]]; then
  if [[ "${NNODES}" == "1" ]]; then
    MASTER_ADDR="127.0.0.1"
  else
    cat >&2 <<EOF
[error] MASTER_ADDR is empty. Set it to rank-0 node intranet IP/hostname.
        For this 2-node launch, run on both nodes with:
        MASTER_ADDR=<node0-ip> NODE_RANK=<0-or-1>
EOF
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
OUTPUT_DIR="${WINDOW32_OUTPUT_DIR:-${SCRIPT_DIR}/runs/manip_long_train_window32_16gpu}"
SCENE_MANIFEST="${WINDOW32_SCENE_MANIFEST:-${OUTPUT_DIR}/manip_trajectory_manifest.txt}"
WRITE_MANIFEST="${WRITE_MANIFEST:-}"

EPOCHS="${EPOCHS:-150}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_SCENES="${MAX_SCENES:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.02}"
BATCH_SIZE="1"
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
VIEW_IDS="${VIEW_IDS:-}"
CAMERA_NAMES="${CAMERA_NAMES:-}"
SAMPLE_STRATEGY="${SAMPLE_STRATEGY:-random_interval}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
RANDOM_STRIDE_MIN="${RANDOM_STRIDE_MIN:-10}"
RANDOM_STRIDE_MAX="${RANDOM_STRIDE_MAX:-60}"
RANDOM_INTERVAL_START="${RANDOM_INTERVAL_START:-first}"
MAX_SAMPLE_FRAMES="${MAX_SAMPLE_FRAMES:-64}"
MIN_SAMPLE_FRAMES="${MIN_SAMPLE_FRAMES:-16}"
# manip_4d_mixed (W/S/M) curriculum
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
# All Manip data now defaults to single-camera trajectory random-interval:
#   - Manip_long3/4 scenes: mode W (realsense_*, stride RANDOM_STRIDE_MIN/MAX)
#   - Manip_long5  scenes: mode T (surround_cam_*, stride T_STRIDE_MIN/MAX) — auto-routed by path
# Mode S/M code is preserved for future curriculum experiments; set non-zero
# weights here to bring them back into the long3/4 mix.
MODE_WEIGHTS_INITIAL="${MODE_WEIGHTS_INITIAL:-S=0.0,W=1.0,M=0.0}"
MODE_WEIGHTS_FINAL="${MODE_WEIGHTS_FINAL:-S=0.0,W=1.0,M=0.0}"
MODE_WARMUP_START="${MODE_WARMUP_START:-2000}"
MODE_WARMUP_END="${MODE_WARMUP_END:-8000}"
# Mode T (Manip_long5 trajectory): single surround_cam, random start, per-step
# random stride. ~1500-2900 frame span at 30fps covers a full pick&place.
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
KV_CACHE_SLIDING_WINDOW="${KV_CACHE_SLIDING_WINDOW:-32}"
DEPTH_FRAMES_CHUNK_SIZE="${DEPTH_FRAMES_CHUNK_SIZE:-2}"
DEPTH_ACTIVATION_CHECKPOINT="${DEPTH_ACTIVATION_CHECKPOINT:-1}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
LOG_EVERY="${LOG_EVERY:-10}"
PRINT_INPUT_EVERY="${PRINT_INPUT_EVERY:-0}"
TENSORBOARD="${TENSORBOARD:-1}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-}"
TENSORBOARD_FLUSH_SECS="${TENSORBOARD_FLUSH_SECS:-30}"
TENSORBOARD_FLUSH_EVERY="${TENSORBOARD_FLUSH_EVERY:-10}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-1}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
VAL_EVERY="${VAL_EVERY:-10000}"
RESUME="${RESUME:-}"
CANONICALIZE_FIRST_FRAME="${CANONICALIZE_FIRST_FRAME:-1}"
FREEZE_DINO_PATCH_EMBED="${FREEZE_DINO_PATCH_EMBED:-1}"
FREEZE_AGGREGATOR="${FREEZE_AGGREGATOR:-0}"
FREEZE_CAMERA="${FREEZE_CAMERA:-0}"
FREEZE_DEPTH="${FREEZE_DEPTH:-0}"
FREEZE_POINT="${FREEZE_POINT:-0}"

# ---------- Cross-dataset curriculum (Manip vs externals) ----------
# When ON, the DataLoader uses CurriculumMixtureSampler:
#   p_manip(step) ramps linearly from MIXTURE_P_MANIP_START at
#   MIXTURE_WARMUP_START to MIXTURE_P_MANIP_END at MIXTURE_WARMUP_END.
# Each batch independently picks Manip (with prob p_manip) OR one external
# uniformly at random. With batch_size=1 this is exactly the "每步在 Manip
# 和 外部之间随机选择" sampling the user asked for.
# *_REPEAT is force-overridden to 1 when this is on.
MIXTURE_CURRICULUM="${MIXTURE_CURRICULUM:-1}"
MIXTURE_P_MANIP_START="${MIXTURE_P_MANIP_START:-0.50}"
MIXTURE_P_MANIP_END="${MIXTURE_P_MANIP_END:-0.90}"
MIXTURE_WARMUP_START="${MIXTURE_WARMUP_START:-2000}"
MIXTURE_WARMUP_END="${MIXTURE_WARMUP_END:-15000}"

# DL3DV mix-in (sampling follows base3d-clean/datasets/dl3dv.py).
# Enabled by default for the cross-dataset curriculum; clear DL3DV_ROOT to disable.
DL3DV_ROOT="${DL3DV_ROOT:-/cpfs/shared/landmark/renkerui/data/dl3dv}"
DL3DV_NUM_VIEWS="${DL3DV_NUM_VIEWS:-0}"           # 0 -> follow MAX_SAMPLE_FRAMES
DL3DV_MIN_VIEWS="${DL3DV_MIN_VIEWS:-0}"           # 0 -> follow MIN_SAMPLE_FRAMES (per-sample clip length is drawn in [min, num])
DL3DV_REPEAT="${DL3DV_REPEAT:-1}"                 # replicate DL3DV samples in ConcatDataset
DL3DV_VAL="${DL3DV_VAL:-0}"                       # 1 -> also mix DL3DV's test split into val
DL3DV_MIN_INTERVAL="${DL3DV_MIN_INTERVAL:-1}"
DL3DV_MAX_INTERVAL="${DL3DV_MAX_INTERVAL:-32}"
DL3DV_VIDEO_PROB="${DL3DV_VIDEO_PROB:-0.8}"
DL3DV_FIX_INTERVAL_PROB="${DL3DV_FIX_INTERVAL_PROB:-0.6}"
DL3DV_BLOCK_SHUFFLE="${DL3DV_BLOCK_SHUFFLE:-16}"

# ScanNet++ v2 mix-in (sampling follows base3d-clean/datasets/scannet.py — NOT
# scannetpp.py — by explicit request). The root must contain per-scene
# subdirs and a valid.json listing scene IDs.
# DISABLED BY DEFAULT — set SCANNETPP_ROOT=/shared/smartbot/renkerui/data/scannetppv2
# (or another valid path) to enable.
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/shared/smartbot/renkerui/data/scannetppv2}"
SCANNETPP_NUM_VIEWS="${SCANNETPP_NUM_VIEWS:-0}"        # 0 -> follow MAX_SAMPLE_FRAMES
SCANNETPP_MIN_VIEWS="${SCANNETPP_MIN_VIEWS:-0}"        # 0 -> follow MIN_SAMPLE_FRAMES (per-sample clip length drawn in [min, num])
SCANNETPP_REPEAT="${SCANNETPP_REPEAT:-1}"              # replicate samples in ConcatDataset
SCANNETPP_VAL="${SCANNETPP_VAL:-0}"                    # 1 -> also mix into val
# Sampling defaults mirror base3d-clean/datasets/scannet.py
SCANNETPP_MIN_INTERVAL="${SCANNETPP_MIN_INTERVAL:-1}"
SCANNETPP_MAX_INTERVAL="${SCANNETPP_MAX_INTERVAL:-30}"
SCANNETPP_VIDEO_PROB="${SCANNETPP_VIDEO_PROB:-0.6}"
SCANNETPP_FIX_INTERVAL_PROB="${SCANNETPP_FIX_INTERVAL_PROB:-0.6}"
SCANNETPP_BLOCK_SHUFFLE="${SCANNETPP_BLOCK_SHUFFLE:-16}"

# TartanAir mix-in (sampling follows base3d-clean/datasets/tartanair.py).
# Disabled by default. Set TARTANAIR_ROOT=/cpfs/shared/landmark/renkerui/data/tartanair
# (or any valid root containing rgb/<scene>/<Easy|Hard>/... + depth/...) to enable.
TARTANAIR_ROOT="${TARTANAIR_ROOT:-/cpfs/shared/landmark/renkerui/data/tartanair}"
TARTANAIR_NUM_VIEWS="${TARTANAIR_NUM_VIEWS:-0}"        # 0 -> follow MAX_SAMPLE_FRAMES
TARTANAIR_MIN_VIEWS="${TARTANAIR_MIN_VIEWS:-0}"        # 0 -> follow MIN_SAMPLE_FRAMES (per-sample clip length drawn in [min, num])
TARTANAIR_REPEAT="${TARTANAIR_REPEAT:-1}"              # replicate samples in ConcatDataset
TARTANAIR_VAL="${TARTANAIR_VAL:-0}"                    # 1 -> also mix into val
# Sampling defaults mirror base3d-clean/datasets/tartanair.py exactly.
TARTANAIR_MIN_INTERVAL="${TARTANAIR_MIN_INTERVAL:-1}"
TARTANAIR_MAX_INTERVAL="${TARTANAIR_MAX_INTERVAL:-32}"
TARTANAIR_VIDEO_PROB="${TARTANAIR_VIDEO_PROB:-0.8}"
TARTANAIR_FIX_INTERVAL_PROB="${TARTANAIR_FIX_INTERVAL_PROB:-0.6}"
TARTANAIR_BLOCK_SHUFFLE="${TARTANAIR_BLOCK_SHUFFLE:-16}"

# DynamicReplica mix-in (sampling follows base3d-clean/datasets/dynamic_replica.py).
# Disabled by default. Set DYNAMIC_REPLICA_ROOT=/shared/smartbot/renkerui/data/dynamic_replica
# (or any valid root containing {train,valid,test}/<scene>/left/{rgb,depth,cam}/) to enable.
DYNAMIC_REPLICA_ROOT="${DYNAMIC_REPLICA_ROOT:-/shared/smartbot/renkerui/data/dynamic_replica}"
DYNAMIC_REPLICA_NUM_VIEWS="${DYNAMIC_REPLICA_NUM_VIEWS:-0}"   # 0 -> follow MAX_SAMPLE_FRAMES
DYNAMIC_REPLICA_MIN_VIEWS="${DYNAMIC_REPLICA_MIN_VIEWS:-0}"   # 0 -> follow MIN_SAMPLE_FRAMES (per-sample clip length drawn in [min, num])
DYNAMIC_REPLICA_REPEAT="${DYNAMIC_REPLICA_REPEAT:-1}"         # replicate samples in ConcatDataset
DYNAMIC_REPLICA_VAL="${DYNAMIC_REPLICA_VAL:-0}"               # 1 -> also mix into val
# Sampling defaults mirror base3d-clean/datasets/dynamic_replica.py exactly.
DYNAMIC_REPLICA_MIN_INTERVAL="${DYNAMIC_REPLICA_MIN_INTERVAL:-1}"
DYNAMIC_REPLICA_MAX_INTERVAL="${DYNAMIC_REPLICA_MAX_INTERVAL:-64}"
DYNAMIC_REPLICA_VIDEO_PROB="${DYNAMIC_REPLICA_VIDEO_PROB:-1.0}"
DYNAMIC_REPLICA_FIX_INTERVAL_PROB="${DYNAMIC_REPLICA_FIX_INTERVAL_PROB:-1.0}"
DYNAMIC_REPLICA_BLOCK_SHUFFLE="${DYNAMIC_REPLICA_BLOCK_SHUFFLE:-16}"

# MapFree mix-in (sampling follows base3d-clean/datasets/mapfree.py).
# Disabled by default. Set MAPFREE_ROOT=/cpfs/shared/landmark/renkerui/data/mapfree
# (the dir must contain valid.json + <scene>/dense{0,1}/{rgb,depth,cam,sky_mask}/) to enable.
MAPFREE_ROOT="${MAPFREE_ROOT:-/cpfs/shared/landmark/renkerui/data/mapfree}"
MAPFREE_NUM_VIEWS="${MAPFREE_NUM_VIEWS:-0}"            # 0 -> follow MAX_SAMPLE_FRAMES
MAPFREE_MIN_VIEWS="${MAPFREE_MIN_VIEWS:-0}"            # 0 -> follow MIN_SAMPLE_FRAMES (per-sample clip length drawn in [min, num])
MAPFREE_REPEAT="${MAPFREE_REPEAT:-1}"                  # replicate samples in ConcatDataset
MAPFREE_VAL="${MAPFREE_VAL:-0}"                        # 1 -> also mix into val
# Sampling defaults mirror base3d-clean/datasets/mapfree.py exactly.
MAPFREE_MIN_INTERVAL="${MAPFREE_MIN_INTERVAL:-1}"
MAPFREE_MAX_INTERVAL="${MAPFREE_MAX_INTERVAL:-64}"
MAPFREE_VIDEO_PROB="${MAPFREE_VIDEO_PROB:-0.8}"
MAPFREE_FIX_INTERVAL_PROB="${MAPFREE_FIX_INTERVAL_PROB:-0.6}"
MAPFREE_BLOCK_SHUFFLE="${MAPFREE_BLOCK_SHUFFLE:-16}"

ARGS=(
  --data_roots "${DATA_ROOT_LONG3}" "${DATA_ROOT_LONG4}" "${DATA_ROOT_LONG5}"
  --oss_uri_roots "${OSS_URI_LONG3},${OSS_URI_LONG4},${OSS_URI_LONG5}"
  --ossutil_bin "${OSSUTIL_BIN}"
  --ossutil_config "${OSSUTIL_CONFIG}"
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
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
  --color_jitter_strength "${COLOR_JITTER_STRENGTH}"
  --color_jitter_prob "${COLOR_JITTER_PROB}"
  --mode_warmup_start "${MODE_WARMUP_START}"
  --mode_warmup_end "${MODE_WARMUP_END}"
  --t_stride_min "${T_STRIDE_MIN}"
  --t_stride_max "${T_STRIDE_MAX}"
  --long5_root_marker "${LONG5_ROOT_MARKER}"
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
  --tensorboard_flush_secs "${TENSORBOARD_FLUSH_SECS}"
  --tensorboard_flush_every "${TENSORBOARD_FLUSH_EVERY}"
  --empty_cache_every "${EMPTY_CACHE_EVERY}"
  --save_every "${SAVE_EVERY}"
  --val_every "${VAL_EVERY}"
)

if [[ -n "${VIEW_IDS}" ]]; then
  ARGS+=(--view_ids "${VIEW_IDS}")
fi

if [[ -n "${CAMERA_NAMES}" ]]; then
  ARGS+=(--camera_names "${CAMERA_NAMES}")
fi

if [[ -n "${SCENE_MANIFEST}" ]]; then
  ARGS+=(--scene_manifest "${SCENE_MANIFEST}")
fi

if [[ -n "${WRITE_MANIFEST}" ]]; then
  ARGS+=(--write_manifest "${WRITE_MANIFEST}")
fi

if [[ -n "${RESUME}" ]]; then
  ARGS+=(--resume "${RESUME}")
fi

# ----- Cross-dataset curriculum (Manip vs externals) -----
if [[ "${MIXTURE_CURRICULUM}" == "1" ]]; then
  ARGS+=(--mixture_curriculum)
else
  ARGS+=(--no-mixture_curriculum)
fi
ARGS+=(--mixture_p_manip_start "${MIXTURE_P_MANIP_START}")
ARGS+=(--mixture_p_manip_end "${MIXTURE_P_MANIP_END}")
ARGS+=(--mixture_warmup_start "${MIXTURE_WARMUP_START}")
ARGS+=(--mixture_warmup_end "${MIXTURE_WARMUP_END}")

# ----- DL3DV mix-in -----
if [[ -n "${DL3DV_ROOT}" ]]; then
  ARGS+=(--dl3dv_root "${DL3DV_ROOT}")
  ARGS+=(--dl3dv_num_views "${DL3DV_NUM_VIEWS}")
  ARGS+=(--dl3dv_min_views "${DL3DV_MIN_VIEWS}")
  ARGS+=(--dl3dv_repeat "${DL3DV_REPEAT}")
  ARGS+=(--dl3dv_min_interval "${DL3DV_MIN_INTERVAL}")
  ARGS+=(--dl3dv_max_interval "${DL3DV_MAX_INTERVAL}")
  ARGS+=(--dl3dv_video_prob "${DL3DV_VIDEO_PROB}")
  ARGS+=(--dl3dv_fix_interval_prob "${DL3DV_FIX_INTERVAL_PROB}")
  ARGS+=(--dl3dv_block_shuffle "${DL3DV_BLOCK_SHUFFLE}")
  if [[ "${DL3DV_VAL}" == "1" ]]; then
    ARGS+=(--dl3dv_val)
  else
    ARGS+=(--no-dl3dv_val)
  fi
fi

# ----- ScanNet++ mix-in (sampling follows scannet.py, not scannetpp.py) -----
if [[ -n "${SCANNETPP_ROOT}" ]]; then
  ARGS+=(--scannetpp_root "${SCANNETPP_ROOT}")
  ARGS+=(--scannetpp_num_views "${SCANNETPP_NUM_VIEWS}")
  ARGS+=(--scannetpp_min_views "${SCANNETPP_MIN_VIEWS}")
  ARGS+=(--scannetpp_repeat "${SCANNETPP_REPEAT}")
  ARGS+=(--scannetpp_min_interval "${SCANNETPP_MIN_INTERVAL}")
  ARGS+=(--scannetpp_max_interval "${SCANNETPP_MAX_INTERVAL}")
  ARGS+=(--scannetpp_video_prob "${SCANNETPP_VIDEO_PROB}")
  ARGS+=(--scannetpp_fix_interval_prob "${SCANNETPP_FIX_INTERVAL_PROB}")
  ARGS+=(--scannetpp_block_shuffle "${SCANNETPP_BLOCK_SHUFFLE}")
  if [[ "${SCANNETPP_VAL}" == "1" ]]; then
    ARGS+=(--scannetpp_val)
  else
    ARGS+=(--no-scannetpp_val)
  fi
fi

# ----- TartanAir mix-in (sampling follows base3d-clean/datasets/tartanair.py) -----
if [[ -n "${TARTANAIR_ROOT}" ]]; then
  ARGS+=(--tartanair_root "${TARTANAIR_ROOT}")
  ARGS+=(--tartanair_num_views "${TARTANAIR_NUM_VIEWS}")
  ARGS+=(--tartanair_min_views "${TARTANAIR_MIN_VIEWS}")
  ARGS+=(--tartanair_repeat "${TARTANAIR_REPEAT}")
  ARGS+=(--tartanair_min_interval "${TARTANAIR_MIN_INTERVAL}")
  ARGS+=(--tartanair_max_interval "${TARTANAIR_MAX_INTERVAL}")
  ARGS+=(--tartanair_video_prob "${TARTANAIR_VIDEO_PROB}")
  ARGS+=(--tartanair_fix_interval_prob "${TARTANAIR_FIX_INTERVAL_PROB}")
  ARGS+=(--tartanair_block_shuffle "${TARTANAIR_BLOCK_SHUFFLE}")
  if [[ "${TARTANAIR_VAL}" == "1" ]]; then
    ARGS+=(--tartanair_val)
  else
    ARGS+=(--no-tartanair_val)
  fi
fi

# ----- DynamicReplica mix-in (sampling follows base3d-clean/datasets/dynamic_replica.py) -----
if [[ -n "${DYNAMIC_REPLICA_ROOT}" ]]; then
  ARGS+=(--dynamic_replica_root "${DYNAMIC_REPLICA_ROOT}")
  ARGS+=(--dynamic_replica_num_views "${DYNAMIC_REPLICA_NUM_VIEWS}")
  ARGS+=(--dynamic_replica_min_views "${DYNAMIC_REPLICA_MIN_VIEWS}")
  ARGS+=(--dynamic_replica_repeat "${DYNAMIC_REPLICA_REPEAT}")
  ARGS+=(--dynamic_replica_min_interval "${DYNAMIC_REPLICA_MIN_INTERVAL}")
  ARGS+=(--dynamic_replica_max_interval "${DYNAMIC_REPLICA_MAX_INTERVAL}")
  ARGS+=(--dynamic_replica_video_prob "${DYNAMIC_REPLICA_VIDEO_PROB}")
  ARGS+=(--dynamic_replica_fix_interval_prob "${DYNAMIC_REPLICA_FIX_INTERVAL_PROB}")
  ARGS+=(--dynamic_replica_block_shuffle "${DYNAMIC_REPLICA_BLOCK_SHUFFLE}")
  if [[ "${DYNAMIC_REPLICA_VAL}" == "1" ]]; then
    ARGS+=(--dynamic_replica_val)
  else
    ARGS+=(--no-dynamic_replica_val)
  fi
fi

# ----- MapFree mix-in (sampling follows base3d-clean/datasets/mapfree.py) -----
if [[ -n "${MAPFREE_ROOT}" ]]; then
  ARGS+=(--mapfree_root "${MAPFREE_ROOT}")
  ARGS+=(--mapfree_num_views "${MAPFREE_NUM_VIEWS}")
  ARGS+=(--mapfree_min_views "${MAPFREE_MIN_VIEWS}")
  ARGS+=(--mapfree_repeat "${MAPFREE_REPEAT}")
  ARGS+=(--mapfree_min_interval "${MAPFREE_MIN_INTERVAL}")
  ARGS+=(--mapfree_max_interval "${MAPFREE_MAX_INTERVAL}")
  ARGS+=(--mapfree_video_prob "${MAPFREE_VIDEO_PROB}")
  ARGS+=(--mapfree_fix_interval_prob "${MAPFREE_FIX_INTERVAL_PROB}")
  ARGS+=(--mapfree_block_shuffle "${MAPFREE_BLOCK_SHUFFLE}")
  if [[ "${MAPFREE_VAL}" == "1" ]]; then
    ARGS+=(--mapfree_val)
  else
    ARGS+=(--no-mapfree_val)
  fi
fi

if [[ -n "${TENSORBOARD_DIR}" ]]; then
  ARGS+=(--tensorboard_dir "${TENSORBOARD_DIR}")
fi

if [[ "${TENSORBOARD}" == "1" ]]; then
  ARGS+=(--tensorboard)
else
  ARGS+=(--no-tensorboard)
fi

if [[ "${CANONICALIZE_FIRST_FRAME}" == "1" ]]; then
  ARGS+=(--canonicalize_first_frame)
else
  ARGS+=(--no-canonicalize_first_frame)
fi

if [[ "${USE_MASK:-0}" == "1" ]]; then
  ARGS+=(--use_mask)
fi

if [[ "${DEPTH_ACTIVATION_CHECKPOINT}" != "1" ]]; then
  ARGS+=(--no_depth_activation_checkpoint)
fi

if [[ "${FREEZE_DINO_PATCH_EMBED}" == "1" ]]; then
  ARGS+=(--freeze_dino_patch_embed)
fi

if [[ "${FREEZE_AGGREGATOR}" == "1" ]]; then
  ARGS+=(--freeze_aggregator)
fi

if [[ "${FREEZE_CAMERA}" == "1" ]]; then
  ARGS+=(--freeze_camera)
fi

if [[ "${FREEZE_DEPTH}" == "1" ]]; then
  ARGS+=(--freeze_depth)
fi

if [[ "${FREEZE_POINT}" == "1" ]]; then
  ARGS+=(--freeze_point)
fi

if [[ "${CPU:-0}" == "1" ]]; then
  ARGS+=(--cpu)
fi

cat <<EOF

========================================
 LingBot-MAP Window32 2-Node/16-GPU DDP Training
========================================
[run]
  workdir       : ${SCRIPT_DIR}
  output_dir    : ${OUTPUT_DIR}
  nnodes        : ${NNODES}
  nproc/node    : ${NPROC_PER_NODE}
  node_rank     : ${NODE_RANK}
  master        : ${MASTER_ADDR}:${MASTER_PORT}
  global_batch  : $(( BATCH_SIZE * ACCUM_STEPS * NNODES * NPROC_PER_NODE ))
  tensorboard   : ${TENSORBOARD} (${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard})

[environment]
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  conda_env     : ${CONDA_ENV}
  python        : ${PYTHON_CMD[*]}
  cuda_alloc    : ${PYTORCH_CUDA_ALLOC_CONF}
  empty_cache   : every ${EMPTY_CACHE_EVERY} optimizer step(s)

[data]
  roots         : ${DATA_ROOT_LONG3}
                  ${DATA_ROOT_LONG4}
                  ${DATA_ROOT_LONG5}
  manifest      : ${SCENE_MANIFEST}
  write_manifest: ${WRITE_MANIFEST:-<disabled>}
  ossutil       : ${OSSUTIL_BIN}
  oss_config    : ${OSSUTIL_CONFIG}
  oss_roots     : ${OSS_URI_LONG3}
                  ${OSS_URI_LONG4}
                  ${OSS_URI_LONG5}
  dl3dv_root    : ${DL3DV_ROOT:-<disabled>}
  dl3dv_views   : [${DL3DV_MIN_VIEWS}, ${DL3DV_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  dl3dv_repeat  : ${DL3DV_REPEAT}     (mix-ratio bias)
  dl3dv_val     : ${DL3DV_VAL}     (also mix into val loader)
  dl3dv_sampler : min_interval=${DL3DV_MIN_INTERVAL} max_interval=${DL3DV_MAX_INTERVAL} \
video_prob=${DL3DV_VIDEO_PROB} fix_interval_prob=${DL3DV_FIX_INTERVAL_PROB} \
block_shuffle=${DL3DV_BLOCK_SHUFFLE}
  scannetpp_root: ${SCANNETPP_ROOT:-<disabled>}
  scannetpp_views: [${SCANNETPP_MIN_VIEWS}, ${SCANNETPP_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  scannetpp_repeat: ${SCANNETPP_REPEAT}     (mix-ratio bias)
  scannetpp_val : ${SCANNETPP_VAL}     (also mix into val loader)
  scannetpp_sampler (from scannet.py, NOT scannetpp.py): \
min_interval=${SCANNETPP_MIN_INTERVAL} max_interval=${SCANNETPP_MAX_INTERVAL} \
video_prob=${SCANNETPP_VIDEO_PROB} fix_interval_prob=${SCANNETPP_FIX_INTERVAL_PROB} \
block_shuffle=${SCANNETPP_BLOCK_SHUFFLE}
  tartanair_root: ${TARTANAIR_ROOT:-<disabled>}
  tartanair_views: [${TARTANAIR_MIN_VIEWS}, ${TARTANAIR_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  tartanair_repeat: ${TARTANAIR_REPEAT}     (mix-ratio bias)
  tartanair_val : ${TARTANAIR_VAL}     (also mix into val loader)
  tartanair_sampler (from base3d-clean/datasets/tartanair.py): \
min_interval=${TARTANAIR_MIN_INTERVAL} max_interval=${TARTANAIR_MAX_INTERVAL} \
video_prob=${TARTANAIR_VIDEO_PROB} fix_interval_prob=${TARTANAIR_FIX_INTERVAL_PROB} \
block_shuffle=${TARTANAIR_BLOCK_SHUFFLE}
  dynamic_replica_root: ${DYNAMIC_REPLICA_ROOT:-<disabled>}
  dynamic_replica_views: [${DYNAMIC_REPLICA_MIN_VIEWS}, ${DYNAMIC_REPLICA_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  dynamic_replica_repeat: ${DYNAMIC_REPLICA_REPEAT}     (mix-ratio bias)
  dynamic_replica_val : ${DYNAMIC_REPLICA_VAL}     (also mix into val loader)
  dynamic_replica_sampler (from base3d-clean/datasets/dynamic_replica.py): \
min_interval=${DYNAMIC_REPLICA_MIN_INTERVAL} max_interval=${DYNAMIC_REPLICA_MAX_INTERVAL} \
video_prob=${DYNAMIC_REPLICA_VIDEO_PROB} fix_interval_prob=${DYNAMIC_REPLICA_FIX_INTERVAL_PROB} \
block_shuffle=${DYNAMIC_REPLICA_BLOCK_SHUFFLE}
  mapfree_root  : ${MAPFREE_ROOT:-<disabled>}
  mapfree_views : [${MAPFREE_MIN_VIEWS}, ${MAPFREE_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  mapfree_repeat: ${MAPFREE_REPEAT}     (mix-ratio bias)
  mapfree_val   : ${MAPFREE_VAL}     (also mix into val loader)
  mapfree_sampler (from base3d-clean/datasets/mapfree.py): \
min_interval=${MAPFREE_MIN_INTERVAL} max_interval=${MAPFREE_MAX_INTERVAL} \
video_prob=${MAPFREE_VIDEO_PROB} fix_interval_prob=${MAPFREE_FIX_INTERVAL_PROB} \
block_shuffle=${MAPFREE_BLOCK_SHUFFLE}

[sampling]
  mode          : ${SEQUENCE_MODE}
  strategy      : ${SAMPLE_STRATEGY}
  random_stride : ${RANDOM_STRIDE_MIN}-${RANDOM_STRIDE_MAX}
  seq_len       : ${MAX_SAMPLE_FRAMES}
  image_size    : ${IMAGE_SIZE}
  depth_chunk   : ${DEPTH_FRAMES_CHUNK_SIZE}
  depth_ckpt    : ${DEPTH_ACTIVATION_CHECKPOINT}
  frame_block   : ${NUM_FRAME_PER_BLOCK}
  kv_window     : ${KV_CACHE_SLIDING_WINDOW}

[manip_4d_mixed curriculum]  (only used when SEQUENCE_MODE=manip_4d_mixed)
  wrist_prefix  : ${WRIST_CAMERA_PREFIX}
  static_prefix : ${STATIC_CAMERA_PREFIX}
  mode S views  : ${S_VIEWS_MIN}-${S_VIEWS_MAX}  (single-timestamp multi-view snapshot)
  mode M grid   : V=${M_VIEWS_MIN}-${M_VIEWS_MAX} (random) x T=dynamic, V*T in [${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}], t-stride ${M_STRIDE_MIN}-${M_STRIDE_MAX}
  mode T (long5): single surround_cam random-interval, stride ${T_STRIDE_MIN}-${T_STRIDE_MAX}, random start, max ${MAX_SAMPLE_FRAMES} frames
  long5 marker  : "${LONG5_ROOT_MARKER}"  (scene paths containing this substring auto-route to mode T, bypass S/W/M)
  weights init  : ${MODE_WEIGHTS_INITIAL}    @ step <= ${MODE_WARMUP_START}  (long3/4 only)
  weights final : ${MODE_WEIGHTS_FINAL}    @ step >= ${MODE_WARMUP_END}  (long3/4 only)

[mixture curriculum]  (cross-dataset Manip-vs-external sampler)
  enabled       : ${MIXTURE_CURRICULUM}    (1 -> CurriculumMixtureSampler; 0 -> plain shuffle)
  p_manip       : ${MIXTURE_P_MANIP_START} -> ${MIXTURE_P_MANIP_END}  (linear over step [${MIXTURE_WARMUP_START}, ${MIXTURE_WARMUP_END}])
  externals     : DL3DV, ScanNet++, TartanAir, DynamicReplica, MapFree (each enabled iff its *_ROOT is set)
  per-ext share : (1 - p_manip) / N_enabled  (uniform over enabled externals)

[augmentation]
  color_jitter  : strength=${COLOR_JITTER_STRENGTH} prob=${COLOR_JITTER_PROB}
                  (RGB only; same params per clip; off for val)

[optimization]
  optimizer     : AdamW
  lr            : ${LR} -> ${MIN_LR}
  weight_decay  : ${WEIGHT_DECAY}
  warmup        : ratio=${WARMUP_RATIO}, steps=${WARMUP_STEPS}
  max_steps     : ${MAX_STEPS}
  epochs        : ${EPOCHS}
  batches/epoch : ${LIMIT_TRAIN_BATCHES}
  canon_first   : ${CANONICALIZE_FIRST_FRAME}    (recenter world to frame-0 c2w=I before anchor-scale)

[freeze]
  dino_patch    : ${FREEZE_DINO_PATCH_EMBED}
  aggregator    : ${FREEZE_AGGREGATOR}
  camera_head   : ${FREEZE_CAMERA}
  depth_head    : ${FREEZE_DEPTH}
  point_head    : ${FREEZE_POINT}
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
