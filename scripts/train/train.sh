#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/scripts/train:${REPO_DIR}/scripts/eval:${PYTHONPATH:-}"
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

DATA_ROOT_LONG6="${DATA_ROOT_LONG6:-/oss-guowenqi/Manip_long6/data}"
OSS_URI_LONG6="${OSS_URI_LONG6:-oss://pjlab-bjpai-sim/guowenqi/Manip_long6/data}"
OSSUTIL_BIN="${OSSUTIL_BIN:-/cpfs/user/guowenqi/ossutil/ossutil}"
OSSUTIL_CONFIG="${OSSUTIL_CONFIG:-/cpfs/user/guowenqi/ossutil/.ossutilconfig}"
OSS_STAGE_FALLBACK="${OSS_STAGE_FALLBACK:-1}"
OSS_STAGE_ROOT="${OSS_STAGE_ROOT:-/cpfs/user/guowenqi/lingbot-map-oss-stage}"
OSS_STAGE_MOUNT_ROOT="${OSS_STAGE_MOUNT_ROOT:-/oss-guowenqi}"
OSS_STAGE_MAX_ENTRIES="${OSS_STAGE_MAX_ENTRIES:-1000}"
OSS_STAGE_OSSUTIL_JOBS="${OSS_STAGE_OSSUTIL_JOBS:-64}"
OSS_STAGE_OSSUTIL_CHECKERS="${OSS_STAGE_OSSUTIL_CHECKERS:-128}"
OSS_STAGE_OSSUTIL_PARALLEL="${OSS_STAGE_OSSUTIL_PARALLEL:-4}"
OSS_STAGE_DELETE_WORKERS="${OSS_STAGE_DELETE_WORKERS:-8}"
OSS_STAGE_DELETE_BATCH="${OSS_STAGE_DELETE_BATCH:-512}"
OSS_STAGE_WORKER_MAX_ENTRIES="${OSS_STAGE_WORKER_MAX_ENTRIES:-25}"
MODEL_PATH="${MODEL_PATH:-${REPO_DIR}/checkpoints/lingbot-map.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/runs/manip_long_train}"
SCENE_MANIFEST="${SCENE_MANIFEST:-${OUTPUT_DIR}/manip_long6_trajectory_manifest.txt}"
WRITE_MANIFEST="${WRITE_MANIFEST:-}"

EPOCHS="${EPOCHS:-150}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_SCENES="${MAX_SCENES:-0}"
VAL_FRACTION="${VAL_FRACTION:-0.02}"
BATCH_SIZE="1"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
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
RANDOM_STRIDE_MIN="${RANDOM_STRIDE_MIN:-2}"
RANDOM_STRIDE_MAX="${RANDOM_STRIDE_MAX:-8}"
W_STRIDE_MIN="${W_STRIDE_MIN:-2}"
W_STRIDE_MAX="${W_STRIDE_MAX:-8}"
MOVING_STRIDE_MIN="${MOVING_STRIDE_MIN:-2}"
MOVING_STRIDE_MAX="${MOVING_STRIDE_MAX:-6}"
FIXED_STRIDE_MIN="${FIXED_STRIDE_MIN:-4}"
FIXED_STRIDE_MAX="${FIXED_STRIDE_MAX:-16}"
LONG6_ROOT_MARKER="${LONG6_ROOT_MARKER:-Manip_long6}"
LONG6_MODE_WEIGHTS="${LONG6_MODE_WEIGHTS:-W=0.40,T=0.45,F=0.15}"
MOVING_CAMERA_PREFIX="${MOVING_CAMERA_PREFIX:-surround_cam_moving}"
FIXED_CAMERA_PREFIX="${FIXED_CAMERA_PREFIX:-surround_cam_fixed}"
RANDOM_INTERVAL_START="${RANDOM_INTERVAL_START:-first}"
MAX_SAMPLE_FRAMES="${MAX_SAMPLE_FRAMES:-64}"
MIN_SAMPLE_FRAMES="${MIN_SAMPLE_FRAMES:-24}"
# Manip_long6 W/T/F curriculum.
WRIST_CAMERA_PREFIX="${WRIST_CAMERA_PREFIX:-realsense}"
STATIC_CAMERA_PREFIX="${STATIC_CAMERA_PREFIX:-surround}"
# Manip_long6 defaults to camera-specific random-interval sampling:
#   - W: wrist realsense cameras, stride W_STRIDE_MIN/MAX
#   - T: moving third-person surround camera, stride MOVING_STRIDE_MIN/MAX
#   - F: fixed third-person surround camera, stride FIXED_STRIDE_MIN/MAX
COLOR_JITTER_STRENGTH="${COLOR_JITTER_STRENGTH:-0.2}"
COLOR_JITTER_PROB="${COLOR_JITTER_PROB:-0.5}"
IMAGE_SIZE="${IMAGE_SIZE:-280}"
DEPTH_SCALE="${DEPTH_SCALE:-0}"
USE_MASK="${USE_MASK:-1}"
NUM_SCALE_FRAMES="${NUM_SCALE_FRAMES:-8}"
NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-1}"
KV_CACHE_SLIDING_WINDOW="${KV_CACHE_SLIDING_WINDOW:-48}"
DEPTH_FRAMES_CHUNK_SIZE="${DEPTH_FRAMES_CHUNK_SIZE:-24}"
DEPTH_ACTIVATION_CHECKPOINT="${DEPTH_ACTIVATION_CHECKPOINT:-1}"
CAMERA_ACTIVATION_CHECKPOINT="${CAMERA_ACTIVATION_CHECKPOINT:-0}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1000}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
LOG_EVERY="${LOG_EVERY:-10}"
PRINT_INPUT_EVERY="${PRINT_INPUT_EVERY:-0}"
TENSORBOARD="${TENSORBOARD:-1}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-}"
TENSORBOARD_FLUSH_SECS="${TENSORBOARD_FLUSH_SECS:-30}"
TENSORBOARD_FLUSH_EVERY="${TENSORBOARD_FLUSH_EVERY:-10}"
EMPTY_CACHE_EVERY="${EMPTY_CACHE_EVERY:-1}"
CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
SAVE_EVERY="${SAVE_EVERY:-10000}"
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
# by relative weights. With batch_size=1 this is exactly the per-step
# Manip-vs-external curriculum, with externals acting as anti-overfit replay.
# *_REPEAT is force-overridden to 1 when this is on.
MIXTURE_CURRICULUM="${MIXTURE_CURRICULUM:-1}"
MIXTURE_P_MANIP_START="${MIXTURE_P_MANIP_START:-0.60}"
MIXTURE_P_MANIP_END="${MIXTURE_P_MANIP_END:-0.80}"
MIXTURE_WARMUP_START="${MIXTURE_WARMUP_START:-2000}"
MIXTURE_WARMUP_END="${MIXTURE_WARMUP_END:-25000}"
MIXTURE_EXTERNAL_WEIGHTS="${MIXTURE_EXTERNAL_WEIGHTS:-dl3dv=4,scannetpp=4,arkitscenes_highres=3.5,hypersim=2.5,adt=2.5,ase=2.5,dynamic_replica=2,tartanair=1.5,co3d=1.5}"

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

# ScanNet++ v2 mix-in.
SCANNETPP_ROOT="${SCANNETPP_ROOT:-/shared/smartbot/renkerui/data/scannetppv2}"
SCANNETPP_NUM_VIEWS="${SCANNETPP_NUM_VIEWS:-0}"        # 0 -> follow MAX_SAMPLE_FRAMES
SCANNETPP_MIN_VIEWS="${SCANNETPP_MIN_VIEWS:-0}"        # 0 -> follow MIN_SAMPLE_FRAMES (per-sample clip length drawn in [min, num])
SCANNETPP_REPEAT="${SCANNETPP_REPEAT:-1}"              # replicate samples in ConcatDataset
SCANNETPP_VAL="${SCANNETPP_VAL:-0}"                    # 1 -> also mix into val
SCANNETPP_MIN_INTERVAL="${SCANNETPP_MIN_INTERVAL:-1}"
SCANNETPP_MAX_INTERVAL="${SCANNETPP_MAX_INTERVAL:-30}"
SCANNETPP_VIDEO_PROB="${SCANNETPP_VIDEO_PROB:-0.6}"
SCANNETPP_FIX_INTERVAL_PROB="${SCANNETPP_FIX_INTERVAL_PROB:-0.6}"
SCANNETPP_BLOCK_SHUFFLE="${SCANNETPP_BLOCK_SHUFFLE:-16}"

# ARKitScenesHighRes mix-in. Enabled by default and sampled exactly like
# base3d-clean/datasets/arkitscenes_highres.py, not like Manip_long.
# Expected layout: {Training,Validation}/<scene>/{scene_metadata.npz,vga_wide,highres_depth}.
ARKITSCENES_HIGHRES_ROOT="${ARKITSCENES_HIGHRES_ROOT:-/shared/smartbot/renkerui/data/arkitscenes_highres}"
ARKITSCENES_HIGHRES_NUM_VIEWS="${ARKITSCENES_HIGHRES_NUM_VIEWS:-0}"  # 0 -> follow MAX_SAMPLE_FRAMES
ARKITSCENES_HIGHRES_MIN_VIEWS="${ARKITSCENES_HIGHRES_MIN_VIEWS:-0}"  # 0 -> follow MIN_SAMPLE_FRAMES (per-sample clip length drawn in [min, num])
ARKITSCENES_HIGHRES_REPEAT="${ARKITSCENES_HIGHRES_REPEAT:-1}"
ARKITSCENES_HIGHRES_VAL="${ARKITSCENES_HIGHRES_VAL:-0}"
ARKITSCENES_HIGHRES_MIN_INTERVAL="${ARKITSCENES_HIGHRES_MIN_INTERVAL:-1}"
ARKITSCENES_HIGHRES_MAX_INTERVAL="${ARKITSCENES_HIGHRES_MAX_INTERVAL:-32}"
ARKITSCENES_HIGHRES_VIDEO_PROB="${ARKITSCENES_HIGHRES_VIDEO_PROB:-0.5}"
ARKITSCENES_HIGHRES_FIX_INTERVAL_PROB="${ARKITSCENES_HIGHRES_FIX_INTERVAL_PROB:-0.5}"
ARKITSCENES_HIGHRES_BLOCK_SHUFFLE="${ARKITSCENES_HIGHRES_BLOCK_SHUFFLE:-16}"

# Hypersim mix-in. Enabled by default and sampled exactly like
# base3d-clean/datasets/hypersim.py, not like Manip_long.
# Expected layout: ai*/{rgb/*.jpg,depth/*.png,pose/*.txt}.
HYPERSIM_ROOT="${HYPERSIM_ROOT:-/shared/smartbot/guowenqi/hypersim}"
HYPERSIM_NUM_VIEWS="${HYPERSIM_NUM_VIEWS:-0}"          # 0 -> follow MAX_SAMPLE_FRAMES
HYPERSIM_MIN_VIEWS="${HYPERSIM_MIN_VIEWS:-0}"          # 0 -> follow MIN_SAMPLE_FRAMES
HYPERSIM_REPEAT="${HYPERSIM_REPEAT:-1}"
HYPERSIM_VAL="${HYPERSIM_VAL:-0}"
# Defaults mirror base3d-clean/datasets/hypersim.py: max_interval=4 and
# block_shuffle=16; omitted probabilities inherit BaseMultiViewDataset defaults.
HYPERSIM_MIN_INTERVAL="${HYPERSIM_MIN_INTERVAL:-1}"
HYPERSIM_MAX_INTERVAL="${HYPERSIM_MAX_INTERVAL:-4}"
HYPERSIM_VIDEO_PROB="${HYPERSIM_VIDEO_PROB:-0.5}"
HYPERSIM_FIX_INTERVAL_PROB="${HYPERSIM_FIX_INTERVAL_PROB:-0.5}"
HYPERSIM_BLOCK_SHUFFLE="${HYPERSIM_BLOCK_SHUFFLE:-16}"

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

# ADT mix-in (sampling follows base3d-clean/datasets/adt.py; NOT Manip_long).
# Enabled by default; clear ADT_ROOT to disable.
ADT_ROOT="${ADT_ROOT:-/shared/smartbot/renkerui/data/adt}"
ADT_NUM_VIEWS="${ADT_NUM_VIEWS:-0}"                  # 0 -> follow MAX_SAMPLE_FRAMES
ADT_MIN_VIEWS="${ADT_MIN_VIEWS:-0}"                  # 0 -> follow MIN_SAMPLE_FRAMES
ADT_REPEAT="${ADT_REPEAT:-1}"                        # replicate samples in ConcatDataset
ADT_VAL="${ADT_VAL:-0}"                              # 1 -> also mix into val
ADT_MAX_DISTANCE="${ADT_MAX_DISTANCE:-128}"          # matches base3d-clean/datasets/adt.py
ADT_ROTATE_CLOCKWISE="${ADT_ROTATE_CLOCKWISE:-1}"        # rotate Aria RGB-D upright and update camera geometry

# ASE mix-in (sampling follows base3d-clean/datasets/ase_20251008.py; NOT Manip_long).
# Enabled by default; clear ASE_ROOT to disable.
ASE_ROOT="${ASE_ROOT-/cpfs/shared/aigc/guowenqi/ase_processed}"
ASE_CACHE_PATH="${ASE_CACHE_PATH-/cpfs/user/guowenqi/base3d-clean/cache/ase_processed_clean_cache.npy}"
ASE_NUM_VIEWS="${ASE_NUM_VIEWS:-0}"                  # 0 -> follow MAX_SAMPLE_FRAMES
ASE_MIN_VIEWS="${ASE_MIN_VIEWS:-0}"                  # 0 -> follow MIN_SAMPLE_FRAMES
ASE_REPEAT="${ASE_REPEAT:-1}"                        # replicate samples in ConcatDataset
ASE_VAL="${ASE_VAL:-0}"                              # 1 -> also mix into val
ASE_MAX_DISTANCE="${ASE_MAX_DISTANCE:-32}"           # matches base3d-clean/datasets/ase_20251008.py

# CO3D mix-in (sampling follows base3d-clean/datasets/co3d_20251006.py; NOT Manip_long).
# Enabled by default; clear CO3D_ROOT to disable.
CO3D_ROOT="${CO3D_ROOT:-/oss-guowenqi/CO3Dv2}"
CO3D_OSS_URI_ROOT="${CO3D_OSS_URI_ROOT:-oss://pjlab-bjpai-sim/guowenqi/CO3Dv2}"
CO3D_BASE3D_ROOT="${CO3D_BASE3D_ROOT:-/cpfs/user/guowenqi/base3d-clean}"
CO3D_NUM_VIEWS="${CO3D_NUM_VIEWS:-0}"                # 0 -> follow MAX_SAMPLE_FRAMES
CO3D_MIN_VIEWS="${CO3D_MIN_VIEWS:-0}"                # 0 -> follow MIN_SAMPLE_FRAMES
CO3D_REPEAT="${CO3D_REPEAT:-1}"                      # replicate samples in ConcatDataset
CO3D_VAL="${CO3D_VAL:-0}"                            # 1 -> also mix CO3D test split into val
CO3D_MASK_BG="${CO3D_MASK_BG:-rand}"                 # true|false|rand, passed to co3d_20251006.py

ARGS=(
  --data_roots "${DATA_ROOT_LONG6}"
  --oss_uri_roots "${OSS_URI_LONG6}"
  --ossutil_bin "${OSSUTIL_BIN}"
  --ossutil_config "${OSSUTIL_CONFIG}"
  --oss_stage_root "${OSS_STAGE_ROOT}"
  --oss_stage_mount_root "${OSS_STAGE_MOUNT_ROOT}"
  --oss_stage_max_entries "${OSS_STAGE_MAX_ENTRIES}"
  --oss_stage_ossutil_jobs "${OSS_STAGE_OSSUTIL_JOBS}"
  --oss_stage_ossutil_checkers "${OSS_STAGE_OSSUTIL_CHECKERS}"
  --oss_stage_ossutil_parallel "${OSS_STAGE_OSSUTIL_PARALLEL}"
  --oss_stage_delete_workers "${OSS_STAGE_DELETE_WORKERS}"
  --oss_stage_delete_batch "${OSS_STAGE_DELETE_BATCH}"
  --oss_stage_worker_max_entries "${OSS_STAGE_WORKER_MAX_ENTRIES}"
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --max_steps "${MAX_STEPS}"
  --max_scenes "${MAX_SCENES}"
  --val_fraction "${VAL_FRACTION}"
  --batch_size "${BATCH_SIZE}"
  --accum_steps "${ACCUM_STEPS}"
  --num_workers "${NUM_WORKERS}"
  --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
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
  --w_stride_min "${W_STRIDE_MIN}"
  --w_stride_max "${W_STRIDE_MAX}"
  --random_interval_start "${RANDOM_INTERVAL_START}"
  --max_sample_frames "${MAX_SAMPLE_FRAMES}"
  --min_sample_frames "${MIN_SAMPLE_FRAMES}"
  --wrist_camera_prefix "${WRIST_CAMERA_PREFIX}"
  --static_camera_prefix "${STATIC_CAMERA_PREFIX}"
  --color_jitter_strength "${COLOR_JITTER_STRENGTH}"
  --color_jitter_prob "${COLOR_JITTER_PROB}"
  --moving_stride_min "${MOVING_STRIDE_MIN}"
  --moving_stride_max "${MOVING_STRIDE_MAX}"
  --fixed_stride_min "${FIXED_STRIDE_MIN}"
  --fixed_stride_max "${FIXED_STRIDE_MAX}"
  --long6_root_marker "${LONG6_ROOT_MARKER}"
  --long6_mode_weights "${LONG6_MODE_WEIGHTS}"
  --moving_camera_prefix "${MOVING_CAMERA_PREFIX}"
  --fixed_camera_prefix "${FIXED_CAMERA_PREFIX}"
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

if [[ "${OSS_STAGE_FALLBACK}" == "1" ]]; then
  ARGS+=(--oss_stage_fallback)
else
  ARGS+=(--no-oss_stage_fallback)
fi

if [[ "${CUDNN_BENCHMARK}" == "1" ]]; then
  ARGS+=(--cudnn_benchmark)
else
  ARGS+=(--no-cudnn_benchmark)
fi

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
ARGS+=(--mixture_external_weights "${MIXTURE_EXTERNAL_WEIGHTS}")

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

# ----- ScanNet++ mix-in -----
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

# ----- ARKitScenesHighRes mix-in (base3d-clean/datasets/arkitscenes_highres.py sampling; NOT Manip_long) -----
if [[ -n "${ARKITSCENES_HIGHRES_ROOT}" ]]; then
  ARGS+=(--arkitscenes_highres_root "${ARKITSCENES_HIGHRES_ROOT}")
  ARGS+=(--arkitscenes_highres_num_views "${ARKITSCENES_HIGHRES_NUM_VIEWS}")
  ARGS+=(--arkitscenes_highres_min_views "${ARKITSCENES_HIGHRES_MIN_VIEWS}")
  ARGS+=(--arkitscenes_highres_repeat "${ARKITSCENES_HIGHRES_REPEAT}")
  ARGS+=(--arkitscenes_highres_min_interval "${ARKITSCENES_HIGHRES_MIN_INTERVAL}")
  ARGS+=(--arkitscenes_highres_max_interval "${ARKITSCENES_HIGHRES_MAX_INTERVAL}")
  ARGS+=(--arkitscenes_highres_video_prob "${ARKITSCENES_HIGHRES_VIDEO_PROB}")
  ARGS+=(--arkitscenes_highres_fix_interval_prob "${ARKITSCENES_HIGHRES_FIX_INTERVAL_PROB}")
  ARGS+=(--arkitscenes_highres_block_shuffle "${ARKITSCENES_HIGHRES_BLOCK_SHUFFLE}")
  if [[ "${ARKITSCENES_HIGHRES_VAL}" == "1" ]]; then
    ARGS+=(--arkitscenes_highres_val)
  else
    ARGS+=(--no-arkitscenes_highres_val)
  fi
fi

# ----- Hypersim mix-in (base3d-clean/datasets/hypersim.py sampling; NOT Manip_long) -----
if [[ -n "${HYPERSIM_ROOT}" ]]; then
  ARGS+=(--hypersim_root "${HYPERSIM_ROOT}")
  ARGS+=(--hypersim_num_views "${HYPERSIM_NUM_VIEWS}")
  ARGS+=(--hypersim_min_views "${HYPERSIM_MIN_VIEWS}")
  ARGS+=(--hypersim_repeat "${HYPERSIM_REPEAT}")
  ARGS+=(--hypersim_min_interval "${HYPERSIM_MIN_INTERVAL}")
  ARGS+=(--hypersim_max_interval "${HYPERSIM_MAX_INTERVAL}")
  ARGS+=(--hypersim_video_prob "${HYPERSIM_VIDEO_PROB}")
  ARGS+=(--hypersim_fix_interval_prob "${HYPERSIM_FIX_INTERVAL_PROB}")
  ARGS+=(--hypersim_block_shuffle "${HYPERSIM_BLOCK_SHUFFLE}")
  if [[ "${HYPERSIM_VAL}" == "1" ]]; then
    ARGS+=(--hypersim_val)
  else
    ARGS+=(--no-hypersim_val)
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

# ----- ADT mix-in (sampling follows base3d-clean/datasets/adt.py; NOT Manip_long) -----
if [[ -n "${ADT_ROOT}" ]]; then
  ARGS+=(--adt_root "${ADT_ROOT}")
  ARGS+=(--adt_num_views "${ADT_NUM_VIEWS}")
  ARGS+=(--adt_min_views "${ADT_MIN_VIEWS}")
  ARGS+=(--adt_repeat "${ADT_REPEAT}")
  ARGS+=(--adt_max_distance "${ADT_MAX_DISTANCE}")
  if [[ "${ADT_ROTATE_CLOCKWISE}" == "1" ]]; then
    ARGS+=(--adt_rotate_clockwise)
  else
    ARGS+=(--no-adt_rotate_clockwise)
  fi
  if [[ "${ADT_VAL}" == "1" ]]; then
    ARGS+=(--adt_val)
  else
    ARGS+=(--no-adt_val)
  fi
fi

# ----- ASE mix-in (sampling follows base3d-clean/datasets/ase_20251008.py; NOT Manip_long) -----
if [[ -n "${ASE_ROOT}" ]]; then
  ARGS+=(--ase_root "${ASE_ROOT}")
  if [[ -n "${ASE_CACHE_PATH}" ]]; then
    ARGS+=(--ase_cache_path "${ASE_CACHE_PATH}")
  fi
  ARGS+=(--ase_num_views "${ASE_NUM_VIEWS}")
  ARGS+=(--ase_min_views "${ASE_MIN_VIEWS}")
  ARGS+=(--ase_repeat "${ASE_REPEAT}")
  ARGS+=(--ase_max_distance "${ASE_MAX_DISTANCE}")
  if [[ "${ASE_VAL}" == "1" ]]; then
    ARGS+=(--ase_val)
  else
    ARGS+=(--no-ase_val)
  fi
fi

# ----- CO3D mix-in (sampling follows base3d-clean/datasets/co3d_20251006.py; NOT Manip_long) -----
if [[ -n "${CO3D_ROOT}" ]]; then
  ARGS+=(--co3d_root "${CO3D_ROOT}")
  ARGS+=(--co3d_base3d_root "${CO3D_BASE3D_ROOT}")
  ARGS+=(--co3d_oss_uri_root "${CO3D_OSS_URI_ROOT}")
  ARGS+=(--co3d_num_views "${CO3D_NUM_VIEWS}")
  ARGS+=(--co3d_min_views "${CO3D_MIN_VIEWS}")
  ARGS+=(--co3d_repeat "${CO3D_REPEAT}")
  ARGS+=(--co3d_mask_bg "${CO3D_MASK_BG}")
  if [[ "${CO3D_VAL}" == "1" ]]; then
    ARGS+=(--co3d_val)
  else
    ARGS+=(--no-co3d_val)
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

if [[ "${DEPTH_ACTIVATION_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--depth_activation_checkpoint)
else
  ARGS+=(--no_depth_activation_checkpoint)
fi
if [[ "${CAMERA_ACTIVATION_CHECKPOINT}" == "1" ]]; then
  ARGS+=(--camera_activation_checkpoint)
else
  ARGS+=(--no_camera_activation_checkpoint)
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
 LingBot-MAP Manip Training
========================================
[run]
  workdir       : ${SCRIPT_DIR}
  output_dir    : ${OUTPUT_DIR}
  tensorboard   : ${TENSORBOARD} (${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard})

[environment]
  cuda_devices  : ${CUDA_VISIBLE_DEVICES}
  conda_env     : ${CONDA_ENV}
  python        : ${PYTHON_CMD[*]}
  cuda_alloc    : ${PYTORCH_CUDA_ALLOC_CONF}
  empty_cache   : every ${EMPTY_CACHE_EVERY} optimizer step(s)

[data]
  roots         : ${DATA_ROOT_LONG6}
  manifest      : ${SCENE_MANIFEST}
  write_manifest: ${WRITE_MANIFEST:-<disabled>}
  ossutil       : ${OSSUTIL_BIN}
  oss_config    : ${OSSUTIL_CONFIG}
  oss_roots     : ${OSS_URI_LONG6}
  oss_fallback  : ${OSS_STAGE_FALLBACK} root=${OSS_STAGE_ROOT} max_entries=${OSS_STAGE_MAX_ENTRIES} worker_max=${OSS_STAGE_WORKER_MAX_ENTRIES}
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
  scannetpp_sampler: \
min_interval=${SCANNETPP_MIN_INTERVAL} max_interval=${SCANNETPP_MAX_INTERVAL} \
video_prob=${SCANNETPP_VIDEO_PROB} fix_interval_prob=${SCANNETPP_FIX_INTERVAL_PROB} \
block_shuffle=${SCANNETPP_BLOCK_SHUFFLE}
  arkitscenes_highres_root: ${ARKITSCENES_HIGHRES_ROOT:-<disabled>}
  arkitscenes_highres_views: [${ARKITSCENES_HIGHRES_MIN_VIEWS}, ${ARKITSCENES_HIGHRES_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  arkitscenes_highres_repeat: ${ARKITSCENES_HIGHRES_REPEAT}
  arkitscenes_highres_val: ${ARKITSCENES_HIGHRES_VAL}
  arkitscenes_highres_sampler (from base3d-clean/datasets/arkitscenes_highres.py; NOT Manip_long): min_interval=${ARKITSCENES_HIGHRES_MIN_INTERVAL} max_interval=${ARKITSCENES_HIGHRES_MAX_INTERVAL} video_prob=${ARKITSCENES_HIGHRES_VIDEO_PROB} fix_interval_prob=${ARKITSCENES_HIGHRES_FIX_INTERVAL_PROB} block_shuffle=${ARKITSCENES_HIGHRES_BLOCK_SHUFFLE}
  hypersim_root : ${HYPERSIM_ROOT:-<disabled>}
  hypersim_views: [${HYPERSIM_MIN_VIEWS}, ${HYPERSIM_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  hypersim_repeat: ${HYPERSIM_REPEAT}
  hypersim_val  : ${HYPERSIM_VAL}
  hypersim_sampler (from base3d-clean/datasets/hypersim.py; NOT Manip_long): \
min_interval=${HYPERSIM_MIN_INTERVAL} max_interval=${HYPERSIM_MAX_INTERVAL} \
video_prob=${HYPERSIM_VIDEO_PROB} fix_interval_prob=${HYPERSIM_FIX_INTERVAL_PROB} \
block_shuffle=${HYPERSIM_BLOCK_SHUFFLE}
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
  adt_root      : ${ADT_ROOT:-<disabled>}
  adt_views     : [${ADT_MIN_VIEWS}, ${ADT_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  adt_repeat    : ${ADT_REPEAT}     (mix-ratio bias)
  adt_val       : ${ADT_VAL}     (also mix into val loader)
  adt_sampler (from base3d-clean/datasets/adt.py; NOT Manip_long): max_distance=${ADT_MAX_DISTANCE}
  adt_rotate_clockwise: ${ADT_ROTATE_CLOCKWISE}
  ase_root      : ${ASE_ROOT:-<disabled>}
  ase_cache     : ${ASE_CACHE_PATH:-<root>/ase_cache.npy}
  ase_views     : [${ASE_MIN_VIEWS}, ${ASE_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  ase_repeat    : ${ASE_REPEAT}     (mix-ratio bias)
  ase_val       : ${ASE_VAL}     (also mix into val loader)
  ase_sampler (from base3d-clean/datasets/ase_20251008.py; NOT Manip_long): max_distance=${ASE_MAX_DISTANCE}
  co3d_root     : ${CO3D_ROOT:-<disabled>}
  co3d_oss_root : ${CO3D_OSS_URI_ROOT}
  co3d_views    : [${CO3D_MIN_VIEWS}, ${CO3D_NUM_VIEWS}] (0 -> follow [MIN,MAX]_SAMPLE_FRAMES=[${MIN_SAMPLE_FRAMES},${MAX_SAMPLE_FRAMES}])
  co3d_repeat   : ${CO3D_REPEAT}     (mix-ratio bias)
  co3d_val      : ${CO3D_VAL}     (also mix CO3D test split into val loader)
  co3d_mask_bg  : ${CO3D_MASK_BG}
  co3d_sampler (from base3d-clean/datasets/co3d_20251006.py; NOT Manip_long): random CO3D views per scene

[sampling]
  mode          : ${SEQUENCE_MODE}
  strategy      : ${SAMPLE_STRATEGY}
  random_stride : ${RANDOM_STRIDE_MIN}-${RANDOM_STRIDE_MAX}
  long6 W stride: ${W_STRIDE_MIN}-${W_STRIDE_MAX}
  long6 moving : ${MOVING_STRIDE_MIN}-${MOVING_STRIDE_MAX}
  long6 fixed  : ${FIXED_STRIDE_MIN}-${FIXED_STRIDE_MAX}
  seq_len       : ${MAX_SAMPLE_FRAMES}
  image_size    : ${IMAGE_SIZE}
  depth_chunk   : ${DEPTH_FRAMES_CHUNK_SIZE}
  depth_ckpt    : ${DEPTH_ACTIVATION_CHECKPOINT}
  camera_ckpt   : ${CAMERA_ACTIVATION_CHECKPOINT}
  frame_block   : ${NUM_FRAME_PER_BLOCK}
  kv_window     : ${KV_CACHE_SLIDING_WINDOW}

[manip_4d_mixed curriculum]  (only used when SEQUENCE_MODE=manip_4d_mixed)
  wrist_prefix  : ${WRIST_CAMERA_PREFIX}
  static_prefix : ${STATIC_CAMERA_PREFIX}
  long6 marker  : "${LONG6_ROOT_MARKER}"
  long6 modes   : ${LONG6_MODE_WEIGHTS}  (W=wrist, T=moving, F=fixed)
  long6 cameras : moving=${MOVING_CAMERA_PREFIX}, fixed=${FIXED_CAMERA_PREFIX}

[mixture curriculum]  (cross-dataset Manip-vs-external sampler)
  enabled       : ${MIXTURE_CURRICULUM}    (1 -> CurriculumMixtureSampler; 0 -> plain shuffle)
  p_manip       : ${MIXTURE_P_MANIP_START} -> ${MIXTURE_P_MANIP_END}  (linear over step [${MIXTURE_WARMUP_START}, ${MIXTURE_WARMUP_END}])
  externals     : DL3DV, ScanNet++, ARKitScenesHighRes, Hypersim, TartanAir, DynamicReplica, ADT, CO3D
  ext weights   : ${MIXTURE_EXTERNAL_WEIGHTS}
  per-ext share : (1 - p_manip) * normalized ext weight

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

if [[ "${LOSS_ONLY:-0}" == "1" ]]; then
  echo "[train.sh] LOSS_ONLY=1: running one forward+loss batch per dataset, no backward/optimizer/checkpoint."
  "${PYTHON_CMD[@]}" loss_only_one_step.py "${ARGS[@]}" "$@"
  exit $?
fi

# Run training in its own process group so Ctrl+C cleanly tears down the whole
# tree (conda run -> python -> DataLoader workers). Without this, Ctrl+C often
# leaves orphan worker processes holding GPU memory.
set -m
TRAIN_PID=""
cleanup() {
  trap - INT TERM EXIT
  if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo
    echo "[train.sh] caught signal, terminating training (pgid=${TRAIN_PID})..."
    kill -TERM "-${TRAIN_PID}" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "${TRAIN_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "-${TRAIN_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap cleanup INT TERM

"${PYTHON_CMD[@]}" "${REPO_DIR}/scripts/train/train.py" "${ARGS[@]}" "$@" &
TRAIN_PID=$!
wait "${TRAIN_PID}"
EXIT_CODE=$?
trap - INT TERM
exit "${EXIT_CODE}"
