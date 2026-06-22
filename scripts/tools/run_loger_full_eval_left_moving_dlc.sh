#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

echo "[loger_full_eval] entered $(date -u +%Y-%m-%dT%H:%M:%SZ) cwd=$(pwd)"

if ! grep -Fq 'default="left_moving_tracks"' "${REPO_DIR}/scripts/eval/eval.py"; then
  echo "[loger_full_eval] ERROR: eval.py default eval_strategy is not left_moving_tracks" >&2
  exit 2
fi
if ! grep -Fq 'default="left_moving_tracks"' "${REPO_DIR}/scripts/eval/eval_loger.py"; then
  echo "[loger_full_eval] ERROR: eval_loger.py default eval_strategy is not left_moving_tracks" >&2
  exit 2
fi
if ! grep -Fq 'EVAL_STRATEGY="${EVAL_STRATEGY:-left_moving_tracks}"' "${REPO_DIR}/scripts/eval/eval_loger.sh"; then
  echo "[loger_full_eval] ERROR: "${REPO_DIR}/scripts/eval/eval_loger.sh" default EVAL_STRATEGY is not left_moving_tracks" >&2
  exit 2
fi

START_TS=$(date +%s)
RUN_ID="${RUN_ID:-loger_full_val449_left_moving_evalpy_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/eval/${RUN_ID}}"
TRAIN_ARGS_JSON="${REPO_DIR}/outputs/runs/manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/args.json"
CUDA_DEVICE_LIST="${CUDA_DEVICE_LIST:-0,1,2,3,4,5,6,7}"
SHARDS_PER_GPU="${SHARDS_PER_GPU:-2}"
EXPECTED_SCENES="${EXPECTED_SCENES:-449}"
EXPECTED_CLIPS="$((EXPECTED_SCENES * 2))"

EVAL_STRATEGY="left_moving_tracks"
EVAL_NUM_FRAMES="64"
EVAL_WRIST_CAMERA_NAME="realsense_left"
EVAL_SURROUND_CAMERA_NAME="surround_cam_moving"
EVAL_SEED="42"
GEOMETRY_NORMALIZATION="none"
CAMERA_ALIGN="sim3"
DEPTH_ALIGN="pi3_scale_shift"
IMAGE_SIZE="0"
POINTCLOUD_METRICS="1"
POINTCLOUD_MAX_POINTS="100000"
POINTCLOUD_ALIGN="pi3_icp"
POINTCLOUD_ICP_THRESHOLD="0.1"
POINTCLOUD_ICP_MAX_ITERATIONS="30"
POINTCLOUD_ICP_BACKEND="auto"
POINTCLOUD_KDTREE_WORKERS="1"
POINTCLOUD_WORKERS="1"
LOGER_INPUT_PREPROCESS="native"
LOGER_NATIVE_WIDTH="504"
LOGER_NATIVE_HEIGHT="280"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICE_LIST}"
GPU_COUNT="${#GPU_IDS[@]}"
SHARD_COUNT="$((GPU_COUNT * SHARDS_PER_GPU))"

if [[ -e "${OUTPUT_ROOT}" ]] && [[ -n "$(find "${OUTPUT_ROOT}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "[loger_full_eval] ERROR: OUTPUT_ROOT exists and is non-empty: ${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}/logs"

echo "[loger_full_eval] output_root=${OUTPUT_ROOT}"
echo "[loger_full_eval] train_args=${TRAIN_ARGS_JSON}"
echo "[loger_full_eval] gpus=${CUDA_DEVICE_LIST} shards=${SHARD_COUNT} eval_strategy=${EVAL_STRATEGY} frames=${EVAL_NUM_FRAMES} expected_clips=${EXPECTED_CLIPS}"
echo "[loger_full_eval] metrics will be saved to ${OUTPUT_ROOT}/combined_metrics.json"

PIDS=()
LOGS=()
for SHARD_IDX in $(seq 0 $((SHARD_COUNT - 1))); do
  GPU="${GPU_IDS[$((SHARD_IDX % GPU_COUNT))]}"
  SHARD_DIR="${OUTPUT_ROOT}/shard_${SHARD_IDX}"
  LOG_PATH="${OUTPUT_ROOT}/logs/shard_${SHARD_IDX}.log"
  mkdir -p "${SHARD_DIR}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU}"
    export TRAIN_ARGS_JSON="${TRAIN_ARGS_JSON}"
    export OUTPUT_DIR="${SHARD_DIR}"
    export SPLIT="val"
    export MAX_SCENES_EVAL="0"
    export EVAL_SHARD_COUNT="${SHARD_COUNT}"
    export EVAL_SHARD_INDEX="${SHARD_IDX}"
    export NUM_WORKERS="${NUM_WORKERS:-8}"
    export DEVICE="cuda"
    export PER_SCENE_CSV="1"
    export SAVE_PREDICTIONS="0"
    export PRINT_EVERY="10"
    export EVAL_STRATEGY="${EVAL_STRATEGY}"
    export EVAL_NUM_FRAMES="${EVAL_NUM_FRAMES}"
    export EVAL_WRIST_CAMERA_NAME="${EVAL_WRIST_CAMERA_NAME}"
    export EVAL_SURROUND_CAMERA_NAME="${EVAL_SURROUND_CAMERA_NAME}"
    export EVAL_SEED="${EVAL_SEED}"
    export GEOMETRY_NORMALIZATION="${GEOMETRY_NORMALIZATION}"
    export CAMERA_ALIGN="${CAMERA_ALIGN}"
    export DEPTH_ALIGN="${DEPTH_ALIGN}"
    export IMAGE_SIZE="${IMAGE_SIZE}"
    export POINTCLOUD_METRICS="${POINTCLOUD_METRICS}"
    export POINTCLOUD_MAX_POINTS="${POINTCLOUD_MAX_POINTS}"
    export POINTCLOUD_ALIGN="${POINTCLOUD_ALIGN}"
    export POINTCLOUD_ICP_THRESHOLD="${POINTCLOUD_ICP_THRESHOLD}"
    export POINTCLOUD_ICP_MAX_ITERATIONS="${POINTCLOUD_ICP_MAX_ITERATIONS}"
    export POINTCLOUD_ICP_BACKEND="${POINTCLOUD_ICP_BACKEND}"
    export POINTCLOUD_KDTREE_WORKERS="${POINTCLOUD_KDTREE_WORKERS}"
    export POINTCLOUD_WORKERS="${POINTCLOUD_WORKERS}"
    export LOGER_INPUT_PREPROCESS="${LOGER_INPUT_PREPROCESS}"
    export LOGER_NATIVE_WIDTH="${LOGER_NATIVE_WIDTH}"
    export LOGER_NATIVE_HEIGHT="${LOGER_NATIVE_HEIGHT}"
    bash "${REPO_DIR}/scripts/eval/eval_loger.sh"
  ) > "${LOG_PATH}" 2>&1 &
  PIDS+=("$!")
  LOGS+=("${LOG_PATH}")
  echo "[loger_full_eval] launched shard=${SHARD_IDX}/${SHARD_COUNT} gpu=${GPU} log=${LOG_PATH}"
done

FAILED=0
for IDX in "${!PIDS[@]}"; do
  PID="${PIDS[$IDX]}"
  if ! wait "${PID}"; then
    echo "[loger_full_eval] ERROR: shard ${IDX} failed, log=${LOGS[$IDX]}" >&2
    tail -120 "${LOGS[$IDX]}" >&2 || true
    FAILED=1
  fi
done
if [[ "${FAILED}" != "0" ]]; then
  exit 1
fi

END_TS=$(date +%s)
ELAPSED_SEC="$((END_TS - START_TS))"

python - "${OUTPUT_ROOT}" "${EXPECTED_SCENES}" "${EXPECTED_CLIPS}" "${ELAPSED_SEC}" <<'PYAGG'
import csv, datetime as dt, glob, json, math, os, statistics, sys
from collections import defaultdict
out=sys.argv[1]; expected_scenes=int(sys.argv[2]); expected_clips=int(sys.argv[3]); elapsed_sec=int(sys.argv[4])
files=sorted(glob.glob(os.path.join(out,'shard_*','per_scene.csv')))
rows=[]; fieldnames=None
for path in files:
    with open(path,newline='') as f:
        reader=csv.DictReader(f)
        if fieldnames is None: fieldnames=reader.fieldnames
        rows.extend(reader)
if len(rows)!=expected_clips:
    raise SystemExit(f'expected {expected_clips} per-scene clip rows ({expected_scenes} scenes * 2 cameras), got {len(rows)}')
if fieldnames is None: raise SystemExit('no per_scene.csv files found')
combined_csv=os.path.join(out,'combined_per_scene.csv')
with open(combined_csv,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
def num(r,k):
    v=r.get(k,'')
    return float(v) if v not in ('',None) else float('nan')
def finite(xs): return [x for x in xs if math.isfinite(x)]
def mean(xs):
    xs=finite(xs); return sum(xs)/len(xs) if xs else None
def median(xs):
    xs=finite(xs); return statistics.median(xs) if xs else None
def dw(rs,k,squared=False):
    total=sum(num(r,'depth_valid_pixels_total') for r in rs)
    if total<=0: return None
    if squared: return math.sqrt(sum((num(r,k)**2)*num(r,'depth_valid_pixels_total') for r in rs)/total)
    return sum(num(r,k)*num(r,'depth_valid_pixels_total') for r in rs)/total
def summarize(rs):
    return {
      'clips': len(rs),
      'depth': {
        'AbsRel': dw(rs,'depth_AbsRel'), 'SqRel': dw(rs,'depth_SqRel'), 'RMSE': dw(rs,'depth_RMSE',True), 'MAE': dw(rs,'depth_MAE'),
        'LogRMSE': dw(rs,'depth_LogRMSE',True), 'log10': dw(rs,'depth_log10'), 'delta<1.25': dw(rs,'depth_delta<1.25'),
        'delta<1.25^2': dw(rs,'depth_delta<1.25^2'), 'delta<1.25^3': dw(rs,'depth_delta<1.25^3'),
        'si-RMSE': dw(rs,'depth_si-RMSE'), 'si-RMSE_note': 'pixel-weighted approximation from per-clip summaries',
        'valid_pixels_total': sum(num(r,'depth_valid_pixels_total') for r in rs), 'align_scale': dw(rs,'depth_align_scale'), 'align_shift': dw(rs,'depth_align_shift')},
      'camera': {
        'ate_rmse_mean': mean([num(r,'cam_ate_rmse') for r in rs]), 'ate_rmse_median': median([num(r,'cam_ate_rmse') for r in rs]),
        'rpe_trans_rmse_mean': mean([num(r,'cam_rpe_trans_rmse') for r in rs]), 'rpe_trans_rmse_median': median([num(r,'cam_rpe_trans_rmse') for r in rs]),
        'rpe_rot_rmse_deg_mean': mean([num(r,'cam_rpe_rot_rmse_deg') for r in rs]), 'rpe_rot_rmse_deg_median': median([num(r,'cam_rpe_rot_rmse_deg') for r in rs]),
        'n_sequences_for_traj': len(rs)},
      'pointcloud': {
        'ACC_mean': mean([num(r,'pc_ACC') for r in rs]), 'ACC_median': median([num(r,'pc_ACC') for r in rs]),
        'Completeness_mean': mean([num(r,'pc_Completeness') for r in rs]), 'Completeness_median': median([num(r,'pc_Completeness') for r in rs]),
        'CD_mean': mean([num(r,'pc_CD') for r in rs]), 'CD_median': median([num(r,'pc_CD') for r in rs]),
        'n_pred_points_mean': mean([num(r,'pc_n_pred_points') for r in rs]), 'n_pred_points_median': median([num(r,'pc_n_pred_points') for r in rs]),
        'n_gt_points_mean': mean([num(r,'pc_n_gt_points') for r in rs]), 'n_gt_points_median': median([num(r,'pc_n_gt_points') for r in rs])}}
groups=defaultdict(list)
for r in rows: groups[r.get('metric_group') or 'unknown'].append(r)
report={
  'model':'LoGeR', 'checkpoint':'/cpfs/user/guowenqi/LoGeR/ckpts/LoGeR/latest.pt',
  'train_args_json':'/outputs/runs/manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/args.json',
  'split':'val', 'eval_strategy':'left_moving_tracks', 'eval_num_frames':64,
  'eval_wrist_camera_name':'realsense_left', 'eval_surround_camera_name':'surround_cam_moving',
  'expected_scenes':expected_scenes, 'expected_clips':expected_clips, 'clips_evaluated':len(rows), 'shard_count':len(files),
  'loger_native_width':504, 'loger_native_height':280, 'pointcloud_metrics':True,
  'aggregation':'overall plus per-camera groups; depth pixel-weighted from per-clip summaries; camera macro over clips; pointcloud macro over clips',
  'elapsed_sec':elapsed_sec, 'elapsed_hms':str(dt.timedelta(seconds=elapsed_sec)), 'combined_per_scene_csv':combined_csv,
  'metrics': {'overall': summarize(rows), 'by_camera': {k:summarize(v) for k,v in sorted(groups.items())}}}
metrics_json=os.path.join(out,'combined_metrics.json')
with open(metrics_json,'w') as f: json.dump(report,f,indent=2,sort_keys=True)
print(json.dumps(report,indent=2,sort_keys=True))
print(f'METRICS_JSON={metrics_json}')
print(f'PER_SCENE_CSV={combined_csv}')
print(f'LOG_DIR={os.path.join(out,"logs")}')
PYAGG
