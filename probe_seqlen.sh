#!/usr/bin/env bash
# probe_seqlen.sh — probe the largest sequence length a single 4090 can train.
#
# Wraps train.sh: keeps the entire training pipeline (AMP, scheduler, optimizer
# state, freezes, modes, image size, depth-chunking, activation checkpointing)
# byte-identical to production, but pins MAX_SAMPLE_FRAMES = MIN_SAMPLE_FRAMES
# = CLIP_LEN = SEQ_LEN so every clip has exactly the length under test (no
# random truncation hiding peak memory).
#
# Usage:
#   ./probe_seqlen.sh <SEQ_LEN>
#   SEQ_LEN=<N> ./probe_seqlen.sh
#
# Examples:
#   ./probe_seqlen.sh 48                 # baseline (current train.sh)
#   ./probe_seqlen.sh 64
#   SEQ_LEN=96 ./probe_seqlen.sh
#   for n in 48 64 80 96 112 128; do ./probe_seqlen.sh "$n" || break; done
#
# Exit code:
#   0           SEQ_LEN survived MAX_STEPS optimizer steps without OOM
#   non-zero    OOM or other failure (inspect stderr / runs/seqlen_probe/T${N}/)
#
# Overrideable env vars (with probe-friendly defaults):
#   MAX_STEPS              total optimizer steps to run (default 50)
#   LIMIT_TRAIN_BATCHES    per-epoch batch cap (default 50)
#   IMAGE_SIZE             input resolution (default 280, matches train.sh)
#   ACCUM_STEPS            grad-accum (default 1, matches train.sh)
#   CUDA_VISIBLE_DEVICES   GPU index (default 0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----- sequence length: positional arg OR env var -----
if [[ -z "${SEQ_LEN:-}" && -n "${1:-}" ]]; then
  SEQ_LEN="$1"
  shift
fi
if [[ -z "${SEQ_LEN:-}" ]]; then
  cat >&2 <<USAGE
usage:
  $(basename "$0") <SEQ_LEN>
  SEQ_LEN=<N> $(basename "$0")
example:
  $(basename "$0") 64
USAGE
  exit 2
fi
if ! [[ "${SEQ_LEN}" =~ ^[0-9]+$ ]] || (( SEQ_LEN < 1 )); then
  echo "[probe_seqlen] SEQ_LEN must be a positive integer, got '${SEQ_LEN}'" >&2
  exit 2
fi

# ----- pin sequence length -----
export CLIP_LEN="${SEQ_LEN}"
export MAX_SAMPLE_FRAMES="${SEQ_LEN}"
export MIN_SAMPLE_FRAMES="${SEQ_LEN}"

# manip_4d_mixed mode S samples a single-timestamp multi-view snapshot; its
# frame count is V (not SEQ_LEN), so it wouldn't actually exercise SEQ_LEN.
# Force the W/T branches that produce true-length trajectories.
# (Current train.sh weights already pin S=0/W=1/M=0, but re-export to make
# the probe robust against a user-customized environment.)
export MODE_WEIGHTS_INITIAL="${MODE_WEIGHTS_INITIAL:-S=0.0,W=1.0,M=0.0}"
export MODE_WEIGHTS_FINAL="${MODE_WEIGHTS_FINAL:-S=0.0,W=1.0,M=0.0}"

# ----- shorten the run, skip val/save, keep everything else real -----
export EPOCHS="${EPOCHS:-1}"
export MAX_STEPS="${MAX_STEPS:-50}"
export LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-50}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0}"
export VAL_EVERY="${VAL_EVERY:-1000000}"
export SAVE_EVERY="${SAVE_EVERY:-1000000}"
export LOG_EVERY="${LOG_EVERY:-1}"
export TENSORBOARD="${TENSORBOARD:-0}"

# ----- per-SEQ_LEN output dir so successive probes don't clobber each other -----
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/runs/seqlen_probe/T${SEQ_LEN}}"

# Allocator fragmentation matters for peak-memory measurements; be explicit so
# the probe result is reproducible regardless of caller's shell.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cat <<EOF

========================================
 LingBot-MAP SeqLen Probe
========================================
  SEQ_LEN              : ${SEQ_LEN}    (CLIP_LEN = MIN = MAX = ${SEQ_LEN})
  IMAGE_SIZE           : ${IMAGE_SIZE:-280}
  BATCH_SIZE           : 1
  ACCUM_STEPS          : ${ACCUM_STEPS:-1}
  MAX_STEPS            : ${MAX_STEPS}
  LIMIT_TRAIN_BATCHES  : ${LIMIT_TRAIN_BATCHES}
  OUTPUT_DIR           : ${OUTPUT_DIR}
  CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES:-0}
  PYTORCH_CUDA_ALLOC   : ${PYTORCH_CUDA_ALLOC_CONF}
----------------------------------------
  exit 0   -> SEQ_LEN fits  (full train step incl. optimizer/AMP)
  non-zero -> OOM or failure (see stderr above)
========================================

EOF

exec "${SCRIPT_DIR}/train.sh" "$@"
