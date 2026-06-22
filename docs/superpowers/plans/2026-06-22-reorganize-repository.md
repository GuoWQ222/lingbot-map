# Reorganize Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize `lingbot-map-copy` into a clearer medium-depth project structure without moving the main package or demo entrypoint.

**Architecture:** Keep public root files stable, move operational scripts into `scripts/*`, and move generated or bulky artifacts into `outputs/`, `checkpoints/`, `reports/`, and `vendor/`. Patch shell scripts that rely on old root-relative locations so they resolve `REPO_DIR` from their new directory.

**Tech Stack:** Bash, Python package layout via `pyproject.toml`, Git worktree metadata already present in the repository.

---

### Task 1: Create Target Directories

**Files:**
- Create directories under `/cpfs/user/guowenqi/lingbot-map-copy`

- [ ] Create `scripts/train`, `scripts/eval`, `scripts/visualize`, `scripts/tools`
- [ ] Create `checkpoints`, `outputs/eval`, `outputs/logs`, `outputs/cache`, `reports`, `vendor`
- [ ] Confirm all directories exist with `find . -maxdepth 2 -type d`

### Task 2: Move Files By Responsibility

**Files:**
- Move root files and directories only; do not delete content.

- [ ] Move train files: `train.py`, `train_multinode.py`, `train*.sh`, `train_sh_loss_formulas.md` to `scripts/train/`
- [ ] Move eval files: `eval*.py`, `eval*.sh`, `aggregate_streamvggt_shards.py`, `run_eval_36shard_moving_allpoints.sh`, `run_streamvggt_8gpu_eval.sh` to `scripts/eval/`
- [ ] Move visualization files: `visualize_*.py`, `run_select_snacks_viser.sh` to `scripts/visualize/`
- [ ] Move tools: `diag_*.py`, `probe_*.sh`, `gct_profile.py`, `loss_only_one_step.py`, `run_one_step_dataset_losses.py`, `split_robo3r_cameras.py`, root `test_*.py`, and `test_4gpu_speed.sh` to `scripts/tools/`
- [ ] Move `lingbot-map.pt` to `checkpoints/`
- [ ] Move eval output directories to `outputs/eval/`
- [ ] Move root logs and `logs/` to `outputs/logs/`
- [ ] Move `.pytest_cache`, `__pycache__`, `ossutil_output`, and `lingbot_map.egg-info` to `outputs/cache/`
- [ ] Move paper, losses, summary, PDF, and docx files to `reports/`
- [ ] Move `cuda-keyring_1.1-1_all.deb` to `vendor/`

### Task 3: Patch Moved Script Paths

**Files:**
- Modify moved shell scripts and README as needed.

- [ ] In `scripts/train/*.sh`, set `REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"` and point Python entrypoints to `${REPO_DIR}/scripts/train/*.py`
- [ ] In `scripts/eval/*.sh`, set `REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"` and point Python entrypoints to `${REPO_DIR}/scripts/eval/*.py`
- [ ] In `scripts/visualize/run_select_snacks_viser.sh`, derive `REPO_DIR` from `../..` and keep running root `demo.py`
- [ ] Replace checkpoint defaults from `${SCRIPT_DIR}/lingbot-map.pt` with `${REPO_DIR}/checkpoints/lingbot-map.pt`
- [ ] Update README references for PDF, checkpoints, and optional training/evaluation script locations

### Task 4: Verify

**Files:**
- Repository tree and scripts.

- [ ] Run `python -m py_compile demo.py scripts/train/train.py scripts/eval/eval.py`
- [ ] Run `bash -n` on moved `.sh` files
- [ ] Run `grep -RInE` to find stale direct references to moved root files
- [ ] Print the final top-level tree with `find . -maxdepth 2`
