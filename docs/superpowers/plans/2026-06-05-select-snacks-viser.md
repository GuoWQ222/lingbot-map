# Select Snacks Viser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run LingBot-MAP checkpoint inference on the selected snack RGB frame directory and serve an interactive viser point cloud.

**Architecture:** Reuse `demo.py` as the inference and viewer entrypoint. Add one shell wrapper with fixed paths and conservative defaults so the same visualization can be relaunched without retyping a long command.

**Tech Stack:** Bash, `conda run -n lingbot-map`, LingBot-MAP `demo.py`, `PointCloudViewer`, viser.

---

### Task 1: Add Launch Script

**Files:**
- Create: `/cpfs/user/guowenqi/lingbot-map/run_select_snacks_viser.sh`

- [x] **Step 1: Create the wrapper**

The script should run:

```bash
/cpfs/user/guowenqi/miniconda3/condabin/conda run -n lingbot-map python /cpfs/user/guowenqi/lingbot-map/demo.py \
  --image_folder /oss-guowenqi/Select_snacks_and_weigh_them/set39-1_collector1_20250712/0000002/observation/cam_left_wrist/color_image/rgb \
  --model_path /cpfs/user/guowenqi/lingbot-map/runs/manip_long_train_64gpu/checkpoint_step_00100000.pt \
  --image_size 280 \
  --model_image_size 280 \
  --mode streaming \
  --stride 7 \
  --use_sdpa \
  --offload_to_cpu \
  --port 8080 \
  --conf_threshold 1.5 \
  --downsample_factor 1 \
  --point_size 0.00001
```

- [x] **Step 2: Validate shell syntax**

Run:

```bash
bash -n /cpfs/user/guowenqi/lingbot-map/run_select_snacks_viser.sh
```

Expected: no output and exit code 0.

### Task 2: Launch Viewer

**Files:**
- Execute: `/cpfs/user/guowenqi/lingbot-map/run_select_snacks_viser.sh`

- [x] **Step 1: Start the viewer**

Run:

```bash
bash /cpfs/user/guowenqi/lingbot-map/run_select_snacks_viser.sh
```

Expected: `demo.py` loads about 63 frames from the 439-frame directory, loads `checkpoint_step_00100000.pt`, completes inference, and starts viser at `http://localhost:8080`.

- [x] **Step 2: Report status**

Report the viewer URL, script path, and any failure details if inference does not reach the viewer stage.
