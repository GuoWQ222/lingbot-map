# LingBot-Map Embodied Fine-tuning

We thank the original LingBot-Map repository and its authors for releasing the
codebase and model foundation that made this embodied fine-tuning work possible.

This fork contains a self-contained training and evaluation pipeline for
fine-tuning LingBot-Map on embodied RGB-D manipulation trajectories.

The pipeline focuses on Manip-style robot data: multi-camera RGB images, real
depth, segmentation masks, intrinsics, and per-frame camera poses. Training uses
LingBot-Map's causal streaming inference path and optimizes camera,
relative-pose, and depth losses. Evaluation reuses the held-out Manip split and
reports depth, camera trajectory, field-of-view, and point-cloud reconstruction
metrics, with separate summaries for wrist/realsense and surround-camera tracks.

<div align="center">
  <img src="assets/teaser.png" width="100%">
</div>

## What is in this fork

- `train.py` / `train.sh`: Manip RGB-D fine-tuning with trajectory discovery,
  preprocessing, data mixing, VGGT-style losses, TensorBoard logging, and
  checkpointing.
- `eval.py` / `eval.sh`: evaluation for one checkpoint or a directory of
  checkpoints, producing `metrics.json`, `per_scene.csv`, and batch
  `summary.csv` outputs.
- `eval_pi3.py`, `eval_vggt.py`, `eval_loger.py`, `eval_scal3r.py`,
  `eval_streamvggt.py`, `eval_ttt3r.py`: companion evaluators for comparison
  against related reconstruction backends.
- `train_64gpu.sh`, `train_8gpu.sh`, `train_window32*.sh`: launch presets for
  larger training runs.
- `train_sh_loss_formulas.md` and `losses.md`: detailed notes for the current
  training objective.

Demo and visualization utilities are available via
`demo.py`, `visualize_scene.py`, and `lingbot_map/vis/`.

## Installation

Create an environment and install the package in editable mode:

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

Optional visualization dependencies:

```bash
pip install -e ".[vis]"
```

FlashInfer is recommended for high-throughput streaming inference. If it is not
installed, the model can use PyTorch SDPA via `--use_sdpa`.

```bash
pip install flashinfer-python -i https://flashinfer.ai/whl/cu128/torch2.9/
```

Evaluation of camera trajectory metrics uses `evo`; point-cloud metrics use
SciPy nearest-neighbor routines:

```bash
pip install evo scipy
```

## Checkpoints

The base LingBot-Map checkpoint is expected by default at:

```text
./lingbot-map.pt
```

You can override it when training or running demos:

```bash
MODEL_PATH=/path/to/lingbot-map.pt bash train.sh

python demo.py --model_path /path/to/checkpoint.pt \
  --image_folder example/church --mask_sky
```

Training checkpoints saved by this fork use the `train.py` format:

- `checkpoint_step_XXXXXXXX.pt`
- `checkpoint_epoch_XXXX.pt`
- `checkpoint_last.pt`

Each checkpoint stores the model, optimizer, scheduler, AMP scaler, RNG state,
training arguments, and a trainable-parameter fingerprint for safer resume.

## Manip data layout

The Manip dataset loader discovers trajectory directories under the roots passed
with `--data_roots`. A trajectory is expected to contain per-camera
subdirectories. Each camera directory should include:

```text
<trajectory>/
  <camera_name>/
    images/
      000000.png
      ...
    depth_real/
      000000.png
      ...
    mask/                 # optional, used when --use_mask is set
      000000.png
      ...
    <pose file>           # discovered by train.py for intrinsics and poses
```

Camera names are used by the sampling code. By default:

- `realsense*` cameras are treated as wrist-camera tracks.
- `surround*` cameras are treated as static/surround tracks.
- paths containing `Manip_long5` are routed to mode `T`, a single surround-camera
  trajectory mode designed for moving surround cameras.

The launcher defaults to internal paths:

```text
/oss-guowenqi/Manip_long3/data
/oss-guowenqi/Manip_long4/data
/oss-guowenqi/Manip_long5/data
```

Override them for your machine:

```bash
DATA_ROOT_LONG3=/path/to/Manip_long3/data \
DATA_ROOT_LONG4=/path/to/Manip_long4/data \
DATA_ROOT_LONG5=/path/to/Manip_long5/data \
bash train.sh
```

`train.py` can also cache trajectory discovery into a manifest. `train.sh`
defaults to:

```text
${OUTPUT_DIR}/manip_trajectory_manifest.txt
```

The train/validation split is deterministic, controlled by `VAL_FRACTION`
and `--seed` (`VAL_FRACTION=0.02`, seed `42` by default).

## Training

The simplest entry point is:

```bash
cd /path/to/lingbot-map
bash train.sh
```

Useful overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
OUTPUT_DIR=runs/manip_long_train \
MODEL_PATH=./lingbot-map.pt \
MAX_STEPS=100000 \
NUM_WORKERS=4 \
bash train.sh
```

The script prints the effective configuration before launch. Important defaults
from `train.sh` are:

| Setting | Default |
| --- | --- |
| Manip roots | `DATA_ROOT_LONG3`, `DATA_ROOT_LONG4`, `DATA_ROOT_LONG5` |
| Output dir | `runs/manip_long_train` |
| Batch size | `1` |
| Gradient accumulation | `ACCUM_STEPS=1` |
| Max steps | `MAX_STEPS=100000` |
| Learning rate | `5e-5` with cosine decay to `1e-8` |
| Weight decay | `0.05` |
| Sequence mode | `manip_4d_mixed` |
| Sampling strategy | `random_interval` |
| Frames per sample | up to `64`, minimum `16` |
| Image size | `280` |
| Depth chunk size | `2` during training |
| Validation fraction | `0.02` |
| TensorBoard | enabled, written to `${OUTPUT_DIR}/tensorboard` |

### Manip sequence sampling

For the current launcher defaults, sampling is effectively:

- Manip_long3/4: mode `W`, a single `realsense*` wrist-camera trajectory with
  random temporal stride in `[10, 60]`.
- Manip_long5: mode `T`, a single `surround_cam*` trajectory with random start
  and stride in `[15, 60]`.

The older `S` and `M` branches remain in the code for multi-view snapshot and
4D grid experiments, but `train.sh` sets:

```text
MODE_WEIGHTS_INITIAL=S=0.0,W=1.0,M=0.0
MODE_WEIGHTS_FINAL=S=0.0,W=1.0,M=0.0
```

so those branches are inactive unless you override the weights.

### Cross-dataset curriculum

`train.sh` enables `MIXTURE_CURRICULUM=1` by default. The DataLoader uses
`CurriculumMixtureSampler` to sample each batch from Manip with probability
`p_manip`, linearly increasing from `0.50` to `0.90` between steps `2000` and
`15000`. The remaining probability is split uniformly across enabled external
datasets.

Supported external mix-ins:

- DL3DV
- ScanNet++ v2
- TartanAir
- DynamicReplica
- MapFree

Each external dataset is enabled if its `*_ROOT` variable is non-empty. Clear a
root to disable that source:

```bash
DL3DV_ROOT= \
SCANNETPP_ROOT= \
TARTANAIR_ROOT= \
DYNAMIC_REPLICA_ROOT= \
MAPFREE_ROOT= \
bash train.sh
```

### Augmentation and preprocessing

Preprocessing is deterministic and applies the same crop/pad geometry to RGB,
depth, masks, and intrinsics. The launcher uses `IMAGE_SIZE=280` and the
`crop` preprocessing mode unless overridden.

Augmentation is RGB-only clip-level ColorJitter. With `COLOR_JITTER_STRENGTH=0.2`
and `COLOR_JITTER_PROB=0.5`, one jitter parameter set is sampled per clip and
applied consistently across all frames. Validation disables ColorJitter.

### Training objective

`VGGTStyleLoss` combines:

```text
L = camera_weight * L_camera
  + relative_pose_weight * L_relative_pose
  + depth_weight * (L_conf_depth + L_reg_depth + L_grad_depth)
```

The launcher uses:

```text
camera_weight=5.0
relative_pose_weight=1.0
depth_weight=1.0
point_weight=0.0
```

Point-map loss is disabled for this fine-tuning path. See
`train_sh_loss_formulas.md` for a fuller derivation of the loss terms and log
keys.

### Training outputs

Every run writes:

```text
<OUTPUT_DIR>/
  args.json
  manip_trajectory_manifest.txt
  tensorboard/
  checkpoint_step_XXXXXXXX.pt
  checkpoint_epoch_XXXX.pt
  checkpoint_last.pt
```

Resume a compatible run with:

```bash
RESUME=/path/to/checkpoint_last.pt bash train.sh
```

## Evaluation

Evaluate one checkpoint:

```bash
CHECKPOINT=/path/to/checkpoint_step_00010000.pt bash eval.sh
```

Evaluate every checkpoint in a run directory:

```bash
CHECKPOINT_DIR=runs/manip_long_train_64gpu \
CHECKPOINT_GLOB='checkpoint_*.pt' \
bash eval.sh
```

`eval.sh` automatically uses `<checkpoint_parent>/args.json` unless
`TRAIN_ARGS_JSON` is provided. This keeps evaluation aligned with the train-time
data roots, manifest, preprocessing, and split.

Important evaluation defaults:

| Setting | Default |
| --- | --- |
| Split | `val` |
| Strategy | `manip_track` |
| Frames | `64`, selected by deterministic linspace |
| Seed | `42` |
| Geometry normalization | `none` |
| Camera alignment | `sim3` |
| Depth alignment | `pi3_scale_shift` |
| Point-cloud metrics | enabled |
| Point-cloud alignment | `pi3_icp` |
| Per-scene CSV | enabled |

`manip_track` evaluates:

- Manip_long3/4 with a single realsense track.
- Manip_long5 with surround-camera tracks. If `EVAL_SURROUND_CAMERA_NAME` is
  empty, all six surround tracks are evaluated; otherwise the named surround
  camera is used.

Examples:

```bash
# Use a single surround camera for Long5.
EVAL_SURROUND_CAMERA_NAME=surround_cam_0 \
CHECKPOINT=/path/to/checkpoint.pt \
bash eval.sh

# Limit evaluation for a quick smoke test.
MAX_SCENES_EVAL=20 \
POINTCLOUD_METRICS=0 \
CHECKPOINT=/path/to/checkpoint.pt \
bash eval.sh
```

Evaluation outputs:

```text
<checkpoint_parent>/eval/<checkpoint_stem>/
  metrics.json
  per_scene.csv
  predictions/          # only when SAVE_PREDICTIONS=1
```

Batch checkpoint mode also writes:

```text
<CHECKPOINT_DIR>/eval/summary.csv
```

`metrics.json` contains an overall summary plus camera-family groups:

```text
modes.<mode>.overall
modes.<mode>.groups.realsense
modes.<mode>.groups.surround
```

Depth metrics are averaged per frame. Camera ATE/RPE metrics are computed with
`evo` and averaged per trajectory. FoV errors are averaged per frame. Point
cloud metrics (`ACC`, `Completeness`, `CD`) are computed per clip with
nearest-neighbor distances.

## Parallel and comparison evaluation

The repository includes launchers for faster or comparative evaluation:

```bash
bash eval_full_parallel.sh
bash eval_fast_parallel.sh

EVAL_BACKEND=ttt3r bash eval.sh
bash eval_pi3.sh
bash eval_vggt.sh
bash eval_loger.sh
bash eval_scal3r.sh
bash eval_streamvggt.sh
```

These scripts share the same idea: keep the Manip validation split and metric
definitions fixed, while changing the backend or parallelism strategy.

## Demo and visualization

Run the viewer:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
  --image_folder example/church --mask_sky
```

For a video:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
  --video_path video.mp4 --fps 10
```

For long sequences:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
  --video_path video.mp4 --fps 10 \
  --mode windowed --window_size 128
```

`visualize_scene.py` can also be used to inspect Manip samples and predictions
from the training/evaluation pipeline.

## Notes for publishing the fork

- `runs/` is intentionally not required for the repository. Training and
  evaluation outputs can be regenerated from checkpoints and data roots.
- The launcher defaults contain local CPFS/OSS paths. Override them in your
  environment or edit the scripts before running on another machine.
- Large binary checkpoints such as `lingbot-map.pt` are usually better hosted as
  release assets, Git LFS objects, or external model files rather than ordinary
  Git blobs.
