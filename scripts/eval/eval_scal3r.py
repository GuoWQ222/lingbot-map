#!/usr/bin/env python3
"""Evaluate Scal3R on the LingBot-MAP Manip validation protocol.

The metrics are reused from eval.py. The eval dataloader mirrors Scal3R native
resize/crop geometry onto GT depth, mask, intrinsics, and world points before
Scal3R is invoked as an external folder-in/folder-out reconstructor.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = REPO_DIR / "outputs" / "runs" / "manip_long_train_64gpu"
DEFAULT_SCAL3R_REPO = REPO_DIR.parent / "Scal3R"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval as E
import eval_model_geometry as EMG
import train as T
from lingbot_map.utils.rotation import mat_to_quat


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Scal3R with LingBot-MAP Manip metrics")
    p.add_argument("--train_args_json", type=str, default=str(DEFAULT_RUN_DIR / "args.json"),
                   help="LingBot-MAP args.json used to recreate the Manip eval split.")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_RUN_DIR / "eval_scal3r"))
    p.add_argument("--scal3r_repo", type=str, default=str(DEFAULT_SCAL3R_REPO))
    p.add_argument("--scal3r_python", type=str, default=sys.executable,
                   help="Python command used to run Scal3R, e.g. 'conda run -n scal3r python'.")
    p.add_argument("--scal3r_config", type=str, default="configs/models/scal3r.yaml")
    p.add_argument("--scal3r_checkpoint", type=str, default="")
    p.add_argument("--scal3r_device", type=str, default="cuda")
    p.add_argument("--scal3r_preprocess_workers", type=int, default=8)
    p.add_argument("--scal3r_block_size", type=int, default=60)
    p.add_argument("--scal3r_overlap_size", type=int, default=30)
    p.add_argument("--scal3r_use_loop", type=int, default=1)
    p.add_argument("--scal3r_use_xyz_align", type=int, default=0)
    p.add_argument("--scal3r_pgo_workers", type=int, default=8)
    p.add_argument("--scal3r_save_xyz", type=int, default=0,
                   help="Save Scal3R PLY outputs. Metrics unproject depth+pose, so 0 is usually enough.")
    p.add_argument("--scal3r_test_use_amp", action="store_true")
    p.add_argument("--force_rerun_scal3r", action="store_true")
    p.add_argument("--keep_scal3r_inputs", action="store_true")

    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max_scenes_eval", type=int, default=0)
    p.add_argument("--eval_shard_count", type=int, default=1,
                   help="Split eval scenes into this many deterministic shards.")
    p.add_argument("--eval_shard_index", type=int, default=0,
                   help="Run only eval scenes with index %% eval_shard_count == eval_shard_index.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per_scene_csv", action="store_true", default=True)
    p.add_argument("--no_per_scene_csv", action="store_false", dest="per_scene_csv")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--print_every", type=int, default=5)

    p.add_argument("--eval_strategy", choices=["manip_track", "wrist_track", "random_static_track", "both", "left_moving_tracks"],
                   default="left_moving_tracks")
    p.add_argument("--eval_num_frames", type=int, default=64)
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_moving")
    p.add_argument("--eval_seed", type=int, default=42)

    p.add_argument("--image_size", type=int, default=518,
                   help="Scal3R proc_max_size; GT is synchronized to Scal3R native geometry before export.")
    p.add_argument("--geometry_normalization", choices=["native", "vggt_independent", "none"],
                   default="none",
                   help="Geometry normalization before metrics. none compares raw GT/pred geometry.")
    p.add_argument("--camera_align", choices=["none", "sim3"], default="sim3",
                   help="Trajectory alignment before camera metrics.")
    p.add_argument("--depth_align", choices=["none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="pi3_scale_shift",
                   help="Depth alignment before metrics.")
    p.add_argument("--secondary_depth_align", choices=["", "none", "median", "lsq", "pi3_scale", "pi3_scale_shift"], default="")
    p.add_argument("--pointcloud_metrics", action="store_true", default=True)
    p.add_argument("--no_pointcloud_metrics", action="store_false", dest="pointcloud_metrics")
    p.add_argument("--pointcloud_max_points", type=int, default=100000)
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"], default="pi3_icp")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1)
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30)
    p.add_argument("--pointcloud_icp_backend", choices=["auto", "open3d", "scipy"], default="open3d")
    p.add_argument("--pointcloud_kdtree_workers", type=int, default=1)
    p.add_argument("--pointcloud_workers", type=int, default=1)
    return p


def args_from_run_json(eval_args: argparse.Namespace) -> argparse.Namespace:
    args_json_path = Path(eval_args.train_args_json)
    if not args_json_path.is_file():
        raise FileNotFoundError(f"args.json not found: {args_json_path}")
    with args_json_path.open("r", encoding="utf-8") as f:
        run_args = json.load(f)

    ns = argparse.Namespace(**run_args)
    for key, value in vars(eval_args).items():
        setattr(ns, key, value)
    ns.train_args_json = str(args_json_path)
    ns.image_size = int(eval_args.image_size)
    ns.model_path = ""
    ns.cpu = (eval_args.device == "cpu")
    ns.batch_size = 1
    ns.geometry_normalization = str(eval_args.geometry_normalization)
    ns.camera_align = str(eval_args.camera_align)
    ns.write_manifest = ns.__dict__.get("write_manifest", None) or None
    return ns


def eval_modes(strategy: str) -> List[str]:
    if strategy == "both":
        return ["wrist_track", "random_static_track"]
    return [strategy]


def build_scal3r_dummy_intrinsics(width: int, height: int, focal_ratio: float = 1.0) -> np.ndarray:
    focal = float(max(int(height), int(width))) * float(focal_ratio)
    return np.array(
        [
            [focal, 0.0, float(width) / 2.0],
            [0.0, focal, float(height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def compute_scal3r_preprocess_geometry(
    width: int,
    height: int,
    proc_max_size: int = 518,
    proc_align_size: int = 14,
    center_crop: bool = True,
    focal_ratio: float = 1.0,
) -> Dict[str, int]:
    """Match Scal3R image_utils.load_and_preprocess_images geometry."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source image size: {width}x{height}")
    proc_align_size = max(1, int(proc_align_size))
    proc_max_size = int(proc_max_size)
    aspect_ratio = float(height) / float(width)
    if proc_max_size > 0:
        target_h = int(aspect_ratio * proc_max_size)
        target_w = proc_max_size
    else:
        target_h = int(height)
        target_w = int(width)
    target_h = max(proc_align_size, target_h // proc_align_size * proc_align_size)
    target_w = max(proc_align_size, target_w // proc_align_size * proc_align_size)

    ratio = max(target_h / max(float(height), 1.0), target_w / max(float(width), 1.0))
    new_height = max(proc_align_size, round(height * ratio / proc_align_size) * proc_align_size)
    new_width = max(proc_align_size, round(width * ratio / proc_align_size) * proc_align_size)

    dummy_k = build_scal3r_dummy_intrinsics(width, height, focal_ratio=focal_ratio)
    dummy_k[0:1] *= new_width / float(width)
    dummy_k[1:2] *= new_height / float(height)
    if center_crop:
        crop_top = int(round(float(dummy_k[1, 2]))) - target_h // 2
        crop_left = int(round(float(dummy_k[0, 2]))) - target_w // 2
    else:
        crop_top = 0
        crop_left = 0
    return {
        "new_width": int(new_width),
        "new_height": int(new_height),
        "crop_left": int(crop_left),
        "crop_top": int(crop_top),
        "crop_width": int(target_w),
        "crop_height": int(target_h),
        "pad_left": 0,
        "pad_right": 0,
        "pad_top": 0,
        "pad_bottom": 0,
    }


class Scal3RNativeEvalDataset(EMG.ModelGeometryEvalDataset):  # type: ignore[misc]
    """Eval sampler whose GT tensors are synchronized to Scal3R native geometry."""

    rgb_resample = Image.Resampling.BICUBIC

    def model_preprocess_geometry(self, width: int, height: int) -> Dict[str, int]:
        return compute_scal3r_preprocess_geometry(
            width,
            height,
            proc_max_size=int(getattr(self, "image_size", 518)),
            proc_align_size=14,
            center_crop=True,
            focal_ratio=1.0,
        )


def build_scal3r_native_eval_loader(args: argparse.Namespace, eval_mode: str = "default") -> Tuple[DataLoader, List[Path]]:
    scenes = T.discover_trajectory_dirs(
        args.data_roots,
        max_scenes=args.max_scenes,
        manifest=args.scene_manifest,
        write_manifest=None,
        oss_uri_roots=T.parse_str_list(args.oss_uri_roots),
        ossutil_bin=args.ossutil_bin,
        ossutil_config=args.ossutil_config,
    )
    if not scenes:
        raise RuntimeError("No Manip trajectories were discovered")

    train_scenes, val_scenes = T.split_scenes(scenes, args.val_fraction, args.seed)
    if args.split == "val":
        eval_scenes = val_scenes
    elif args.split == "train":
        eval_scenes = train_scenes
    else:
        eval_scenes = list(scenes)

    if args.max_scenes_eval > 0:
        eval_scenes = eval_scenes[: args.max_scenes_eval]
    shard_count = max(1, int(getattr(args, "eval_shard_count", 1)))
    shard_index = int(getattr(args, "eval_shard_index", 0))
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"eval_shard_index must be in [0, {shard_count}), got {shard_index}")
    if shard_count > 1:
        total_eval_scenes = len(eval_scenes)
        eval_scenes = eval_scenes[shard_index::shard_count]
        print(f"[eval_scal3r] shard {shard_index}/{shard_count}: {len(eval_scenes)} of {total_eval_scenes} scenes")
    if not eval_scenes:
        raise RuntimeError(f"split={args.split} produced no scenes")

    view_ids = T.parse_int_list(args.view_ids)
    camera_names = T.parse_str_list(args.camera_names)
    common_kwargs = dict(
        clip_len=args.clip_len,
        image_size=int(args.image_size),
        patch_size=14,
        preprocess_mode="crop",
        sequence_mode=args.sequence_mode,
        view_ids=view_ids,
        camera_names=camera_names,
        sample_strategy=args.sample_strategy,
        frame_stride=args.frame_stride,
        random_stride_min=args.random_stride_min,
        random_stride_max=args.random_stride_max,
        random_interval_start=args.random_interval_start,
        max_sample_frames=args.max_sample_frames,
        min_sample_frames=args.min_sample_frames,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_mask=args.use_mask,
        invert_cam_extrinsics=args.invert_cam_extrinsics,
        samples_per_scene=args.samples_per_scene,
        wrist_camera_prefix=args.wrist_camera_prefix,
        static_camera_prefix=args.static_camera_prefix,
        w_stride_min=getattr(args, "w_stride_min", getattr(args, "random_stride_min", 2)),
        w_stride_max=getattr(args, "w_stride_max", getattr(args, "random_stride_max", 8)),
        moving_stride_min=getattr(args, "moving_stride_min", 2),
        moving_stride_max=getattr(args, "moving_stride_max", 6),
        fixed_stride_min=getattr(args, "fixed_stride_min", 4),
        fixed_stride_max=getattr(args, "fixed_stride_max", 16),
        long6_root_marker=getattr(args, "long6_root_marker", "Manip_long6"),
        long6_mode_weights=T.parse_mode_weights(getattr(args, "long6_mode_weights", "W=0.40,T=0.45,F=0.15")),
        moving_camera_prefix=getattr(args, "moving_camera_prefix", "surround_cam_moving"),
        fixed_camera_prefix=getattr(args, "fixed_camera_prefix", "surround_cam_fixed"),
        color_jitter_strength=0.0,
        color_jitter_prob=0.0,
    )

    if eval_mode == "default":
        raise ValueError("Scal3R native geometry requires a deterministic eval track mode, not train_default")
    if eval_mode not in {"left_moving_tracks", "manip_track", "wrist_track", "random_static_track"}:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")
    dataset = Scal3RNativeEvalDataset(
        eval_scenes,
        eval_mode=eval_mode,
        eval_num_frames=int(args.eval_num_frames),
        eval_wrist_camera_name=str(args.eval_wrist_camera_name),
        eval_surround_camera_name=str(getattr(args, "eval_surround_camera_name", "")),
        eval_seed=int(args.eval_seed),
        **common_kwargs,
    )
    dataset.set_global_step(1)

    nw = max(0, int(args.num_workers))
    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available() and args.device != "cpu",
        drop_last=False,
        collate_fn=T.collate_rgbd_sequences,
    )
    if nw > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **loader_kwargs), eval_scenes


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        keep.append(ch if ch.isalnum() or ch in "-_." else "_")
    return "".join(keep).strip("._") or "scene"


def save_batch_images_for_scal3r(images: torch.Tensor, input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    images_cpu = images[0].detach().cpu().float().clamp(0.0, 1.0)
    for index, image in enumerate(images_cpu):
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array, mode="RGB").save(input_dir / f"{index:06d}.png")


def run_scal3r(args: argparse.Namespace, input_dir: Path, result_dir: Path) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = result_dir / "runtime"
    cmd = shlex.split(str(args.scal3r_python)) + [
        "-m", "scal3r.run",
        "--config", str(args.scal3r_config),
        "--input_dir", str(input_dir),
        "--output_dir", str(result_dir),
        "--runtime_dir", str(runtime_dir),
        "--device", str(args.scal3r_device),
        "--preprocess_workers", str(int(args.scal3r_preprocess_workers)),
        "--block_size", str(int(args.scal3r_block_size)),
        "--overlap_size", str(int(args.scal3r_overlap_size)),
        "--use_loop", str(int(args.scal3r_use_loop)),
        "--use_xyz_align", str(int(args.scal3r_use_xyz_align)),
        "--pgo_workers", str(int(args.scal3r_pgo_workers)),
        "--save_dpt", "1",
        "--save_xyz", str(int(args.scal3r_save_xyz)),
    ]
    if str(args.scal3r_checkpoint):
        cmd.extend(["--checkpoint", str(args.scal3r_checkpoint)])
    if bool(args.scal3r_test_use_amp):
        cmd.append("--test_use_amp")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.scal3r_repo}:{env.get('PYTHONPATH', '')}"
    env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    subprocess.run(cmd, cwd=str(args.scal3r_repo), env=env, check=True)


def load_scal3r_intrinsics(intri_path: Path, n_frames: int) -> np.ndarray:
    import cv2

    storage = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise FileNotFoundError(f"cannot open Scal3R intrinsics: {intri_path}")
    try:
        intrinsics = []
        for index in range(n_frames):
            node = storage.getNode(f"K_{index:06d}")
            mat = node.mat()
            if mat is None:
                raise KeyError(f"K_{index:06d} not found in {intri_path}")
            intrinsics.append(np.asarray(mat, dtype=np.float32))
        return np.stack(intrinsics, axis=0)
    finally:
        storage.release()


def load_scal3r_depths(depth_dir: Path, n_frames: int, target_hw: Tuple[int, int]) -> torch.Tensor:
    import cv2

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    depths = []
    for index in range(n_frames):
        path = depth_dir / f"{index:06d}.exr"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"missing Scal3R depth: {path}")
        depth_np = np.asarray(depth, dtype=np.float32)
        if depth_np.ndim == 3:
            depth_np = depth_np[..., 0]
        depths.append(torch.from_numpy(depth_np.copy()))
    depth_t = torch.stack(depths, dim=0)[None, ..., None].float()
    target_h, target_w = target_hw
    if tuple(depth_t.shape[2:4]) != (target_h, target_w):
        bs, ss, hh, ww, cc = depth_t.shape
        resized = F.interpolate(
            depth_t.permute(0, 1, 4, 2, 3).reshape(bs * ss, cc, hh, ww),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        depth_t = resized.reshape(bs, ss, cc, target_h, target_w).permute(0, 1, 3, 4, 2)
    return depth_t


def load_scal3r_c2w(mat_path: Path, n_frames: int) -> torch.Tensor:
    raw = np.loadtxt(mat_path, dtype=np.float64).reshape(n_frames, 4, 4)
    c2w = raw.copy()
    for index in range(n_frames):
        rot = c2w[index, :3, :3]
        det = float(np.linalg.det(rot))
        if math.isfinite(det) and abs(det) > 1e-12:
            scale = math.copysign(abs(det) ** (1.0 / 3.0), det)
            if abs(scale) > 1e-12:
                c2w[index, :3, :3] = rot / scale
        u, _, vt = np.linalg.svd(c2w[index, :3, :3])
        proj = u @ vt
        if np.linalg.det(proj) < 0:
            u[:, -1] *= -1.0
            proj = u @ vt
        c2w[index, :3, :3] = proj
        c2w[index, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return torch.from_numpy(c2w.astype(np.float32))[None]


def c2w_intrinsics_to_pose_enc(c2w: torch.Tensor, intrinsics: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = image_hw
    quat = mat_to_quat(c2w[..., :3, :3])
    fov_h = 2 * torch.atan((h / 2.0) / intrinsics[..., 1, 1].clamp(min=1e-6))
    fov_w = 2 * torch.atan((w / 2.0) / intrinsics[..., 0, 0].clamp(min=1e-6))
    return torch.cat([c2w[..., :3, 3], quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def load_scal3r_predictions(result_dir: Path, target_hw: Tuple[int, int], device: torch.device) -> Dict[str, torch.Tensor]:
    mat_path = result_dir / "mat.txt"
    intri_path = result_dir / "intri.yml"
    if not mat_path.is_file():
        raise FileNotFoundError(f"missing Scal3R pose file: {mat_path}")
    c2w_rows = np.loadtxt(mat_path, dtype=np.float64)
    n_frames = int(c2w_rows.reshape(-1, 16).shape[0])
    c2w = load_scal3r_c2w(mat_path, n_frames)
    depth = load_scal3r_depths(result_dir / "depths", n_frames, target_hw)
    intr_np = load_scal3r_intrinsics(intri_path, n_frames)
    intrinsics = torch.from_numpy(intr_np)[None].float()
    pose_enc = c2w_intrinsics_to_pose_enc(c2w, intrinsics, target_hw)
    return {
        "depth": depth.to(device),
        "camera_c2w": c2w.to(device),
        "intrinsics": intrinsics.to(device),
        "pose_enc": pose_enc.to(device),
    }


def _se3_4x4_to_3x4_np(c2w: np.ndarray) -> np.ndarray:
    return c2w[:, :3, :4]


def compute_camera_metrics_evo_from_c2w(
    pred_c2w: torch.Tensor,
    gt_extrinsics_w2c: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    align_mode: str = "sim3",
) -> Optional[Dict[str, float]]:
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D

    mask = valid_frame_mask.bool()
    if int(mask.sum().item()) < 3:
        return None
    pred_4x4 = pred_c2w.float()
    if pred_4x4.shape[-2:] != (4, 4):
        full = torch.eye(4, dtype=pred_4x4.dtype, device=pred_4x4.device).repeat(pred_4x4.shape[0], 1, 1)
        full[:, :3, :4] = pred_4x4
        pred_4x4 = full
    gt_c2w_3x4 = T.w2c_to_c2w_extrinsics(gt_extrinsics_w2c.unsqueeze(0).float())[0]

    mask_np = mask.detach().cpu().numpy()
    pred_np = pred_4x4.detach().cpu().numpy()[mask_np]
    gt_np = E._se3_3x4_to_4x4_np(gt_c2w_3x4.detach().cpu().numpy())[mask_np]
    if pred_np.shape[0] < 3 or not (np.all(np.isfinite(pred_np)) and np.all(np.isfinite(gt_np))):
        return None

    timestamps = np.arange(pred_np.shape[0], dtype=np.float64)
    pred_traj = PoseTrajectory3D(poses_se3=list(pred_np), timestamps=timestamps)
    gt_traj = PoseTrajectory3D(poses_se3=list(gt_np), timestamps=timestamps.copy())
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)
    if align_mode not in {"none", "sim3"}:
        raise ValueError(f"Unknown camera_align mode: {align_mode}")
    align_trajectory = align_mode == "sim3"
    try:
        ate_result = main_ape.ape(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=align_trajectory, correct_scale=align_trajectory,
        )
        rpe_rot_result = main_rpe.rpe(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=align_trajectory, correct_scale=align_trajectory,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
        )
        rpe_trans_result = main_rpe.rpe(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=align_trajectory, correct_scale=align_trajectory,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[eval_scal3r] evo trajectory metrics failed ({str(exc).splitlines()[0][:120]}); skipping scene")
        return None
    return {
        "ate_rmse": float(ate_result.stats["rmse"]),
        "rpe_trans_rmse": float(rpe_trans_result.stats["rmse"]),
        "rpe_rot_rmse_deg": float(rpe_rot_result.stats["rmse"]),
        "n_frames_used": float(pred_np.shape[0]),
    }


def compute_fov_errors_from_intrinsics(
    pred_intrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
    image_hw: Tuple[int, int],
    valid_frame_mask: torch.Tensor,
) -> Optional[List[Dict[str, float]]]:
    mask = valid_frame_mask.bool()
    if int(mask.sum().item()) == 0:
        return None
    h, w = image_hw
    pred_fov_h = 2 * torch.atan((h / 2.0) / pred_intrinsics[..., 1, 1].clamp(min=1e-6))
    pred_fov_w = 2 * torch.atan((w / 2.0) / pred_intrinsics[..., 0, 0].clamp(min=1e-6))
    gt_fov_h = 2 * torch.atan((h / 2.0) / gt_intrinsics[..., 1, 1].clamp(min=1e-6))
    gt_fov_w = 2 * torch.atan((w / 2.0) / gt_intrinsics[..., 0, 0].clamp(min=1e-6))
    fov_h_err = torch.rad2deg((pred_fov_h - gt_fov_h).abs())[mask].detach().cpu().numpy()
    fov_w_err = torch.rad2deg((pred_fov_w - gt_fov_w).abs())[mask].detach().cpu().numpy()
    return [{"fov_h_deg_mae": float(h_err), "fov_w_deg_mae": float(w_err)}
            for h_err, w_err in zip(fov_h_err, fov_w_err)]


@contextlib.contextmanager
def scal3r_input_dir(args: argparse.Namespace, base_output_dir: Path, scene_name: str):
    if bool(args.keep_scal3r_inputs):
        path = base_output_dir / "scal3r_inputs" / safe_name(scene_name)
        path.mkdir(parents=True, exist_ok=True)
        yield path
    else:
        with tempfile.TemporaryDirectory(prefix=f"scal3r_{safe_name(scene_name)}_") as tmp:
            yield Path(tmp)


def ensure_scal3r_result(
    args: argparse.Namespace,
    images_t: torch.Tensor,
    output_dir: Path,
    scene_name: str,
) -> Path:
    result_dir = output_dir / "scal3r_results" / safe_name(scene_name)
    if not bool(args.force_rerun_scal3r) and (result_dir / "mat.txt").is_file() and (result_dir / "depths").is_dir():
        return result_dir
    with scal3r_input_dir(args, output_dir, scene_name) as input_dir:
        save_batch_images_for_scal3r(images_t, input_dir)
        run_scal3r(args, input_dir, result_dir)
    return result_dir


@torch.no_grad()
def evaluate_one_mode(args: argparse.Namespace, device: torch.device, eval_mode: str, output_dir: Path) -> Dict[str, object]:
    # build_scal3r_native_eval_loader keeps eval.py ordering: max_scenes_eval first,
    # then eval_shard_count/eval_shard_index.
    loader, eval_scenes = build_scal3r_native_eval_loader(args, eval_mode=eval_mode)
    print(f"[eval_scal3r] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
          f"mode={eval_mode}, frames={args.eval_num_frames}, image_size={args.image_size}")

    all_depth_pixels = E.DepthPixelAccumulator()
    secondary_depth_pixels: Optional[E.DepthPixelAccumulator] = None
    per_scene_traj: List[Dict[str, float]] = []
    all_pointcloud_rows: List[Dict[str, float]] = []

    report_group_names = ("realsense_left", "surround_cam_moving", "realsense", "surround", "other")
    group_depth_pixels: Dict[str, E.DepthPixelAccumulator] = {
        name: E.DepthPixelAccumulator() for name in report_group_names
    }
    group_secondary_depth_pixels: Dict[str, E.DepthPixelAccumulator] = {}
    group_traj: Dict[str, List[Dict[str, float]]] = {name: [] for name in report_group_names}
    group_pointcloud_rows: Dict[str, List[Dict[str, float]]] = {name: [] for name in report_group_names}
    group_counts: Dict[str, int] = {name: 0 for name in report_group_names}
    pointcloud_queue = E.AsyncPointcloudMetricQueue(
        args,
        all_pointcloud_rows,
        group_pointcloud_rows,
        log_prefix="[eval_scal3r]",
    )

    per_scene_rows: List[Dict[str, object]] = []
    skipped = 0
    evaluated = 0
    secondary_align = str(getattr(args, "secondary_depth_align", "") or "")
    if secondary_align == str(args.depth_align):
        secondary_align = ""
    if secondary_align:
        secondary_depth_pixels = E.DepthPixelAccumulator()
        group_secondary_depth_pixels = {name: E.DepthPixelAccumulator() for name in report_group_names}

    for batch_idx, batch in enumerate(loader, start=1):
        scene_field = batch["scene"]
        scene_name = scene_field[0] if isinstance(scene_field, list) else str(scene_field)
        mode_field = batch["sample_mode"]
        sample_mode = mode_field[0] if isinstance(mode_field, list) else str(mode_field)
        metric_group = E.metric_group_from_sample_mode(sample_mode)
        if metric_group not in group_counts:
            group_counts[metric_group] = 0
            group_depth_pixels[metric_group] = E.DepthPixelAccumulator()
            if secondary_align:
                group_secondary_depth_pixels[metric_group] = E.DepthPixelAccumulator()
            group_traj[metric_group] = []
            group_pointcloud_rows[metric_group] = []

        images_for_scal3r = cast(torch.Tensor, batch["images"]).clone()
        geometry_normalization = str(getattr(args, "geometry_normalization", "none"))
        if geometry_normalization == "native":
            if getattr(args, "canonicalize_first_frame", True):
                batch = T.canonicalize_to_first_frame(batch)
            if getattr(args, "normalize_scene", True):
                batch = T.normalize_scene_batch(
                    batch,
                    num_anchor_frames=min(args.num_scale_frames, int(images_for_scal3r.shape[1])),
                )
        elif geometry_normalization == "vggt_independent":
            batch = E.vggt_normalize_gt_batch(batch)
        elif geometry_normalization == "none":
            batch = dict(batch)
        else:
            raise ValueError(f"Unknown geometry_normalization: {geometry_normalization}")
        batch = T.to_device(batch, device)
        images_t = cast(torch.Tensor, batch["images"])
        depths_t = cast(torch.Tensor, batch["depths"])
        point_masks_t = cast(torch.Tensor, batch["point_masks"])
        extrinsics_t = cast(torch.Tensor, batch["extrinsics"])
        intrinsics_t = cast(torch.Tensor, batch["intrinsics"])
        world_points_t = cast(torch.Tensor, batch["world_points"])
        image_hw = (int(images_t.shape[-2]), int(images_t.shape[-1]))

        try:
            result_dir = ensure_scal3r_result(args, images_for_scal3r, output_dir, scene_name)
            predictions = load_scal3r_predictions(result_dir, image_hw, device)
            if geometry_normalization == "vggt_independent":
                predictions = E.vggt_normalize_predictions(predictions, point_masks_t.bool())
            elif "world_points" not in predictions:
                predictions = dict(predictions)
                predictions["world_points"] = E.unproject_depth_to_world_from_pose(
                    cast(torch.Tensor, predictions["depth"]).float(),
                    cast(torch.Tensor, predictions["pose_enc"]).float(),
                )
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[eval_scal3r] batch {batch_idx} Scal3R failed for {scene_name} ({str(exc).splitlines()[0][:160]}); skipping")
            continue

        per_frame_valid = point_masks_t.sum(dim=(-1, -2)) > int(getattr(args, "min_valid_pixels", 100))
        row: Dict[str, object] = {
            "scene": scene_name,
            "sample_mode": sample_mode,
            "metric_group": metric_group,
            "n_frames": int(images_t.shape[1]),
        }
        gt_scale_t = batch.get("vggt_gt_scale")
        if torch.is_tensor(gt_scale_t):
            row["gt_geometry_scale"] = float(gt_scale_t[0].detach().cpu().item())
        pred_scale_t = predictions.get("vggt_pred_scale")
        if torch.is_tensor(pred_scale_t):
            row["pred_geometry_scale"] = float(pred_scale_t[0].detach().cpu().item())

        pred_depth = cast(torch.Tensor, predictions["depth"]).float()
        mask = point_masks_t.bool() & per_frame_valid[..., None, None]
        clip_depth = E.DepthPixelAccumulator()
        clip_depth.update(pred_depth, depths_t.float(), mask, align=str(args.depth_align))
        all_depth_pixels.update(pred_depth, depths_t.float(), mask, align=str(args.depth_align))
        group_depth_pixels[metric_group].update(pred_depth, depths_t.float(), mask, align=str(args.depth_align))
        for key, value in clip_depth.summary().items():
            row[f"depth_{args.depth_align}_{key}"] = value

        if secondary_align and secondary_depth_pixels is not None:
            clip_secondary_depth = E.DepthPixelAccumulator()
            clip_secondary_depth.update(pred_depth, depths_t.float(), mask, align=secondary_align)
            secondary_depth_pixels.update(pred_depth, depths_t.float(), mask, align=secondary_align)
            group_secondary_depth_pixels[metric_group].update(pred_depth, depths_t.float(), mask, align=secondary_align)
            for key, value in clip_secondary_depth.summary().items():
                row[f"depth_{secondary_align}_{key}"] = value

        pred_c2w = T.pose_encoding_to_c2w_matrix(cast(torch.Tensor, predictions["pose_enc"]))[0].float()
        traj_metrics = compute_camera_metrics_evo_from_c2w(
            pred_c2w,
            extrinsics_t[0],
            per_frame_valid[0],
            align_mode=str(getattr(args, "camera_align", "sim3")),
        )
        if traj_metrics is not None:
            per_scene_traj.append(traj_metrics)
            group_traj[metric_group].append(traj_metrics)
            row["cam_ate_rmse"] = traj_metrics["ate_rmse"]
            row["cam_rpe_trans_rmse"] = traj_metrics["rpe_trans_rmse"]
            row["cam_rpe_rot_rmse_deg"] = traj_metrics["rpe_rot_rmse_deg"]
            row["cam_n_frames_used"] = traj_metrics["n_frames_used"]


        if bool(args.pointcloud_metrics):
            pointcloud_queue.submit(
                cast(torch.Tensor, predictions["world_points"]).float(),
                world_points_t.float(),
                mask,
                row,
                metric_group,
                scene_name,
            )

        if args.save_predictions:
            pred_path = output_dir / "predictions" / f"{safe_name(scene_name)}.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pred_path,
                depth=pred_depth[0, ..., 0].detach().cpu().numpy().astype(np.float32),
                pose_enc=cast(torch.Tensor, predictions["pose_enc"])[0].detach().cpu().numpy().astype(np.float32),
                camera_c2w=pred_c2w.detach().cpu().numpy().astype(np.float32),
                intrinsics=cast(torch.Tensor, predictions["intrinsics"])[0].detach().cpu().numpy().astype(np.float32),
                world_points=cast(torch.Tensor, predictions["world_points"])[0].detach().cpu().numpy().astype(np.float32),
                gt_depth=depths_t[0].detach().cpu().numpy().astype(np.float32),
                point_masks=point_masks_t[0].detach().cpu().numpy().astype(np.uint8),
                gt_extrinsics=extrinsics_t[0].detach().cpu().numpy().astype(np.float32),
                gt_intrinsics=intrinsics_t[0].detach().cpu().numpy().astype(np.float32),
            )

        per_scene_rows.append(row)
        group_counts[metric_group] += 1
        evaluated += 1
        if args.print_every > 0 and (batch_idx % args.print_every == 0 or batch_idx == len(loader)):
            running = all_depth_pixels.summary()
            print(f"[eval_scal3r] [{batch_idx}/{len(loader)}] evaluated={evaluated} skipped={skipped} "
                  f"AbsRel({args.depth_align})={running.get('AbsRel', float('nan')):.4f}")

    pointcloud_queue.close()

    if evaluated == 0:
        raise RuntimeError(f"No batches evaluated for eval_mode={eval_mode}.")

    summary: Dict[str, object] = {
        "model": "Scal3R",
        "scal3r_repo": str(args.scal3r_repo),
        "scal3r_config": str(args.scal3r_config),
        "scal3r_checkpoint": str(args.scal3r_checkpoint),
        "train_args_json": str(args.train_args_json),
        "split": args.split,
        "eval_mode": eval_mode,
        "eval_shard_count": int(getattr(args, "eval_shard_count", 1)),
        "eval_shard_index": int(getattr(args, "eval_shard_index", 0)),
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "geometry_normalization": str(getattr(args, "geometry_normalization", "none")),
        "camera_align": str(getattr(args, "camera_align", "sim3")),
        "depth_align": str(args.depth_align),
        "input_geometry_sync": "scal3r_native_geometry+lingbot_map_sync",
        "secondary_depth_align": secondary_align,
        "pointcloud_source": "Scal3R depth plus Scal3R c2w/intrinsics unprojection",
        "aggregation": "overall plus realsense_left/surround_cam_moving camera-track groups; depth=Pi3-style pixel-weighted after per-clip alignment; trajectory=per-sequence macro; pointcloud=per-clip macro",
        "pointcloud_align": str(args.pointcloud_align),
        "pointcloud_icp_backend": str(getattr(args, "pointcloud_icp_backend", "open3d")),
        "pointcloud_kdtree_workers": int(getattr(args, "pointcloud_kdtree_workers", 1)),
        "pointcloud_workers": int(getattr(args, "pointcloud_workers", 1)),
        "scenes_skipped": int(skipped),
    }
    overall_summary = E.summarize_metric_group(
        per_scene_traj, all_pointcloud_rows, evaluated,
        depth_pixels=all_depth_pixels,
    )
    summary["overall"] = overall_summary
    if "depth" in overall_summary:
        summary["depth"] = overall_summary["depth"]
    if secondary_depth_pixels is not None:
        secondary_summary = secondary_depth_pixels.summary()
        if secondary_summary:
            summary[f"depth_{secondary_align}"] = secondary_summary
    if "camera" in overall_summary:
        summary["camera"] = overall_summary["camera"]
    if "pointcloud" in overall_summary:
        summary["pointcloud"] = overall_summary["pointcloud"]

    groups: Dict[str, object] = {}
    for group_name in report_group_names:
        if group_counts.get(group_name, 0) <= 0:
            continue
        group_summary = E.summarize_metric_group(
            group_traj[group_name],
            group_pointcloud_rows[group_name],
            group_counts[group_name],
            depth_pixels=group_depth_pixels[group_name],
        )
        if secondary_align and group_name in group_secondary_depth_pixels:
            secondary_group_summary = group_secondary_depth_pixels[group_name].summary()
            if secondary_group_summary:
                group_summary[f"depth_{secondary_align}"] = secondary_group_summary
        groups[group_name] = group_summary
    if "realsense_left" in groups and "surround_cam_moving" in groups:
        groups["realsense_left+surround_cam_moving"] = overall_summary
    summary["groups"] = groups

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[eval_scal3r] wrote {metrics_path}")

    if args.per_scene_csv and per_scene_rows:
        csv_path = output_dir / "per_scene.csv"
        keys = sorted({key for row in per_scene_rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_scene_rows)
        print(f"[eval_scal3r] wrote {csv_path}")
    return summary


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_scal3r] output_dir={base_output_dir}")
    print(f"[eval_scal3r] scal3r_python={args.scal3r_python}")

    overall: Dict[str, object] = {
        "model": "Scal3R",
        "train_args_json": str(args.train_args_json),
        "split": args.split,
        "eval_strategy": args.eval_strategy,
        "eval_shard_count": int(getattr(args, "eval_shard_count", 1)),
        "eval_shard_index": int(getattr(args, "eval_shard_index", 0)),
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "geometry_normalization": str(getattr(args, "geometry_normalization", "none")),
        "camera_align": str(getattr(args, "camera_align", "sim3")),
        "modes": {},
    }
    modes = eval_modes(args.eval_strategy)
    for mode_name in modes:
        out_dir = base_output_dir if len(modes) == 1 else (base_output_dir / mode_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[eval_scal3r] === mode: {mode_name} ===")
        cast(Dict[str, object], overall["modes"])[mode_name] = evaluate_one_mode(args, device, mode_name, out_dir)

    summary_path = base_output_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    print(f"[eval_scal3r] wrote {summary_path}")
    return overall


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = args_from_run_json(eval_args)
    evaluate(args)


if __name__ == "__main__":
    main()
