#!/usr/bin/env python3
"""Evaluate Scal3R on the LingBot-MAP Manip validation protocol.

The dataloader and metrics are reused from eval.py. Scal3R is invoked as an
external folder-in/folder-out reconstructor, then its files are adapted to the
same tensor contract used by the other baseline evaluators.
"""

from __future__ import annotations

import argparse
import copy
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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "manip_long_train_64gpu"
DEFAULT_SCAL3R_REPO = SCRIPT_DIR.parent / "Scal3R"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval as E
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

    p.add_argument("--eval_strategy", choices=["manip_track", "wrist_track", "random_static_track", "both"],
                   default="manip_track")
    p.add_argument("--eval_num_frames", type=int, default=64)
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_0")
    p.add_argument("--eval_seed", type=int, default=42)

    p.add_argument("--image_size", type=int, default=518,
                   help="Input size used by the LingBot loader before saving images for Scal3R.")
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
    if int(getattr(args, "eval_shard_count", 1)) > 1:
        shard_count = int(args.eval_shard_count)
        shard_index = int(args.eval_shard_index)
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(f"eval_shard_index={shard_index} outside [0, {shard_count})")
        all_args = copy.copy(args)
        all_args.max_scenes_eval = 0
        _, all_eval_scenes = E.build_eval_loader(all_args, eval_mode=eval_mode)
        shard_scenes = all_eval_scenes[shard_index::shard_count]
        shard_manifest = output_dir / f"shard_{shard_index:02d}_of_{shard_count:02d}_manifest.txt"
        shard_manifest.parent.mkdir(parents=True, exist_ok=True)
        with shard_manifest.open("w", encoding="utf-8") as handle:
            for scene in shard_scenes:
                handle.write(f"{scene}\n")
        shard_args = copy.copy(args)
        shard_args.scene_manifest = str(shard_manifest)
        shard_args.max_scenes_eval = 0
        shard_args.split = "all"
        loader, eval_scenes = E.build_eval_loader(shard_args, eval_mode=eval_mode)
    else:
        loader, eval_scenes = E.build_eval_loader(args, eval_mode=eval_mode)
    print(f"[eval_scal3r] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
          f"mode={eval_mode}, frames={args.eval_num_frames}, image_size={args.image_size}")

    all_depth_frames: List[Dict[str, float]] = []
    secondary_depth_frames: List[Dict[str, float]] = []
    per_scene_traj: List[Dict[str, float]] = []
    all_fov_frames: List[Dict[str, float]] = []
    all_pointcloud_rows: List[Dict[str, float]] = []

    group_depth_frames: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_secondary_depth_frames: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_traj: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_fov_frames: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_pointcloud_rows: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_counts: Dict[str, int] = {"realsense": 0, "surround": 0, "other": 0}

    per_scene_rows: List[Dict[str, object]] = []
    skipped = 0
    evaluated = 0
    secondary_align = str(getattr(args, "secondary_depth_align", "") or "")
    if secondary_align == str(args.depth_align):
        secondary_align = ""

    for batch_idx, batch in enumerate(loader, start=1):
        scene_field = batch["scene"]
        scene_name = scene_field[0] if isinstance(scene_field, list) else str(scene_field)
        mode_field = batch["sample_mode"]
        sample_mode = mode_field[0] if isinstance(mode_field, list) else str(mode_field)
        metric_group = E.metric_group_from_sample_mode(sample_mode)
        for dct in (group_counts, group_depth_frames, group_secondary_depth_frames, group_traj,
                    group_fov_frames, group_pointcloud_rows):
            if metric_group not in dct:
                dct[metric_group] = 0 if dct is group_counts else []

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
        scene_frames = E.compute_depth_per_frame_metrics(
            pred_depth, depths_t.float(), mask, align=str(args.depth_align),
        )
        all_depth_frames.extend(scene_frames)
        group_depth_frames[metric_group].extend(scene_frames)
        if scene_frames:
            for key, value in E.mean_over_frames(scene_frames).items():
                row[f"depth_{args.depth_align}_{key}"] = value

        if secondary_align:
            secondary_frames = E.compute_depth_per_frame_metrics(
                pred_depth, depths_t.float(), mask, align=secondary_align,
            )
            secondary_depth_frames.extend(secondary_frames)
            group_secondary_depth_frames[metric_group].extend(secondary_frames)
            if secondary_frames:
                for key, value in E.mean_over_frames(secondary_frames).items():
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

        fov_rows = compute_fov_errors_from_intrinsics(
            cast(torch.Tensor, predictions["intrinsics"])[0].float(),
            intrinsics_t[0].float(),
            image_hw,
            per_frame_valid[0],
        )
        if fov_rows is not None:
            all_fov_frames.extend(fov_rows)
            group_fov_frames[metric_group].extend(fov_rows)
            row["cam_fov_h_deg_mae"] = float(np.mean([r["fov_h_deg_mae"] for r in fov_rows]))
            row["cam_fov_w_deg_mae"] = float(np.mean([r["fov_w_deg_mae"] for r in fov_rows]))

        if bool(args.pointcloud_metrics):
            try:
                pc_metrics = E.compute_pointcloud_metrics(
                    cast(torch.Tensor, predictions["world_points"]).float(),
                    world_points_t.float(),
                    mask,
                    max_points=int(args.pointcloud_max_points),
                    align=str(args.pointcloud_align),
                    icp_threshold=float(args.pointcloud_icp_threshold),
                    icp_max_iterations=int(args.pointcloud_icp_max_iterations),
                )
            except Exception as exc:  # noqa: BLE001
                pc_metrics = None
                print(f"[eval_scal3r] pointcloud failed for {scene_name} ({str(exc).splitlines()[0][:120]}); skipping")
            if pc_metrics is not None:
                all_pointcloud_rows.append(pc_metrics)
                group_pointcloud_rows[metric_group].append(pc_metrics)
                for key, value in pc_metrics.items():
                    row[f"pc_{key}"] = value

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
            running = E.mean_over_frames(all_depth_frames)
            print(f"[eval_scal3r] [{batch_idx}/{len(loader)}] evaluated={evaluated} skipped={skipped} "
                  f"AbsRel({args.depth_align})={running.get('AbsRel', float('nan')):.4f}")

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
        "secondary_depth_align": secondary_align,
        "pointcloud_source": "Scal3R depth plus Scal3R c2w/intrinsics unprojection",
        "pointcloud_align": str(args.pointcloud_align),
        "scenes_skipped": int(skipped),
    }
    overall_summary = E.summarize_metric_group(
        all_depth_frames, per_scene_traj, all_fov_frames, all_pointcloud_rows, evaluated
    )
    summary["overall"] = overall_summary
    if "depth" in overall_summary:
        summary["depth"] = overall_summary["depth"]
    if secondary_depth_frames:
        summary[f"depth_{secondary_align}"] = E.mean_over_frames(secondary_depth_frames)
    if "camera" in overall_summary:
        summary["camera"] = overall_summary["camera"]
    if "pointcloud" in overall_summary:
        summary["pointcloud"] = overall_summary["pointcloud"]

    groups: Dict[str, object] = {}
    for group_name in sorted(group_counts.keys()):
        if group_counts.get(group_name, 0) <= 0:
            continue
        group_summary = E.summarize_metric_group(
            group_depth_frames[group_name],
            group_traj[group_name],
            group_fov_frames[group_name],
            group_pointcloud_rows[group_name],
            group_counts[group_name],
        )
        if group_secondary_depth_frames.get(group_name):
            group_summary[f"depth_{secondary_align}"] = E.mean_over_frames(group_secondary_depth_frames[group_name])
        groups[group_name] = group_summary
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
