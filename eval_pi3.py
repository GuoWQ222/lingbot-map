#!/usr/bin/env python3
"""Evaluate Pi3 on the LingBot-MAP Manip eval protocol.

This intentionally reuses LingBot-MAP's dataloader and metric implementations,
but swaps the model side to Pi3:

  Pi3 local_points[..., 2] -> depth
  Pi3 points               -> world point cloud
  Pi3 camera_poses         -> camera-to-world poses
  recover_focal_shift      -> approximate pixel intrinsics for FoV metrics
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train as T

LBE = importlib.import_module("eval")


def build_argparser() -> argparse.ArgumentParser:
    default_pi3_ckpt = Path("/cpfs/user/guowenqi/Pi3/ckpts/model.safetensors")
    p = argparse.ArgumentParser(description="Evaluate Pi3 with LingBot-MAP Manip metrics")
    p.add_argument("--train_args_json", type=str, required=True,
                   help="LingBot-MAP args.json used only to recreate the Manip eval dataset config.")
    p.add_argument("--pi3_repo", type=str, default="/cpfs/user/guowenqi/Pi3",
                   help="Local Pi3 repository path.")
    p.add_argument("--pi3_checkpoint", type=str,
                   default=str(default_pi3_ckpt) if default_pi3_ckpt.is_file() else "",
                   help="Local Pi3 checkpoint. Empty falls back to Pi3.from_pretrained().")
    p.add_argument("--pi3_pretrained_model_name_or_path", type=str, default="yyfz233/Pi3",
                   help="Used only when --pi3_checkpoint is empty.")
    p.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "eval" / "pi3"))
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max_scenes_eval", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per_scene_csv", action="store_true")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--print_every", type=int, default=5)
    p.add_argument("--eval_strategy", choices=["train_default", "manip_track", "wrist_track", "random_static_track", "both"],
                   default="manip_track")
    p.add_argument("--eval_num_frames", type=int, default=64)
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_0")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--depth_align", choices=["none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="pi3_scale_shift",
                   help="Depth alignment before metrics. Pi3 modes compute one alignment over the whole clip.")
    p.add_argument("--camera_align", choices=["none", "sim3"], default="sim3",
                   help="Trajectory alignment before camera metrics. none compares VGGT-normalized poses directly; "
                        "sim3 uses evo align=True, correct_scale=True.")
    p.add_argument("--image_size", type=int, default=0,
                   help="Override image_size from train_args_json. Pi3 requires H/W divisible by 14.")
    p.add_argument("--geometry_normalization", choices=["native", "vggt_independent", "none"],
                   default="none",
                   help="Geometry frame/scale used before metrics. none keeps native GT and Pi3 outputs; "
                        "vggt_independent canonicalizes GT by GT frame 0 and predictions by predicted frame 0, "
                        "then scales each by its own mean valid point distance.")
    p.add_argument("--pointcloud_metrics", action="store_true")
    p.add_argument("--pointcloud_max_points", type=int, default=100000)
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"],
                   default="pi3_icp")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1)
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30)
    p.add_argument("--recover_focal", action=argparse.BooleanOptionalAction, default=True,
                   help="Recover intrinsics from Pi3 local point maps for FoV metrics.")
    p.add_argument("--focal_mask_threshold", type=float, default=0.1,
                   help="Use sigmoid(conf) above this threshold for focal recovery.")
    p.add_argument("--focal_downsample_size", type=int, nargs=2, metavar=("H", "W"), default=(64, 64))
    return p


def coerce_pi3_args_from_json(eval_args: argparse.Namespace) -> argparse.Namespace:
    args_json_path = Path(eval_args.train_args_json)
    if not args_json_path.is_file():
        raise FileNotFoundError(f"args.json not found: {args_json_path}")
    with open(args_json_path, "r", encoding="utf-8") as f:
        train_args_dict = json.load(f)

    ns = argparse.Namespace(**train_args_dict)
    ns.checkpoint = eval_args.pi3_checkpoint or eval_args.pi3_pretrained_model_name_or_path
    ns.train_args_json = str(args_json_path)
    ns.output_dir = eval_args.output_dir
    ns.split = eval_args.split
    ns.max_scenes_eval = eval_args.max_scenes_eval
    ns.num_workers = eval_args.num_workers
    ns.device = eval_args.device
    ns.per_scene_csv = eval_args.per_scene_csv
    ns.save_predictions = eval_args.save_predictions
    ns.print_every = eval_args.print_every
    ns.eval_strategy = eval_args.eval_strategy
    ns.eval_num_frames = eval_args.eval_num_frames
    ns.eval_wrist_camera_name = eval_args.eval_wrist_camera_name
    ns.eval_surround_camera_name = eval_args.eval_surround_camera_name
    ns.eval_seed = eval_args.eval_seed
    ns.depth_align = eval_args.depth_align
    ns.camera_align = str(eval_args.camera_align)
    ns.geometry_normalization = str(eval_args.geometry_normalization)
    ns.pointcloud_metrics = bool(eval_args.pointcloud_metrics)
    ns.pointcloud_max_points = int(eval_args.pointcloud_max_points)
    ns.pointcloud_align = str(eval_args.pointcloud_align)
    ns.pointcloud_icp_threshold = float(eval_args.pointcloud_icp_threshold)
    ns.pointcloud_icp_max_iterations = int(eval_args.pointcloud_icp_max_iterations)
    ns.pi3_repo = eval_args.pi3_repo
    ns.pi3_checkpoint = eval_args.pi3_checkpoint
    ns.pi3_pretrained_model_name_or_path = eval_args.pi3_pretrained_model_name_or_path
    ns.recover_focal = bool(eval_args.recover_focal)
    ns.focal_mask_threshold = float(eval_args.focal_mask_threshold)
    ns.focal_downsample_size = tuple(int(x) for x in eval_args.focal_downsample_size)
    if int(eval_args.image_size) > 0:
        ns.image_size = int(eval_args.image_size)
    ns.model_path = ""
    ns.cpu = (eval_args.device == "cpu")
    ns.batch_size = 1
    ns.write_manifest = ns.__dict__.get("write_manifest", None) or None
    return ns


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = width / height
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5
    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width,
                       dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height,
                       dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], dim=-1)


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray) -> Tuple[np.float32, np.float32]:
    """Solve min |focal * xy / (z + shift) - uv|."""
    from scipy.optimize import least_squares

    uv = uv.reshape(-1, 2)
    xy = xyz[..., :2].reshape(-1, 2)
    z = xyz[..., 2].reshape(-1)

    def fn(uv_: np.ndarray, xy_: np.ndarray, z_: np.ndarray, shift: np.ndarray) -> np.ndarray:
        xy_proj = xy_ / (z_ + shift)[:, None]
        denom = np.square(xy_proj).sum()
        if denom <= 0:
            return np.zeros_like(uv_).ravel()
        focal = (xy_proj * uv_).sum() / denom
        return (focal * xy_proj - uv_).ravel()

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm")
    optim_shift = solution["x"].squeeze().astype(np.float32)
    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()
    return optim_shift, np.float32(optim_focal)


def recover_focal_shift(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    downsample_size: Tuple[int, int] = (64, 64),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pi3 demo focal recovery, kept local to avoid importing visualization deps."""
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    points_flat = points.reshape(-1, *shape[-3:])
    mask_flat = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)

    points_lr = F.interpolate(points_flat.permute(0, 3, 1, 2), downsample_size, mode="nearest").permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode="nearest").squeeze(0).permute(1, 2, 0)
    mask_lr = None
    if mask_flat is not None:
        mask_lr = F.interpolate(mask_flat.to(torch.float32).unsqueeze(1), downsample_size, mode="nearest").squeeze(1) > 0

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    mask_lr_np = None if mask_lr is None else mask_lr.cpu().numpy()
    optim_shift: List[float] = []
    optim_focal: List[float] = []
    for i in range(points_flat.shape[0]):
        pts_i = points_lr_np[i] if mask_lr_np is None else points_lr_np[i][mask_lr_np[i]]
        uv_i = uv_lr_np if mask_lr_np is None else uv_lr_np[mask_lr_np[i]]
        finite = np.isfinite(pts_i).all(axis=-1)
        pts_i = pts_i[finite]
        uv_i = uv_i[finite]
        if uv_i.shape[0] < 2:
            optim_focal.append(1.0)
            optim_shift.append(0.0)
            continue
        shift_i, focal_i = solve_optimal_focal_shift(uv_i, pts_i)
        optim_focal.append(float(focal_i))
        optim_shift.append(float(shift_i))

    focal_t = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    shift_t = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    return focal_t, shift_t


def intrinsics_from_recovered_focal(focal: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Convert Pi3 focal (relative to half diagonal) into pixel intrinsics."""
    aspect_ratio = width / height
    sqrt_term = math.sqrt(1.0 + aspect_ratio ** 2)
    fx_norm = focal / 2.0 * sqrt_term / aspect_ratio
    fy_norm = focal / 2.0 * sqrt_term
    k = torch.zeros(*focal.shape, 3, 3, device=focal.device, dtype=focal.dtype)
    k[..., 0, 0] = fx_norm * width
    k[..., 1, 1] = fy_norm * height
    k[..., 0, 2] = 0.5 * width
    k[..., 1, 2] = 0.5 * height
    k[..., 2, 2] = 1.0
    return k


class Pi3EvalAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        recover_focal: bool = True,
        focal_mask_threshold: float = 0.1,
        focal_downsample_size: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.model = model
        self.recover_focal = recover_focal
        self.focal_mask_threshold = focal_mask_threshold
        self.focal_downsample_size = focal_downsample_size

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Pi3 eval expects images [B,S,3,H,W], got {tuple(images.shape)}")
        h, w = int(images.shape[-2]), int(images.shape[-1])
        if h % 14 != 0 or w % 14 != 0:
            raise ValueError(
                f"Pi3 requires image H/W divisible by 14, got H={h}, W={w}. "
                "Set IMAGE_SIZE to a multiple of 14 in eval_pi3.sh."
            )

        pred = self.model(images)
        local_points = pred["local_points"].float()
        conf_logits = pred["conf"][..., 0].float()
        depth_conf = torch.sigmoid(conf_logits)
        out: Dict[str, torch.Tensor] = {
            "depth": local_points[..., 2:3],
            "depth_conf": depth_conf,
            "world_points": pred["points"].float(),
            "camera_c2w": pred["camera_poses"].float(),
        }
        if self.recover_focal:
            focal_mask = (depth_conf > self.focal_mask_threshold) & torch.isfinite(local_points).all(dim=-1)
            focal, shift = recover_focal_shift(local_points, focal_mask, downsample_size=self.focal_downsample_size)
            intrinsics = intrinsics_from_recovered_focal(focal, h, w)
            out["intrinsics"] = intrinsics
            out["focal_shift"] = shift
        return out


def build_pi3_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    pi3_repo = Path(args.pi3_repo).resolve()
    if str(pi3_repo) not in sys.path:
        sys.path.insert(0, str(pi3_repo))
    from pi3.models.pi3 import Pi3

    ckpt = str(getattr(args, "pi3_checkpoint", "") or "")
    if ckpt:
        model = Pi3().to(device).eval()
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file
            weight = load_file(ckpt)
        else:
            weight = torch.load(ckpt, map_location=device, weights_only=False)
            if isinstance(weight, dict):
                weight = weight.get("model", weight.get("state_dict", weight))
        model.load_state_dict(weight)
        print(f"[eval_pi3] loaded Pi3 checkpoint: {ckpt}")
    else:
        model = Pi3.from_pretrained(str(args.pi3_pretrained_model_name_or_path)).to(device).eval()
        print(f"[eval_pi3] loaded Pi3 from_pretrained: {args.pi3_pretrained_model_name_or_path}")
    return Pi3EvalAdapter(
        model,
        recover_focal=bool(getattr(args, "recover_focal", True)),
        focal_mask_threshold=float(getattr(args, "focal_mask_threshold", 0.1)),
        focal_downsample_size=tuple(getattr(args, "focal_downsample_size", (64, 64))),
    ).to(device).eval()


def compute_camera_metrics_evo_from_c2w(
    pred_c2w: torch.Tensor,
    gt_extrinsics_w2c: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    align_mode: str = "none",
) -> Optional[Dict[str, float]]:
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D

    mask = valid_frame_mask.bool()
    if int(mask.sum().item()) < 3:
        return None
    if pred_c2w.shape[-2:] == (4, 4):
        pred_c2w_3x4 = pred_c2w[..., :3, :]
    else:
        pred_c2w_3x4 = pred_c2w
    gt_c2w_3x4 = T.w2c_to_c2w_extrinsics(gt_extrinsics_w2c.unsqueeze(0).float())[0]

    mask_np = mask.detach().cpu().numpy()
    pred_4x4 = LBE._se3_3x4_to_4x4_np(pred_c2w_3x4.detach().cpu().numpy())[mask_np]
    gt_4x4 = LBE._se3_3x4_to_4x4_np(gt_c2w_3x4.detach().cpu().numpy())[mask_np]
    if pred_4x4.shape[0] < 3 or not (np.all(np.isfinite(pred_4x4)) and np.all(np.isfinite(gt_4x4))):
        return None
    pred_4x4 = _project_pose_rotations_to_so3(pred_4x4)
    gt_4x4 = _project_pose_rotations_to_so3(gt_4x4)

    timestamps = np.arange(pred_4x4.shape[0], dtype=np.float64)
    pred_traj = PoseTrajectory3D(poses_se3=list(pred_4x4), timestamps=timestamps)
    gt_traj = PoseTrajectory3D(poses_se3=list(gt_4x4), timestamps=timestamps.copy())
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
        print(f"[eval_pi3] evo trajectory metrics failed ({str(exc).splitlines()[0][:120]}); skipping scene")
        return None
    return {
        "ate_rmse": float(ate_result.stats["rmse"]),
        "rpe_trans_rmse": float(rpe_trans_result.stats["rmse"]),
        "rpe_rot_rmse_deg": float(rpe_rot_result.stats["rmse"]),
        "n_frames_used": float(pred_4x4.shape[0]),
    }


def _project_pose_rotations_to_so3(poses_4x4: np.ndarray) -> np.ndarray:
    """Project slightly non-orthonormal rotations to the nearest SO(3)."""
    out = poses_4x4.copy()
    for i in range(out.shape[0]):
        u, _, vt = np.linalg.svd(out[i, :3, :3])
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1.0
            r = u @ vt
        out[i, :3, :3] = r
        out[i, 3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=out.dtype)
    return out


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
    fov_h_err_deg = torch.rad2deg((pred_fov_h - gt_fov_h).abs())[mask].detach().cpu().numpy()
    fov_w_err_deg = torch.rad2deg((pred_fov_w - gt_fov_w).abs())[mask].detach().cpu().numpy()
    return [{"fov_h_deg_mae": float(h_err), "fov_w_deg_mae": float(w_err)}
            for h_err, w_err in zip(fov_h_err_deg, fov_w_err_deg)]


def _camera_c2w_to_4x4(camera_c2w: torch.Tensor) -> torch.Tensor:
    if camera_c2w.shape[-2:] == (4, 4):
        return camera_c2w
    if camera_c2w.shape[-2:] == (3, 4):
        return T.se3_3x4_to_4x4(camera_c2w)
    raise ValueError(f"Expected camera_c2w with shape [...,3,4] or [...,4,4], got {tuple(camera_c2w.shape)}")


def _transform_world_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(points[..., :1])
    points_h = torch.cat([points, ones], dim=-1)
    transformed = torch.einsum("bij,bshwj->bshwi", transform.to(dtype=points.dtype), points_h)
    return transformed[..., :3]


def vggt_normalize_pi3_predictions(
    predictions: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Canonicalize Pi3 predictions by predicted frame 0 and scale by predicted points."""
    camera_c2w = _camera_c2w_to_4x4(predictions["camera_c2w"].float())
    first_w2c = T.inverse_se3(camera_c2w[:, 0])
    canon_c2w = torch.matmul(first_w2c.unsqueeze(1), camera_c2w)
    canon_world_points = _transform_world_points(predictions["world_points"].float(), first_w2c)
    scale = LBE._mean_valid_point_scale(canon_world_points, valid_mask)

    norm_c2w = canon_c2w.clone()
    norm_c2w[:, :, :3, 3] = norm_c2w[:, :, :3, 3] / scale[:, None, None]

    out = dict(predictions)
    out["camera_c2w"] = T.check_and_fix_inf_nan(norm_c2w, "vggt_pi3_camera_c2w", hard_max=None)
    out["world_points"] = T.check_and_fix_inf_nan(
        canon_world_points / scale[:, None, None, None, None],
        "vggt_pi3_world_points",
        hard_max=None,
    )
    out["depth"] = T.check_and_fix_inf_nan(
        predictions["depth"].float() / scale[:, None, None, None, None],
        "vggt_pi3_depth",
        hard_max=None,
    )
    out["vggt_pred_scale"] = scale
    return out


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_pi3] output_dir = {base_output_dir}")

    model = build_pi3_model(args, device)
    modes = {
        "train_default": ["default"],
        "manip_track": ["manip_track"],
        "wrist_track": ["wrist_track"],
        "random_static_track": ["random_static_track"],
        "both": ["wrist_track", "random_static_track"],
    }[args.eval_strategy]

    overall: Dict[str, object] = {
        "model": "Pi3",
        "checkpoint": args.checkpoint,
        "pi3_repo": str(args.pi3_repo),
        "train_args_json": str(args.train_args_json),
        "split": args.split,
        "eval_strategy": args.eval_strategy,
        "eval_num_frames": int(args.eval_num_frames),
        "recover_focal": bool(getattr(args, "recover_focal", True)),
        "focal_mask_threshold": float(getattr(args, "focal_mask_threshold", 0.1)),
        "focal_downsample_size": list(getattr(args, "focal_downsample_size", (64, 64))),
        "geometry_normalization": str(getattr(args, "geometry_normalization", "vggt_independent")),
        "depth_align": str(args.depth_align),
        "camera_align": str(getattr(args, "camera_align", "none")),
        "pointcloud_metrics": bool(getattr(args, "pointcloud_metrics", False)),
        "pointcloud_max_points": int(getattr(args, "pointcloud_max_points", 100000)),
        "pointcloud_align": str(getattr(args, "pointcloud_align", "none")),
        "pointcloud_icp_threshold": float(getattr(args, "pointcloud_icp_threshold", 0.1)),
        "pointcloud_icp_max_iterations": int(getattr(args, "pointcloud_icp_max_iterations", 30)),
        "modes": {},
    }

    for mode_name in modes:
        out_dir = base_output_dir if len(modes) == 1 else (base_output_dir / mode_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[eval_pi3] === mode: {mode_name} ===")
        cast(Dict[str, object], overall["modes"])[mode_name] = _evaluate_one_mode(args, model, device, mode_name, out_dir)

    summary_path = base_output_dir / "metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    print(f"[eval_pi3] wrote {summary_path}")
    return overall


@torch.no_grad()
def _evaluate_one_mode(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    eval_mode: str,
    output_dir: Path,
) -> Dict[str, object]:
    loader, eval_scenes = LBE.build_eval_loader(args, eval_mode=eval_mode)
    print(f"[eval_pi3] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
          f"eval_mode={eval_mode}, num_frames={args.eval_num_frames}")

    amp_enabled = bool(getattr(args, "amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "bf16") == "bf16" else torch.float16
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else contextlib.nullcontext()

    all_depth_frames: List[Dict[str, float]] = []
    per_scene_traj: List[Dict[str, float]] = []
    all_fov_frames: List[Dict[str, float]] = []
    all_pointcloud_rows: List[Dict[str, float]] = []

    group_depth_frames: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_traj: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_fov_frames: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_pointcloud_rows: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_counts: Dict[str, int] = {"realsense": 0, "surround": 0, "other": 0}

    per_scene_rows: List[Dict[str, object]] = []
    skipped = 0
    evaluated = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images_before_norm = cast(torch.Tensor, batch["images"])
        geometry_normalization = str(getattr(args, "geometry_normalization", "vggt_independent"))
        if geometry_normalization == "vggt_independent":
            batch = LBE.vggt_normalize_gt_batch(batch)
        elif geometry_normalization == "native":
            if getattr(args, "canonicalize_first_frame", True):
                batch = T.canonicalize_to_first_frame(batch)
            if getattr(args, "normalize_scene", True):
                batch = T.normalize_scene_batch(
                    batch,
                    num_anchor_frames=min(args.num_scale_frames, int(images_before_norm.shape[1])),
                )
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

        try:
            with amp_ctx:
                predictions = model(images_t)
        except Exception as exc:
            skipped += 1
            print(f"[eval_pi3] batch {batch_idx} forward failed ({exc}); skipping")
            continue

        if geometry_normalization == "vggt_independent":
            predictions = vggt_normalize_pi3_predictions(predictions, point_masks_t.bool())

        scene_field = batch["scene"]
        scene_name = scene_field[0] if isinstance(scene_field, list) else str(scene_field)
        mode_field = batch["sample_mode"]
        sample_mode = mode_field[0] if isinstance(mode_field, list) else str(mode_field)
        metric_group = LBE.metric_group_from_sample_mode(sample_mode)
        group_counts.setdefault(metric_group, 0)
        group_depth_frames.setdefault(metric_group, [])
        group_traj.setdefault(metric_group, [])
        group_fov_frames.setdefault(metric_group, [])
        group_pointcloud_rows.setdefault(metric_group, [])

        per_frame_valid = point_masks_t.sum(dim=(-1, -2)) > int(getattr(args, "min_valid_pixels", 100))
        row: Dict[str, object] = {
            "scene": scene_name,
            "sample_mode": sample_mode,
            "metric_group": metric_group,
            "n_frames": int(images_t.shape[1]),
        }

        pred_depth = predictions["depth"].float()
        gt_depth = depths_t.float()
        mask = point_masks_t.bool() & per_frame_valid[..., None, None]
        scene_frames = LBE.compute_depth_per_frame_metrics(pred_depth, gt_depth, mask, align=str(args.depth_align))
        all_depth_frames.extend(scene_frames)
        group_depth_frames[metric_group].extend(scene_frames)
        if scene_frames:
            scene_macro = LBE.mean_over_frames(scene_frames)
            for k, v in scene_macro.items():
                row[f"depth_{k}"] = v

        pred_c2w = predictions["camera_c2w"][0].float()
        traj_metrics = compute_camera_metrics_evo_from_c2w(
            pred_c2w,
            extrinsics_t[0],
            per_frame_valid[0],
            align_mode=str(getattr(args, "camera_align", "none")),
        )
        if traj_metrics is not None:
            per_scene_traj.append(traj_metrics)
            group_traj[metric_group].append(traj_metrics)
            row["cam_ate_rmse"] = traj_metrics["ate_rmse"]
            row["cam_rpe_trans_rmse"] = traj_metrics["rpe_trans_rmse"]
            row["cam_rpe_rot_rmse_deg"] = traj_metrics["rpe_rot_rmse_deg"]
            row["cam_n_frames_used"] = traj_metrics["n_frames_used"]

        pred_intr = predictions.get("intrinsics")
        if pred_intr is not None:
            image_hw = (int(images_t.shape[-2]), int(images_t.shape[-1]))
            fov_rows = compute_fov_errors_from_intrinsics(pred_intr[0].float(), intrinsics_t[0].float(), image_hw, per_frame_valid[0])
            if fov_rows is not None:
                all_fov_frames.extend(fov_rows)
                group_fov_frames[metric_group].extend(fov_rows)
                row["cam_fov_h_deg_mae"] = float(np.mean([r["fov_h_deg_mae"] for r in fov_rows]))
                row["cam_fov_w_deg_mae"] = float(np.mean([r["fov_w_deg_mae"] for r in fov_rows]))

        if bool(getattr(args, "pointcloud_metrics", False)):
            pc_valid = point_masks_t.bool() & per_frame_valid[..., None, None]
            try:
                pc_metrics = LBE.compute_pointcloud_metrics(
                    predictions["world_points"].float(),
                    world_points_t.float(),
                    pc_valid,
                    max_points=int(getattr(args, "pointcloud_max_points", 100000)),
                    align=str(getattr(args, "pointcloud_align", "none")),
                    icp_threshold=float(getattr(args, "pointcloud_icp_threshold", 0.1)),
                    icp_max_iterations=int(getattr(args, "pointcloud_icp_max_iterations", 30)),
                )
            except Exception as exc:  # noqa: BLE001
                pc_metrics = None
                print(f"[eval_pi3] pointcloud metrics failed for {scene_name} ({str(exc).splitlines()[0][:120]}); skipping")
            if pc_metrics is not None:
                all_pointcloud_rows.append(pc_metrics)
                group_pointcloud_rows[metric_group].append(pc_metrics)
                for k, v in pc_metrics.items():
                    row[f"pc_{k}"] = v

        if args.save_predictions:
            pred_path = output_dir / "predictions" / f"{scene_name}.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                pred_path,
                depth=predictions["depth"][0, ..., 0].float().detach().cpu().numpy().astype(np.float32),
                depth_conf=predictions["depth_conf"][0].float().detach().cpu().numpy().astype(np.float32),
                camera_c2w=predictions["camera_c2w"][0].float().detach().cpu().numpy().astype(np.float32),
                world_points=predictions["world_points"][0].float().detach().cpu().numpy().astype(np.float32),
                intrinsics=(predictions["intrinsics"][0].float().detach().cpu().numpy().astype(np.float32)
                            if "intrinsics" in predictions else np.empty(0, dtype=np.float32)),
                focal_shift=(predictions["focal_shift"][0].float().detach().cpu().numpy().astype(np.float32)
                             if "focal_shift" in predictions else np.empty(0, dtype=np.float32)),
                gt_depth=depths_t[0].float().detach().cpu().numpy().astype(np.float32),
                point_masks=point_masks_t[0].detach().cpu().numpy().astype(np.uint8),
                extrinsics=extrinsics_t[0].float().detach().cpu().numpy().astype(np.float32),
                gt_intrinsics=intrinsics_t[0].float().detach().cpu().numpy().astype(np.float32),
            )

        per_scene_rows.append(row)
        evaluated += 1
        group_counts[metric_group] += 1
        if args.print_every > 0 and (batch_idx % args.print_every == 0 or batch_idx == len(loader)):
            running_depth = LBE.mean_over_frames(all_depth_frames)
            running_msg = (f"AbsRel={running_depth.get('AbsRel', float('nan')):.4f} "
                           f"RMSE={running_depth.get('RMSE', float('nan')):.4f} "
                           f"d1={running_depth.get('delta<1.25', float('nan')):.4f}") if running_depth else "no-depth"
            cam_msg = ""
            if per_scene_traj:
                cam_msg = (f" ATE={np.mean([m['ate_rmse'] for m in per_scene_traj]):.4f}"
                           f" RPEt={np.mean([m['rpe_trans_rmse'] for m in per_scene_traj]):.4f}"
                           f" RPEr={np.mean([m['rpe_rot_rmse_deg'] for m in per_scene_traj]):.3f}deg")
            pc_msg = ""
            if all_pointcloud_rows:
                pc_msg = (f" PC_ACC={np.mean([m['ACC'] for m in all_pointcloud_rows]):.4f}"
                          f" PC_Comp={np.mean([m['Completeness'] for m in all_pointcloud_rows]):.4f}"
                          f" PC_CD={np.mean([m['CD'] for m in all_pointcloud_rows]):.4f}")
            print(f"[eval_pi3] [{batch_idx}/{len(loader)}] scene={scene_name} mode={sample_mode} "
                  f"frames={int(images_t.shape[1])} {running_msg}{cam_msg}{pc_msg} skipped={skipped}")

    if evaluated == 0:
        raise RuntimeError(f"No batches evaluated for eval_mode={eval_mode}.")

    summary: Dict[str, object] = {
        "eval_mode": eval_mode,
        "scenes_evaluated": evaluated,
        "scenes_skipped": skipped,
        "geometry_normalization": str(getattr(args, "geometry_normalization", "vggt_independent")),
        "depth_align": str(getattr(args, "depth_align", "none")),
        "camera_align": str(getattr(args, "camera_align", "none")),
        "pointcloud_align": str(getattr(args, "pointcloud_align", "none")),
        "aggregation": "LingBot-MAP metrics; depth=per-frame macro; trajectory=per-sequence macro; fov=recovered Pi3 intrinsics per-frame macro; pointcloud=per-clip macro",
    }
    overall_summary = LBE.summarize_metric_group(all_depth_frames, per_scene_traj, all_fov_frames, all_pointcloud_rows, evaluated)
    if "depth" in overall_summary:
        summary["depth"] = overall_summary["depth"]
    if "camera" in overall_summary:
        summary["camera"] = overall_summary["camera"]
    if "pointcloud" in overall_summary:
        summary["pointcloud"] = overall_summary["pointcloud"]
    summary["overall"] = overall_summary
    summary["groups"] = {
        group_name: LBE.summarize_metric_group(
            group_depth_frames[group_name],
            group_traj[group_name],
            group_fov_frames[group_name],
            group_pointcloud_rows[group_name],
            group_counts[group_name],
        )
        for group_name in ("realsense", "surround", "other")
        if group_counts.get(group_name, 0) > 0
    }

    summary_path = output_dir / "metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[eval_pi3] wrote {summary_path}")

    if args.per_scene_csv and per_scene_rows:
        keys = sorted({k for row in per_scene_rows for k in row.keys()})
        csv_path = output_dir / "per_scene.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_scene_rows)
        print(f"[eval_pi3] wrote {csv_path}")

    print()
    print("=" * 60)
    print(f"Model         : Pi3")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Split         : {args.split} ({evaluated} scenes evaluated, {skipped} skipped)")
    print(f"Eval mode     : {eval_mode}  num_frames={args.eval_num_frames}")
    depth_summary = cast(Optional[Dict[str, float]], summary.get("depth"))
    if depth_summary is not None:
        print(f"Depth metrics (geometry_normalization={getattr(args, 'geometry_normalization', 'vggt_independent')}, depth_align={args.depth_align}):")
        for k, v in depth_summary.items():
            print(f"  {k:18s} = {v:.6f}" if isinstance(v, float) else f"  {k:18s} = {v}")
    camera_summary = cast(Optional[Dict[str, float]], summary.get("camera"))
    if camera_summary is not None:
        print(f"Camera metrics (align={getattr(args, 'camera_align', 'none')}; Pi3 c2w poses; FoV from recover_focal_shift intrinsics):")
        for k, v in camera_summary.items():
            print(f"  {k:22s} = {v:.6f}" if isinstance(v, float) else f"  {k:22s} = {v}")
    pointcloud_summary = cast(Optional[Dict[str, float]], summary.get("pointcloud"))
    if pointcloud_summary is not None:
        print(f"Point cloud metrics (align={getattr(args, 'pointcloud_align', 'none')}):")
        for k, v in pointcloud_summary.items():
            print(f"  {k:22s} = {v:.6f}" if isinstance(v, float) else f"  {k:22s} = {v}")
    print("=" * 60)
    return summary


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = coerce_pi3_args_from_json(eval_args)
    random.seed(int(args.eval_seed))
    np.random.seed(int(args.eval_seed))
    torch.manual_seed(int(args.eval_seed))
    evaluate(args)


if __name__ == "__main__":
    main()
