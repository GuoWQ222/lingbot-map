"""Evaluate pretrained VGGT on the LingBot-MAP Manip validation split.

This script reuses eval.py's Manip split/sampling and metric helpers, but swaps
in the upstream VGGT model. By default it keeps GT and prediction geometry in
their native frames/scales and only applies alignment inside the metric helpers,
matching eval.py/eval_pi3.py:

  1. images are resized/cropped on the VGGT-native grid: width=518,
     patch-size divisible, and center-crop height by default
  2. depth metrics can use Pi3-style sequence scale/shift alignment
  3. camera metrics can use Sim(3) alignment via evo
  4. point-cloud metrics use VGGT depth + pose unprojection, not the point head
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = REPO_DIR / "outputs" / "runs" / "manip_long_train_64gpu"
DEFAULT_VGGT_REPO = REPO_DIR.parent / "vggt"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval as E
import train as T
from lingbot_map.utils.pose_enc import extri_intri_to_pose_encoding, pose_encoding_to_extri_intri


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate VGGT on the LingBot-MAP Manip val split")
    p.add_argument("--train_args_json", type=str, default=str(DEFAULT_RUN_DIR / "args.json"),
                   help="Training args.json used only to recreate the Manip validation split.")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_RUN_DIR / "eval_vggt"),
                   help="Where to write metrics.json, per_scene.csv, and optional predictions.")
    p.add_argument("--vggt_repo", type=str, default=str(DEFAULT_VGGT_REPO),
                   help="Path to a local VGGT checkout to put on PYTHONPATH.")
    p.add_argument("--model_name", type=str, default="facebook/VGGT-1B",
                   help="Hugging Face model id for VGGT.from_pretrained.")
    p.add_argument("--model_weights", type=str, default="",
                   help="Optional local VGGT .pt/.pth state_dict. If set, avoids from_pretrained.")
    p.add_argument("--strict_load", action="store_true",
                   help="Use strict=True when loading --model_weights.")

    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max_scenes_eval", type=int, default=0)
    p.add_argument("--eval_shard_count", type=int, default=1,
                   help="Split eval scenes into this many deterministic shards.")
    p.add_argument("--eval_shard_index", type=int, default=0,
                   help="Run only eval scenes with index %% eval_shard_count == eval_shard_index.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per_scene_csv", action="store_true", default=True)
    p.add_argument("--no_per_scene_csv", action="store_false", dest="per_scene_csv")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--print_every", type=int, default=5)

    p.add_argument("--eval_strategy", choices=["manip_track", "wrist_track", "random_static_track", "both", "left_moving_tracks"],
                   default="left_moving_tracks")
    p.add_argument("--eval_num_frames", type=int, default=64)
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_moving",
                   help="For left_moving_tracks/manip_track Long6. Empty string evaluates all surround cameras in manip_track.")
    p.add_argument("--eval_seed", type=int, default=42)

    p.add_argument("--image_size", type=int, default=518,
                   help="VGGT default image size. Keep 518 unless intentionally ablated.")
    p.add_argument("--vggt_preprocess_mode", choices=["crop", "pad"], default="crop",
                   help="VGGT-native image geometry. crop matches vggt.utils.load_fn default; "
                        "pad preserves all pixels on a square 518 grid.")
    p.add_argument("--depth_align", choices=["none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="pi3_scale_shift",
                   help="Primary depth metric alignment. Pi3 modes align once over the sequence.")
    p.add_argument("--secondary_depth_align", choices=["", "none", "median", "lsq", "pi3_scale", "pi3_scale_shift"], default="",
                   help="Optional extra depth summary with another alignment mode.")
    p.add_argument("--geometry_normalization", choices=["vggt_independent", "none"], default="none",
                   help="Geometry frame/scale before metrics. none leaves GT/pred untouched; "
                        "vggt_independent keeps the old VGGT first-camera + scale normalization.")
    p.add_argument("--camera_align", choices=["sim3", "none"], default="sim3",
                   help="Camera metric alignment after geometry normalization.")
    p.add_argument("--pointcloud_metrics", action="store_true", default=True)
    p.add_argument("--no_pointcloud_metrics", action="store_false", dest="pointcloud_metrics")
    p.add_argument("--pointcloud_max_points", type=int, default=100000)
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"],
                   default="pi3_icp")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1)
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30)
    p.add_argument("--pointcloud_icp_backend", choices=["auto", "open3d", "scipy"], default="open3d")
    p.add_argument("--pointcloud_kdtree_workers", type=int, default=1)
    p.add_argument("--pointcloud_workers", type=int, default=1)

    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    return p


def args_from_run_json(eval_args: argparse.Namespace) -> argparse.Namespace:
    args_json_path = Path(eval_args.train_args_json)
    if not args_json_path.is_file():
        raise FileNotFoundError(f"args.json not found: {args_json_path}")
    with args_json_path.open("r", encoding="utf-8") as f:
        run_args = json.load(f)

    ns = argparse.Namespace(**run_args)
    ns.train_args_json = str(args_json_path)
    ns.output_dir = eval_args.output_dir
    ns.vggt_repo = eval_args.vggt_repo
    ns.model_name = eval_args.model_name
    ns.model_weights = eval_args.model_weights
    ns.strict_load = bool(eval_args.strict_load)

    ns.split = eval_args.split
    ns.max_scenes_eval = eval_args.max_scenes_eval
    ns.eval_shard_count = int(eval_args.eval_shard_count)
    ns.eval_shard_index = int(eval_args.eval_shard_index)
    ns.num_workers = eval_args.num_workers
    ns.device = eval_args.device
    ns.per_scene_csv = bool(eval_args.per_scene_csv)
    ns.save_predictions = bool(eval_args.save_predictions)
    ns.print_every = eval_args.print_every

    ns.eval_strategy = eval_args.eval_strategy
    ns.eval_num_frames = eval_args.eval_num_frames
    ns.eval_wrist_camera_name = eval_args.eval_wrist_camera_name
    ns.eval_surround_camera_name = eval_args.eval_surround_camera_name
    ns.eval_seed = eval_args.eval_seed

    ns.input_preprocess = "vggt_native"
    ns.image_size = int(eval_args.image_size)
    ns.patch_size = 14
    ns.vggt_preprocess_mode = str(getattr(eval_args, "vggt_preprocess_mode", "crop"))
    ns.preprocess_mode = ns.vggt_preprocess_mode
    ns.depth_align = eval_args.depth_align
    ns.secondary_depth_align = eval_args.secondary_depth_align
    ns.geometry_normalization = str(eval_args.geometry_normalization)
    ns.camera_align = str(eval_args.camera_align)
    ns.pointcloud_metrics = bool(eval_args.pointcloud_metrics)
    ns.pointcloud_max_points = int(eval_args.pointcloud_max_points)
    ns.pointcloud_align = eval_args.pointcloud_align
    ns.pointcloud_icp_threshold = float(eval_args.pointcloud_icp_threshold)
    ns.pointcloud_icp_max_iterations = int(eval_args.pointcloud_icp_max_iterations)
    ns.pointcloud_icp_backend = str(eval_args.pointcloud_icp_backend)
    ns.pointcloud_kdtree_workers = int(eval_args.pointcloud_kdtree_workers)
    ns.pointcloud_workers = int(eval_args.pointcloud_workers)
    ns.amp = bool(eval_args.amp)
    ns.amp_dtype = eval_args.amp_dtype

    # eval.build_eval_loader expects these training knobs to exist.
    ns.write_manifest = None
    ns.batch_size = 1
    ns.cpu = (ns.device == "cpu")
    return ns


def _to_homogeneous_w2c(extrinsics: torch.Tensor) -> torch.Tensor:
    b, s = extrinsics.shape[:2]
    out = torch.zeros((b, s, 4, 4), device=extrinsics.device, dtype=extrinsics.dtype)
    out[:, :, :3, :] = extrinsics
    out[:, :, 3, 3] = 1.0
    return out


def _invert_se3_4x4(se3: torch.Tensor) -> torch.Tensor:
    r = se3[..., :3, :3]
    t = se3[..., :3, 3]
    rt = r.transpose(-1, -2)
    out = torch.zeros_like(se3)
    out[..., :3, :3] = rt
    out[..., :3, 3] = -(rt @ t[..., None]).squeeze(-1)
    out[..., 3, 3] = 1.0
    return out


def _mean_valid_point_scale(points: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_mask.to(device=points.device, dtype=torch.bool)
    finite = torch.isfinite(points).all(dim=-1)
    valid = mask & finite
    dist = points.float().norm(dim=-1)
    valid_f = valid.float()
    valid_count = valid_f.sum(dim=(1, 2, 3))
    valid_sum = (dist * valid_f).sum(dim=(1, 2, 3))

    finite_f = finite.float()
    finite_count = finite_f.sum(dim=(1, 2, 3))
    finite_sum = (dist * finite_f).sum(dim=(1, 2, 3))
    fallback = torch.where(
        finite_count > 0,
        finite_sum / finite_count.clamp(min=1.0),
        torch.ones_like(finite_count),
    )
    scale = torch.where(valid_count > 0, valid_sum / valid_count.clamp(min=1.0), fallback)
    return scale.clamp(min=1e-6, max=1e6)


def _transform_world_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(points[..., :1])
    points_h = torch.cat([points, ones], dim=-1)
    transformed = torch.einsum("bij,bshwj->bshwi", transform.to(dtype=points.dtype), points_h)
    return transformed[..., :3]


def _depth_accumulator_state(acc: E.DepthPixelAccumulator) -> Dict[str, float]:
    return {
        "valid_pixels": float(acc.valid_pixels),
        "abs_rel_sum": float(acc.abs_rel_sum),
        "sq_rel_sum": float(acc.sq_rel_sum),
        "sq_error_sum": float(acc.sq_error_sum),
        "abs_error_sum": float(acc.abs_error_sum),
        "log10_sum": float(acc.log10_sum),
        "log_diff_sum": float(acc.log_diff_sum),
        "log_diff_sq_sum": float(acc.log_diff_sq_sum),
        "delta1_count": float(acc.delta1_count),
        "delta2_count": float(acc.delta2_count),
        "delta3_count": float(acc.delta3_count),
        "align_scale_sum": float(np.sum(acc.align_scales)) if acc.align_scales else 0.0,
        "align_scale_count": float(len(acc.align_scales)),
        "align_shift_sum": float(np.sum(acc.align_shifts)) if acc.align_shifts else 0.0,
        "align_shift_count": float(len(acc.align_shifts)),
    }


def vggt_normalize_gt_batch(batch: Dict[str, object]) -> Tuple[Dict[str, object], torch.Tensor]:
    """Mirror VGGT training/train_utils/normalization.py without the CPU assert."""
    extrinsics = cast(torch.Tensor, batch["extrinsics"]).float()
    depths = cast(torch.Tensor, batch["depths"]).float()
    world_points = cast(torch.Tensor, batch["world_points"]).float()
    point_masks = cast(torch.Tensor, batch["point_masks"]).bool()

    extr_h = _to_homogeneous_w2c(extrinsics)
    first_c2w = _invert_se3_4x4(extr_h[:, 0])
    new_extr_h = extr_h @ first_c2w[:, None]

    r0 = extrinsics[:, 0, :3, :3]
    t0 = extrinsics[:, 0, :3, 3]
    new_world = torch.einsum("bshwi,bji->bshwj", world_points, r0) + t0[:, None, None, None, :]

    avg_scale = _mean_valid_point_scale(new_world, point_masks)

    new_extr_h[:, :, :3, 3] = new_extr_h[:, :, :3, 3] / avg_scale[:, None, None]
    normalized = dict(batch)
    normalized["extrinsics"] = new_extr_h[:, :, :3, :]
    normalized["depths"] = depths / avg_scale[:, None, None, None]
    normalized["world_points"] = new_world / avg_scale[:, None, None, None, None]
    normalized["point_masks"] = point_masks
    normalized["vggt_avg_scale"] = avg_scale
    normalized["vggt_gt_scale"] = avg_scale
    return normalized, avg_scale


def _extract_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a state_dict-like object, got {type(obj)!r}")
    for key in ("model", "state_dict", "model_state_dict"):
        child = obj.get(key)
        if isinstance(child, dict):
            obj = child
            break
    state = cast(Dict[str, torch.Tensor], obj)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def build_vggt_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    vggt_repo = Path(args.vggt_repo)
    if vggt_repo.is_dir() and str(vggt_repo) not in sys.path:
        sys.path.insert(0, str(vggt_repo))
    from vggt.models.vggt import VGGT

    if args.model_weights:
        model = VGGT()
        loaded = torch.load(args.model_weights, map_location="cpu")
        state = _extract_state_dict(loaded)
        missing, unexpected = model.load_state_dict(state, strict=bool(args.strict_load))
        if missing or unexpected:
            print(f"[eval_vggt] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    else:
        model = VGGT.from_pretrained(args.model_name)
    return model.to(device).eval()


def _eval_modes(strategy: str) -> List[str]:
    if strategy == "both":
        return ["wrist_track", "random_static_track"]
    return [strategy]


def _vggt_pose_encoding_to_c2w(
    pose_encoding: torch.Tensor,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """VGGT pose encoding stores OpenCV W2C extrinsics; convert to C2W."""
    pred_w2c, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding.float(),
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True,
    )
    pred_c2w = T.w2c_to_c2w_extrinsics(pred_w2c)
    return pred_c2w, intrinsics


def compute_vggt_camera_metrics_evo(
    pred_pose_enc: torch.Tensor,
    gt_extrinsics_w2c: torch.Tensor,
    image_hw: Tuple[int, int],
    valid_frame_mask: torch.Tensor,
    align_mode: str = "sim3",
) -> Optional[Dict[str, float]]:
    """Trajectory metrics for VGGT, whose pose head outputs W2C extrinsics."""
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D

    if align_mode == "sim3":
        do_align = True
    elif align_mode == "none":
        do_align = False
    else:
        raise ValueError(f"Unknown camera_align mode: {align_mode}")

    mask = valid_frame_mask.bool()
    if int(mask.sum().item()) < 3:
        return None

    pred_c2w, _ = _vggt_pose_encoding_to_c2w(pred_pose_enc.unsqueeze(0), image_hw)
    gt_c2w = T.w2c_to_c2w_extrinsics(gt_extrinsics_w2c.unsqueeze(0).float())

    pred_c2w_np = pred_c2w[0].detach().cpu().numpy()
    gt_c2w_np = gt_c2w[0].detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    pred_4x4 = E._se3_3x4_to_4x4_np(pred_c2w_np)[mask_np]
    gt_4x4 = E._se3_3x4_to_4x4_np(gt_c2w_np)[mask_np]

    if not (np.all(np.isfinite(pred_4x4)) and np.all(np.isfinite(gt_4x4))):
        return None
    if pred_4x4.shape[0] < 3:
        return None

    timestamps = np.arange(pred_4x4.shape[0], dtype=np.float64)
    pred_traj = PoseTrajectory3D(poses_se3=list(pred_4x4), timestamps=timestamps)
    gt_traj = PoseTrajectory3D(poses_se3=list(gt_4x4), timestamps=timestamps.copy())
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    try:
        ate_result = main_ape.ape(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=do_align, correct_scale=do_align,
        )
        rpe_rot_result = main_rpe.rpe(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=do_align, correct_scale=do_align,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
        )
        rpe_trans_result = main_rpe.rpe(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=do_align, correct_scale=do_align,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0][:120]
        print(f"[eval_vggt] evo trajectory metrics failed ({msg}); skipping scene")
        return None

    return {
        "ate_rmse": float(ate_result.stats["rmse"]),
        "rpe_trans_rmse": float(rpe_trans_result.stats["rmse"]),
        "rpe_rot_rmse_deg": float(rpe_rot_result.stats["rmse"]),
        "n_frames_used": float(pred_4x4.shape[0]),
    }


def unproject_depth_to_world_from_vggt_pose(
    pred_depth: torch.Tensor,
    pred_pose_enc: torch.Tensor,
) -> torch.Tensor:
    """Build world points from VGGT depth plus VGGT W2C pose encoding."""
    depth = pred_depth[..., 0].float()
    B, S, H, W = depth.shape
    pose = pred_pose_enc.float()
    c2w_3x4, intrinsics = _vggt_pose_encoding_to_c2w(pose, (int(H), int(W)))
    c2w = T.se3_3x4_to_4x4(c2w_3x4)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
    intrinsics_inv = torch.inverse(intrinsics.to(dtype=depth.dtype))
    camera_dirs = torch.einsum("bsij,hwj->bshwi", intrinsics_inv, pixel_coords)
    camera_points = camera_dirs * depth[..., None]
    ones = torch.ones(B, S, H, W, 1, device=depth.device, dtype=depth.dtype)
    camera_points_h = torch.cat([camera_points, ones], dim=-1)
    world_points_h = torch.einsum("bsij,bshwj->bshwi", c2w.to(dtype=depth.dtype), camera_points_h)
    return world_points_h[..., :3]


def vggt_normalize_predictions(
    predictions: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
    image_hw: Tuple[int, int],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Canonicalize VGGT predictions using their own frame 0 and point scale."""
    fallback_scale = torch.ones(
        int(valid_mask.shape[0]),
        device=valid_mask.device,
        dtype=torch.float32,
    )
    if "pose_enc" not in predictions or "depth" not in predictions:
        return predictions, fallback_scale

    pose_orig = cast(torch.Tensor, predictions["pose_enc"])
    depth_orig = cast(torch.Tensor, predictions["depth"])
    pose = pose_orig.float()
    pred_depth = depth_orig.float()

    pred_w2c, pred_intrinsics = pose_encoding_to_extri_intri(
        pose,
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True,
    )
    pred_w2c_h = _to_homogeneous_w2c(pred_w2c.float())
    first_w2c = pred_w2c_h[:, 0]
    first_c2w = _invert_se3_4x4(first_w2c)
    canon_w2c_h = pred_w2c_h @ first_c2w[:, None]

    pred_world_points = unproject_depth_to_world_from_vggt_pose(pred_depth, pose)
    canon_world_points = _transform_world_points(pred_world_points, first_w2c)
    pred_scale = _mean_valid_point_scale(canon_world_points, valid_mask)

    norm_w2c_h = canon_w2c_h.clone()
    norm_w2c_h[:, :, :3, 3] = norm_w2c_h[:, :, :3, 3] / pred_scale[:, None, None]
    norm_pose = extri_intri_to_pose_encoding(
        norm_w2c_h[:, :, :3, :],
        pred_intrinsics,
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
    ).to(dtype=pose_orig.dtype)

    out = dict(predictions)
    out["pose_enc"] = T.check_and_fix_inf_nan(norm_pose, "vggt_pred_pose_enc", hard_max=None)
    out["depth"] = T.check_and_fix_inf_nan(
        (pred_depth / pred_scale[:, None, None, None, None]).to(dtype=depth_orig.dtype),
        "vggt_pred_depth",
        hard_max=None,
    )
    out["world_points"] = T.check_and_fix_inf_nan(
        canon_world_points / pred_scale[:, None, None, None, None],
        "vggt_pred_world_points",
        hard_max=None,
    )
    out["vggt_pred_scale"] = pred_scale
    return out, pred_scale


@torch.no_grad()
def evaluate_one_mode(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    eval_mode: str,
    output_dir: Path,
) -> Dict[str, object]:
    if int(getattr(args, "eval_shard_count", 1)) > 1:
        import copy
        shard_count = int(args.eval_shard_count)
        shard_index = int(args.eval_shard_index)
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(f"eval_shard_index={shard_index} outside [0, {shard_count})")
        all_args = copy.copy(args)
        all_args.max_scenes_eval = 0
        all_args.eval_shard_count = 1
        all_args.eval_shard_index = 0
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
        shard_args.eval_shard_count = 1
        shard_args.eval_shard_index = 0
        # The shard manifest already contains the requested split's scenes.
        shard_args.split = "all"
        loader, eval_scenes = E.build_eval_loader(shard_args, eval_mode=eval_mode)
    else:
        loader, eval_scenes = E.build_eval_loader(args, eval_mode=eval_mode)
    print(f"[eval_vggt] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
          f"mode={eval_mode}, frames={args.eval_num_frames}, image_size={args.image_size}, "
          f"preprocess={getattr(args, 'input_preprocess', 'unknown')}:"
          f"{getattr(args, 'vggt_preprocess_mode', getattr(args, 'preprocess_mode', 'unknown'))}, "
          f"patch_size={getattr(args, 'patch_size', 'n/a')}")

    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_ctx = (torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_enabled else contextlib.nullcontext())

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
        log_prefix="[eval_vggt]",
    )

    per_scene_rows: List[Dict[str, object]] = []
    skipped = 0
    evaluated = 0

    secondary_align = str(getattr(args, "secondary_depth_align", "") or "")
    if secondary_align == args.depth_align:
        secondary_align = ""
    if secondary_align:
        secondary_depth_pixels = E.DepthPixelAccumulator()
        group_secondary_depth_pixels = {name: E.DepthPixelAccumulator() for name in report_group_names}

    for batch_idx, batch in enumerate(loader, start=1):
        geometry_normalization = str(getattr(args, "geometry_normalization", "none"))
        if geometry_normalization == "vggt_independent":
            batch = T.to_device(batch, device)
            batch, avg_scale = vggt_normalize_gt_batch(batch)
        elif geometry_normalization == "none":
            batch = T.to_device(dict(batch), device)
            avg_scale = None
        else:
            raise ValueError(f"Unknown geometry_normalization: {geometry_normalization}")

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
            print(f"[eval_vggt] batch {batch_idx} forward failed ({exc}); skipping")
            continue

        image_hw = (int(images_t.shape[-2]), int(images_t.shape[-1]))
        if geometry_normalization == "vggt_independent":
            predictions, pred_avg_scale = vggt_normalize_predictions(
                cast(Dict[str, torch.Tensor], predictions),
                point_masks_t.bool(),
                image_hw=image_hw,
            )
        else:
            pred_avg_scale = None

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

        per_frame_valid = point_masks_t.sum(dim=(-1, -2)) > int(getattr(args, "min_valid_pixels", 100))
        depth_row: Dict[str, object] = {
            "scene": scene_name,
            "sample_mode": sample_mode,
            "metric_group": metric_group,
            "n_frames": int(images_t.shape[1]),
        }
        if torch.is_tensor(avg_scale):
            depth_row["vggt_avg_scale"] = float(avg_scale[0].detach().cpu().item())
            depth_row["vggt_gt_scale"] = float(avg_scale[0].detach().cpu().item())
        if torch.is_tensor(pred_avg_scale):
            depth_row["vggt_pred_scale"] = float(pred_avg_scale[0].detach().cpu().item())

        if "depth" in predictions:
            pred_depth = cast(torch.Tensor, predictions["depth"]).float()
            mask = point_masks_t.bool() & per_frame_valid[..., None, None]
            clip_depth = E.DepthPixelAccumulator()
            clip_depth.update(pred_depth, depths_t.float(), mask, align=str(args.depth_align))
            all_depth_pixels.update(pred_depth, depths_t.float(), mask, align=str(args.depth_align))
            group_depth_pixels[metric_group].update(pred_depth, depths_t.float(), mask, align=str(args.depth_align))
            for key, value in clip_depth.summary().items():
                depth_row[f"depth_{args.depth_align}_{key}"] = value

            if secondary_align and secondary_depth_pixels is not None:
                clip_secondary_depth = E.DepthPixelAccumulator()
                clip_secondary_depth.update(pred_depth, depths_t.float(), mask, align=secondary_align)
                secondary_depth_pixels.update(pred_depth, depths_t.float(), mask, align=secondary_align)
                group_secondary_depth_pixels[metric_group].update(pred_depth, depths_t.float(), mask, align=secondary_align)
                for key, value in clip_secondary_depth.summary().items():
                    depth_row[f"depth_{secondary_align}_{key}"] = value

        if "pose_enc" in predictions:
            pred_pose = cast(torch.Tensor, predictions["pose_enc"])[0].float()
            traj_metrics = compute_vggt_camera_metrics_evo(
                pred_pose,
                extrinsics_t[0],
                image_hw=image_hw,
                valid_frame_mask=per_frame_valid[0],
                align_mode=str(getattr(args, "camera_align", "sim3")),
            )
            if traj_metrics is not None:
                per_scene_traj.append(traj_metrics)
                group_traj[metric_group].append(traj_metrics)
                depth_row["cam_ate_rmse"] = traj_metrics["ate_rmse"]
                depth_row["cam_rpe_trans_rmse"] = traj_metrics["rpe_trans_rmse"]
                depth_row["cam_rpe_rot_rmse_deg"] = traj_metrics["rpe_rot_rmse_deg"]
                depth_row["cam_n_frames_used"] = traj_metrics["n_frames_used"]


        if bool(args.pointcloud_metrics) and "depth" in predictions and "pose_enc" in predictions:
            pred_world_points = unproject_depth_to_world_from_vggt_pose(
                cast(torch.Tensor, predictions["depth"]).float(),
                cast(torch.Tensor, predictions["pose_enc"]).float(),
            )
            pc_valid = point_masks_t.bool() & per_frame_valid[..., None, None]
            pointcloud_queue.submit(
                pred_world_points.float(),
                world_points_t.float(),
                pc_valid,
                depth_row,
                metric_group,
                scene_name,
            )

        if args.save_predictions and "depth" in predictions:
            pred_path = output_dir / "predictions" / f"{scene_name}.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pred_payload = {
                "depth": cast(torch.Tensor, predictions["depth"])[0, ..., 0].float().detach().cpu().numpy().astype(np.float32),
                "pose_enc": (cast(torch.Tensor, predictions.get("pose_enc", torch.empty(0)))[0].float().detach().cpu().numpy().astype(np.float32)
                             if "pose_enc" in predictions else np.empty(0, dtype=np.float32)),
                "gt_depth": depths_t[0].float().detach().cpu().numpy().astype(np.float32),
                "point_masks": point_masks_t[0].detach().cpu().numpy().astype(np.uint8),
                "extrinsics": extrinsics_t[0].float().detach().cpu().numpy().astype(np.float32),
                "intrinsics": intrinsics_t[0].float().detach().cpu().numpy().astype(np.float32),
            }
            if torch.is_tensor(avg_scale):
                pred_payload["vggt_avg_scale"] = avg_scale[0].detach().cpu().numpy().astype(np.float32)
                pred_payload["vggt_gt_scale"] = avg_scale[0].detach().cpu().numpy().astype(np.float32)
            if torch.is_tensor(pred_avg_scale):
                pred_payload["vggt_pred_scale"] = pred_avg_scale[0].detach().cpu().numpy().astype(np.float32)
            np.savez_compressed(pred_path, **pred_payload)

        per_scene_rows.append(depth_row)
        group_counts[metric_group] += 1
        evaluated += 1
        if batch_idx % max(1, int(args.print_every)) == 0:
            running = all_depth_pixels.summary()
            print(f"[eval_vggt] {batch_idx}/{len(loader)} evaluated={evaluated} skipped={skipped} "
                  f"AbsRel({args.depth_align})={running.get('AbsRel', float('nan')):.4f}")

    pointcloud_queue.close()

    summary: Dict[str, object] = {
        "model": args.model_weights or args.model_name,
        "train_args_json": args.train_args_json,
        "split": args.split,
        "eval_mode": eval_mode,
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "input_preprocess": str(getattr(args, "input_preprocess", "vggt_native")),
        "vggt_preprocess_mode": str(getattr(args, "vggt_preprocess_mode", getattr(args, "preprocess_mode", "crop"))),
        "input_geometry_sync": "model_native_geometry+lingbot_map_sync",
        "patch_size": int(getattr(args, "patch_size", 14)),
        "geometry_normalization": str(getattr(args, "geometry_normalization", "none")),
        "depth_align": str(args.depth_align),
        "secondary_depth_align": secondary_align,
        "camera_align": str(getattr(args, "camera_align", "sim3")),
        "pointcloud_source": "depth_plus_pose_unprojection",
        "aggregation": "overall plus realsense_left/surround_cam_moving camera-track groups; depth=Pi3-style pixel-weighted after per-clip alignment; trajectory=per-sequence macro; pointcloud=per-clip macro",
        "pointcloud_align": str(args.pointcloud_align),
        "pointcloud_icp_backend": str(getattr(args, "pointcloud_icp_backend", "open3d")),
        "pointcloud_kdtree_workers": int(getattr(args, "pointcloud_kdtree_workers", 1)),
        "pointcloud_workers": int(getattr(args, "pointcloud_workers", 1)),
        "eval_shard_count": int(getattr(args, "eval_shard_count", 1)),
        "eval_shard_index": int(getattr(args, "eval_shard_index", 0)),
        "skipped": int(skipped),
    }
    overall_summary = E.summarize_metric_group(
        per_scene_traj, all_pointcloud_rows, evaluated,
        depth_pixels=all_depth_pixels,
    )
    summary["overall"] = overall_summary
    summary["depth_accumulator"] = _depth_accumulator_state(all_depth_pixels)
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
        group_summary["depth_accumulator"] = _depth_accumulator_state(group_depth_pixels[group_name])
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
    print(f"[eval_vggt] wrote {metrics_path}")

    if args.per_scene_csv and per_scene_rows:
        csv_path = output_dir / "per_scene.csv"
        keys = sorted({key for row in per_scene_rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_scene_rows:
                writer.writerow(row)
        print(f"[eval_vggt] wrote {csv_path}")

    return summary


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_vggt] output_dir={base_output_dir}")
    print(f"[eval_vggt] model={args.model_weights or args.model_name}")

    model = build_vggt_model(args, device)
    overall: Dict[str, object] = {
        "model": args.model_weights or args.model_name,
        "train_args_json": args.train_args_json,
        "split": args.split,
        "eval_strategy": args.eval_strategy,
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "input_preprocess": str(getattr(args, "input_preprocess", "vggt_native")),
        "vggt_preprocess_mode": str(getattr(args, "vggt_preprocess_mode", getattr(args, "preprocess_mode", "crop"))),
        "patch_size": int(getattr(args, "patch_size", 14)),
        "modes": {},
    }
    modes = _eval_modes(args.eval_strategy)
    for mode_name in modes:
        out_dir = base_output_dir if len(modes) == 1 else (base_output_dir / mode_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[eval_vggt] === mode: {mode_name} ===")
        cast(Dict[str, object], overall["modes"])[mode_name] = evaluate_one_mode(
            args, model, device, mode_name, out_dir
        )

    summary_path = base_output_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    print(f"[eval_vggt] wrote {summary_path}")
    return overall


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = args_from_run_json(eval_args)
    evaluate(args)


if __name__ == "__main__":
    main()
