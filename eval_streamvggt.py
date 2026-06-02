"""Evaluate pretrained StreamVGGT on the LingBot-MAP Manip validation split.

This script reuses eval.py's Manip split/sampling and metric helpers, but swaps
in the StreamVGGT model. It keeps GT and prediction geometry in their native
frames/scales and only applies alignment inside the metric helpers, matching
eval.sh/eval.py:

  1. images are resized/cropped by the LingBot Manip loader at image_size=518
  2. depth metrics can use Pi3-style sequence scale/shift alignment
  3. camera metrics can use Sim(3) alignment via evo
  4. point-cloud metrics use StreamVGGT depth + pose unprojection
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
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "manip_long_train_64gpu"
DEFAULT_STREAMVGGT_REPO = SCRIPT_DIR.parent / "StreamVGGT"
DEFAULT_STREAMVGGT_WEIGHTS = DEFAULT_STREAMVGGT_REPO / "checkpoints.pth"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval as E
import train as T
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate StreamVGGT on the LingBot-MAP Manip val split")
    p.add_argument("--train_args_json", type=str, default=str(DEFAULT_RUN_DIR / "args.json"),
                   help="Training args.json used only to recreate the Manip validation split.")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_RUN_DIR / "eval_streamvggt"),
                   help="Where to write metrics.json, per_scene.csv, and optional predictions.")
    p.add_argument("--streamvggt_repo", type=str, default=str(DEFAULT_STREAMVGGT_REPO),
                   help="Path to the local StreamVGGT checkout.")
    p.add_argument("--model_weights", type=str, default=str(DEFAULT_STREAMVGGT_WEIGHTS),
                   help="Local StreamVGGT .pt/.pth state_dict.")
    p.add_argument("--strict_load", action="store_true",
                   help="Use strict=True when loading --model_weights.")
    p.add_argument("--forward_mode", choices=["stream", "full"], default="stream",
                   help="stream uses StreamVGGT.inference cache path; full calls the batched forward path.")

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

    p.add_argument("--eval_strategy", choices=["manip_track", "wrist_track", "random_static_track", "both"],
                   default="manip_track")
    p.add_argument("--eval_num_frames", type=int, default=64)
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_0",
                   help="For manip_track Long5. Empty string evaluates all 6 surround cameras.")
    p.add_argument("--eval_seed", type=int, default=42)

    p.add_argument("--image_size", type=int, default=518,
                   help="StreamVGGT default image size. Keep 518 unless intentionally ablated.")
    p.add_argument("--depth_align", choices=["none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="pi3_scale_shift",
                   help="Primary depth metric alignment. Pi3 modes align once over the sequence.")
    p.add_argument("--secondary_depth_align", choices=["", "none", "median", "lsq", "pi3_scale", "pi3_scale_shift"], default="",
                   help="Optional extra depth summary with another alignment mode.")
    p.add_argument("--camera_align", choices=["sim3", "none"], default="sim3",
                   help="Camera metric alignment inside trajectory metrics.")
    p.add_argument("--pointcloud_metrics", action="store_true", default=True)
    p.add_argument("--no_pointcloud_metrics", action="store_false", dest="pointcloud_metrics")
    p.add_argument("--pointcloud_max_points", type=int, default=100000)
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"],
                   default="pi3_icp")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1)
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30)

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
    ns.streamvggt_repo = eval_args.streamvggt_repo
    ns.model_weights = eval_args.model_weights
    ns.strict_load = bool(eval_args.strict_load)
    ns.forward_mode = str(eval_args.forward_mode)

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

    ns.image_size = int(eval_args.image_size)
    ns.depth_align = eval_args.depth_align
    ns.secondary_depth_align = eval_args.secondary_depth_align
    ns.camera_align = str(eval_args.camera_align)
    ns.pointcloud_metrics = bool(eval_args.pointcloud_metrics)
    ns.pointcloud_max_points = int(eval_args.pointcloud_max_points)
    ns.pointcloud_align = eval_args.pointcloud_align
    ns.pointcloud_icp_threshold = float(eval_args.pointcloud_icp_threshold)
    ns.pointcloud_icp_max_iterations = int(eval_args.pointcloud_icp_max_iterations)
    ns.amp = bool(eval_args.amp)
    ns.amp_dtype = eval_args.amp_dtype

    # eval.build_eval_loader expects these training knobs to exist.
    ns.write_manifest = None
    ns.batch_size = 1
    ns.cpu = (ns.device == "cpu")
    return ns


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


class StreamVGGTAdapter(nn.Module):
    def __init__(self, model: nn.Module, forward_mode: str = "stream") -> None:
        super().__init__()
        if forward_mode not in {"stream", "full"}:
            raise ValueError(f"Unknown forward_mode: {forward_mode}")
        self.model = model
        self.forward_mode = forward_mode

    @staticmethod
    def _stack_ress(ress: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        predictions: Dict[str, torch.Tensor] = {}
        if not ress:
            return predictions
        if "depth" in ress[0]:
            predictions["depth"] = torch.stack([r["depth"] for r in ress], dim=1)
        if "depth_conf" in ress[0]:
            predictions["depth_conf"] = torch.stack([r["depth_conf"] for r in ress], dim=1)
        if "camera_pose" in ress[0]:
            predictions["pose_enc"] = torch.stack([r["camera_pose"] for r in ress], dim=1)
        if "pts3d_in_other_view" in ress[0]:
            predictions["world_points"] = torch.stack([r["pts3d_in_other_view"] for r in ress], dim=1)
        if "conf" in ress[0]:
            predictions["world_points_conf"] = torch.stack([r["conf"] for r in ress], dim=1)
        return predictions

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"Expected images with shape [B,S,C,H,W], got {tuple(images.shape)}")
        batch_size, seq_len = images.shape[:2]
        if self.forward_mode == "stream":
            if batch_size != 1:
                raise ValueError("StreamVGGT streaming eval expects batch_size=1")
            frames = [{"img": images[:, idx]} for idx in range(seq_len)]
            output = self.model.inference(frames)
        else:
            views = [{"img": images[:, idx]} for idx in range(seq_len)]
            output = self.model(views)
        return self._stack_ress(cast(List[Dict[str, torch.Tensor]], output.ress))


def _ensure_transformers_model_output() -> None:
    try:
        from transformers.file_utils import ModelOutput as _ModelOutput  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types

    class ModelOutput:
        pass

    transformers_mod = types.ModuleType("transformers")
    file_utils_mod = types.ModuleType("transformers.file_utils")
    file_utils_mod.ModelOutput = ModelOutput
    transformers_mod.file_utils = file_utils_mod
    sys.modules.setdefault("transformers", transformers_mod)
    sys.modules["transformers.file_utils"] = file_utils_mod


def build_streamvggt_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    streamvggt_repo = Path(args.streamvggt_repo)
    streamvggt_src = streamvggt_repo / "src"
    for candidate in (streamvggt_src, streamvggt_repo):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(1, str(candidate))
    _ensure_transformers_model_output()
    from streamvggt.models.streamvggt import StreamVGGT

    weights_path = Path(args.model_weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"StreamVGGT weights not found: {weights_path}")

    model = StreamVGGT()
    loaded = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(loaded)
    strict = bool(args.strict_load)
    result = model.load_state_dict(state, strict=strict)
    missing = list(getattr(result, "missing_keys", []))
    unexpected = list(getattr(result, "unexpected_keys", []))
    if missing or unexpected:
        print(f"[eval_streamvggt] load_state_dict missing={len(missing)} unexpected={len(unexpected)} strict={strict}")
    model = model.to(device).eval()
    return StreamVGGTAdapter(model, forward_mode=str(args.forward_mode)).to(device).eval()


def _eval_modes(strategy: str) -> List[str]:
    if strategy == "both":
        return ["wrist_track", "random_static_track"]
    return [strategy]


def _streamvggt_pose_encoding_to_c2w(
    pose_encoding: torch.Tensor,
    image_hw: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """StreamVGGT pose encoding stores OpenCV W2C extrinsics; convert to C2W."""
    pred_w2c, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding.float(),
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True,
    )
    pred_c2w = T.w2c_to_c2w_extrinsics(pred_w2c)
    return pred_c2w, intrinsics


def compute_streamvggt_camera_metrics_evo(
    pred_pose_enc: torch.Tensor,
    gt_extrinsics_w2c: torch.Tensor,
    image_hw: Tuple[int, int],
    valid_frame_mask: torch.Tensor,
    align_mode: str = "sim3",
) -> Optional[Dict[str, float]]:
    """Trajectory metrics for StreamVGGT, whose pose head outputs W2C extrinsics."""
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

    pred_c2w, _ = _streamvggt_pose_encoding_to_c2w(pred_pose_enc.unsqueeze(0), image_hw)
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
        print(f"[eval_streamvggt] evo trajectory metrics failed ({msg}); skipping scene")
        return None

    return {
        "ate_rmse": float(ate_result.stats["rmse"]),
        "rpe_trans_rmse": float(rpe_trans_result.stats["rmse"]),
        "rpe_rot_rmse_deg": float(rpe_rot_result.stats["rmse"]),
        "n_frames_used": float(pred_4x4.shape[0]),
    }


def unproject_depth_to_world_from_streamvggt_pose(
    pred_depth: torch.Tensor,
    pred_pose_enc: torch.Tensor,
) -> torch.Tensor:
    """Build world points from StreamVGGT depth plus StreamVGGT W2C pose encoding."""
    depth = pred_depth[..., 0].float()
    B, S, H, W = depth.shape
    pose = pred_pose_enc.float()
    c2w_3x4, intrinsics = _streamvggt_pose_encoding_to_c2w(pose, (int(H), int(W)))
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


def evaluate_one_mode(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    eval_mode: str,
    output_dir: Path,
) -> Dict[str, object]:
    # eval.build_eval_loader already applies eval_shard_count/eval_shard_index.
    # Keep sharding in one place; doing an outer manifest shard here and then
    # passing the same shard args into eval.py would shard twice.
    loader, eval_scenes = E.build_eval_loader(args, eval_mode=eval_mode)
    print(f"[eval_streamvggt] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
          f"mode={eval_mode}, frames={args.eval_num_frames}, image_size={args.image_size}")

    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    amp_ctx = (torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_enabled else contextlib.nullcontext())

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
    if secondary_align == args.depth_align:
        secondary_align = ""

    for batch_idx, batch in enumerate(loader, start=1):
        batch = T.to_device(dict(batch), device)

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
            print(f"[eval_streamvggt] batch {batch_idx} forward failed ({exc}); skipping")
            continue

        image_hw = (int(images_t.shape[-2]), int(images_t.shape[-1]))

        scene_field = batch["scene"]
        scene_name = scene_field[0] if isinstance(scene_field, list) else str(scene_field)
        mode_field = batch["sample_mode"]
        sample_mode = mode_field[0] if isinstance(mode_field, list) else str(mode_field)
        metric_group = E.metric_group_from_sample_mode(sample_mode)
        for dct in (group_counts, group_depth_frames, group_secondary_depth_frames, group_traj,
                    group_fov_frames, group_pointcloud_rows):
            if metric_group not in dct:
                dct[metric_group] = 0 if dct is group_counts else []

        per_frame_valid = point_masks_t.sum(dim=(-1, -2)) > int(getattr(args, "min_valid_pixels", 100))
        depth_row: Dict[str, object] = {
            "scene": scene_name,
            "sample_mode": sample_mode,
            "metric_group": metric_group,
            "n_frames": int(images_t.shape[1]),
        }
        if "depth" in predictions:
            pred_depth = cast(torch.Tensor, predictions["depth"]).float()
            mask = point_masks_t.bool() & per_frame_valid[..., None, None]
            scene_frames = E.compute_depth_per_frame_metrics(
                pred_depth, depths_t.float(), mask, align=str(args.depth_align),
            )
            all_depth_frames.extend(scene_frames)
            group_depth_frames[metric_group].extend(scene_frames)
            if scene_frames:
                for key, value in E.mean_over_frames(scene_frames).items():
                    depth_row[f"depth_{args.depth_align}_{key}"] = value

            if secondary_align:
                secondary_frames = E.compute_depth_per_frame_metrics(
                    pred_depth, depths_t.float(), mask, align=secondary_align,
                )
                secondary_depth_frames.extend(secondary_frames)
                group_secondary_depth_frames[metric_group].extend(secondary_frames)
                if secondary_frames:
                    for key, value in E.mean_over_frames(secondary_frames).items():
                        depth_row[f"depth_{secondary_align}_{key}"] = value

        if "pose_enc" in predictions:
            pred_pose = cast(torch.Tensor, predictions["pose_enc"])[0].float()
            traj_metrics = compute_streamvggt_camera_metrics_evo(
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

            fov_rows = E.compute_fov_per_frame_errors(
                pred_pose,
                intrinsics_t[0],
                image_hw=image_hw,
                valid_frame_mask=per_frame_valid[0],
            )
            if fov_rows is not None:
                all_fov_frames.extend(fov_rows)
                group_fov_frames[metric_group].extend(fov_rows)
                depth_row["cam_fov_h_deg_mae"] = float(np.mean([r["fov_h_deg_mae"] for r in fov_rows]))
                depth_row["cam_fov_w_deg_mae"] = float(np.mean([r["fov_w_deg_mae"] for r in fov_rows]))

        if bool(args.pointcloud_metrics) and "depth" in predictions and "pose_enc" in predictions:
            pred_world_points = unproject_depth_to_world_from_streamvggt_pose(
                cast(torch.Tensor, predictions["depth"]).float(),
                cast(torch.Tensor, predictions["pose_enc"]).float(),
            )
            pc_valid = point_masks_t.bool() & per_frame_valid[..., None, None]
            try:
                pc_metrics = E.compute_pointcloud_metrics(
                    pred_world_points.float(),
                    world_points_t.float(),
                    pc_valid,
                    max_points=int(args.pointcloud_max_points),
                    align=str(args.pointcloud_align),
                    icp_threshold=float(args.pointcloud_icp_threshold),
                    icp_max_iterations=int(args.pointcloud_icp_max_iterations),
                )
            except Exception as exc:
                pc_metrics = None
                print(f"[eval_streamvggt] pointcloud failed for {scene_name} ({str(exc).splitlines()[0][:120]}); skipping")
            if pc_metrics is not None:
                all_pointcloud_rows.append(pc_metrics)
                group_pointcloud_rows[metric_group].append(pc_metrics)
                for key, value in pc_metrics.items():
                    depth_row[f"pc_{key}"] = value

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
            np.savez_compressed(pred_path, **pred_payload)

        per_scene_rows.append(depth_row)
        group_counts[metric_group] += 1
        evaluated += 1
        if batch_idx % max(1, int(args.print_every)) == 0:
            running = E.mean_over_frames(all_depth_frames)
            print(f"[eval_streamvggt] {batch_idx}/{len(loader)} evaluated={evaluated} skipped={skipped} "
                  f"AbsRel({args.depth_align})={running.get('AbsRel', float('nan')):.4f}")

    summary: Dict[str, object] = {
        "model": args.model_weights,
        "train_args_json": args.train_args_json,
        "split": args.split,
        "eval_mode": eval_mode,
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "geometry_normalization": "none",
        "depth_align": str(args.depth_align),
        "secondary_depth_align": secondary_align,
        "camera_align": str(getattr(args, "camera_align", "sim3")),
        "pointcloud_source": "streamvggt_depth_plus_pose_unprojection",
        "pointcloud_align": str(args.pointcloud_align),
        "eval_shard_count": int(getattr(args, "eval_shard_count", 1)),
        "eval_shard_index": int(getattr(args, "eval_shard_index", 0)),
        "skipped": int(skipped),
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
    print(f"[eval_streamvggt] wrote {metrics_path}")

    if args.per_scene_csv and per_scene_rows:
        csv_path = output_dir / "per_scene.csv"
        keys = sorted({key for row in per_scene_rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_scene_rows:
                writer.writerow(row)
        print(f"[eval_streamvggt] wrote {csv_path}")

    return summary


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_streamvggt] output_dir={base_output_dir}")
    print(f"[eval_streamvggt] model={args.model_weights}")

    model = build_streamvggt_model(args, device)
    overall: Dict[str, object] = {
        "model": args.model_weights,
        "train_args_json": args.train_args_json,
        "split": args.split,
        "eval_strategy": args.eval_strategy,
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "modes": {},
    }
    modes = _eval_modes(args.eval_strategy)
    for mode_name in modes:
        out_dir = base_output_dir if len(modes) == 1 else (base_output_dir / mode_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[eval_streamvggt] === mode: {mode_name} ===")
        cast(Dict[str, object], overall["modes"])[mode_name] = evaluate_one_mode(
            args, model, device, mode_name, out_dir
        )

    summary_path = base_output_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    print(f"[eval_streamvggt] wrote {summary_path}")
    return overall


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = args_from_run_json(eval_args)
    evaluate(args)


if __name__ == "__main__":
    main()
