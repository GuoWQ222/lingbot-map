#!/usr/bin/env python3
"""Evaluate TTT3R/CUT3R on the LingBot-MAP Manip validation protocol.

This adapter reuses eval.py's deterministic Manip sampler and metric helpers.
It does not normalize GT or predictions into a first-camera frame; depth,
camera, and point-cloud alignment are applied only inside the metric functions,
matching eval.sh's default metric-time alignment behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "runs" / "manip_long_train_64gpu"
DEFAULT_TTT3R_REPO = SCRIPT_DIR.parent / "TTT3R"
DEFAULT_TTT3R_CHECKPOINT = DEFAULT_TTT3R_REPO / "src" / "cut3r_512_dpt_4_64.pth"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval as E
import train as T
from lingbot_map.utils.rotation import mat_to_quat


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate TTT3R with LingBot-MAP Manip metrics")
    p.add_argument("--train_args_json", type=str, default=str(DEFAULT_RUN_DIR / "args.json"),
                   help="LingBot-MAP args.json used to recreate the Manip eval split.")
    p.add_argument("--output_dir", type=str, default=str(DEFAULT_RUN_DIR / "eval_ttt3r"))
    p.add_argument("--ttt3r_repo", type=str, default=str(DEFAULT_TTT3R_REPO))
    p.add_argument("--ttt3r_checkpoint", type=str, default=str(DEFAULT_TTT3R_CHECKPOINT))
    p.add_argument("--model_update_type", choices=["ttt3r", "cut3r"], default="ttt3r")
    p.add_argument("--reset_interval", type=int, default=1000000,
                   help="TTT3R recurrent state reset interval. Large default means no reset within eval clips.")

    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max_scenes_eval", type=int, default=0)
    p.add_argument("--eval_num_shards", type=int, default=1,
                   help="Split eval clips across this many deterministic shards.")
    p.add_argument("--eval_shard_index", type=int, default=0,
                   help="0-based shard index when eval_num_shards > 1.")
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

    p.add_argument("--image_size", type=int, default=512,
                   help="Input size used by the LingBot loader before TTT3R inference.")
    p.add_argument("--geometry_normalization", choices=["none"], default="none",
                   help="Kept for eval.sh compatibility. TTT3R adapter intentionally supports only none.")
    p.add_argument("--camera_align", choices=["none", "sim3"], default="sim3")
    p.add_argument("--depth_align", choices=["none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="pi3_scale_shift")
    p.add_argument("--secondary_depth_align", choices=["", "none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="")
    p.add_argument("--pointcloud_metrics", action="store_true", default=True)
    p.add_argument("--no_pointcloud_metrics", action="store_false", dest="pointcloud_metrics")
    p.add_argument("--pointcloud_max_points", type=int, default=100000)
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"],
                   default="pi3_icp")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1)
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30)
    p.add_argument("--focal_mode", type=str, default="weiszfeld",
                   help="TTT3R focal recovery mode passed to estimate_focal_knowing_depth.")
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
    # eval.py's shared dataloader has both a scene-level shard field
    # (eval_shard_count) and a later dataset-level field (eval_num_shards).
    # TTT3R uses scene-level sharding only, so disable the second pass.
    ns.eval_shard_count = int(eval_args.eval_num_shards)
    ns.eval_num_shards = 1
    ns.train_args_json = str(args_json_path)
    ns.model_path = ""
    ns.cpu = (eval_args.device == "cpu")
    ns.batch_size = 1
    ns.write_manifest = ns.__dict__.get("write_manifest", None) or None
    ns.geometry_normalization = "none"
    ns.image_size = int(eval_args.image_size)
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


def patch_transformers_for_legacy_torch() -> None:
    try:
        import transformers
    except Exception:
        return

    pre_trained_model = getattr(transformers, "PreTrainedModel", None)
    module_name = getattr(pre_trained_model, "__module__", "")
    if pre_trained_model is not None and "dummy_pt_objects" not in module_name:
        return

    class _PretrainedConfig:
        model_type = ""

        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _PreTrainedModel(torch.nn.Module):
        config_class = _PretrainedConfig
        base_model_prefix = ""

        def __init__(self, config=None):
            super().__init__()
            self.config = config


        def from_pretrained(cls, *args, **kwargs):
            raise NotImplementedError("HF from_pretrained is not available in this eval shim")

    transformers.PretrainedConfig = _PretrainedConfig
    transformers.PreTrainedModel = _PreTrainedModel


def setup_ttt3r_imports(ttt3r_repo: str, checkpoint: str) -> None:
    repo = Path(ttt3r_repo).resolve()
    src = repo / "src"
    for path in (str(Path(checkpoint).resolve().parent), str(src), str(repo)):
        if path not in sys.path:
            sys.path.insert(0, path)
    patch_transformers_for_legacy_torch()


def load_ttt3r_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    setup_ttt3r_imports(args.ttt3r_repo, args.ttt3r_checkpoint)
    try:
        from src.dust3r.model import ARCroco3DStereo, load_model
        if hasattr(ARCroco3DStereo, "_backends"):
            ARCroco3DStereo._backends = []
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Failed to import TTT3R dependency {exc.name!r}. "
            f"Install {Path(args.ttt3r_repo) / 'requirements.txt'} in the eval environment, "
            "or run eval_ttt3r.sh with CONDA_ENV/PYTHON_BIN pointing at an environment that has TTT3R dependencies."
        ) from exc

    ckpt = Path(args.ttt3r_checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"TTT3R checkpoint not found: {ckpt}")
    print(f"[eval_ttt3r] loading {ckpt}")
    model = load_model(str(ckpt), device="cpu").to(device)
    model.config.model_update_type = str(args.model_update_type)
    model.eval()
    return model


def make_ttt3r_views(images: torch.Tensor, reset_interval: int, target_size: int) -> List[Dict[str, object]]:
    """Convert [1, S, 3, H, W] LingBot images in [0,1] to TTT3R view dicts."""
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError(f"Expected images [1,S,3,H,W], got {tuple(images.shape)}")
    images_cpu = images.detach().cpu().float().clamp(0.0, 1.0)
    _, seq_len, _, source_height, source_width = images_cpu.shape
    target_size = max(16, int(target_size))

    # Match TTT3R demo.py -> dust3r.utils.image.load_images(size=target_size):
    # resize the long edge while preserving aspect ratio, then center-crop to
    # a shape compatible with the 16px patch grid.
    source_long_edge = max(int(source_width), int(source_height))
    if target_size == 224:
        resize_long_edge = round(target_size * max(source_width / source_height, source_height / source_width))
    else:
        resize_long_edge = target_size
    target_width = int(round(source_width * resize_long_edge / source_long_edge))
    target_height = int(round(source_height * resize_long_edge / source_long_edge))
    images_cpu = torch.nn.functional.interpolate(
        images_cpu[0], size=(target_height, target_width), mode="bilinear", align_corners=False
    ).unsqueeze(0)

    width, height = target_width, target_height
    cx, cy = width // 2, height // 2
    if target_size == 224:
        half_w = half_h = min(cx, cy)
    else:
        half_w = ((2 * cx) // 16) * 8
        half_h = ((2 * cy) // 16) * 8
        if width == height:
            half_h = int(3 * half_w / 4)
    left, right = cx - half_w, cx + half_w
    top, bottom = cy - half_h, cy + half_h
    images_cpu = images_cpu[..., top:bottom, left:right]
    height, width = int(bottom - top), int(right - left)

    views: List[Dict[str, object]] = []
    reset_every = max(1, int(reset_interval))
    for index in range(seq_len):
        img = images_cpu[0, index:index + 1] * 2.0 - 1.0
        views.append({
            "img": img,
            "ray_map": torch.full((1, 6, height, width), torch.nan, dtype=img.dtype),
            "true_shape": torch.tensor([[height, width]], dtype=torch.int32),
            "idx": index,
            "instance": str(index),
            "camera_pose": torch.eye(4, dtype=torch.float32).unsqueeze(0),
            "img_mask": torch.tensor([True]),
            "ray_mask": torch.tensor([False]),
            "update": torch.tensor([True]),
            "reset": torch.tensor([(index + 1) % reset_every == 0]),
        })
    return views


def crop_predictions_to_hw(predictions: Dict[str, torch.Tensor], image_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    height, width = image_hw
    for key in ("depth", "depth_conf", "world_points"):
        value = predictions.get(key)
        if value is None or value.ndim < 5:
            continue
        src_h, src_w, channels = int(value.shape[-3]), int(value.shape[-2]), int(value.shape[-1])
        if src_h == height and src_w == width:
            continue
        leading = value.shape[:-3]
        flat = value.reshape(-1, src_h, src_w, channels).permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(flat, size=(height, width), mode="bilinear", align_corners=False)
        predictions[key] = resized.permute(0, 2, 3, 1).reshape(*leading, height, width, channels)
    return predictions


def c2w_intrinsics_to_pose_enc(c2w: torch.Tensor, intrinsics: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
    height, width = image_hw
    quat = mat_to_quat(c2w[..., :3, :3])
    fov_h = 2 * torch.atan((height / 2.0) / intrinsics[..., 1, 1].clamp(min=1e-6))
    fov_w = 2 * torch.atan((width / 2.0) / intrinsics[..., 0, 0].clamp(min=1e-6))
    return torch.cat([c2w[..., :3, 3], quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()


def ttt3r_outputs_to_predictions(outputs: Dict[str, object], args: argparse.Namespace, device: torch.device) -> Dict[str, torch.Tensor]:
    setup_ttt3r_imports(args.ttt3r_repo, args.ttt3r_checkpoint)
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import matrix_cumprod

    preds = list(cast(List[Dict[str, torch.Tensor]], outputs["pred"]))
    views = list(cast(List[Dict[str, object]], outputs["views"]))
    reset_mask = torch.cat([cast(torch.Tensor, view["reset"]).detach().cpu().bool() for view in views], dim=0)
    shifted_reset = torch.cat([torch.tensor([False]), reset_mask[:-1]], dim=0)
    preds = [pred for pred, mask in zip(preds, shifted_reset) if not bool(mask)]
    views = [view for view, mask in zip(views, shifted_reset) if not bool(mask)]
    reset_mask = reset_mask[~shifted_reset]

    pts_self = torch.cat([pred["pts3d_in_self_view"].detach().cpu().float() for pred in preds], dim=0)
    conf_self = torch.cat([pred["conf_self"].detach().cpu().float() for pred in preds], dim=0)
    c2w_list = [pose_encoding_to_camera(pred["camera_pose"].detach().cpu().clone()).float() for pred in preds]

    if bool(reset_mask.any().item()):
        c2w_cat = torch.cat(c2w_list, dim=0)
        identity = torch.eye(4, dtype=c2w_cat.dtype, device=c2w_cat.device)
        reset_poses = torch.where(reset_mask[:, None, None], c2w_cat, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        c2w = torch.einsum("sij,sjk->sik", shifted_bases, c2w_cat)
    else:
        c2w = torch.cat(c2w_list, dim=0)

    seq_len, height, width, _ = pts_self.shape
    pp = torch.tensor([width // 2, height // 2], dtype=pts_self.dtype, device=pts_self.device).repeat(seq_len, 1)
    focal = estimate_focal_knowing_depth(pts_self, pp, focal_mode=str(args.focal_mode)).detach().cpu().float()

    intrinsics = torch.eye(3, dtype=torch.float32).repeat(seq_len, 1, 1)
    intrinsics[:, 0, 0] = focal
    intrinsics[:, 1, 1] = focal
    intrinsics[:, 0, 2] = pp[:, 0].cpu()
    intrinsics[:, 1, 2] = pp[:, 1].cpu()

    ones = torch.ones_like(pts_self[..., :1])
    pts_h = torch.cat([pts_self, ones], dim=-1)
    world_points = torch.einsum("sij,shwj->shwi", c2w, pts_h)[..., :3]
    depth = pts_self[..., 2][None, ..., None]
    pose_enc = c2w_intrinsics_to_pose_enc(c2w[None], intrinsics[None], (height, width))

    return {
        "depth": depth.to(device),
        "depth_conf": conf_self[None].to(device),
        "camera_c2w": c2w[None].to(device),
        "intrinsics": intrinsics[None].to(device),
        "pose_enc": pose_enc.to(device),
        "world_points": world_points[None].to(device),
    }


def evaluate_one_mode(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    eval_mode: str,
    output_dir: Path,
) -> Dict[str, object]:
    setup_ttt3r_imports(args.ttt3r_repo, args.ttt3r_checkpoint)
    from src.dust3r.inference import inference_recurrent_lighter

    loader, eval_scenes = E.build_eval_loader(args, eval_mode=eval_mode)
    print(f"[eval_ttt3r] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
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

        images_for_ttt3r = cast(torch.Tensor, batch["images"]).clone()
        batch = T.to_device(dict(batch), device)
        images_t = cast(torch.Tensor, batch["images"])
        depths_t = cast(torch.Tensor, batch["depths"])
        point_masks_t = cast(torch.Tensor, batch["point_masks"])
        extrinsics_t = cast(torch.Tensor, batch["extrinsics"])
        intrinsics_t = cast(torch.Tensor, batch["intrinsics"])
        world_points_t = cast(torch.Tensor, batch["world_points"])
        image_hw = (int(images_t.shape[-2]), int(images_t.shape[-1]))

        try:
            views = make_ttt3r_views(images_for_ttt3r, reset_interval=int(args.reset_interval), target_size=int(args.image_size))
            outputs, _ = inference_recurrent_lighter(views, model, str(device), verbose=False)
            predictions = ttt3r_outputs_to_predictions(outputs, args, device)
            predictions = crop_predictions_to_hw(predictions, image_hw)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[eval_ttt3r] batch {batch_idx} TTT3R failed for {scene_name} "
                  f"({str(exc).splitlines()[0][:160]}); skipping")
            continue

        per_frame_valid = point_masks_t.sum(dim=(-1, -2)) > int(getattr(args, "min_valid_pixels", 100))
        row: Dict[str, object] = {
            "scene": scene_name,
            "sample_mode": sample_mode,
            "metric_group": metric_group,
            "n_frames": int(images_t.shape[1]),
        }

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

        pred_pose = cast(torch.Tensor, predictions["pose_enc"])[0].float()
        traj_metrics = E.compute_camera_metrics_evo(
            pred_pose,
            extrinsics_t[0],
            image_hw=image_hw,
            valid_frame_mask=per_frame_valid[0],
            align_mode=str(getattr(args, "camera_align", "sim3")),
        )
        if traj_metrics is not None:
            per_scene_traj.append(traj_metrics)
            group_traj[metric_group].append(traj_metrics)
            row["cam_ate_rmse"] = traj_metrics["ate_rmse"]
            row["cam_rpe_trans_rmse"] = traj_metrics["rpe_trans_rmse"]
            row["cam_rpe_rot_rmse_deg"] = traj_metrics["rpe_rot_rmse_deg"]
            row["cam_n_frames_used"] = traj_metrics["n_frames_used"]

        fov_rows = E.compute_fov_per_frame_errors(
            pred_pose,
            intrinsics_t[0].float(),
            image_hw=image_hw,
            valid_frame_mask=per_frame_valid[0],
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
                print(f"[eval_ttt3r] pointcloud failed for {scene_name} "
                      f"({str(exc).splitlines()[0][:120]}); skipping")
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
                depth_conf=cast(torch.Tensor, predictions["depth_conf"])[0].detach().cpu().numpy().astype(np.float32),
                pose_enc=pred_pose.detach().cpu().numpy().astype(np.float32),
                camera_c2w=cast(torch.Tensor, predictions["camera_c2w"])[0].detach().cpu().numpy().astype(np.float32),
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
            cam_msg = ""
            if per_scene_traj:
                cam_msg = f" ATE={float(np.mean([m['ate_rmse'] for m in per_scene_traj])):.4f}"
            pc_msg = ""
            if all_pointcloud_rows:
                pc_msg = f" PC_CD={float(np.mean([m['CD'] for m in all_pointcloud_rows])):.4f}"
            print(f"[eval_ttt3r] [{batch_idx}/{len(loader)}] evaluated={evaluated} skipped={skipped} "
                  f"AbsRel({args.depth_align})={running.get('AbsRel', float('nan')):.4f}{cam_msg}{pc_msg}")

    if evaluated == 0:
        raise RuntimeError(f"No batches evaluated for eval_mode={eval_mode}.")

    summary: Dict[str, object] = {
        "model": "TTT3R",
        "ttt3r_repo": str(args.ttt3r_repo),
        "ttt3r_checkpoint": str(args.ttt3r_checkpoint),
        "model_update_type": str(args.model_update_type),
        "train_args_json": str(args.train_args_json),
        "split": args.split,
        "eval_mode": eval_mode,
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "geometry_normalization": "none",
        "camera_align": str(getattr(args, "camera_align", "sim3")),
        "depth_align": str(args.depth_align),
        "secondary_depth_align": secondary_align,
        "pointcloud_source": "TTT3R pts3d_in_self_view transformed by TTT3R c2w",
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
    print(f"[eval_ttt3r] wrote {metrics_path}")

    if args.per_scene_csv and per_scene_rows:
        csv_path = output_dir / "per_scene.csv"
        keys = sorted({key for row in per_scene_rows for key in row.keys()})
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(per_scene_rows)
        print(f"[eval_ttt3r] wrote {csv_path}")
    return summary


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_ttt3r] output_dir={base_output_dir}")
    print("[eval_ttt3r] geometry_normalization=none; alignment is applied only inside metric helpers")
    model = load_ttt3r_model(args, device)

    overall: Dict[str, object] = {
        "model": "TTT3R",
        "ttt3r_repo": str(args.ttt3r_repo),
        "ttt3r_checkpoint": str(args.ttt3r_checkpoint),
        "model_update_type": str(args.model_update_type),
        "train_args_json": str(args.train_args_json),
        "split": args.split,
        "eval_strategy": args.eval_strategy,
        "eval_num_frames": int(args.eval_num_frames),
        "image_size": int(args.image_size),
        "geometry_normalization": "none",
        "camera_align": str(getattr(args, "camera_align", "sim3")),
        "depth_align": str(args.depth_align),
        "pointcloud_align": str(args.pointcloud_align),
        "modes": {},
    }
    modes = eval_modes(args.eval_strategy)
    for mode_name in modes:
        out_dir = base_output_dir if len(modes) == 1 else (base_output_dir / mode_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[eval_ttt3r] === mode: {mode_name} ===")
        cast(Dict[str, object], overall["modes"])[mode_name] = evaluate_one_mode(args, model, device, mode_name, out_dir)

    summary_path = base_output_dir / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    print(f"[eval_ttt3r] wrote {summary_path}")
    return overall


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = args_from_run_json(eval_args)
    evaluate(args)


if __name__ == "__main__":
    main()
