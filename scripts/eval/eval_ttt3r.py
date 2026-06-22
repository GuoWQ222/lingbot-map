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
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = REPO_DIR / "outputs" / "runs" / "manip_long_train_64gpu"
DEFAULT_TTT3R_REPO = REPO_DIR.parent / "TTT3R"
DEFAULT_TTT3R_CHECKPOINT = DEFAULT_TTT3R_REPO / "src" / "cut3r_512_dpt_4_64.pth"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import eval as E
import train as T
import eval_model_geometry as EMG
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

    p.add_argument("--eval_strategy", choices=["manip_track", "wrist_track", "random_static_track", "both", "left_moving_tracks"],
                   default="left_moving_tracks")
    p.add_argument("--eval_num_frames", type=int, default=64)
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_moving")
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
    p.add_argument("--pointcloud_icp_backend", choices=["auto", "open3d", "scipy"], default="open3d")
    p.add_argument("--pointcloud_kdtree_workers", type=int, default=1)
    p.add_argument("--pointcloud_workers", type=int, default=1)
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


def compute_ttt3r_preprocess_geometry(width: int, height: int, target_size: int) -> Dict[str, int]:
    target_size = max(16, int(target_size))
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source image size: {width}x{height}")
    source_long_edge = max(width, height)
    if target_size == 224:
        resize_long_edge = round(target_size * max(width / height, height / width))
    else:
        resize_long_edge = target_size
    target_width = int(round(width * resize_long_edge / source_long_edge))
    target_height = int(round(height * resize_long_edge / source_long_edge))

    cx, cy = target_width // 2, target_height // 2
    if target_size == 224:
        half_w = half_h = min(cx, cy)
    else:
        half_w = ((2 * cx) // 16) * 8
        half_h = ((2 * cy) // 16) * 8
        if target_width == target_height:
            half_h = int(3 * half_w / 4)
    left, right = cx - half_w, cx + half_w
    top, bottom = cy - half_h, cy + half_h
    return {
        "new_width": target_width,
        "new_height": target_height,
        "crop_left": left,
        "crop_top": top,
        "crop_width": right - left,
        "crop_height": bottom - top,
        "pad_left": 0,
        "pad_right": 0,
        "pad_top": 0,
        "pad_bottom": 0,
    }


class TTT3RNativeEvalDataset(E.EvalLinspaceDataset):  # type: ignore[misc]
    """Eval sampler with raw RGB-D loaded directly onto TTT3R's image grid."""

    def _load_one(
        self,
        scene_dir: Path,
        entry: T.FrameEntry,
        jitter_params: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = Image.open(entry.rgb_path)
        if rgb.mode == "RGBA":
            background = Image.new("RGBA", rgb.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(background, rgb)
        rgb = rgb.convert("RGB")

        width, height = rgb.size
        geometry = compute_ttt3r_preprocess_geometry(width, height, int(self.image_size))
        intrinsics_raw, extrinsics = self._load_camera_for_entry(scene_dir, entry)
        intrinsics = T.preprocess_intrinsics(intrinsics_raw, width, height, geometry)

        rgb = T.apply_preprocess_to_image(
            rgb,
            geometry,
            resample=Image.Resampling.BICUBIC,
            fill=(255, 255, 255),
        )
        if jitter_params is not None:
            rgb = self._apply_color_jitter(rgb, jitter_params)
        image_tensor = TF.to_tensor(rgb)

        if entry.depth_path.suffix.lower() == ".npy":
            depth_raw = np.load(entry.depth_path).astype(np.float32)
            depth_img = Image.fromarray(depth_raw).convert("F")
            depth_img = T.apply_preprocess_to_image(
                depth_img,
                geometry,
                resample=Image.Resampling.NEAREST,
                fill=0,
            )
            depth_np = np.array(depth_img, dtype=np.float32, copy=True)
        else:
            depth_img = Image.open(entry.depth_path)
            depth_img = T.apply_preprocess_to_image(
                depth_img,
                geometry,
                resample=Image.Resampling.NEAREST,
                fill=0,
            )
            depth_raw = np.array(depth_img, copy=True)
            depth_dtype = depth_raw.dtype
            if depth_raw.ndim == 3:
                depth_raw = depth_raw.astype(np.float32).mean(axis=2)
            depth_scale = float(self.depth_scale)
            if depth_scale <= 0:
                if depth_dtype == np.uint16:
                    depth_scale = 10000.0
                elif np.issubdtype(depth_dtype, np.integer):
                    depth_scale = float(np.iinfo(depth_dtype).max)
                else:
                    depth_scale = 1.0
            depth_np = depth_raw.astype(np.float32) / depth_scale

        if entry.mask_path is not None:
            mask_img = Image.open(entry.mask_path)
            mask_img = T.apply_preprocess_to_image(
                mask_img,
                geometry,
                resample=Image.Resampling.NEAREST,
                fill=0,
            )
            mask_np = np.asarray(mask_img)
            if mask_np.ndim == 3:
                mask_np = mask_np[..., 0]
        else:
            mask_np = np.ones_like(depth_np, dtype=np.uint8)

        valid = np.isfinite(depth_np) & (depth_np > self.min_depth)
        if self.max_depth > 0:
            valid &= depth_np < self.max_depth
        if self.use_mask:
            valid &= mask_np > 0

        depth = torch.from_numpy(depth_np).float()
        point_mask = torch.from_numpy(valid.astype(np.bool_))
        world_points = T.depth_to_world_points(depth, intrinsics.float(), extrinsics.float())
        world_points = torch.where(point_mask[..., None], world_points, torch.zeros_like(world_points))

        return image_tensor, depth, point_mask, intrinsics.float(), extrinsics.float(), world_points.float()




class TTT3RNativeEvalDataset(EMG.ModelGeometryEvalDataset):  # type: ignore[misc]
    """Eval sampler whose GT tensors are synchronized to TTT3R native geometry."""

    rgb_resample = Image.Resampling.BICUBIC

    def model_preprocess_geometry(self, width: int, height: int) -> Dict[str, int]:
        return compute_ttt3r_preprocess_geometry(width, height, int(self.image_size))


def build_ttt3r_eval_loader(args: argparse.Namespace, eval_mode: str) -> Tuple[DataLoader, List[Path]]:
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
        print(f"[eval_ttt3r] shard {shard_index}/{shard_count}: {len(eval_scenes)} of {total_eval_scenes} scenes")
    if not eval_scenes:
        raise RuntimeError(f"split={args.split} produced no scenes")

    if eval_mode not in {"left_moving_tracks", "manip_track", "wrist_track", "random_static_track"}:
        raise ValueError(f"Unknown eval_mode for TTT3R native loader: {eval_mode}")

    dataset = TTT3RNativeEvalDataset(
        eval_scenes,
        eval_mode=eval_mode,
        eval_num_frames=int(args.eval_num_frames),
        eval_wrist_camera_name=str(args.eval_wrist_camera_name),
        eval_surround_camera_name=str(getattr(args, "eval_surround_camera_name", "")),
        eval_seed=int(args.eval_seed),
        clip_len=args.clip_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        sequence_mode=args.sequence_mode,
        view_ids=T.parse_int_list(args.view_ids),
        camera_names=T.parse_str_list(args.camera_names),
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


def crop_predictions_to_hw(predictions: Dict[str, torch.Tensor], image_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
    height, width = image_hw
    for key in ("depth", "depth_conf", "world_points"):
        value = predictions.get(key)
        if value is None or value.ndim < 5:
            continue
        src_h, src_w = int(value.shape[-3]), int(value.shape[-2])
        if src_h == height and src_w == width:
            continue
        raise ValueError(
            f"TTT3R prediction shape for {key} is {src_h}x{src_w}, "
            f"but metric grid is {height}x{width}; refusing to stretch predictions."
        )
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

    loader, eval_scenes = build_ttt3r_eval_loader(args, eval_mode=eval_mode)
    print(f"[eval_ttt3r] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
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
        log_prefix="[eval_ttt3r]",
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
            running = all_depth_pixels.summary()
            cam_msg = ""
            if per_scene_traj:
                cam_msg = f" ATE={float(np.mean([m['ate_rmse'] for m in per_scene_traj])):.4f}"
            pc_msg = ""
            if all_pointcloud_rows:
                pc_msg = f" PC_CD={float(np.mean([m['CD'] for m in all_pointcloud_rows])):.4f}"
            print(f"[eval_ttt3r] [{batch_idx}/{len(loader)}] evaluated={evaluated} skipped={skipped} "
                  f"AbsRel({args.depth_align})={running.get('AbsRel', float('nan')):.4f}{cam_msg}{pc_msg}")

    pointcloud_queue.close()

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
        "input_geometry_sync": "model_native_geometry+lingbot_map_sync",
        "secondary_depth_align": secondary_align,
        "pointcloud_source": "TTT3R pts3d_in_self_view transformed by TTT3R c2w on the TTT3R-native metric grid",
        "pointcloud_align": str(args.pointcloud_align),
        "pointcloud_icp_backend": str(getattr(args, "pointcloud_icp_backend", "open3d")),
        "pointcloud_kdtree_workers": int(getattr(args, "pointcloud_kdtree_workers", 1)),
        "pointcloud_workers": int(getattr(args, "pointcloud_workers", 1)),
        "aggregation": "overall plus realsense_left/surround_cam_moving camera-track groups; depth=Pi3-style pixel-weighted after per-clip alignment; trajectory=per-sequence macro; pointcloud=per-clip macro",
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
