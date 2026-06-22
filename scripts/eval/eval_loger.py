#!/usr/bin/env python3
"""Evaluate LoGeR on the LingBot-MAP Manip eval protocol.

This reuses eval_pi3.py's Manip dataloader and metrics, but swaps the model
loader to LoGeR. Geometry is kept in the native LoGeR / GT frames by default;
alignment is applied only inside metric helpers via --depth_align,
--camera_align, and --pointcloud_align.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parents[1]
for _path in (_REPO_DIR, _REPO_DIR / "scripts" / "train", _REPO_DIR / "scripts" / "eval", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import inspect
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms

import eval_pi3 as PI3E
import eval_model_geometry as EMG


SCRIPT_DIR = Path(__file__).resolve().parent


def _maybe_parse_sequence(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = yaml.safe_load(stripped)
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
            except Exception:
                pass
    return value


def _load_loger_config(config_path: str) -> Dict[str, Any]:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"LoGeR config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _model_kwargs_from_config(pi3_cls: type[nn.Module], config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        model_config = {}

    signature = inspect.signature(pi3_cls.__init__)
    valid_kwargs = {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }

    kwargs: Dict[str, Any] = {}
    for key, value in model_config.items():
        if key not in valid_kwargs:
            continue
        if key in {"ttt_insert_after", "attn_insert_after"}:
            value = _maybe_parse_sequence(value)
        kwargs[key] = value

    if bool(getattr(args, "loger_pi3x", False)):
        kwargs["pi3x"] = True
        kwargs["pi3x_metric"] = bool(getattr(args, "loger_pi3x_metric", True))
    return kwargs


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, (list, tuple)):
        values = [_int_or_default(v, default) for v in value]
        return max(values) if values else int(default)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "" or stripped.lower() in {"none", "null", "default"}:
            return int(default)
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = yaml.safe_load(stripped)
                return _int_or_default(parsed, default)
            except Exception:
                return int(default)
    try:
        return int(round(float(value)))
    except Exception:
        return int(default)


def _flatten_collated_rgb_paths(rgb_paths_batch: object) -> List[List[str]]:
    if not isinstance(rgb_paths_batch, (list, tuple)) or not rgb_paths_batch:
        raise ValueError("LoGeR native input mode requires non-empty rgb_paths in the eval batch")
    if all(isinstance(path, str) for path in rgb_paths_batch):
        return [[str(path) for path in cast(Sequence[str], rgb_paths_batch)]]
    if all(isinstance(item, tuple) for item in rgb_paths_batch):
        frame_major = cast(Sequence[Tuple[object, ...]], rgb_paths_batch)
        batch_size = len(frame_major[0])
        if batch_size <= 0 or any(len(frame_paths) != batch_size for frame_paths in frame_major):
            raise ValueError("Inconsistent collated rgb_paths batch shape")
        return [[str(frame_paths[batch_idx]) for frame_paths in frame_major] for batch_idx in range(batch_size)]
    if all(isinstance(item, (list, tuple)) for item in rgb_paths_batch):
        return [[str(path) for path in cast(Sequence[object], sample_paths)] for sample_paths in cast(Sequence[object], rgb_paths_batch)]
    raise ValueError(f"Unsupported rgb_paths batch shape: {type(rgb_paths_batch)!r}")


def _load_loger_native_images_from_paths(
    image_paths: Sequence[str],
    *,
    target_hw: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not image_paths:
        raise ValueError("LoGeR native image loader received no image paths")
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid LoGeR native target size: H={target_h}, W={target_w}")
    to_tensor = transforms.ToTensor()
    frames = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            resized = img.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
        frames.append(to_tensor(resized).to(dtype=dtype))
    return torch.stack(frames, dim=0).unsqueeze(0).to(device, non_blocking=True)


def load_loger_native_images_from_batch(
    args: argparse.Namespace,
    batch: Dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    rgb_path_groups = _flatten_collated_rgb_paths(batch.get("rgb_paths"))
    reference = batch.get("images")
    if not torch.is_tensor(reference):
        raise ValueError("LoGeR native input mode needs batch['images'] to infer target H/W")
    target_hw = (int(reference.shape[-2]), int(reference.shape[-1]))
    dtype = reference.dtype if reference.is_floating_point() else torch.float32
    videos = [
        _load_loger_native_images_from_paths(paths, target_hw=target_hw, device=device, dtype=dtype).squeeze(0)
        for paths in rgb_path_groups
    ]
    if not videos:
        raise RuntimeError("LoGeR native image loader returned an empty batch")
    return torch.stack(videos, dim=0)




def compute_loger_preprocess_geometry(
    width: int,
    height: int,
    target_width: int = 0,
    target_height: int = 0,
    pixel_limit: int = 255000,
) -> Dict[str, int]:
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid source image size: {width}x{height}")
    if int(target_width) > 0 or int(target_height) > 0:
        new_width = int(target_width) if int(target_width) > 0 else int(target_height)
        new_height = int(target_height) if int(target_height) > 0 else int(target_width)
    else:
        scale = math.sqrt(float(pixel_limit) / float(width * height)) if width * height > 0 else 1.0
        width_target = width * scale
        height_target = height * scale
        k = round(width_target / 14)
        m = round(height_target / 14)
        while max(1, k) * 14 * max(1, m) * 14 > int(pixel_limit):
            if k / max(1, m) > width_target / max(1.0, height_target):
                k -= 1
            else:
                m -= 1
        new_width = max(1, k) * 14
        new_height = max(1, m) * 14
    return {
        "new_width": max(14, int(new_width)),
        "new_height": max(14, int(new_height)),
        "crop_left": 0,
        "crop_top": 0,
        "crop_width": max(14, int(new_width)),
        "crop_height": max(14, int(new_height)),
        "pad_left": 0,
        "pad_right": 0,
        "pad_top": 0,
        "pad_bottom": 0,
    }


class LoGeRNativeEvalDataset(EMG.ModelGeometryEvalDataset):  # type: ignore[misc]
    """Eval sampler whose GT tensors are synchronized to LoGeR native geometry."""

    rgb_resample = Image.Resampling.LANCZOS
    native_width = 504
    native_height = 280

    def model_preprocess_geometry(self, width: int, height: int) -> Dict[str, int]:
        target_w = int(getattr(self, "loger_native_width", 0) or self.native_width)
        target_h = int(getattr(self, "loger_native_height", 0) or self.native_height)
        if target_w <= 0 and target_h <= 0:
            target = int(getattr(self, "image_size", 0))
            target_w = target_h = target
        return compute_loger_preprocess_geometry(width, height, target_w, target_h)


def _forward_kwargs_from_config(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        model_config = {}
    training = config.get("training_settings", {})
    if not isinstance(training, dict):
        training = {}

    se3_config = bool(model_config.get("se3", config.get("se3", False)))
    se3_value = se3_config if getattr(args, "loger_se3", None) is None else bool(args.loger_se3)

    window_size = (
        args.loger_window_size
        if args.loger_window_size is not None
        else training.get("window_size", -1)
    )
    overlap_size = (
        args.loger_overlap_size
        if args.loger_overlap_size is not None
        else training.get("overlap_size", 0)
    )
    reset_every = (
        args.loger_reset_every
        if args.loger_reset_every is not None
        else training.get("reset_every", 0)
    )
    num_iterations = (
        args.loger_num_iterations
        if args.loger_num_iterations is not None
        else config.get("num_iterations", 1)
    )

    return {
        "window_size": _int_or_default(window_size, -1),
        "overlap_size": _int_or_default(overlap_size, 0),
        "reset_every": _int_or_default(reset_every, 0),
        "num_iterations": _int_or_default(num_iterations, 1),
        "sim3": bool(config.get("sim3", False)) or bool(getattr(args, "loger_sim3", False)),
        "sim3_scale_mode": str(getattr(args, "loger_sim3_scale_mode", "median")),
        "se3": se3_value,
        "turn_off_ttt": bool(getattr(args, "loger_no_ttt", False)),
        "turn_off_swa": bool(getattr(args, "loger_no_swa", False)),
    }


class LoGeREvalAdapter(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        forward_kwargs: Dict[str, Any],
        recover_focal: bool = True,
        focal_mask_threshold: float = 0.1,
        focal_downsample_size: Tuple[int, int] = (64, 64),
    ) -> None:
        super().__init__()
        self.model = model
        self.forward_kwargs = dict(forward_kwargs)
        self.recover_focal = recover_focal
        self.focal_mask_threshold = focal_mask_threshold
        self.focal_downsample_size = focal_downsample_size

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError(f"LoGeR eval expects images [B,S,3,H,W], got {tuple(images.shape)}")
        h, w = int(images.shape[-2]), int(images.shape[-1])
        if h % 14 != 0 or w % 14 != 0:
            raise ValueError(
                f"LoGeR requires image H/W divisible by 14, got H={h}, W={w}. "
                "Set IMAGE_SIZE to a multiple of 14 in eval_loger.sh."
            )

        pred = self.model(images, **self.forward_kwargs, no_detach=True)
        local_points = pred["local_points"].float()
        conf_logits = pred.get("conf")
        if conf_logits is None:
            depth_conf = torch.ones_like(local_points[..., 2])
        else:
            depth_conf = torch.sigmoid(conf_logits[..., 0].float())

        out: Dict[str, torch.Tensor] = {
            "depth": local_points[..., 2:3],
            "depth_conf": depth_conf,
            "world_points": pred["points"].float(),
            "camera_c2w": pred["camera_poses"].float(),
        }
        if self.recover_focal:
            focal_mask = (depth_conf > self.focal_mask_threshold) & torch.isfinite(local_points).all(dim=-1)
            focal, shift = PI3E.recover_focal_shift(
                local_points,
                focal_mask,
                downsample_size=self.focal_downsample_size,
            )
            out["intrinsics"] = PI3E.intrinsics_from_recovered_focal(focal, h, w)
            out["focal_shift"] = shift
        return out


def build_loger_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    loger_repo = Path(args.loger_repo).resolve()
    if str(loger_repo) not in sys.path:
        sys.path.insert(0, str(loger_repo))
    if bool(getattr(args, "loger_disable_compile", True)):
        torch.compile = (  # type: ignore[assignment]
            lambda fn=None, *compile_args, **compile_kwargs:
            (fn if fn is not None else (lambda real_fn: real_fn))
        )
    from loger.models.pi3 import Pi3

    config = _load_loger_config(str(args.loger_config))
    model_kwargs = _model_kwargs_from_config(Pi3, config, args)
    forward_kwargs = _forward_kwargs_from_config(config, args)
    model = Pi3(**model_kwargs)

    ckpt = Path(args.loger_checkpoint)
    if not ckpt.is_file():
        raise FileNotFoundError(f"LoGeR checkpoint not found: {ckpt}")
    weight = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(weight, dict) and "model_state_dict" in weight:
        state = weight["model_state_dict"]
    elif isinstance(weight, dict) and "model" in weight:
        state = weight["model"]
    elif isinstance(weight, dict) and "state_dict" in weight:
        state = weight["state_dict"]
    else:
        state = weight
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    info = model.load_state_dict(state, strict=bool(args.loger_strict_load))
    print(
        f"[eval_loger] loaded {ckpt} "
        f"(missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}, "
        f"strict={bool(args.loger_strict_load)})"
    )
    print(f"[eval_loger] model kwargs: {model_kwargs}")
    print(f"[eval_loger] forward kwargs: {forward_kwargs}")

    return LoGeREvalAdapter(
        model.to(device).eval(),
        forward_kwargs=forward_kwargs,
        recover_focal=bool(getattr(args, "recover_focal", True)),
        focal_mask_threshold=float(getattr(args, "focal_mask_threshold", 0.1)),
        focal_downsample_size=tuple(getattr(args, "focal_downsample_size", (64, 64))),
    ).to(device).eval()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate LoGeR with LingBot-MAP Manip metrics")
    p.add_argument("--train_args_json", type=str, required=True,
                   help="LingBot-MAP args.json used only to recreate the Manip eval dataset config.")
    p.add_argument("--loger_repo", type=str, default="/cpfs/user/guowenqi/LoGeR")
    p.add_argument("--loger_checkpoint", type=str, default="/cpfs/user/guowenqi/LoGeR/ckpts/LoGeR/latest.pt")
    p.add_argument("--loger_config", type=str, default="/cpfs/user/guowenqi/LoGeR/ckpts/LoGeR/original_config.yaml")
    p.add_argument("--loger_strict_load", action="store_true")
    p.add_argument("--loger_window_size", type=int, default=None)
    p.add_argument("--loger_overlap_size", type=int, default=None)
    p.add_argument("--loger_reset_every", type=int, default=None)
    p.add_argument("--loger_num_iterations", type=int, default=None)
    p.add_argument("--loger_sim3", action="store_true")
    p.add_argument("--loger_se3", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--loger_sim3_scale_mode", type=str, default="median",
                   choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all", "sim3_avg1"])
    p.add_argument("--loger_no_ttt", action="store_true")
    p.add_argument("--loger_no_swa", action="store_true")
    p.add_argument("--loger_pi3x", action="store_true")
    p.add_argument("--loger_pi3x_metric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loger_disable_compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--loger_native_width", type=int, default=504,
                   help="LoGeR native input width; set <=0 with height <=0 to use LoGeR pixel-limit resizing.")
    p.add_argument("--loger_native_height", type=int, default=280,
                   help="LoGeR native input height; set <=0 with width <=0 to use LoGeR pixel-limit resizing.")
    p.add_argument("--loger_input_preprocess", choices=["native", "lingbot"], default="native",
                   help="native reloads raw rgb_paths with LoGeR-style RGB/LANCZOS/ToTensor; lingbot reuses LingBot-MAP tensors.")

    p.add_argument("--output_dir", type=str, default=str(REPO_DIR / "outputs" / "eval" / "loger"))
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max_scenes_eval", type=int, default=0)
    p.add_argument("--eval_shard_count", type=int, default=1)
    p.add_argument("--eval_shard_index", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per_scene_csv", action="store_true")
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--print_every", type=int, default=5)
    p.add_argument("--eval_strategy", choices=["train_default", "manip_track", "wrist_track", "random_static_track", "both", "left_moving_tracks"],
                   default="left_moving_tracks")
    p.add_argument("--eval_num_frames", type=int, default=64)
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_moving")
    p.add_argument("--eval_seed", type=int, default=42)
    p.add_argument("--depth_align", choices=["none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="pi3_scale_shift")
    p.add_argument("--camera_align", choices=["none", "sim3"], default="sim3")
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--geometry_normalization", choices=["native", "vggt_independent", "none"],
                   default="none")
    p.add_argument("--pointcloud_metrics", dest="pointcloud_metrics", action="store_true", default=True)
    p.add_argument("--no_pointcloud_metrics", dest="pointcloud_metrics", action="store_false")
    p.add_argument("--pointcloud_max_points", type=int, default=100000)
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"],
                   default="pi3_icp")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1)
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30)
    p.add_argument("--pointcloud_icp_backend", choices=["auto", "open3d", "scipy"], default="open3d")
    p.add_argument("--pointcloud_kdtree_workers", type=int, default=1)
    p.add_argument("--pointcloud_workers", type=int, default=1)
    p.add_argument("--recover_focal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--focal_mask_threshold", type=float, default=0.1)
    p.add_argument("--focal_downsample_size", type=int, nargs=2, metavar=("H", "W"), default=(64, 64))
    return p


def coerce_loger_args_from_json(eval_args: argparse.Namespace) -> argparse.Namespace:
    args_json_path = Path(eval_args.train_args_json)
    if not args_json_path.is_file():
        raise FileNotFoundError(f"args.json not found: {args_json_path}")
    with args_json_path.open("r", encoding="utf-8") as f:
        train_args_dict = json.load(f)

    ns = argparse.Namespace(**train_args_dict)
    ns.checkpoint = str(eval_args.loger_checkpoint)
    ns.train_args_json = str(args_json_path)
    ns.output_dir = eval_args.output_dir
    ns.split = eval_args.split
    ns.max_scenes_eval = eval_args.max_scenes_eval
    ns.eval_shard_count = int(eval_args.eval_shard_count)
    ns.eval_shard_index = int(eval_args.eval_shard_index)
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
    ns.pointcloud_icp_backend = str(eval_args.pointcloud_icp_backend)
    ns.pointcloud_kdtree_workers = int(eval_args.pointcloud_kdtree_workers)
    ns.pointcloud_workers = int(eval_args.pointcloud_workers)
    ns.recover_focal = bool(eval_args.recover_focal)
    ns.focal_mask_threshold = float(eval_args.focal_mask_threshold)
    ns.focal_downsample_size = tuple(int(x) for x in eval_args.focal_downsample_size)

    ns.loger_repo = str(eval_args.loger_repo)
    ns.loger_checkpoint = str(eval_args.loger_checkpoint)
    ns.loger_config = str(eval_args.loger_config)
    ns.loger_strict_load = bool(eval_args.loger_strict_load)
    ns.loger_window_size = eval_args.loger_window_size
    ns.loger_overlap_size = eval_args.loger_overlap_size
    ns.loger_reset_every = eval_args.loger_reset_every
    ns.loger_num_iterations = eval_args.loger_num_iterations
    ns.loger_sim3 = bool(eval_args.loger_sim3)
    ns.loger_se3 = eval_args.loger_se3
    ns.loger_sim3_scale_mode = str(eval_args.loger_sim3_scale_mode)
    ns.loger_no_ttt = bool(eval_args.loger_no_ttt)
    ns.loger_no_swa = bool(eval_args.loger_no_swa)
    ns.loger_pi3x = bool(eval_args.loger_pi3x)
    ns.loger_pi3x_metric = bool(eval_args.loger_pi3x_metric)
    ns.loger_disable_compile = bool(eval_args.loger_disable_compile)
    ns.loger_input_preprocess = str(eval_args.loger_input_preprocess)
    ns.loger_native_width = int(eval_args.loger_native_width)
    ns.loger_native_height = int(eval_args.loger_native_height)

    if int(eval_args.image_size) > 0:
        ns.image_size = int(eval_args.image_size)

    # eval_pi3.evaluate records args.pi3_repo/pi3_input_mode in the JSON; keep
    # those reused fields informative for LoGeR.
    ns.pi3_repo = ns.loger_repo
    ns.pi3_input_mode = "native" if ns.loger_input_preprocess == "native" else "lingbot"
    if ns.pi3_input_mode == "native" and ns.loger_native_width > 0:
        ns.pi3_native_width = ns.loger_native_width
    else:
        ns.pi3_native_width = int(ns.__dict__.get("image_size", 518))
    ns.model_path = ""
    ns.cpu = (eval_args.device == "cpu")
    ns.batch_size = 1
    ns.write_manifest = ns.__dict__.get("write_manifest", None) or None
    return ns


def _patch_metrics_json(args: argparse.Namespace) -> None:
    metrics_path = Path(args.output_dir) / "metrics.json"
    if not metrics_path.is_file():
        return
    with metrics_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    data["model"] = "LoGeR"
    data["checkpoint"] = str(args.loger_checkpoint)
    data["loger_repo"] = str(args.loger_repo)
    data["loger_config"] = str(args.loger_config)
    data["loger_window_size"] = args.loger_window_size
    data["loger_overlap_size"] = args.loger_overlap_size
    data["loger_reset_every"] = args.loger_reset_every
    data["loger_input_preprocess"] = str(getattr(args, "loger_input_preprocess", "native"))
    data["loger_native_width"] = int(getattr(args, "loger_native_width", 504))
    data["loger_native_height"] = int(getattr(args, "loger_native_height", 280))
    data["input_geometry_sync"] = "model_native_geometry+lingbot_map_sync" if str(getattr(args, "loger_input_preprocess", "native")) == "native" else "lingbot_map"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = coerce_loger_args_from_json(eval_args)
    random.seed(int(args.eval_seed))
    np.random.seed(int(args.eval_seed))
    torch.manual_seed(int(args.eval_seed))

    PI3E.build_pi3_model = build_loger_model
    if str(getattr(args, "loger_input_preprocess", "native")) == "native":
        LoGeRNativeEvalDataset.native_width = int(getattr(args, "loger_native_width", 504))
        LoGeRNativeEvalDataset.native_height = int(getattr(args, "loger_native_height", 280))
        PI3E.Pi3NativeEvalDataset = LoGeRNativeEvalDataset
        PI3E.load_pi3_native_images_from_paths = load_loger_native_images_from_batch
    PI3E.evaluate(args)
    _patch_metrics_json(args)


if __name__ == "__main__":
    main()
