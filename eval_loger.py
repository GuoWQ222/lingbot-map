#!/usr/bin/env python3
"""Evaluate LoGeR on the LingBot-MAP Manip eval protocol.

This reuses eval_pi3.py's Manip dataloader and metrics, but swaps the model
loader to LoGeR. Geometry is kept in the native LoGeR / GT frames by default;
alignment is applied only inside metric helpers via --depth_align,
--camera_align, and --pointcloud_align.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

import eval_pi3 as PI3E


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

    p.add_argument("--output_dir", type=str, default=str(SCRIPT_DIR / "eval" / "loger"))
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--max_scenes_eval", type=int, default=0)
    p.add_argument("--eval_shard_count", type=int, default=1)
    p.add_argument("--eval_shard_index", type=int, default=0)
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
                   default="pi3_scale_shift")
    p.add_argument("--camera_align", choices=["none", "sim3"], default="sim3")
    p.add_argument("--image_size", type=int, default=0)
    p.add_argument("--geometry_normalization", choices=["native", "vggt_independent", "none"],
                   default="none")
    p.add_argument("--pointcloud_metrics", action="store_true")
    p.add_argument("--pointcloud_max_points", type=int, default=100000)
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"],
                   default="pi3_icp")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1)
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30)
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

    # eval_pi3.evaluate records args.pi3_repo in the JSON. Point it at LoGeR so
    # the reused summary stays informative.
    ns.pi3_repo = ns.loger_repo
    if int(eval_args.image_size) > 0:
        ns.image_size = int(eval_args.image_size)
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
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = coerce_loger_args_from_json(eval_args)
    random.seed(int(args.eval_seed))
    np.random.seed(int(args.eval_seed))
    torch.manual_seed(int(args.eval_seed))

    PI3E.build_pi3_model = build_loger_model
    PI3E.evaluate(args)
    _patch_metrics_json(args)


if __name__ == "__main__":
    main()
