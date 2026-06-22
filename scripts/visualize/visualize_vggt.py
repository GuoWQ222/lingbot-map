#!/usr/bin/env python3
"""Run VGGT on an image folder and visualize predictions with PointCloudViewer."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parents[1]
for _path in (_REPO_DIR, _REPO_DIR / "scripts" / "train", _REPO_DIR / "scripts" / "eval", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from demo import prepare_for_visualization
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.vis import PointCloudViewer


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VGGT folder inference + viser viewer")
    p.add_argument("--image_folder", required=True)
    p.add_argument("--vggt_repo", default="/cpfs/user/guowenqi/vggt")
    p.add_argument("--model_name", default="facebook/VGGT-1B")
    p.add_argument("--model_weights", default="/cpfs/user/guowenqi/vggt/model.pt")
    p.add_argument("--strict_load", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image_size", type=int, default=518)
    p.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    p.add_argument("--num_frames", type=int, default=64)
    p.add_argument("--sample_mode", choices=["uniform", "stride"], default="uniform")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--first_k", type=int, default=None)
    p.add_argument("--port", type=int, default=8101)
    p.add_argument("--conf_threshold", type=float, default=1.5)
    p.add_argument("--conf_filter_mode", choices=["percentile", "absolute"], default="percentile")
    p.add_argument("--downsample_factor", type=int, default=1)
    p.add_argument("--point_size", type=float, default=1e-5)
    p.add_argument("--use_point_map", action="store_true")
    p.add_argument("--mask_sky", action="store_true")
    p.add_argument("--depth_stride", type=int, default=1)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    return p


def list_images(image_folder: Path) -> list[Path]:
    paths = sorted(p for p in image_folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No image files found in {image_folder}")
    return paths


def sample_images(paths: list[Path], num_frames: int, sample_mode: str, stride: int, first_k: int | None) -> list[Path]:
    if first_k is not None:
        paths = paths[: max(0, first_k)]
    if sample_mode == "stride":
        return paths[:: max(1, stride)]
    if len(paths) <= num_frames:
        return paths
    indices = np.linspace(0, len(paths) - 1, num_frames, dtype=np.int64)
    return [paths[int(i)] for i in indices]


def load_images(paths: list[Path], image_size: int, mode: str) -> torch.Tensor:
    # VGGT's helper is fixed at 518, so use it directly for the canonical case.
    if image_size == 518:
        from vggt.utils.load_fn import load_and_preprocess_images

        return load_and_preprocess_images([str(p) for p in paths], mode=mode)

    from PIL import Image
    from torchvision import transforms as TF

    to_tensor = TF.ToTensor()
    images = []
    for path in paths:
        img = Image.open(path)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        img = img.convert("RGB")
        width, height = img.size
        if mode == "pad":
            if width >= height:
                new_width = image_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = image_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = image_size
            new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = to_tensor(img)
        if mode == "crop" and new_height > image_size:
            start_y = (new_height - image_size) // 2
            tensor = tensor[:, start_y : start_y + image_size, :]
        if mode == "pad":
            h_padding = image_size - tensor.shape[1]
            w_padding = image_size - tensor.shape[2]
            if h_padding > 0 or w_padding > 0:
                top = h_padding // 2
                bottom = h_padding - top
                left = w_padding // 2
                right = w_padding - left
                tensor = torch.nn.functional.pad(tensor, (left, right, top, bottom), mode="constant", value=1.0)
        images.append(tensor)

    shapes = {tuple(img.shape[-2:]) for img in images}
    if len(shapes) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        padded = []
        for img in images:
            h_padding = max_h - img.shape[1]
            w_padding = max_w - img.shape[2]
            top = h_padding // 2
            bottom = h_padding - top
            left = w_padding // 2
            right = w_padding - left
            padded.append(torch.nn.functional.pad(img, (left, right, top, bottom), mode="constant", value=1.0))
        images = padded
    return torch.stack(images, dim=0)


def extract_state_dict(obj: object) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Expected state_dict-like object, got {type(obj)!r}")
    for key in ("model", "state_dict", "model_state_dict"):
        child = obj.get(key)
        if isinstance(child, dict):
            obj = child
            break
    state = obj
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    return state


def build_vggt(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    vggt_repo = Path(args.vggt_repo)
    if vggt_repo.is_dir() and str(vggt_repo) not in sys.path:
        sys.path.insert(0, str(vggt_repo))
    from vggt.models.vggt import VGGT

    if args.model_weights:
        model = VGGT()
        loaded = torch.load(args.model_weights, map_location="cpu")
        missing, unexpected = model.load_state_dict(extract_state_dict(loaded), strict=bool(args.strict_load))
        if missing or unexpected:
            print(f"[model] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    else:
        model = VGGT.from_pretrained(args.model_name)
    return model.to(device).eval()


def confidence_to_percent_rank(conf: np.ndarray) -> np.ndarray:
    out = np.zeros(conf.shape, dtype=np.float32)
    valid = np.isfinite(conf)
    vals = conf[valid].astype(np.float64, copy=False)
    if vals.size == 0:
        return out
    sorted_vals = np.sort(vals)
    ranks = np.searchsorted(sorted_vals, vals, side="right") / float(vals.size) * 100.0
    out[valid] = ranks.astype(np.float32, copy=False)
    return out


def postprocess(predictions: Dict[str, torch.Tensor], images: torch.Tensor) -> Dict[str, torch.Tensor]:
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    out = dict(predictions)
    out["extrinsic"] = extrinsic_4x4[..., :3, :4]
    out["intrinsic"] = intrinsic
    out.pop("pose_enc_list", None)
    out["images"] = images
    return out


def main() -> None:
    args = build_parser().parse_args()
    vggt_repo = Path(args.vggt_repo)
    if vggt_repo.is_dir() and str(vggt_repo) not in sys.path:
        sys.path.insert(0, str(vggt_repo))
    image_folder = Path(args.image_folder)
    paths = sample_images(list_images(image_folder), args.num_frames, args.sample_mode, args.stride, args.first_k)
    if not paths:
        raise RuntimeError("No images selected")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    print("\n================================================================")
    print("                  VGGT Demo (PointCloudViewer)                  ")
    print("================================================================")
    print(f"[run]\n  device              : {device}\n  port                : {args.port}\n  model               : {args.model_weights or args.model_name}")
    print(f"\n[input]\n  source              : {image_folder}\n  selected_files      : {len(paths)}")
    print(f"  sample_mode         : {args.sample_mode}")
    print(f"  first,last          : {paths[0].name}, {paths[-1].name}")

    images = load_images(paths, args.image_size, args.preprocess_mode)
    print(f"  preprocessed        : {images.shape[0]} frames, {images.shape[-1]}x{images.shape[-2]} ({args.preprocess_mode})")
    model = build_vggt(args, device)
    images_dev = images.to(device, non_blocking=True)[None]
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp and device.type == "cuda"):
        predictions = model(images_dev)
    predictions = postprocess(predictions, images_dev)

    vis_predictions = prepare_for_visualization(predictions)
    if args.conf_filter_mode == "percentile":
        for key in ("world_points_conf", "depth_conf"):
            if key in vis_predictions and vis_predictions[key] is not None:
                vis_predictions[key] = confidence_to_percent_rank(np.asarray(vis_predictions[key]))
        threshold_label = "Confidence Percent"
        threshold_min = 0.0
        threshold_max = 100.0
        threshold_step = 0.1
    else:
        threshold_label = "Visibility Threshold"
        threshold_min = 1.0
        threshold_max = 5.0
        threshold_step = 0.01

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"\n[inference]\n  gpu_peak_alloc_gb   : {peak:.2f}")
    print(f"\n[viewer]\n  url                 : http://localhost:{args.port}")
    print(f"  conf_filter_mode    : {args.conf_filter_mode}\n  conf_threshold      : {args.conf_threshold}")
    print("================================================================\n")

    viewer = PointCloudViewer(
        pred_dict=vis_predictions,
        device="cpu",
        port=args.port,
        show_camera=True,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
        use_point_map=args.use_point_map,
        mask_sky=args.mask_sky,
        image_folder=str(image_folder),
        depth_stride=args.depth_stride,
        vis_threshold_label=threshold_label,
        vis_threshold_min=threshold_min,
        vis_threshold_max=threshold_max,
        vis_threshold_step=threshold_step,
    )
    viewer.run()


if __name__ == "__main__":
    main()
