#!/usr/bin/env python3
"""Train LingBot-MAP on Manip RGB-D trajectories.

This is intentionally self-contained: Manip trajectory discovery, image/depth
preprocessing, VGGT-style losses, LingBot-MAP anchor-frame normalization, and
the training loop all live in this file so it can be adjusted quickly for early
experiments.
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
import random
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as TF
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message="Failed to JIT torch c dlpack extension.*",
    category=UserWarning,
)

from lingbot_map.utils.pose_enc import extri_intri_to_pose_encoding
from lingbot_map.utils.rotation import quat_to_mat


DEFAULT_DATA_ROOTS = [
    "/oss-guowenqi/Manip_long3/data",
    "/oss-guowenqi/Manip_long4/data",
    "/oss-guowenqi/Manip_long5/data",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIP_CAMERA_NAMES = (
    "realsense_left",
    "realsense_right",
    "surround_cam_0",
    "surround_cam_1",
    "surround_cam_2",
    "surround_cam_3",
    "surround_cam_4",
    "surround_cam_5",
)

# data_vis.py writes camera2env poses in the simulator camera frame. This
# rotation maps OpenCV camera coordinates (x-right, y-down, z-forward) into
# that frame before the camera2env transform is applied.
OPENCV_TO_GENMANIP_CAMERA_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None or value.strip() == "":
        return None
    return [int(x) for x in value.split(",") if x.strip() != ""]


def parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None or value.strip() == "":
        return None
    return [x.strip() for x in value.split(",") if x.strip() != ""]


def parse_mode_weights(value: Optional[str]) -> Optional[Dict[str, float]]:
    if value is None or value.strip() == "":
        return None
    out: Dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"mode-weights entry '{item}' is missing '=' separator (expected e.g. 'S=0.3')"
            )
        key, val = item.split("=", 1)
        key = key.strip().upper()
        try:
            out[key] = float(val.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"mode-weights entry '{item}' has non-numeric value") from exc
    return out or None


def check_and_fix_inf_nan(
    tensor: torch.Tensor,
    name: str,
    hard_max: Optional[float] = 1e6,
) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        return tensor
    if hard_max is None:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(
        tensor.clamp(min=-hard_max, max=hard_max),
        nan=0.0,
        posinf=hard_max,
        neginf=-hard_max,
    )


def inverse_se3(transform: torch.Tensor) -> torch.Tensor:
    """Invert an SE(3) matrix with shape (..., 4, 4)."""
    rot = transform[..., :3, :3]
    trans = transform[..., :3, 3:4]
    inv = torch.zeros_like(transform)
    inv[..., :3, :3] = rot.transpose(-1, -2)
    inv[..., :3, 3:4] = -torch.matmul(rot.transpose(-1, -2), trans)
    inv[..., 3, 3] = 1.0
    return inv


def se3_3x4_to_4x4(transform: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 4) SE(3) matrices to homogeneous (..., 4, 4)."""
    full = torch.zeros(transform.shape[:-2] + (4, 4), dtype=transform.dtype, device=transform.device)
    full[..., :3, :4] = transform
    full[..., 3, 3] = 1.0
    return full


def w2c_to_c2w_extrinsics(extrinsics: torch.Tensor) -> torch.Tensor:
    """Convert OpenCV world-to-camera extrinsics to camera-to-world matrices."""
    return inverse_se3(se3_3x4_to_4x4(extrinsics))[..., :3, :4]


def pose_encoding_to_c2w_matrix(pose_encoding: torch.Tensor) -> torch.Tensor:
    """Convert LingBot-MAP absT_quaR_FoV pose encoding to c2w SE(3)."""
    rotation = quat_to_mat(pose_encoding[..., 3:7])
    translation = pose_encoding[..., :3]
    c2w = torch.zeros(pose_encoding.shape[:-1] + (4, 4), dtype=pose_encoding.dtype, device=pose_encoding.device)
    c2w[..., :3, :3] = rotation
    c2w[..., :3, 3] = translation
    c2w[..., 3, 3] = 1.0
    return c2w


def _iter_concat_leaves(dataset):
    """Yield each leaf sub-dataset, descending into ConcatDataset recursively."""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for child in dataset.datasets:
            yield from _iter_concat_leaves(child)
    else:
        yield dataset


def _propagate_global_step(dataset, step: int) -> None:
    """Call ``set_global_step(step)`` on every leaf that implements it.

    No-op on leaves without the method, so it's safe to call regardless of
    which mix-ins are active.
    """
    for leaf in _iter_concat_leaves(dataset):
        setter = getattr(leaf, "set_global_step", None)
        if callable(setter):
            setter(step)


def _find_dataset_with_attr(dataset, attr: str):
    """Return the first leaf sub-dataset that exposes ``attr``, else None."""
    for leaf in _iter_concat_leaves(dataset):
        if hasattr(leaf, attr):
            return leaf
    return None


def to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _trainable_param_fingerprint(model: nn.Module) -> List[Tuple[str, Tuple[int, ...]]]:
    return [
        (name, tuple(param.shape))
        for name, param in model.named_parameters()
        if param.requires_grad
    ]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "trainable_fingerprint": _trainable_param_fingerprint(model),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)


def _interpolate_pos_embed(ckpt_pe: torch.Tensor, model_pe: torch.Tensor, key: str) -> torch.Tensor:
    """Bilinearly interpolate a ViT-style position embedding to a new patch grid.

    Expects shape [1, 1+N^2, C] (with cls token) or [1, N^2, C]; cls token (if any)
    is preserved verbatim.
    """
    if ckpt_pe.dim() != 3 or model_pe.dim() != 3 or ckpt_pe.shape[-1] != model_pe.shape[-1]:
        raise ValueError(
            f"cannot interpolate pos_embed for {key}: "
            f"ckpt {tuple(ckpt_pe.shape)} vs model {tuple(model_pe.shape)}"
        )
    ckpt_n, model_n = ckpt_pe.shape[1], model_pe.shape[1]
    has_cls = (int(round(ckpt_n**0.5))**2 != ckpt_n)
    if has_cls:
        cls = ckpt_pe[:, :1]
        patch = ckpt_pe[:, 1:]
        target = model_n - 1
    else:
        cls = None
        patch = ckpt_pe
        target = model_n
    src = int(round(patch.shape[1] ** 0.5))
    dst = int(round(target ** 0.5))
    if src * src != patch.shape[1] or dst * dst != target:
        raise ValueError(
            f"non-square patch grid for {key}: src_n={patch.shape[1]} dst_n={target}"
        )
    C = patch.shape[-1]
    grid = patch.reshape(1, src, src, C).permute(0, 3, 1, 2).float()  # [1,C,src,src]
    grid = torch.nn.functional.interpolate(grid, size=(dst, dst), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, dst * dst, C).to(ckpt_pe.dtype)
    if cls is not None:
        return torch.cat([cls, grid], dim=1)
    return grid


def load_state_dict_flexible(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
    map_location: str = "cpu",
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    model_state = model.state_dict()
    interpolated: list[str] = []
    dropped: list[tuple[str, tuple, tuple]] = []
    for key in list(state.keys()):
        if key not in model_state:
            continue
        src_shape = tuple(state[key].shape)
        dst_shape = tuple(model_state[key].shape)
        if src_shape == dst_shape:
            continue
        if "pos_embed" in key:
            try:
                state[key] = _interpolate_pos_embed(state[key], model_state[key], key)
                interpolated.append(f"{key}: {src_shape} -> {dst_shape}")
            except Exception as exc:
                print(f"[model] pos_embed interpolation failed for {key}: {exc}")
                dropped.append((key, src_shape, dst_shape))
                state.pop(key)
        else:
            dropped.append((key, src_shape, dst_shape))
            state.pop(key)

    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(
        f"[model] checkpoint: {checkpoint_path} "
        f"(missing={len(missing)}, unexpected={len(unexpected)}, "
        f"interpolated={len(interpolated)}, dropped={len(dropped)})"
    )
    for line in interpolated:
        print(f"[model] interpolated {line}")
    for key, s, d in dropped:
        print(f"[model] dropped (shape mismatch) {key}: {s} -> {d}")
    if missing:
        print("[model] first missing keys:", missing[:8])
    if unexpected:
        print("[model] first unexpected keys:", unexpected[:8])


def freeze_by_name(model: nn.Module, substrings: Sequence[str]) -> int:
    if not substrings:
        return 0
    frozen = 0
    for name, param in model.named_parameters():
        if any(token in name for token in substrings):
            if param.requires_grad:
                frozen += param.numel()
            param.requires_grad_(False)
    return frozen


def count_trainable_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


# -----------------------------------------------------------------------------
# Manip dataset
# -----------------------------------------------------------------------------


def manip_camera_sort_key(camera_name: str) -> Tuple[int, object]:
    if camera_name in MANIP_CAMERA_NAMES:
        return (0, MANIP_CAMERA_NAMES.index(camera_name))
    return (1, camera_name)


def find_manip_pose_path(camera_dir: Path, trajectory_dir: Path, camera_name: str) -> Path:
    in_camera_dir = camera_dir / f"{camera_name}_pose.txt"
    if in_camera_dir.is_file():
        return in_camera_dir
    return trajectory_dir / f"{camera_name}_pose.txt"


def read_trajectory_manifest(manifest_path: Path) -> List[Path]:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return [Path(line.strip()) for line in handle if line.strip()]


def write_trajectory_manifest(manifest_path: Path, trajectories: Sequence[Path]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(f"{trajectory}\n")


def _valid_trajectory_name(name: str) -> bool:
    return bool(name) and not name.startswith(".") and "claim" not in name.lower()


def _parse_ossutil_dir_name(raw_line: str, oss_uri_root: str) -> Optional[str]:
    token = raw_line.strip()
    if not token:
        return None

    root = oss_uri_root.rstrip("/") + "/"
    if not token.startswith(root):
        return None
    rel = token[len(root):].strip("/")
    if not rel or "/" in rel:
        return None
    return rel


def discover_trajectory_dirs_with_ossutil(
    roots: Sequence[str],
    oss_uri_roots: Sequence[str],
    ossutil_bin: str,
    ossutil_config: str = "",
) -> List[Path]:
    if len(roots) != len(oss_uri_roots):
        raise ValueError("--oss_uri_roots must have the same length as --data_roots")

    trajectories: List[Path] = []
    for root, oss_uri_root in zip(roots, oss_uri_roots):
        command = [ossutil_bin, "ls"]
        if ossutil_config:
            command.extend(["-c", ossutil_config])
        command.extend(["-d", "--short-format", oss_uri_root.rstrip("/") + "/"])
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        root_path = Path(root)
        for line in result.stdout.splitlines():
            name = _parse_ossutil_dir_name(line, oss_uri_root)
            if name is None or not _valid_trajectory_name(name):
                continue
            trajectories.append(root_path / name)
    return sorted(trajectories)


def discover_trajectory_dirs(
    roots: Sequence[str],
    max_scenes: int = 0,
    manifest: Optional[str] = None,
    write_manifest: Optional[str] = None,
    oss_uri_roots: Optional[Sequence[str]] = None,
    ossutil_bin: str = "ossutil",
    ossutil_config: str = "",
) -> List[Path]:
    manifest_path = Path(manifest) if manifest else None
    if manifest_path is not None and manifest_path.is_file():
        trajectories = read_trajectory_manifest(manifest_path)
        print(f"[data] manifest: loaded {len(trajectories)} Manip trajectories from {manifest_path}")
        return trajectories[:max_scenes] if max_scenes > 0 else trajectories

    if manifest_path is not None:
        print(f"[data] manifest missing; scanning Manip roots and caching to {manifest_path}")

    trajectories: List[Path] = []
    if oss_uri_roots:
        try:
            trajectories = discover_trajectory_dirs_with_ossutil(roots, oss_uri_roots, ossutil_bin, ossutil_config)
            print(f"[data] discovered {len(trajectories)} candidate Manip trajectories with ossutil")
        except Exception as exc:
            print(f"[data][warn] ossutil trajectory scan failed ({exc}); falling back to mounted path scan")

    if not trajectories:
        for root in roots:
            root_path = Path(root)
            if not root_path.exists():
                print(f"[data][warn] data root does not exist: {root_path}")
                continue
            with os.scandir(root_path) as iterator:
                for entry in iterator:
                    if entry.is_dir() and _valid_trajectory_name(entry.name):
                        trajectories.append(Path(entry.path))

        trajectories = sorted(trajectories)
        print(f"[data] discovered {len(trajectories)} candidate Manip trajectories with mounted path scan")
    write_targets: List[Path] = []
    if manifest_path is not None:
        write_targets.append(manifest_path)
    if write_manifest:
        explicit_path = Path(write_manifest)
        if explicit_path not in write_targets:
            write_targets.append(explicit_path)
    for target in write_targets:
        write_trajectory_manifest(target, trajectories)
        print(f"[data] wrote {len(trajectories)} Manip trajectories to manifest: {target}")

    return trajectories[:max_scenes] if max_scenes > 0 else trajectories


def parse_manip_frame_stem(stem: str) -> int:
    if stem.isdigit():
        return int(stem)
    for token in reversed(stem.replace("-", "_").split("_")):
        if token.isdigit():
            return int(token)
    raise ValueError(f"Cannot parse Manip frame id from {stem}")

def split_scenes(
    scenes: Sequence[Path],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Path], List[Path]]:
    scenes = list(scenes)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    if val_fraction <= 0 or len(scenes) < 2:
        return scenes, []
    val_count = max(1, int(round(len(scenes) * val_fraction)))
    val = scenes[:val_count]
    train = scenes[val_count:]
    return train, val


def compute_preprocess_geometry(
    width: int,
    height: int,
    image_size: int,
    patch_size: int,
    mode: str,
) -> Dict[str, int]:
    if mode == "pad":
        if width >= height:
            new_width = image_size
            new_height = round(height * (new_width / width) / patch_size) * patch_size
        else:
            new_height = image_size
            new_width = round(width * (new_height / height) / patch_size) * patch_size
        new_width = max(patch_size, new_width)
        new_height = max(patch_size, new_height)
        pad_left = (image_size - new_width) // 2
        pad_right = image_size - new_width - pad_left
        pad_top = (image_size - new_height) // 2
        pad_bottom = image_size - new_height - pad_top
        return {
            "new_width": new_width,
            "new_height": new_height,
            "crop_left": 0,
            "crop_top": 0,
            "crop_width": new_width,
            "crop_height": new_height,
            "pad_left": max(0, pad_left),
            "pad_right": max(0, pad_right),
            "pad_top": max(0, pad_top),
            "pad_bottom": max(0, pad_bottom),
        }

    if mode != "crop":
        raise ValueError(f"Unknown preprocess mode: {mode}")

    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    new_height = max(patch_size, new_height)
    crop_top = max(0, (new_height - image_size) // 2)
    crop_height = min(new_height, image_size)
    return {
        "new_width": new_width,
        "new_height": new_height,
        "crop_left": 0,
        "crop_top": crop_top,
        "crop_width": new_width,
        "crop_height": crop_height,
        "pad_left": 0,
        "pad_right": 0,
        "pad_top": 0,
        "pad_bottom": 0,
    }


def apply_preprocess_to_image(
    image: Image.Image,
    geometry: Dict[str, int],
    resample: Image.Resampling,
    fill: object,
) -> Image.Image:
    image = image.resize(
        (geometry["new_width"], geometry["new_height"]),
        resample=resample,
    )
    left = geometry["crop_left"]
    top = geometry["crop_top"]
    right = left + geometry["crop_width"]
    bottom = top + geometry["crop_height"]
    image = image.crop((left, top, right, bottom))

    if any(geometry[key] > 0 for key in ("pad_left", "pad_right", "pad_top", "pad_bottom")):
        image = ImageOps.expand(
            image,
            border=(
                geometry["pad_left"],
                geometry["pad_top"],
                geometry["pad_right"],
                geometry["pad_bottom"],
            ),
            fill=fill,
        )
    return image


def preprocess_intrinsics(
    intrinsics: np.ndarray,
    width: int,
    height: int,
    geometry: Dict[str, int],
) -> torch.Tensor:
    intrinsics = intrinsics.astype(np.float32).copy()
    scale_x = geometry["new_width"] / float(width)
    scale_y = geometry["new_height"] / float(height)

    intrinsics[0, 0] *= scale_x
    intrinsics[0, 2] *= scale_x
    intrinsics[1, 1] *= scale_y
    intrinsics[1, 2] *= scale_y

    intrinsics[0, 2] += geometry["pad_left"] - geometry["crop_left"]
    intrinsics[1, 2] += geometry["pad_top"] - geometry["crop_top"]
    return torch.from_numpy(intrinsics[:3, :3])


def depth_to_world_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    """Unproject a depth map with OpenCV w2c extrinsics to world points."""
    height, width = depth.shape
    dtype = depth.dtype
    device = depth.device

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    z = depth
    x = (x_grid - intrinsics[0, 2]) / intrinsics[0, 0].clamp(min=1e-6) * z
    y = (y_grid - intrinsics[1, 2]) / intrinsics[1, 1].clamp(min=1e-6) * z
    camera_points = torch.stack([x, y, z, torch.ones_like(z)], dim=-1)

    extrinsic_4x4 = torch.eye(4, dtype=dtype, device=device)
    extrinsic_4x4[:3, :4] = extrinsics.to(dtype=dtype, device=device)
    c2w = inverse_se3(extrinsic_4x4)
    world_points = torch.matmul(camera_points, c2w.transpose(0, 1))[..., :3]
    return world_points


@dataclass(frozen=True)
class FrameEntry:
    frame_id: int
    view_id: int
    rgb_path: Path
    depth_path: Path
    mask_path: Optional[Path]
    camera_name: Optional[str] = None
    pose_path: Optional[Path] = None


def quat_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 0:
        raise ValueError("Invalid zero-norm quaternion")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def manip_camera2env_pose_to_opencv_w2c(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    camera_to_env = np.eye(4, dtype=np.float32)
    camera_to_env[:3, :3] = quat_wxyz_to_rotation_matrix(quat_wxyz)
    camera_to_env[:3, 3] = np.asarray(position, dtype=np.float32)

    opencv_to_genmanip = np.eye(4, dtype=np.float32)
    opencv_to_genmanip[:3, :3] = OPENCV_TO_GENMANIP_CAMERA_ROTATION
    opencv_camera_to_env = camera_to_env @ opencv_to_genmanip
    return np.linalg.inv(opencv_camera_to_env).astype(np.float32)


def read_manip_camera_pose_file(pose_path: Path) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    intrinsic_rows: List[List[float]] = []
    extrinsics_by_frame: Dict[int, np.ndarray] = {}
    with open(pose_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                payload = stripped[1:].strip()
                parts = payload.split()
                if len(parts) == 3:
                    try:
                        intrinsic_rows.append([float(value) for value in parts])
                    except ValueError:
                        pass
                continue

            parts = stripped.split()
            if len(parts) < 8:
                continue
            frame_id = int(float(parts[0]))
            position = np.asarray([float(value) for value in parts[1:4]], dtype=np.float32)
            quat_wxyz = np.asarray([float(value) for value in parts[4:8]], dtype=np.float32)
            extrinsics_by_frame[frame_id] = manip_camera2env_pose_to_opencv_w2c(position, quat_wxyz)

    if len(intrinsic_rows) < 3:
        raise ValueError(f"No 3x3 intrinsics found in {pose_path}")
    if not extrinsics_by_frame:
        raise ValueError(f"No per-frame poses found in {pose_path}")
    return np.asarray(intrinsic_rows[:3], dtype=np.float32), extrinsics_by_frame


class ManipTrajectoryDataset(Dataset):
    def __init__(
        self,
        scene_dirs: Sequence[Path],
        clip_len: int,
        image_size: int,
        patch_size: int,
        preprocess_mode: str = "crop",
        sequence_mode: str = "all_views",
        view_ids: Optional[Sequence[int]] = None,
        camera_names: Optional[Sequence[str]] = None,
        sample_strategy: str = "fixed_stride",
        frame_stride: int = 1,
        random_stride_min: int = 6,
        random_stride_max: int = 64,
        random_interval_start: str = "first",
        max_sample_frames: int = 0,
        min_sample_frames: int = 1,
        depth_scale: float = 1000.0,
        min_depth: float = 1e-6,
        max_depth: float = 0.0,
        use_mask: bool = False,
        invert_cam_extrinsics: bool = False,
        samples_per_scene: int = 1,
        # manip_4d_mixed mode hyperparameters
        wrist_camera_prefix: str = "realsense",
        static_camera_prefix: str = "surround",
        m_stride_min: int = 2,
        m_stride_max: int = 8,
        s_views_min: int = 4,
        s_views_max: int = 8,
        m_num_views: int = 4,
        m_num_times: int = 4,
        m_views_min: int = 3,
        m_views_max: int = 6,
        mode_weights_initial: Optional[Dict[str, float]] = None,
        mode_weights_final: Optional[Dict[str, float]] = None,
        mode_warmup_start: int = 2000,
        mode_warmup_end: int = 8000,
        # Mode T (single-camera trajectory random-interval) — used exclusively
        # for Manip_long5 scenes, which have moving surround cameras.
        t_stride_min: int = 15,
        t_stride_max: int = 60,
        long5_root_marker: str = "Manip_long5",
        color_jitter_strength: float = 0.0,
        color_jitter_prob: float = 0.5,
    ) -> None:
        self.scene_dirs = list(scene_dirs)
        self.clip_len = max(1, clip_len)
        self.image_size = image_size
        self.patch_size = patch_size
        self.preprocess_mode = preprocess_mode
        self.sequence_mode = sequence_mode
        self.view_ids = set(view_ids) if view_ids is not None else None
        self.camera_names = set(camera_names) if camera_names is not None else None
        self.sample_strategy = sample_strategy
        self.frame_stride = max(1, frame_stride)
        self.random_stride_min = max(1, random_stride_min)
        self.random_stride_max = max(self.random_stride_min, random_stride_max)
        self.random_interval_start = random_interval_start
        self.max_sample_frames = max(0, max_sample_frames)
        self.min_sample_frames = max(1, min_sample_frames)
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.use_mask = use_mask
        self.invert_cam_extrinsics = invert_cam_extrinsics
        self.samples_per_scene = max(1, samples_per_scene)
        self._entries_cache: "OrderedDict[Path, List[FrameEntry]]" = OrderedDict()
        self._pose_cache: "OrderedDict[Path, Tuple[np.ndarray, Dict[int, np.ndarray]]]" = OrderedDict()
        # LRU bounds (per-worker). Sized so the caches together hold ~720
        # scenes worth of state on the assumption of ~8 cameras/scene; with
        # ~26.5 MB per scene worst-case this caps each worker at ~19 GB
        # (~76 GB total across 4 workers, leaving headroom on a 100 GB host).
        self._entries_cache_max = 720
        self._pose_cache_max = 5760

        # manip_4d_mixed configuration
        self.wrist_camera_prefix = wrist_camera_prefix
        self.static_camera_prefix = static_camera_prefix
        self.m_stride_min = max(1, m_stride_min)
        self.m_stride_max = max(self.m_stride_min, m_stride_max)
        self.s_views_min = max(1, s_views_min)
        self.s_views_max = max(self.s_views_min, s_views_max)
        self.m_num_views = max(1, m_num_views)
        self.m_num_times = max(0, m_num_times)
        self.m_views_min = max(2, m_views_min)
        self.m_views_max = max(self.m_views_min, m_views_max)
        default_initial = {"S": 0.70, "W": 0.20, "M": 0.10}
        default_final = {"S": 0.30, "W": 0.40, "M": 0.30}
        self.mode_weights_initial = self._normalize_mode_weights(mode_weights_initial or default_initial)
        self.mode_weights_final = self._normalize_mode_weights(mode_weights_final or default_final)
        self.mode_warmup_start = max(0, mode_warmup_start)
        self.mode_warmup_end = max(self.mode_warmup_start, mode_warmup_end)
        # Mode T (single-camera trajectory random-interval) for Manip_long5
        # scenes: surround cameras follow a moving circular/linear trajectory,
        # so each surround camera by itself is a 4D-rich sequence (camera pose
        # + scene dynamics both change frame-to-frame). Long3/4 retains the
        # S/W/M curriculum; long5 always routes to mode T regardless of
        # mode_weights, with its own stride range.
        self.t_stride_min = max(1, t_stride_min)
        self.t_stride_max = max(self.t_stride_min, t_stride_max)
        self.long5_root_marker = long5_root_marker or ""
        # Shared with worker processes so the main loop can advertise the
        # current optimizer step and drive the curriculum schedule.
        self._step_counter = multiprocessing.Value("i", 0)
        self._cam_classify_cache: Dict[Path, Tuple[Dict[int, List[FrameEntry]], Dict[Tuple[int, int], FrameEntry], List[int], List[int]]] = {}

        # Per-clip RGB ColorJitter (depth/mask/geometry untouched). Same params
        # are sampled once per __getitem__ and broadcast across all frames in
        # the clip so cross-view / cross-time appearance stays consistent.
        self.color_jitter_strength = max(0.0, float(color_jitter_strength))
        self.color_jitter_prob = float(np.clip(color_jitter_prob, 0.0, 1.0))
        if self.color_jitter_strength > 0:
            self._color_jitter = TF.ColorJitter(
                brightness=self.color_jitter_strength,
                contrast=self.color_jitter_strength,
                saturation=self.color_jitter_strength,
                hue=min(0.5, self.color_jitter_strength * 0.25),
            )
        else:
            self._color_jitter = None

    @staticmethod
    def _normalize_mode_weights(weights: Dict[str, float]) -> Dict[str, float]:
        cleaned = {key: max(0.0, float(value)) for key, value in weights.items() if key in {"S", "W", "M"}}
        total = sum(cleaned.values())
        if total <= 0:
            return {"S": 1 / 3, "W": 1 / 3, "M": 1 / 3}
        return {key: value / total for key, value in cleaned.items()}

    def set_global_step(self, step: int) -> None:
        """Called from the main training loop to advance the curriculum schedule."""
        with self._step_counter.get_lock():
            self._step_counter.value = int(step)

    def _current_step(self) -> int:
        return int(self._step_counter.value)

    def _compute_mode_weights(self, step: int) -> Dict[str, float]:
        if step <= self.mode_warmup_start or self.mode_warmup_end == self.mode_warmup_start:
            return dict(self.mode_weights_initial)
        if step >= self.mode_warmup_end:
            return dict(self.mode_weights_final)
        ratio = (step - self.mode_warmup_start) / (self.mode_warmup_end - self.mode_warmup_start)
        out: Dict[str, float] = {}
        for key in ("S", "W", "M"):
            a = self.mode_weights_initial.get(key, 0.0)
            b = self.mode_weights_final.get(key, 0.0)
            out[key] = a + (b - a) * ratio
        # numerical safety renormalization
        total = sum(out.values()) or 1.0
        return {key: value / total for key, value in out.items()}

    def __len__(self) -> int:
        return len(self.scene_dirs) * self.samples_per_scene

    def _entries_for_scene(self, scene_dir: Path) -> List[FrameEntry]:
        cached = self._entries_cache.get(scene_dir)
        if cached is not None:
            self._entries_cache.move_to_end(scene_dir)
            return cached

        entries = sorted(self._manip_entries_for_scene(scene_dir), key=lambda item: (item.frame_id, item.view_id))
        self._entries_cache[scene_dir] = entries
        while len(self._entries_cache) > self._entries_cache_max:
            self._entries_cache.popitem(last=False)
        return entries

    def _manip_entries_for_scene(self, scene_dir: Path) -> List[FrameEntry]:
        camera_dirs: List[Tuple[str, Path, Path]] = []
        with os.scandir(scene_dir) as iterator:
            for entry in iterator:
                if not entry.is_dir() or entry.name.startswith("."):
                    continue
                camera_dir = Path(entry.path)
                if not (camera_dir / "images").is_dir() or not (camera_dir / "depth_real").is_dir():
                    continue
                pose_path = find_manip_pose_path(camera_dir, scene_dir, entry.name)
                if not pose_path.is_file():
                    continue
                camera_dirs.append((entry.name, camera_dir, pose_path))

        camera_dirs.sort(key=lambda item: manip_camera_sort_key(item[0]))
        entries: List[FrameEntry] = []
        for view_id, (camera_name, camera_dir, pose_path) in enumerate(camera_dirs):
            if self.view_ids is not None and view_id not in self.view_ids:
                continue
            if self.camera_names is not None and camera_name not in self.camera_names:
                continue

            image_dir = camera_dir / "images"
            depth_dir = camera_dir / "depth_real"
            mask_dir = camera_dir / "mask"
            has_mask_dir = mask_dir.is_dir()
            with os.scandir(image_dir) as image_iter:
                image_entries = sorted(
                    (item for item in image_iter if item.name and not item.name.startswith(".")),
                    key=lambda item: item.name,
                )
            for image_entry in image_entries:
                rgb_path = Path(image_entry.path)
                if rgb_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    frame_id = parse_manip_frame_stem(rgb_path.stem)
                except ValueError:
                    continue
                depth_path = depth_dir / f"{rgb_path.stem}.png"
                mask_path = mask_dir / f"{rgb_path.stem}.png"
                entries.append(
                    FrameEntry(
                        frame_id=frame_id,
                        view_id=view_id,
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        mask_path=mask_path if has_mask_dir else None,
                        camera_name=camera_name,
                        pose_path=pose_path,
                    )
                )
        return entries

    def _entries_for_sequence_mode(self, entries: Sequence[FrameEntry]) -> List[FrameEntry]:
        if self.sequence_mode == "single_view":
            by_view: Dict[int, List[FrameEntry]] = {}
            for entry in entries:
                by_view.setdefault(entry.view_id, []).append(entry)
            view_entries = [sorted(items, key=lambda item: item.frame_id) for items in by_view.values() if items]
            if not view_entries:
                raise RuntimeError("No entries after single-view grouping")
            if self.sample_strategy == "fixed_stride":
                usable = [items for items in view_entries if len(items) >= self.clip_len]
                return random.choice(usable if usable else view_entries)
            return random.choice(view_entries)
        if self.sequence_mode == "all_views":
            return list(entries)
        if self.sequence_mode == "manip_4d_mixed":
            # Returned list is already the final selection; downstream sampling
            # is bypassed via the marker stored on the instance.
            return list(entries)
        raise ValueError(f"Unknown sequence mode: {self.sequence_mode}")

    # ------------------------------------------------------------------
    # manip_4d_mixed: W (wrist trajectory) / S (static multi-view) / M (4D grid)
    # ------------------------------------------------------------------
    def _classify_cameras(
        self,
        entries: Sequence[FrameEntry],
    ) -> Tuple[Dict[int, List[FrameEntry]], Dict[Tuple[int, int], FrameEntry], List[int], List[int]]:
        by_view: Dict[int, List[FrameEntry]] = {}
        by_vt: Dict[Tuple[int, int], FrameEntry] = {}
        cam_name_by_view: Dict[int, str] = {}
        for entry in entries:
            by_view.setdefault(entry.view_id, []).append(entry)
            by_vt[(entry.view_id, entry.frame_id)] = entry
            if entry.camera_name and entry.view_id not in cam_name_by_view:
                cam_name_by_view[entry.view_id] = entry.camera_name
        for view_id in by_view:
            by_view[view_id].sort(key=lambda item: item.frame_id)

        wrist_views: List[int] = []
        static_views: List[int] = []
        for view_id, name in cam_name_by_view.items():
            lower = name.lower() if name else ""
            if lower.startswith(self.wrist_camera_prefix):
                wrist_views.append(view_id)
            elif lower.startswith(self.static_camera_prefix):
                static_views.append(view_id)
        wrist_views.sort()
        static_views.sort()
        return by_view, by_vt, wrist_views, static_views

    def _sample_mode_W(
        self,
        by_view: Dict[int, List[FrameEntry]],
        wrist_views: Sequence[int],
        static_views: Sequence[int],
    ) -> List[FrameEntry]:
        candidate_views = list(wrist_views) or list(static_views) or list(by_view.keys())
        view_id = random.choice(candidate_views)
        camera_entries = by_view[view_id]
        if not camera_entries:
            raise RuntimeError("Mode W: empty wrist camera entries")

        # Always start at a random offset within the wrist trajectory so we
        # don't bias towards the very beginning of every episode.
        if len(camera_entries) > 1:
            start = random.randint(0, len(camera_entries) - 1)
        else:
            start = 0
        max_frames = self.max_sample_frames if self.max_sample_frames > 0 else self.clip_len

        def walk(direction: int) -> List[FrameEntry]:
            sel = [camera_entries[start]]
            idx = start
            while len(sel) < max_frames:
                step = random.randint(self.random_stride_min, self.random_stride_max)
                nxt = idx + direction * step
                if nxt < 0 or nxt >= len(camera_entries):
                    break
                sel.append(camera_entries[nxt])
                idx = nxt
            return sel

        # Try forward first; if it can't reach min_sample_frames (start is too
        # close to the end), walk backward from the same start instead so the
        # late-trajectory phase still gets sampled with true random intervals
        # rather than a uniform linspace. Mirrors mode T (Manip_long5).
        selected = walk(direction=+1)
        if len(selected) < self.min_sample_frames:
            backward = walk(direction=-1)
            backward.reverse()  # chronological order
            if len(backward) > len(selected):
                selected = backward

        if len(selected) < self.min_sample_frames and len(camera_entries) >= self.min_sample_frames:
            indices = np.linspace(0, len(camera_entries) - 1, self.min_sample_frames)
            selected = [camera_entries[int(round(idx))] for idx in indices]
        return selected

    def _sample_mode_S(
        self,
        by_view: Dict[int, List[FrameEntry]],
        by_vt: Dict[Tuple[int, int], FrameEntry],
        static_views: Sequence[int],
        wrist_views: Sequence[int],
    ) -> List[FrameEntry]:
        candidate_views = list(static_views)
        if len(candidate_views) < 2:
            candidate_views = list(static_views) + list(wrist_views)
        if len(candidate_views) < 2:
            candidate_views = list(by_view.keys())
        if len(candidate_views) < 2:
            raise RuntimeError("Mode S: fewer than 2 views available")

        n_views_max = min(self.s_views_max, len(candidate_views))
        n_views_min = min(self.s_views_min, n_views_max)
        n_views = random.randint(n_views_min, n_views_max)
        selected_views = random.sample(candidate_views, n_views)

        per_view_frames = [set(entry.frame_id for entry in by_view[v]) for v in selected_views]
        common_frames = set.intersection(*per_view_frames) if per_view_frames else set()
        if not common_frames:
            # Fallback: use the most frequent frame_id
            counts: Dict[int, int] = {}
            for frames in per_view_frames:
                for frame_id in frames:
                    counts[frame_id] = counts.get(frame_id, 0) + 1
            common_frames = {fid for fid, c in counts.items() if c == max(counts.values())}
        if not common_frames:
            raise RuntimeError("Mode S: no shared frames across selected static views")

        frame_id = random.choice(sorted(common_frames))
        selected: List[FrameEntry] = []
        for view_id in selected_views:
            entry = by_vt.get((view_id, frame_id))
            if entry is not None:
                selected.append(entry)
        if len(selected) < max(2, self.s_views_min):
            raise RuntimeError("Mode S: failed to assemble enough views at chosen timestamp")
        return selected

    def _sample_mode_M(
        self,
        by_view: Dict[int, List[FrameEntry]],
        by_vt: Dict[Tuple[int, int], FrameEntry],
        static_views: Sequence[int],
        wrist_views: Sequence[int],
    ) -> List[FrameEntry]:
        candidate_views = list(static_views)
        if len(candidate_views) < 2:
            candidate_views = list(by_view.keys())
        if len(candidate_views) < 2:
            raise RuntimeError("Mode M: fewer than 2 views available")

        n_views_max = min(self.m_views_max, len(candidate_views))
        n_views_min = min(self.m_views_min, n_views_max)
        n_views = random.randint(n_views_min, n_views_max)
        selected_views = random.sample(candidate_views, n_views)

        per_view_frames = [sorted(set(entry.frame_id for entry in by_view[v])) for v in selected_views]
        common_set = set(per_view_frames[0])
        for frames in per_view_frames[1:]:
            common_set &= set(frames)
        common_sorted = sorted(common_set)
        if not common_sorted:
            raise RuntimeError("Mode M: no shared frames across selected views")

        # Dynamic timestep count: aim for V*T inside [min_sample_frames,
        # max_sample_frames]; m_num_times (>0) is an additional upper cap.
        max_total = self.max_sample_frames if self.max_sample_frames > 0 else self.clip_len
        min_total = max(2 * n_views, self.min_sample_frames)
        t_max = max(1, max_total // n_views)
        t_min = max(1, (min_total + n_views - 1) // n_views)  # ceil(min_total / n_views)
        t_min = min(t_min, t_max)
        if self.m_num_times > 0:
            t_max = min(t_max, self.m_num_times)
            t_min = min(t_min, t_max)
        t_target = random.randint(t_min, t_max)

        # Bias start so the median-stride walk has room for full t_target; if
        # the actual stride pushes past the end of common_sorted, accept the
        # early termination (output may be < min_sample_frames in that case).
        median_stride = max(1, (self.m_stride_min + self.m_stride_max) // 2)
        desired_span = (t_target - 1) * median_stride + 1
        max_start = max(0, len(common_sorted) - desired_span)
        start_pos = random.randint(0, max_start)

        positions = [start_pos]
        cur = start_pos
        while len(positions) < t_target:
            cur += random.randint(self.m_stride_min, self.m_stride_max)
            if cur >= len(common_sorted):
                break  # early termination at end of common-frame sequence
            positions.append(cur)

        selected: List[FrameEntry] = []
        # Group by time so the model sees (t0,v0)...(t0,vN), (t1,v0)..., etc.
        for pos in positions:
            frame_id = common_sorted[pos]
            for view_id in selected_views:
                entry = by_vt.get((view_id, frame_id))
                if entry is not None:
                    selected.append(entry)
        if len(selected) < max(2, n_views):
            raise RuntimeError("Mode M: failed to assemble enough (t,v) entries")
        return selected

    # ------------------------------------------------------------------
    # Mode T: single-camera trajectory with per-step random stride.
    # Used exclusively for Manip_long5, where surround cameras move along a
    # circular/linear trajectory and each is a 4D-rich sequence on its own.
    # Algorithm mirrors mode W but draws from surround cameras and uses
    # t_stride_min/max instead of random_stride_min/max.
    # ------------------------------------------------------------------
    def _sample_mode_T(
        self,
        by_view: Dict[int, List[FrameEntry]],
        static_views: Sequence[int],
        wrist_views: Sequence[int],
    ) -> List[FrameEntry]:
        candidate_views = list(static_views)
        # Long5 should normally have surround cams; fall back to wrist or any
        # available view if a scene is malformed, so a single bad scene does
        # not blow up the whole epoch.
        if not candidate_views:
            candidate_views = list(wrist_views) or list(by_view.keys())
        if not candidate_views:
            raise RuntimeError("Mode T: no candidate cameras")
        view_id = random.choice(candidate_views)
        camera_entries = by_view[view_id]
        if not camera_entries:
            raise RuntimeError("Mode T: empty camera entries")

        # Random start anywhere in the trajectory: long5 trajectories are
        # ~2300-3000 frames, so always starting from frame 0 would heavily
        # bias the model towards the early phase of every pick&place.
        if len(camera_entries) > 1:
            start = random.randint(0, len(camera_entries) - 1)
        else:
            start = 0
        max_frames = self.max_sample_frames if self.max_sample_frames > 0 else self.clip_len

        def walk(direction: int) -> List[FrameEntry]:
            sel = [camera_entries[start]]
            idx = start
            while len(sel) < max_frames:
                step = random.randint(self.t_stride_min, self.t_stride_max)
                nxt = idx + direction * step
                if nxt < 0 or nxt >= len(camera_entries):
                    break
                sel.append(camera_entries[nxt])
                idx = nxt
            return sel

        # Try forward first; if it can't reach min_sample_frames (start is too
        # close to the end), walk backward from the same start instead so the
        # late-trajectory phase still gets sampled with true random intervals
        # rather than a uniform linspace.
        selected = walk(direction=+1)
        if len(selected) < self.min_sample_frames:
            backward = walk(direction=-1)
            backward.reverse()  # chronological order
            if len(backward) > len(selected):
                selected = backward

        # Last-resort linspace bridge: only triggers when neither forward nor
        # backward walk can fit min_sample_frames (very rare — short scenes).
        if len(selected) < self.min_sample_frames and len(camera_entries) >= self.min_sample_frames:
            indices = np.linspace(0, len(camera_entries) - 1, self.min_sample_frames)
            selected = [camera_entries[int(round(idx))] for idx in indices]
        return selected

    def _is_long5_scene(self, scene_dir: Optional[Path]) -> bool:
        if scene_dir is None or not self.long5_root_marker:
            return False
        return self.long5_root_marker.lower() in str(scene_dir).lower()

    def _sample_manip_4d_mixed(
        self,
        entries: Sequence[FrameEntry],
        scene_dir: Optional[Path] = None,
    ) -> Tuple[List[FrameEntry], str]:
        by_view, by_vt, wrist_views, static_views = self._classify_cameras(entries)
        # Manip_long5 routing: surround cameras move along a circular/linear
        # trajectory, so we always sample as a single-camera random-interval
        # walk over a surround_cam (mode T). We bypass the S/W/M curriculum
        # entirely — its weights are designed for the static-surround layout
        # of Manip_long3/4 and don't apply to long5.
        if self._is_long5_scene(scene_dir):
            return self._sample_mode_T(by_view, static_views, wrist_views), "T"
        weights = self._compute_mode_weights(self._current_step())
        modes = ["S", "W", "M"]
        weight_list = [weights.get(key, 0.0) for key in modes]
        if sum(weight_list) <= 0:
            weight_list = [1.0, 1.0, 1.0]
        # Try modes in weighted-random order, with graceful fallback if the
        # chosen mode is infeasible for this episode.
        order: List[str] = []
        remaining = list(modes)
        remaining_weights = list(weight_list)
        while remaining:
            if sum(remaining_weights) > 0:
                pick = random.choices(remaining, weights=remaining_weights, k=1)[0]
            else:
                # All remaining modes have zero probability: append in
                # random order so the mode-priority does not bias fallbacks.
                pick = random.choice(remaining)
            order.append(pick)
            idx = remaining.index(pick)
            remaining.pop(idx)
            remaining_weights.pop(idx)
        last_error: Optional[Exception] = None
        for mode in order:
            try:
                if mode == "W":
                    return self._sample_mode_W(by_view, wrist_views, static_views), "W"
                if mode == "S":
                    return self._sample_mode_S(by_view, by_vt, static_views, wrist_views), "S"
                if mode == "M":
                    return self._sample_mode_M(by_view, by_vt, static_views, wrist_views), "M"
            except Exception as exc:  # noqa: BLE001 - we want to fallthrough to next mode
                last_error = exc
                continue
        raise RuntimeError(f"manip_4d_mixed: all modes failed. last_error={last_error}")

    def _sample_fixed_stride(self, entries_for_clip: Sequence[FrameEntry]) -> List[FrameEntry]:
        span = (self.clip_len - 1) * self.frame_stride + 1
        if len(entries_for_clip) >= span:
            start = random.randint(0, len(entries_for_clip) - span)
            return list(entries_for_clip[start:start + span:self.frame_stride])

        if len(entries_for_clip) >= self.clip_len:
            indices = np.linspace(0, len(entries_for_clip) - 1, self.clip_len)
            return [entries_for_clip[int(round(idx))] for idx in indices]

        selected = list(entries_for_clip)
        while len(selected) < self.clip_len:
            selected.append(selected[-1])
        return selected

    def _sample_random_interval(self, entries_for_clip: Sequence[FrameEntry]) -> List[FrameEntry]:
        if not entries_for_clip:
            raise RuntimeError("No entries available for random-interval sampling")

        if self.random_interval_start == "random" and len(entries_for_clip) > 1:
            start = random.randint(0, len(entries_for_clip) - 1)
        else:
            start = 0

        selected = [entries_for_clip[start]]
        index = start
        while True:
            interval = random.randint(self.random_stride_min, self.random_stride_max)
            next_index = index + interval
            if next_index >= len(entries_for_clip):
                break
            selected.append(entries_for_clip[next_index])
            index = next_index
            if self.max_sample_frames > 0 and len(selected) >= self.max_sample_frames:
                break
        return selected

    def _sample_entries(
        self,
        entries: Sequence[FrameEntry],
        scene_dir: Optional[Path] = None,
    ) -> Tuple[List[FrameEntry], str]:
        if not entries:
            raise RuntimeError("No RGB-D entries found")

        if self.sequence_mode == "manip_4d_mixed":
            return self._sample_manip_4d_mixed(entries, scene_dir)

        entries_for_clip = self._entries_for_sequence_mode(entries)
        if self.sample_strategy == "fixed_stride":
            return self._sample_fixed_stride(entries_for_clip), "legacy"
        if self.sample_strategy == "random_interval":
            selected = self._sample_random_interval(entries_for_clip)
            if len(selected) < self.min_sample_frames and len(entries_for_clip) >= self.min_sample_frames:
                indices = np.linspace(0, len(entries_for_clip) - 1, self.min_sample_frames)
                selected = [entries_for_clip[int(round(idx))] for idx in indices]
            return selected, "legacy"
        raise ValueError(f"Unknown sample strategy: {self.sample_strategy}")

    def _load_camera_for_entry(
        self,
        scene_dir: Path,
        entry: FrameEntry,
    ) -> Tuple[np.ndarray, torch.Tensor]:
        if entry.pose_path is None:
            raise ValueError(f"Missing Manip camera pose file for {entry.rgb_path}")

        cached_pose = self._pose_cache.get(entry.pose_path)
        if cached_pose is None:
            cached_pose = read_manip_camera_pose_file(entry.pose_path)
            self._pose_cache[entry.pose_path] = cached_pose
            while len(self._pose_cache) > self._pose_cache_max:
                self._pose_cache.popitem(last=False)
        else:
            self._pose_cache.move_to_end(entry.pose_path)
        intrinsics, extrinsics_by_frame = cached_pose
        if entry.frame_id not in extrinsics_by_frame:
            raise ValueError(f"Frame {entry.frame_id} not found in {entry.pose_path}")
        extrinsic_4x4 = extrinsics_by_frame[entry.frame_id]
        if self.invert_cam_extrinsics:
            extrinsic_4x4 = np.linalg.inv(extrinsic_4x4).astype(np.float32)
        return intrinsics, torch.from_numpy(extrinsic_4x4[:3, :4].astype(np.float32))

    def _sample_color_jitter_params(self) -> Optional[Tuple]:
        """Sample one ColorJitter param tuple to share across a whole clip.

        Returns None when jitter is disabled or skipped this iteration. Sampling
        once per clip (instead of per-frame) preserves cross-view / cross-time
        appearance consistency, which matters for the multi-view / 4D modes.
        """
        if self._color_jitter is None:
            return None
        if random.random() >= self.color_jitter_prob:
            return None
        return TF.ColorJitter.get_params(
            self._color_jitter.brightness,
            self._color_jitter.contrast,
            self._color_jitter.saturation,
            self._color_jitter.hue,
        )

    @staticmethod
    def _apply_color_jitter(image: Image.Image, params: Tuple) -> Image.Image:
        fn_idx, b_factor, c_factor, s_factor, h_factor = params
        for fn_id in fn_idx:
            if fn_id == 0 and b_factor is not None:
                image = TF.functional.adjust_brightness(image, b_factor)
            elif fn_id == 1 and c_factor is not None:
                image = TF.functional.adjust_contrast(image, c_factor)
            elif fn_id == 2 and s_factor is not None:
                image = TF.functional.adjust_saturation(image, s_factor)
            elif fn_id == 3 and h_factor is not None:
                image = TF.functional.adjust_hue(image, h_factor)
        return image

    def _load_one(
        self,
        scene_dir: Path,
        entry: FrameEntry,
        jitter_params: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = Image.open(entry.rgb_path)
        if rgb.mode == "RGBA":
            background = Image.new("RGBA", rgb.size, (255, 255, 255, 255))
            rgb = Image.alpha_composite(background, rgb)
        rgb = rgb.convert("RGB")

        width, height = rgb.size
        geometry = compute_preprocess_geometry(
            width,
            height,
            self.image_size,
            self.patch_size,
            self.preprocess_mode,
        )

        intrinsics_raw, extrinsics = self._load_camera_for_entry(scene_dir, entry)
        intrinsics = preprocess_intrinsics(intrinsics_raw, width, height, geometry)

        rgb = apply_preprocess_to_image(
            rgb,
            geometry,
            resample=Image.Resampling.BICUBIC,
            fill=(255, 255, 255),
        )
        if jitter_params is not None:
            rgb = self._apply_color_jitter(rgb, jitter_params)
        image_tensor = TF.ToTensor()(rgb)

        depth_img = Image.open(entry.depth_path)
        depth_img = apply_preprocess_to_image(
            depth_img,
            geometry,
            resample=Image.Resampling.NEAREST,
            fill=0,
        )
        depth_raw = np.asarray(depth_img)
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
            mask_img = apply_preprocess_to_image(
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
        world_points = depth_to_world_points(depth, intrinsics.float(), extrinsics.float())
        world_points = torch.where(point_mask[..., None], world_points, torch.zeros_like(world_points))

        return image_tensor, depth, point_mask, intrinsics.float(), extrinsics.float(), world_points.float()

    def __getitem__(self, index: int) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        start_scene = index % len(self.scene_dirs)
        for attempt in range(min(16, len(self.scene_dirs))):
            scene_dir = self.scene_dirs[(start_scene + attempt) % len(self.scene_dirs)]
            try:
                entries = self._entries_for_scene(scene_dir)
                selected, mode_label = self._sample_entries(entries, scene_dir)
                jitter_params = self._sample_color_jitter_params()
                loaded = [self._load_one(scene_dir, entry, jitter_params) for entry in selected]
                images, depths, masks, intrinsics, extrinsics, world_points = zip(*loaded)
                return {
                    "images": torch.stack(list(images), dim=0),
                    "depths": torch.stack(list(depths), dim=0),
                    "point_masks": torch.stack(list(masks), dim=0),
                    "intrinsics": torch.stack(list(intrinsics), dim=0),
                    "extrinsics": torch.stack(list(extrinsics), dim=0),
                    "world_points": torch.stack(list(world_points), dim=0),
                    "frame_ids": torch.tensor([entry.frame_id for entry in selected], dtype=torch.long),
                    "view_ids": torch.tensor([entry.view_id for entry in selected], dtype=torch.long),
                    "scene": scene_dir.name,
                    "sample_mode": mode_label,
                }
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load a valid sample near index {index}: {last_error}")


def _pad_sequence_tensor(values: Sequence[torch.Tensor], fill_value: float = 0.0) -> torch.Tensor:
    max_len = max(int(value.shape[0]) for value in values)
    out_shape = (len(values), max_len) + tuple(values[0].shape[1:])
    out = values[0].new_full(out_shape, fill_value)
    for batch_idx, value in enumerate(values):
        out[batch_idx, : value.shape[0]] = value
    return out


def _pad_camera_sequence(values: Sequence[torch.Tensor], key: str) -> torch.Tensor:
    max_len = max(int(value.shape[0]) for value in values)
    out_shape = (len(values), max_len) + tuple(values[0].shape[1:])
    out = values[0].new_zeros(out_shape)
    if key == "intrinsics":
        out[:, :, 0, 0] = 1.0
        out[:, :, 1, 1] = 1.0
        out[:, :, 2, 2] = 1.0
    elif key == "extrinsics":
        out[:, :, 0, 0] = 1.0
        out[:, :, 1, 1] = 1.0
        out[:, :, 2, 2] = 1.0
    for batch_idx, value in enumerate(values):
        out[batch_idx, : value.shape[0]] = value
    return out


def collate_rgbd_sequences(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
    sequence_lengths = torch.tensor([int(sample["images"].shape[0]) for sample in samples], dtype=torch.long)
    batch: Dict[str, object] = {
        "images": _pad_sequence_tensor([sample["images"] for sample in samples]),
        "depths": _pad_sequence_tensor([sample["depths"] for sample in samples]),
        "point_masks": _pad_sequence_tensor([sample["point_masks"] for sample in samples], fill_value=0),
        "intrinsics": _pad_camera_sequence([sample["intrinsics"] for sample in samples], "intrinsics"),
        "extrinsics": _pad_camera_sequence([sample["extrinsics"] for sample in samples], "extrinsics"),
        "world_points": _pad_sequence_tensor([sample["world_points"] for sample in samples]),
        "frame_ids": _pad_sequence_tensor([sample["frame_ids"] for sample in samples], fill_value=-1),
        "view_ids": _pad_sequence_tensor([sample["view_ids"] for sample in samples], fill_value=-1),
        "sequence_lengths": sequence_lengths,
        "scene": [sample["scene"] for sample in samples],
        "sample_mode": [str(sample.get("sample_mode", "legacy")) for sample in samples],
    }
    return batch

def canonicalize_to_first_frame(batch: Dict[str, object]) -> Dict[str, object]:
    """Re-express extrinsics and world_points so frame 0's c2w is identity.

    Mirrors VGGT's first-frame canonicalization
    (`vggt/training/train_utils/normalization.py`). The pretrained LingBot-MAP
    camera head was trained to emit identity for frame 0; without this step,
    finetuning on datasets that ship absolute simulator/world poses pays a
    large gratuitous loss on the global rigid offset between the dataset's
    world frame and the first camera. Apply BEFORE `normalize_scene_batch`
    so the anchor-scale step still operates on the canonicalized geometry.
    """
    extr_w2c = batch["extrinsics"]                            # (B, F, 3, 4)
    world_points = batch["world_points"]                      # (B, F, H, W, 3)

    extr_44 = se3_3x4_to_4x4(extr_w2c)                        # (B, F, 4, 4)
    inv_extr_0 = inverse_se3(extr_44[:, 0])                   # (B, 4, 4)  == c2w[0]
    new_extr_44 = torch.matmul(extr_44, inv_extr_0.unsqueeze(1))  # E_i @ inv(E_0)
    new_extr_w2c = new_extr_44[..., :3, :4]

    R0 = extr_w2c[:, 0, :3, :3]                               # (B, 3, 3)
    t0 = extr_w2c[:, 0, :3, 3]                                # (B, 3)
    new_world = torch.einsum("bij,bfhwj->bfhwi", R0, world_points) + t0[:, None, None, None, :]

    batch = dict(batch)
    batch["extrinsics"] = check_and_fix_inf_nan(new_extr_w2c, "extrinsics", hard_max=None)
    batch["world_points"] = check_and_fix_inf_nan(new_world, "world_points", hard_max=None)
    return batch


def normalize_scene_batch(batch: Dict[str, object], num_anchor_frames: int) -> Dict[str, object]:
    """LingBot-MAP paper normalization.

    The paper fixes scale from the anchor frames instead of using the full video
    or re-centering into the first camera coordinate system. Let X_anchor be the
    valid ground-truth point cloud from the first n anchor frames. We compute

        s = mean_{x in X_anchor} ||x||_2

    and divide ground-truth depths, point coordinates, and camera translations
    by s. Camera rotations and intrinsics are left unchanged.
    """
    extrinsics = batch["extrinsics"]
    depths = batch["depths"]
    world_points = batch["world_points"]
    point_masks = batch["point_masks"]

    assert torch.is_tensor(extrinsics)
    assert torch.is_tensor(depths)
    assert torch.is_tensor(world_points)
    assert torch.is_tensor(point_masks)

    _, seq_len = extrinsics.shape[:2]
    anchor_frames = min(max(1, int(num_anchor_frames)), seq_len)

    anchor_mask = point_masks[:, :anchor_frames].float()
    anchor_dist = world_points[:, :anchor_frames].norm(dim=-1)
    anchor_count = anchor_mask.sum(dim=(1, 2, 3))
    anchor_sum = (anchor_dist * anchor_mask).sum(dim=(1, 2, 3))

    # Rare corrupt samples can have no valid depth in the anchor frames. In that
    # case use the sampled clip as a fallback so the batch stays trainable.
    all_mask = point_masks.float()
    all_count = all_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    all_sum = (world_points.norm(dim=-1) * all_mask).sum(dim=(1, 2, 3))
    scale = torch.where(anchor_count > 0, anchor_sum / anchor_count.clamp(min=1.0), all_sum / all_count)
    scale = scale.clamp(min=1e-6, max=1e6)

    new_extrinsics = extrinsics.clone()
    new_depths = depths / scale[:, None, None, None]
    new_world = world_points / scale[:, None, None, None, None]
    new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / scale[:, None, None]

    batch = dict(batch)
    batch["extrinsics"] = check_and_fix_inf_nan(new_extrinsics, "extrinsics", hard_max=None)
    batch["world_points"] = check_and_fix_inf_nan(new_world, "world_points", hard_max=None)
    batch["depths"] = check_and_fix_inf_nan(new_depths, "depths", hard_max=None)
    batch["anchor_scale"] = scale
    return batch


# -----------------------------------------------------------------------------
# VGGT-style losses
# -----------------------------------------------------------------------------



class VGGTStyleLoss(nn.Module):
    def __init__(
        self,
        camera_weight: float = 5.0,
        depth_weight: float = 1.0,
        relative_pose_weight: float = 1.0,
        camera_loss_type: str = "l1",
        camera_gamma: float = 0.6,
        weight_trans: float = 1.0,
        weight_rot: float = 1.0,
        weight_focal: float = 0.5,
        relative_trans_weight: float = 1.0,
        relative_pose_window: int = 64,
        depth_gradient_loss_fn: Optional[str] = "grad",
        loss_gamma: float = 1.0,
        loss_alpha: float = 0.2,
        valid_range: float = 0.98,
        min_valid_pixels: int = 100,
    ) -> None:
        super().__init__()
        self.camera_weight = camera_weight
        self.depth_weight = depth_weight
        self.relative_pose_weight = relative_pose_weight
        self.camera_loss_type = camera_loss_type
        self.camera_gamma = camera_gamma
        self.weight_trans = weight_trans
        self.weight_rot = weight_rot
        self.weight_focal = weight_focal
        self.relative_trans_weight = relative_trans_weight
        self.relative_pose_window = relative_pose_window
        self.depth_gradient_loss_fn = depth_gradient_loss_fn
        self.loss_gamma = loss_gamma
        self.loss_alpha = loss_alpha
        self.valid_range = valid_range
        self.min_valid_pixels = min_valid_pixels

    def forward(self, predictions: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        total = None
        losses: Dict[str, torch.Tensor] = {}

        if self.camera_weight > 0 and "pose_enc_list" in predictions:
            camera_losses = compute_camera_loss(
                predictions,
                batch,
                loss_type=self.camera_loss_type,
                gamma=self.camera_gamma,
                weight_trans=self.weight_trans,
                weight_rot=self.weight_rot,
                weight_focal=self.weight_focal,
                min_valid_pixels=self.min_valid_pixels,
            )
            losses.update(camera_losses)
            weighted = camera_losses["loss_camera"] * self.camera_weight
            total = weighted if total is None else total + weighted

        if self.relative_pose_weight > 0 and "pose_enc_list" in predictions:
            relative_losses = compute_relative_pose_loss(
                predictions,
                batch,
                gamma=self.camera_gamma,
                relative_pose_window=self.relative_pose_window,
                relative_trans_weight=self.relative_trans_weight,
                min_valid_pixels=self.min_valid_pixels,
            )
            losses.update(relative_losses)
            weighted = relative_losses["loss_relative_pose"] * self.relative_pose_weight
            total = weighted if total is None else total + weighted

        if self.depth_weight > 0 and "depth" in predictions:
            depth_losses = compute_depth_loss(
                predictions,
                batch,
                gamma=self.loss_gamma,
                alpha=self.loss_alpha,
                gradient_loss_fn=self.depth_gradient_loss_fn,
                valid_range=self.valid_range,
                min_valid_pixels=self.min_valid_pixels,
            )
            losses.update(depth_losses)
            weighted = (
                depth_losses["loss_conf_depth"]
                + depth_losses["loss_reg_depth"]
                + depth_losses["loss_grad_depth"]
            ) * self.depth_weight
            total = weighted if total is None else total + weighted

        if total is None:
            any_pred = next(iter(predictions.values()))
            total = any_pred.float().mean() * 0.0
        losses["objective"] = total
        losses["loss_objective"] = total
        return losses

def compute_camera_loss(
    pred_dict: Dict[str, torch.Tensor],
    batch_data: Dict[str, torch.Tensor],
    loss_type: str = "l1",
    gamma: float = 0.6,
    weight_trans: float = 1.0,
    weight_rot: float = 1.0,
    weight_focal: float = 0.5,
    min_valid_pixels: int = 100,
) -> Dict[str, torch.Tensor]:
    pred_pose_encodings = pred_dict["pose_enc_list"]
    point_masks = batch_data["point_masks"]
    valid_frame_mask = point_masks.sum(dim=(-1, -2)) > min_valid_pixels
    n_stages = len(pred_pose_encodings)

    # The dataloader keeps OpenCV world-to-camera extrinsics for depth
    # unprojection. LingBot-MAP supervises camera-to-world poses, so invert
    # before converting to absT_quaR_FoV.
    gt_c2w = w2c_to_c2w_extrinsics(batch_data["extrinsics"])
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_c2w,
        batch_data["intrinsics"],
        batch_data["images"].shape[-2:],
        pose_encoding_type="absT_quaR_FoV",
    )

    total_loss_t = total_loss_r = total_loss_fl = None
    for stage_idx, pred_pose_stage in enumerate(pred_pose_encodings):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        if valid_frame_mask.sum() == 0:
            loss_t = pred_pose_stage.float().mean() * 0.0
            loss_r = pred_pose_stage.float().mean() * 0.0
            loss_fl = pred_pose_stage.float().mean() * 0.0
        else:
            loss_t, loss_r, loss_fl = camera_loss_single(
                pred_pose_stage[valid_frame_mask].clone(),
                gt_pose_encoding[valid_frame_mask].clone(),
                loss_type=loss_type,
            )

        loss_t = loss_t * stage_weight
        loss_r = loss_r * stage_weight
        loss_fl = loss_fl * stage_weight
        total_loss_t = loss_t if total_loss_t is None else total_loss_t + loss_t
        total_loss_r = loss_r if total_loss_r is None else total_loss_r + loss_r
        total_loss_fl = loss_fl if total_loss_fl is None else total_loss_fl + loss_fl

    avg_loss_t = total_loss_t / n_stages
    avg_loss_r = total_loss_r / n_stages
    avg_loss_fl = total_loss_fl / n_stages
    loss_camera = avg_loss_t * weight_trans + avg_loss_r * weight_rot + avg_loss_fl * weight_focal
    return {
        "loss_camera": loss_camera,
        "loss_T": avg_loss_t,
        "loss_R": avg_loss_r,
        "loss_FL": avg_loss_fl,
    }



def camera_loss_single(
    pred_pose_enc: torch.Tensor,
    gt_pose_enc: torch.Tensor,
    loss_type: str = "l1",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_type == "l1":
        loss_t = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_r = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_fl = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        loss_t = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_r = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_fl = (pred_pose_enc[..., 7:]).sub(gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown camera loss type: {loss_type}")

    loss_t = check_and_fix_inf_nan(loss_t, "loss_T").clamp(max=100).mean()
    loss_r = check_and_fix_inf_nan(loss_r, "loss_R").mean()
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_FL").mean()
    return loss_t, loss_r, loss_fl


def rotation_geodesic_loss(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    residual = torch.matmul(pred_rot.transpose(-1, -2), gt_rot)
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(min=-1.0, max=1.0)
    skew_vec = torch.stack(
        [
            residual[..., 2, 1] - residual[..., 1, 2],
            residual[..., 0, 2] - residual[..., 2, 0],
            residual[..., 1, 0] - residual[..., 0, 1],
        ],
        dim=-1,
    )
    sine = 0.5 * torch.linalg.norm(skew_vec, dim=-1)
    return torch.atan2(sine, cosine)


def relative_pose_loss_single(
    pred_c2w: torch.Tensor,
    gt_c2w: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    relative_pose_window: int = 64,
    relative_trans_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_losses = []
    batch_rot_losses = []
    batch_trans_losses = []

    for batch_idx in range(pred_c2w.shape[0]):
        valid_idx = torch.nonzero(valid_frame_mask[batch_idx], as_tuple=False).flatten()
        if valid_idx.numel() < 2:
            continue

        order = torch.arange(valid_idx.numel(), device=valid_idx.device)
        order_i = order[:, None].expand(-1, valid_idx.numel()).reshape(-1)
        order_j = order[None, :].expand(valid_idx.numel(), -1).reshape(-1)
        pair_mask = order_i != order_j
        if relative_pose_window > 0:
            pair_mask &= (order_i - order_j).abs() < relative_pose_window
        if pair_mask.sum() == 0:
            continue

        idx_i = valid_idx[order_i[pair_mask]]
        idx_j = valid_idx[order_j[pair_mask]]

        pred_i_inv = inverse_se3(pred_c2w[batch_idx, idx_i])
        gt_i_inv = inverse_se3(gt_c2w[batch_idx, idx_i])
        pred_rel = torch.matmul(pred_i_inv, pred_c2w[batch_idx, idx_j])
        gt_rel = torch.matmul(gt_i_inv, gt_c2w[batch_idx, idx_j])

        rot_loss = rotation_geodesic_loss(pred_rel[..., :3, :3], gt_rel[..., :3, :3]).mean()
        trans_loss = F.l1_loss(pred_rel[..., :3, 3], gt_rel[..., :3, 3], reduction="mean")
        total = rot_loss + relative_trans_weight * trans_loss

        batch_losses.append(total)
        batch_rot_losses.append(rot_loss)
        batch_trans_losses.append(trans_loss)

    if not batch_losses:
        dummy = pred_c2w.float().mean() * 0.0
        return dummy, dummy, dummy

    return (
        check_and_fix_inf_nan(torch.stack(batch_losses).mean(), "loss_relative_pose"),
        check_and_fix_inf_nan(torch.stack(batch_rot_losses).mean(), "loss_relative_rot"),
        check_and_fix_inf_nan(torch.stack(batch_trans_losses).mean(), "loss_relative_trans"),
    )


def compute_relative_pose_loss(
    pred_dict: Dict[str, torch.Tensor],
    batch_data: Dict[str, torch.Tensor],
    gamma: float = 0.6,
    relative_pose_window: int = 64,
    relative_trans_weight: float = 1.0,
    min_valid_pixels: int = 100,
) -> Dict[str, torch.Tensor]:
    pred_pose_encodings = pred_dict["pose_enc_list"]
    point_masks = batch_data["point_masks"]
    valid_frame_mask = point_masks.sum(dim=(-1, -2)) > min_valid_pixels
    gt_c2w = se3_3x4_to_4x4(w2c_to_c2w_extrinsics(batch_data["extrinsics"]))
    n_stages = len(pred_pose_encodings)

    total_loss = total_rot = total_trans = None
    for stage_idx, pred_pose_stage in enumerate(pred_pose_encodings):
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_c2w = pose_encoding_to_c2w_matrix(pred_pose_stage)
        loss_rel, loss_rot, loss_trans = relative_pose_loss_single(
            pred_c2w,
            gt_c2w,
            valid_frame_mask,
            relative_pose_window=relative_pose_window,
            relative_trans_weight=relative_trans_weight,
        )
        loss_rel = loss_rel * stage_weight
        loss_rot = loss_rot * stage_weight
        loss_trans = loss_trans * stage_weight
        total_loss = loss_rel if total_loss is None else total_loss + loss_rel
        total_rot = loss_rot if total_rot is None else total_rot + loss_rot
        total_trans = loss_trans if total_trans is None else total_trans + loss_trans

    return {
        "loss_relative_pose": total_loss / n_stages,
        "loss_relative_rot": total_rot / n_stages,
        "loss_relative_trans": total_trans / n_stages,
    }


def compute_depth_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    gamma: float = 1.0,
    alpha: float = 0.2,
    gradient_loss_fn: Optional[str] = "grad",
    valid_range: float = 0.98,
    min_valid_pixels: int = 100,
) -> Dict[str, torch.Tensor]:
    pred_depth = predictions["depth"]
    pred_depth_conf = predictions["depth_conf"]
    gt_depth = check_and_fix_inf_nan(batch["depths"], "gt_depth", hard_max=None)[..., None]
    gt_depth_mask = batch["point_masks"].clone()

    # Drop frames with too few valid pixels (matches compute_camera_loss semantics).
    per_frame_valid = gt_depth_mask.sum(dim=(-1, -2)) > min_valid_pixels  # [B, S]
    gt_depth_mask = gt_depth_mask & per_frame_valid[..., None, None]

    if not gt_depth_mask.any():
        dummy = pred_depth.float().mean() * 0.0
        return {
            "loss_conf_depth": dummy,
            "loss_reg_depth": dummy,
            "loss_grad_depth": dummy,
        }

    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_depth,
        gt_depth,
        gt_depth_mask,
        conf=pred_depth_conf,
        gradient_loss_fn=gradient_loss_fn,
        gamma=gamma,
        alpha=alpha,
        valid_range=valid_range,
    )
    return {
        "loss_conf_depth": loss_conf,
        "loss_reg_depth": loss_reg,
        "loss_grad_depth": loss_grad,
    }


def regression_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    conf: torch.Tensor,
    gradient_loss_fn: Optional[str] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
    valid_range: float = 0.98,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len, height, width, channels = pred.shape
    mask = mask.bool()
    conf = conf.clamp(min=1e-6)

    loss_reg_values = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg_values = check_and_fix_inf_nan(loss_reg_values, "loss_reg")

    loss_conf_values = gamma * loss_reg_values * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf_values = check_and_fix_inf_nan(loss_conf_values, "loss_conf")

    if gradient_loss_fn:
        conf_for_grad = conf.reshape(batch_size * seq_len, height, width) if "conf" in gradient_loss_fn else None
        if "normal" in gradient_loss_fn:
            loss_grad = gradient_loss_multi_scale_wrapper(
                pred.reshape(batch_size * seq_len, height, width, channels),
                gt.reshape(batch_size * seq_len, height, width, channels),
                mask.reshape(batch_size * seq_len, height, width),
                scales=3,
                gradient_loss_fn=normal_loss,
                conf=conf_for_grad,
            )
        elif "grad" in gradient_loss_fn:
            loss_grad = gradient_loss_multi_scale_wrapper(
                pred.reshape(batch_size * seq_len, height, width, channels),
                gt.reshape(batch_size * seq_len, height, width, channels),
                mask.reshape(batch_size * seq_len, height, width),
                scales=4,
                gradient_loss_fn=gradient_loss,
                conf=conf_for_grad,
            )
        else:
            loss_grad = pred.float().mean() * 0.0
    else:
        loss_grad = pred.float().mean() * 0.0

    if loss_conf_values.numel() > 0:
        if valid_range > 0:
            loss_conf_values = filter_by_quantile(loss_conf_values, valid_range)
        loss_conf = check_and_fix_inf_nan(loss_conf_values, "loss_conf").mean()
    else:
        loss_conf = pred.float().mean() * 0.0

    if loss_reg_values.numel() > 0:
        if valid_range > 0:
            loss_reg_values = filter_by_quantile(loss_reg_values, valid_range)
        loss_reg = check_and_fix_inf_nan(loss_reg_values, "loss_reg").mean()
    else:
        loss_reg = pred.float().mean() * 0.0

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scales: int = 4,
    gradient_loss_fn=None,
    conf: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    total = prediction.float().mean() * 0.0
    for scale in range(scales):
        step = 2 ** scale
        total = total + gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )
    return total / scales


def gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    conf: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
) -> torch.Tensor:
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    valid_per_batch = torch.sum(mask, dim=(1, 2, 3))
    diff = (prediction - target) * mask

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    grad_x = (grad_x * mask_x).clamp(max=100)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    grad_y = (grad_y * mask_y).clamp(max=100)

    if conf is not None:
        conf = conf.clamp(min=1e-6)[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]
        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    divisor = torch.sum(valid_per_batch)
    if divisor <= 0:
        return prediction.float().mean() * 0.0
    grad = torch.sum(grad_x, dim=(1, 2, 3)) + torch.sum(grad_y, dim=(1, 2, 3))
    return torch.sum(grad) / divisor


def normal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cos_eps: float = 1e-8,
    conf: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    alpha: float = 0.2,
) -> torch.Tensor:
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals, gt_valids = point_map_to_normal(target, mask, eps=cos_eps)
    all_valid = pred_valids & gt_valids

    if torch.sum(all_valid) < 10:
        return prediction.float().mean() * 0.0

    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()
    dot = torch.sum(pred_normals * gt_normals, dim=-1).clamp(-1 + cos_eps, 1 - cos_eps)
    loss = 1 - dot
    if loss.numel() < 10:
        return prediction.float().mean() * 0.0
    loss = check_and_fix_inf_nan(loss, "normal_loss")

    if conf is not None:
        conf = conf.clamp(min=1e-6)[None, ...].expand(4, -1, -1, -1)
        conf = conf[all_valid].clone()
        loss = gamma * loss * conf - alpha * torch.log(conf)
    return loss.mean()


def point_map_to_normal(
    point_map: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.amp.autocast("cuda", enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=0)
        pts = F.pad(
            point_map.float().permute(0, 3, 1, 2),
            (1, 1, 1, 1),
            mode="constant",
            value=0,
        ).permute(0, 2, 3, 1)

        center = pts[:, 1:-1, 1:-1, :]
        up = pts[:, :-2, 1:-1, :]
        left = pts[:, 1:-1, :-2, :]
        down = pts[:, 2:, 1:-1, :]
        right = pts[:, 1:-1, 2:, :]

        up_dir = up - center
        left_dir = left - center
        down_dir = down - center
        right_dir = right - center

        n1 = torch.cross(up_dir, left_dir, dim=-1)
        n2 = torch.cross(left_dir, down_dir, dim=-1)
        n3 = torch.cross(down_dir, right_dir, dim=-1)
        n4 = torch.cross(right_dir, up_dir, dim=-1)

        v1 = padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1]
        v3 = padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1]

        normals = torch.stack([n1, n2, n3, n4], dim=0)
        valids = torch.stack([v1, v2, v3, v4], dim=0)
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)
    return normals, valids


def filter_by_quantile(
    loss_tensor: torch.Tensor,
    valid_range: float,
    min_elements: int = 1000,
    hard_max: float = 100.0,
) -> torch.Tensor:
    if loss_tensor.numel() <= min_elements:
        return loss_tensor
    loss_tensor = loss_tensor.clamp(max=hard_max)
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = torch.minimum(
        quantile_thresh,
        torch.as_tensor(hard_max, device=loss_tensor.device, dtype=loss_tensor.dtype),
    )
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(input_tensor: torch.Tensor, q: float) -> torch.Tensor:
    q = float(q)
    if q < 0 or q > 1:
        raise ValueError(f"q must be in [0, 1], got {q}")
    flat = input_tensor.reshape(-1)
    if flat.numel() == 0:
        return input_tensor.new_tensor(0.0)
    k = round(q * (flat.numel() - 1)) + 1
    return torch.kthvalue(flat, k).values


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, Optional[DataLoader], int, int]:
    scenes = discover_trajectory_dirs(
        args.data_roots,
        max_scenes=args.max_scenes,
        manifest=args.scene_manifest,
        write_manifest=args.write_manifest,
        oss_uri_roots=parse_str_list(args.oss_uri_roots),
        ossutil_bin=args.ossutil_bin,
        ossutil_config=args.ossutil_config,
    )
    if not scenes:
        raise RuntimeError("No Manip trajectories were discovered")

    train_scenes, val_scenes = split_scenes(scenes, args.val_fraction, args.seed)
    view_ids = parse_int_list(args.view_ids)
    camera_names = parse_str_list(args.camera_names)

    common_kwargs = dict(
        clip_len=args.clip_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
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
        m_stride_min=args.m_stride_min,
        m_stride_max=args.m_stride_max,
        s_views_min=args.s_views_min,
        s_views_max=args.s_views_max,
        m_num_views=args.m_num_views,
        m_num_times=args.m_num_times,
        m_views_min=args.m_views_min,
        m_views_max=args.m_views_max,
        mode_weights_initial=parse_mode_weights(args.mode_weights_initial),
        mode_weights_final=parse_mode_weights(args.mode_weights_final),
        mode_warmup_start=args.mode_warmup_start,
        mode_warmup_end=args.mode_warmup_end,
        t_stride_min=args.t_stride_min,
        t_stride_max=args.t_stride_max,
        long5_root_marker=args.long5_root_marker,
        color_jitter_strength=args.color_jitter_strength,
        color_jitter_prob=args.color_jitter_prob,
    )
    train_dataset = ManipTrajectoryDataset(train_scenes, **common_kwargs)
    val_kwargs = dict(common_kwargs, color_jitter_strength=0.0, color_jitter_prob=0.0)
    val_dataset = ManipTrajectoryDataset(val_scenes, **val_kwargs) if val_scenes else None

    # Track sub-dataset boundaries for the (optional) curriculum mixture
    # sampler. The Manip block is always at offset 0; each subsequent external
    # mix-in appends its (name, size) here as it is added below.
    manip_train_size = len(train_dataset)
    external_train: List[Tuple[str, int]] = []

    # When the runtime curriculum is enabled, static *_REPEAT duplication would
    # double-count externals against the sampler's dynamic weights. Force them
    # to 1 here and warn the user.
    if getattr(args, "mixture_curriculum", False):
        for ext_name in ("dl3dv", "scannetpp", "tartanair", "dynamic_replica", "mapfree"):
            repeat_attr = f"{ext_name}_repeat"
            if getattr(args, repeat_attr, 1) != 1:
                print(
                    f"[mixture] --mixture_curriculum is on; overriding "
                    f"--{repeat_attr}={getattr(args, repeat_attr)} -> 1 "
                    f"(the sampler controls mix ratio dynamically)."
                )
                setattr(args, repeat_attr, 1)

    # ----- DL3DV mix-in (sampling follows base3d-clean/datasets/dl3dv.py) -----
    if getattr(args, "dl3dv_root", "") and args.dl3dv_root.strip():
        from lingbot_map.data.dl3dv import DL3DVTrajectoryDataset
        dl3dv_num_views = args.dl3dv_num_views if args.dl3dv_num_views > 0 else args.max_sample_frames
        dl3dv_min_views = args.dl3dv_min_views if args.dl3dv_min_views > 0 else args.min_sample_frames
        dl3dv_min_views = min(dl3dv_min_views, dl3dv_num_views)
        dl3dv_common = dict(
            root=args.dl3dv_root.strip(),
            num_views=dl3dv_num_views,
            min_views=dl3dv_min_views,
            image_size=args.image_size,
            patch_size=args.patch_size,
            preprocess_mode=args.preprocess_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            samples_per_scene=args.samples_per_scene,
            min_interval=args.dl3dv_min_interval,
            max_interval=args.dl3dv_max_interval,
            video_prob=args.dl3dv_video_prob,
            fix_interval_prob=args.dl3dv_fix_interval_prob,
            block_shuffle=args.dl3dv_block_shuffle,
        )
        dl3dv_train = DL3DVTrajectoryDataset(
            split="train",
            color_jitter_strength=args.color_jitter_strength,
            color_jitter_prob=args.color_jitter_prob,
            verbose=True,
            **dl3dv_common,
        )
        if args.dl3dv_repeat > 1:
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset] + [dl3dv_train] * int(args.dl3dv_repeat)
            )
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dl3dv_train])
        external_train.append(("dl3dv", len(dl3dv_train)))
        print(
            f"[data] DL3DV mixed into train: {len(dl3dv_train)} samples"
            f" (x{args.dl3dv_repeat}), num_views=[{dl3dv_min_views}, {dl3dv_num_views}]"
        )
        if getattr(args, "dl3dv_val", False):
            dl3dv_val = DL3DVTrajectoryDataset(
                split="test",
                color_jitter_strength=0.0,
                color_jitter_prob=0.0,
                verbose=True,
                **dl3dv_common,
            )
            if val_dataset is None:
                val_dataset = dl3dv_val
            else:
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, dl3dv_val])
            print(f"[data] DL3DV mixed into val: {len(dl3dv_val)} samples")

    # ----- ScanNet++ v2 mix-in (sampling follows base3d-clean/datasets/scannet.py) -----
    if getattr(args, "scannetpp_root", "") and args.scannetpp_root.strip():
        from lingbot_map.data.scannetpp import ScanNetppTrajectoryDataset
        scannetpp_num_views = (
            args.scannetpp_num_views if args.scannetpp_num_views > 0 else args.max_sample_frames
        )
        scannetpp_min_views = (
            args.scannetpp_min_views if args.scannetpp_min_views > 0 else args.min_sample_frames
        )
        scannetpp_min_views = min(scannetpp_min_views, scannetpp_num_views)
        scannetpp_common = dict(
            root=args.scannetpp_root.strip(),
            num_views=scannetpp_num_views,
            min_views=scannetpp_min_views,
            image_size=args.image_size,
            patch_size=args.patch_size,
            preprocess_mode=args.preprocess_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            samples_per_scene=args.samples_per_scene,
            min_interval=args.scannetpp_min_interval,
            max_interval=args.scannetpp_max_interval,
            video_prob=args.scannetpp_video_prob,
            fix_interval_prob=args.scannetpp_fix_interval_prob,
            block_shuffle=args.scannetpp_block_shuffle,
        )
        scannetpp_train = ScanNetppTrajectoryDataset(
            split="train",
            color_jitter_strength=args.color_jitter_strength,
            color_jitter_prob=args.color_jitter_prob,
            verbose=True,
            **scannetpp_common,
        )
        if args.scannetpp_repeat > 1:
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset] + [scannetpp_train] * int(args.scannetpp_repeat)
            )
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, scannetpp_train])
        external_train.append(("scannetpp", len(scannetpp_train)))
        print(
            f"[data] ScanNet++ mixed into train: {len(scannetpp_train)} samples"
            f" (x{args.scannetpp_repeat}), num_views=[{scannetpp_min_views}, {scannetpp_num_views}]"
        )
        if getattr(args, "scannetpp_val", False):
            scannetpp_val = ScanNetppTrajectoryDataset(
                split="test",
                color_jitter_strength=0.0,
                color_jitter_prob=0.0,
                verbose=True,
                **scannetpp_common,
            )
            if val_dataset is None:
                val_dataset = scannetpp_val
            else:
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, scannetpp_val])
            print(f"[data] ScanNet++ mixed into val: {len(scannetpp_val)} samples")

    # ----- TartanAir mix-in (sampling follows base3d-clean/datasets/tartanair.py) -----
    if getattr(args, "tartanair_root", "") and args.tartanair_root.strip():
        from lingbot_map.data.tartanair import TartanAirTrajectoryDataset
        tartanair_num_views = (
            args.tartanair_num_views if args.tartanair_num_views > 0 else args.max_sample_frames
        )
        tartanair_min_views = (
            args.tartanair_min_views if args.tartanair_min_views > 0 else args.min_sample_frames
        )
        tartanair_min_views = min(tartanair_min_views, tartanair_num_views)
        tartanair_common = dict(
            root=args.tartanair_root.strip(),
            num_views=tartanair_num_views,
            min_views=tartanair_min_views,
            image_size=args.image_size,
            patch_size=args.patch_size,
            preprocess_mode=args.preprocess_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            samples_per_scene=args.samples_per_scene,
            min_interval=args.tartanair_min_interval,
            max_interval=args.tartanair_max_interval,
            video_prob=args.tartanair_video_prob,
            fix_interval_prob=args.tartanair_fix_interval_prob,
            block_shuffle=args.tartanair_block_shuffle,
        )
        tartanair_train = TartanAirTrajectoryDataset(
            split="train",
            color_jitter_strength=args.color_jitter_strength,
            color_jitter_prob=args.color_jitter_prob,
            verbose=True,
            **tartanair_common,
        )
        if args.tartanair_repeat > 1:
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset] + [tartanair_train] * int(args.tartanair_repeat)
            )
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, tartanair_train])
        external_train.append(("tartanair", len(tartanair_train)))
        print(
            f"[data] TartanAir mixed into train: {len(tartanair_train)} samples"
            f" (x{args.tartanair_repeat}), num_views=[{tartanair_min_views}, {tartanair_num_views}]"
        )
        if getattr(args, "tartanair_val", False):
            tartanair_val = TartanAirTrajectoryDataset(
                split="test",
                color_jitter_strength=0.0,
                color_jitter_prob=0.0,
                verbose=True,
                **tartanair_common,
            )
            if val_dataset is None:
                val_dataset = tartanair_val
            else:
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, tartanair_val])
            print(f"[data] TartanAir mixed into val: {len(tartanair_val)} samples")

    # ----- DynamicReplica mix-in (sampling follows base3d-clean/datasets/dynamic_replica.py) -----
    if getattr(args, "dynamic_replica_root", "") and args.dynamic_replica_root.strip():
        from lingbot_map.data.dynamic_replica import DynamicReplicaTrajectoryDataset
        dynamic_replica_num_views = (
            args.dynamic_replica_num_views if args.dynamic_replica_num_views > 0 else args.max_sample_frames
        )
        dynamic_replica_min_views = (
            args.dynamic_replica_min_views if args.dynamic_replica_min_views > 0 else args.min_sample_frames
        )
        dynamic_replica_min_views = min(dynamic_replica_min_views, dynamic_replica_num_views)
        dynamic_replica_common = dict(
            root=args.dynamic_replica_root.strip(),
            num_views=dynamic_replica_num_views,
            min_views=dynamic_replica_min_views,
            image_size=args.image_size,
            patch_size=args.patch_size,
            preprocess_mode=args.preprocess_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            samples_per_scene=args.samples_per_scene,
            min_interval=args.dynamic_replica_min_interval,
            max_interval=args.dynamic_replica_max_interval,
            video_prob=args.dynamic_replica_video_prob,
            fix_interval_prob=args.dynamic_replica_fix_interval_prob,
            block_shuffle=args.dynamic_replica_block_shuffle,
        )
        dynamic_replica_train = DynamicReplicaTrajectoryDataset(
            split="train",
            color_jitter_strength=args.color_jitter_strength,
            color_jitter_prob=args.color_jitter_prob,
            verbose=True,
            **dynamic_replica_common,
        )
        if args.dynamic_replica_repeat > 1:
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset] + [dynamic_replica_train] * int(args.dynamic_replica_repeat)
            )
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dynamic_replica_train])
        external_train.append(("dynamic_replica", len(dynamic_replica_train)))
        print(
            f"[data] DynamicReplica mixed into train: {len(dynamic_replica_train)} samples"
            f" (x{args.dynamic_replica_repeat}), num_views=[{dynamic_replica_min_views}, {dynamic_replica_num_views}]"
        )
        if getattr(args, "dynamic_replica_val", False):
            dynamic_replica_val = DynamicReplicaTrajectoryDataset(
                split="valid",
                color_jitter_strength=0.0,
                color_jitter_prob=0.0,
                verbose=True,
                **dynamic_replica_common,
            )
            if val_dataset is None:
                val_dataset = dynamic_replica_val
            else:
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, dynamic_replica_val])
            print(f"[data] DynamicReplica mixed into val: {len(dynamic_replica_val)} samples")

    # ----- MapFree mix-in (sampling follows base3d-clean/datasets/mapfree.py) -----
    if getattr(args, "mapfree_root", "") and args.mapfree_root.strip():
        from lingbot_map.data.mapfree import MapfreeTrajectoryDataset
        mapfree_num_views = (
            args.mapfree_num_views if args.mapfree_num_views > 0 else args.max_sample_frames
        )
        mapfree_min_views = (
            args.mapfree_min_views if args.mapfree_min_views > 0 else args.min_sample_frames
        )
        mapfree_min_views = min(mapfree_min_views, mapfree_num_views)
        mapfree_common = dict(
            root=args.mapfree_root.strip(),
            num_views=mapfree_num_views,
            min_views=mapfree_min_views,
            image_size=args.image_size,
            patch_size=args.patch_size,
            preprocess_mode=args.preprocess_mode,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            samples_per_scene=args.samples_per_scene,
            min_interval=args.mapfree_min_interval,
            max_interval=args.mapfree_max_interval,
            video_prob=args.mapfree_video_prob,
            fix_interval_prob=args.mapfree_fix_interval_prob,
            block_shuffle=args.mapfree_block_shuffle,
        )
        mapfree_train = MapfreeTrajectoryDataset(
            split="train",
            color_jitter_strength=args.color_jitter_strength,
            color_jitter_prob=args.color_jitter_prob,
            verbose=True,
            **mapfree_common,
        )
        if args.mapfree_repeat > 1:
            train_dataset = torch.utils.data.ConcatDataset(
                [train_dataset] + [mapfree_train] * int(args.mapfree_repeat)
            )
        else:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, mapfree_train])
        external_train.append(("mapfree", len(mapfree_train)))
        print(
            f"[data] MapFree mixed into train: {len(mapfree_train)} samples"
            f" (x{args.mapfree_repeat}), num_views=[{mapfree_min_views}, {mapfree_num_views}]"
        )
        if getattr(args, "mapfree_val", False):
            mapfree_val = MapfreeTrajectoryDataset(
                split="train",  # mapfree has no separate val split; reuse train universe
                color_jitter_strength=0.0,
                color_jitter_prob=0.0,
                verbose=True,
                **mapfree_common,
            )
            if val_dataset is None:
                val_dataset = mapfree_val
            else:
                val_dataset = torch.utils.data.ConcatDataset([val_dataset, mapfree_val])
            print(f"[data] MapFree mixed into val: {len(mapfree_val)} samples")

    # ----- Optional: step-aware curriculum mixture sampler -----
    # Active only when --mixture_curriculum is set AND at least one external
    # mix-in is enabled. Otherwise we fall back to the original
    # ``shuffle=True`` (uniform over all ConcatDataset indices) behavior.
    mixture_sampler = None
    if getattr(args, "mixture_curriculum", False):
        if not external_train:
            print(
                "[mixture] --mixture_curriculum requested but no external "
                "mix-in is enabled; falling back to plain shuffle."
            )
        else:
            from lingbot_map.data.mixture_sampler import CurriculumMixtureSampler
            ext_names = [n for n, _ in external_train]
            ext_sizes = [s for _, s in external_train]
            # epoch_length = LIMIT_TRAIN_BATCHES with batch_size=1, so the
            # sampler hands the DataLoader exactly one epoch worth of indices.
            epoch_length = max(1, int(args.limit_train_batches)) if args.limit_train_batches > 0 else len(train_dataset)
            mixture_sampler = CurriculumMixtureSampler(
                manip_size=manip_train_size,
                external_sizes=ext_sizes,
                external_names=ext_names,
                epoch_length=epoch_length,
                p_manip_start=args.mixture_p_manip_start,
                p_manip_end=args.mixture_p_manip_end,
                warmup_start=args.mixture_warmup_start,
                warmup_end=args.mixture_warmup_end,
                seed=args.seed,
            )
            initial_weights = mixture_sampler.get_dataset_weights(0)
            print(
                f"[mixture] curriculum sampler active: manip={manip_train_size}, "
                f"externals={dict(external_train)}, epoch_length={epoch_length}"
            )
            print(
                f"[mixture] schedule: p_manip {args.mixture_p_manip_start:.3f} "
                f"-> {args.mixture_p_manip_end:.3f} over steps "
                f"[{args.mixture_warmup_start}, {args.mixture_warmup_end}]"
            )
            print(f"[mixture] initial dataset weights (step=0): {initial_weights}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(mixture_sampler is None),
        sampler=mixture_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_rgbd_sequences,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(1, min(args.num_workers, 4)) if args.num_workers > 0 else 0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_rgbd_sequences,
        )
    return train_loader, val_loader, len(train_scenes), len(val_scenes)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=max(args.max_frame_num, args.clip_len, args.max_sample_frames),
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        enable_point=False,
        enable_local_point=False,
        use_gradient_checkpoint=not args.no_gradient_checkpoint,
    )

    if args.model_path:
        load_state_dict_flexible(
            model,
            args.model_path,
            strict=args.strict_load,
            map_location="cpu",
        )

    use_depth_ckpt = not args.no_depth_activation_checkpoint
    if getattr(model, "depth_head", None) is not None and hasattr(model.depth_head, "use_activation_checkpoint"):
        model.depth_head.use_activation_checkpoint = use_depth_ckpt
        print(f"[model] depth activation checkpoint: {int(use_depth_ckpt)}")

    freeze_tokens = []
    if args.freeze_dino_patch_embed:
        freeze_tokens.append("aggregator.patch_embed")
    if args.freeze_aggregator:
        freeze_tokens.append("aggregator")
    if args.freeze_camera:
        freeze_tokens.append("camera_head")
    if args.freeze_depth:
        freeze_tokens.append("depth_head")
    if args.freeze_point:
        freeze_tokens.append("point_head")
    frozen_params = freeze_by_name(model, freeze_tokens)
    if freeze_tokens:
        print(
            f"[model] frozen parameters: {frozen_params / 1e6:.2f}M "
            f"matching {', '.join(freeze_tokens)}"
        )

    return model.to(device)


def build_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = max(1, steps_per_epoch * args.epochs // max(1, args.accum_steps))

    if args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(round(total_steps * args.warmup_ratio))
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))

    min_lr_ratio = 0.0
    if args.lr > 0:
        min_lr_ratio = max(0.0, min(1.0, args.min_lr / args.lr))

    def lr_lambda(step: int) -> float:
        step = max(0, min(step, total_steps))
        if warmup_steps > 0 and step < warmup_steps:
            progress = float(step) / float(max(1, warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * progress
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    effective_ratio = warmup_steps / float(max(1, total_steps))
    print(
        f"[optim] AdamW lr={args.lr:.3e} min_lr={args.min_lr:.3e} "
        f"weight_decay={args.weight_decay:.3g} total_steps={total_steps} "
        f"warmup_steps={warmup_steps} warmup_ratio={effective_ratio:.4f} "
        f"cosine_steps={max(0, total_steps - warmup_steps)}"
    )
    return optimizer, scheduler


def loss_to_float_dict(losses: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {
        key: float(value.detach().float().cpu())
        for key, value in losses.items()
        if torch.is_tensor(value) and value.ndim == 0
    }


def _use_color() -> bool:
    val = os.environ.get("FORCE_COLOR", "").lower()
    if val in ("1", "true", "yes", "on"):
        return True
    val = os.environ.get("NO_COLOR", "").lower()
    if val in ("1", "true", "yes", "on"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


_USE_COLOR = _use_color()


def _c(code: str, text: object) -> str:
    if not _USE_COLOR:
        return str(text)
    return f"\033[{code}m{text}\033[0m"


def _dim(t):     return _c("2", t)
def _bold(t):    return _c("1", t)
def _cyan(t):    return _c("36", t)
def _green(t):   return _c("32", t)
def _yellow(t):  return _c("33", t)
def _magenta(t): return _c("35", t)
def _red(t):     return _c("31", t)
def _blue(t):    return _c("34", t)


def _fmt_duration(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "--"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"{d}d{h}h"


_SEP = lambda: _dim(" │ ")


def format_metrics(metrics: Dict[str, float], keys: Sequence[str]) -> str:
    chunks = []
    for key in keys:
        if key in metrics:
            chunks.append(f"{key}={metrics[key]:.4f}")
    return " ".join(chunks)


_LOSS_COMPONENT_LABELS: Tuple[Tuple[str, str], ...] = (
    ("loss_camera",         "cam"),
    ("loss_conf_depth",     "depth"),
    ("loss_reg_depth",      "reg"),
    ("loss_grad_depth",     "grad"),
    ("loss_relative_pose",  "rpose"),
    ("loss_T",              "T"),
    ("loss_R",              "R"),
    ("loss_FL",             "FL"),
)


def format_loss_components(averaged: Dict[str, float]) -> str:
    parts = []
    for key, label in _LOSS_COMPONENT_LABELS:
        if key in averaged:
            parts.append(f"{_dim(label)} {averaged[key]:.3f}")
    return " ".join(parts)


_MANIP_MODES = ("W", "T", "S", "M")


def _extract_sample_mode(batch: Dict[str, object]) -> str:
    mode = batch.get("sample_mode") if isinstance(batch, dict) else None
    if isinstance(mode, list) and mode:
        return str(mode[0])
    if isinstance(mode, str):
        return mode
    return ""


def format_batch_source(batch: Dict[str, object]) -> str:
    mode = _extract_sample_mode(batch)
    if not mode:
        return "?"
    return "manip" if mode in _MANIP_MODES else mode


def format_batch_mode(batch: Dict[str, object]) -> str:
    mode = _extract_sample_mode(batch)
    return mode if mode in _MANIP_MODES else ""


def format_batch_input(batch: Dict[str, object]) -> str:
    images = batch.get("images")
    mode = batch.get("sample_mode")
    mode_str = ""
    if isinstance(mode, list) and mode:
        mode_str = f" mode={mode[0]}" if len(mode) == 1 else f" modes={mode}"
    if torch.is_tensor(images):
        return f"images={tuple(images.shape)}{mode_str}"
    return f"images=<unavailable>{mode_str}"



CURRENT_OOM_CONTEXT: Dict[str, object] = {}
_OOM_HOOK_INSTALLED = False
_OOM_ALLOC_RE = re.compile(r"Tried to allocate\s+([0-9.]+)\s+([KMGTP]?iB|bytes?)")

def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"

def is_cuda_oom_exception(exc: BaseException) -> bool:
    oom_type = getattr(torch, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    cuda_oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_type is not None and isinstance(exc, cuda_oom_type):
        return True
    return "CUDA out of memory" in str(exc)

def format_cuda_oom_report(exc: BaseException) -> str:
    match = _OOM_ALLOC_RE.search(str(exc))
    requested = f"{match.group(1)} {match.group(2)}" if match else "unknown"
    lines = ["[cuda-oom] CUDA out of memory report", f"  requested: {requested}"]
    if CURRENT_OOM_CONTEXT:
        lines.append("  context: " + " ".join(f"{k}={v}" for k, v in CURRENT_OOM_CONTEXT.items()))
    if torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            free_b, total_b = torch.cuda.mem_get_info(dev)
            allocated = torch.cuda.memory_allocated(dev)
            reserved = torch.cuda.memory_reserved(dev)
            lines.append(f"  cuda: device={dev} free={format_bytes(free_b)} total={format_bytes(total_b)}")
            lines.append(f"  pytorch: allocated={format_bytes(allocated)} reserved={format_bytes(reserved)} reserved_unallocated={format_bytes(max(0, reserved - allocated))} max_allocated={format_bytes(torch.cuda.max_memory_allocated(dev))} max_reserved={format_bytes(torch.cuda.max_memory_reserved(dev))}")
        except Exception as mem_exc:
            lines.append(f"  cuda: failed to query memory ({mem_exc})")
    else:
        lines.append("  cuda: unavailable")
    return "\n".join(lines)

def install_cuda_oom_hook() -> None:
    global _OOM_HOOK_INSTALLED
    if _OOM_HOOK_INSTALLED:
        return
    previous_hook = sys.excepthook
    def hook(exc_type, exc, traceback):
        if is_cuda_oom_exception(exc):
            try:
                tqdm.write(format_cuda_oom_report(exc))
            except Exception:
                print(format_cuda_oom_report(exc), file=sys.stderr)
        previous_hook(exc_type, exc, traceback)
    sys.excepthook = hook
    _OOM_HOOK_INSTALLED = True

def create_tensorboard_writer(args: argparse.Namespace, output_dir: Path):
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:
        print(f"[tensorboard][warn] unavailable ({exc}); scalar visualization disabled")
        return None

    log_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir), flush_secs=args.tensorboard_flush_secs)
    writer.add_text("config/args", json.dumps(vars(args), indent=2, sort_keys=True), 0)
    print(f"[tensorboard] logs: {log_dir}")
    return writer


def write_tensorboard_scalars(writer, prefix: str, metrics: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in sorted(metrics.items()):
        value = float(value)
        if math.isfinite(value):
            writer.add_scalar(f"{prefix}/{key}", value, step)


def tensorboard_input_metrics(batch: Dict[str, object]) -> Dict[str, float]:
    images = batch.get("images")
    if not torch.is_tensor(images) or images.ndim < 5:
        return {}
    return {
        "batch_size": float(images.shape[0]),
        "sequence_length": float(images.shape[1]),
        "input_channels": float(images.shape[2]),
        "input_height": float(images.shape[-2]),
        "input_width": float(images.shape[-1]),
    }


_ALL_SAMPLE_MODES = ("S", "W", "M", "legacy")


def per_mode_loss_metrics(batch: Dict[str, object], scalar_losses: Dict[str, float]) -> Dict[str, float]:
    """Emit `<mode>/loss_*` scalars for the modes present in this batch.

    With batch_size==1 this is just the single sample's mode; for larger
    batches we emit metrics for each mode that appears (averaged across the
    samples that share that mode is not possible without per-sample losses,
    so we record the batch-level loss tagged by every mode in the batch).
    """
    modes = batch.get("sample_mode")
    if not isinstance(modes, list) or not modes:
        return {}
    out: Dict[str, float] = {}
    seen: set = set()
    for mode in modes:
        if mode in seen:
            continue
        seen.add(mode)
        for key, value in scalar_losses.items():
            out[f"mode_{mode}/{key}"] = float(value)
    # Also record the active modes as one-hot counters so we can see how the
    # curriculum scheduler distributes samples over time.
    for mode in _ALL_SAMPLE_MODES:
        out[f"mode_count/{mode}"] = float(1.0 if mode in seen else 0.0)
    return out


@torch.no_grad()
def run_validation(
    model: nn.Module,
    criterion: VGGTStyleLoss,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    limit = args.limit_val_batches if args.limit_val_batches > 0 else len(val_loader)

    iterator = iter(val_loader)
    for _ in range(limit):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if args.canonicalize_first_frame:
            batch = canonicalize_to_first_frame(batch)
        if args.normalize_scene:
            batch = normalize_scene_batch(
                batch,
                num_anchor_frames=min(args.num_scale_frames, int(batch["images"].shape[1])),
            )
        batch = to_device(batch, device)

        model.clean_kv_cache()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(
                batch["images"],
                num_frame_for_scale=min(args.num_scale_frames, int(batch["images"].shape[1])),
                num_frame_per_block=args.num_frame_per_block,
                depth_frames_chunk_size=args.depth_frames_chunk_size,
                causal_inference=True,
            )
            losses = criterion(predictions, batch)
        model.clean_kv_cache()

        scalar_losses = loss_to_float_dict(losses)
        for key, value in scalar_losses.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1

    if count == 0:
        return {}
    return {key: value / count for key, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    install_cuda_oom_hook()
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)

    train_loader, val_loader, train_scene_count, val_scene_count = build_dataloaders(args)
    effective_train_batches = len(train_loader)
    if args.limit_train_batches > 0:
        effective_train_batches = min(effective_train_batches, args.limit_train_batches)
    print(f"[data] split: train_scenes={train_scene_count}, val_scenes={val_scene_count}")
    print(f"[data] train batches: total={len(train_loader)}, running={effective_train_batches}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model = build_model(args, device)
    trainable, total = count_trainable_params(model)
    print(f"[model] trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

    steps_per_epoch = effective_train_batches
    optimizer, scheduler = build_optimizer_and_scheduler(model, args, steps_per_epoch)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)

        # Refuse to load optimizer state if the trainable-parameter fingerprint
        # has shifted (e.g. user toggled --freeze_* between runs). Param groups
        # are positional, so silently loading state into mismatched groups would
        # corrupt momentum/variance buffers without any visible error.
        saved_fp = ckpt.get("trainable_fingerprint")
        current_fp = _trainable_param_fingerprint(model)
        if saved_fp is not None and saved_fp != current_fp:
            saved_set = {n for n, _ in saved_fp}
            current_set = {n for n, _ in current_fp}
            added = sorted(current_set - saved_set)[:8]
            removed = sorted(saved_set - current_set)[:8]
            raise RuntimeError(
                "Resume aborted: trainable-parameter set has changed since the "
                "checkpoint was saved (likely due to a different --freeze_* "
                "configuration). Optimizer state would be misapplied.\n"
                f"  added (now trainable): {added}\n"
                f"  removed (now frozen) : {removed}"
            )
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])

        rng_state = ckpt.get("rng")
        if rng_state is not None:
            try:
                random.setstate(rng_state["python"])
                np.random.set_state(rng_state["numpy"])
                torch.set_rng_state(rng_state["torch"])
                if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                    torch.cuda.set_rng_state_all(rng_state["cuda"])
            except Exception as exc:
                print(f"[resume] warning: failed to restore RNG state ({exc})")

        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        print(f"[resume] {args.resume} epoch={start_epoch}, step={global_step}")

    criterion = VGGTStyleLoss(
        camera_weight=args.camera_weight,
        depth_weight=args.depth_weight,
        relative_pose_weight=args.relative_pose_weight,
        camera_loss_type=args.camera_loss_type,
        camera_gamma=args.camera_gamma,
        weight_trans=args.weight_trans,
        weight_rot=args.weight_rot,
        weight_focal=args.weight_focal,
        relative_trans_weight=args.relative_trans_weight,
        relative_pose_window=args.relative_pose_window,
        depth_gradient_loss_fn=args.depth_gradient_loss_fn,
        loss_gamma=args.loss_gamma,
        loss_alpha=args.loss_alpha,
        valid_range=args.valid_range,
        min_valid_pixels=args.min_valid_pixels,
    ).to(device)

    metric_keys = [
        "loss_objective",
        "loss_camera",
        "loss_T",
        "loss_R",
        "loss_FL",
        "loss_relative_pose",
        "loss_relative_rot",
        "loss_relative_trans",
        "loss_conf_depth",
        "loss_reg_depth",
        "loss_grad_depth",
    ]

    tb_writer = create_tensorboard_writer(args, output_dir)

    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    total_train_steps = args.max_steps if args.max_steps > 0 else max(
        1, math.ceil(steps_per_epoch * args.epochs / max(1, args.accum_steps))
    )
    step_progress = tqdm(
        total=total_train_steps,
        initial=min(global_step, total_train_steps),
        desc=_cyan(_bold("train")),
        unit="step",
        dynamic_ncols=True,
        bar_format=(
            "{desc} {percentage:3.0f}%|{bar}| "
            f"{_dim('{n_fmt}/{total_fmt}')} "
            f"[{_dim('{elapsed}<{remaining}, {rate_fmt}')}] "
            "{postfix}"
        ),
        colour="cyan",
        smoothing=0.1,
    )
    last_input_print_step = -1
    last_logged_loss: Optional[float] = None
    initial_global_step = global_step
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running: Dict[str, float] = {}
        running_count = 0

        limit = steps_per_epoch
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= limit:
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            if args.canonicalize_first_frame:
                batch = canonicalize_to_first_frame(batch)
            if args.normalize_scene:
                batch = normalize_scene_batch(
                    batch,
                    num_anchor_frames=min(args.num_scale_frames, int(batch["images"].shape[1])),
                )
            batch = to_device(batch, device)
            input_desc = format_batch_input(batch)
            CURRENT_OOM_CONTEXT.clear()
            CURRENT_OOM_CONTEXT.update({"phase": "prepare", "step": global_step, "epoch": epoch + 1, "batch": f"{batch_idx + 1}/{limit}", "input": input_desc, "depth_chunk": args.depth_frames_chunk_size})
            if (
                args.print_input_every > 0
                and global_step != last_input_print_step
                and (global_step == 0 or global_step % args.print_input_every == 0)
            ):
                tqdm.write(
                    f"[input] step={global_step} epoch={epoch + 1} "
                    f"batch={batch_idx + 1}/{limit} {input_desc}"
                )
                last_input_print_step = global_step

            model.clean_kv_cache()
            CURRENT_OOM_CONTEXT["phase"] = "forward+loss"
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                predictions = model(
                    batch["images"],
                    num_frame_for_scale=min(args.num_scale_frames, int(batch["images"].shape[1])),
                    num_frame_per_block=args.num_frame_per_block,
                    depth_frames_chunk_size=args.depth_frames_chunk_size,
                    causal_inference=True,
                )
                losses = criterion(predictions, batch)
                loss = losses["objective"] / max(1, args.accum_steps)

            if not torch.isfinite(loss.detach()):
                tqdm.write("[warn] non-finite loss, skipping batch")
                model.clean_kv_cache()
                optimizer.zero_grad(set_to_none=True)
                continue

            CURRENT_OOM_CONTEXT["phase"] = "backward"
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            model.clean_kv_cache()

            scalar_losses = loss_to_float_dict(losses)
            for key, value in scalar_losses.items():
                running[key] = running.get(key, 0.0) + value
            running_count += 1

            should_step = (batch_idx + 1) % args.accum_steps == 0
            if should_step:
                CURRENT_OOM_CONTEXT["phase"] = "optimizer_step"
                if args.grad_clip_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        args.grad_clip_norm,
                    )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if (
                    device.type == "cuda"
                    and args.empty_cache_every > 0
                    and global_step % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()
                step_progress.update(1)
                current_loss = scalar_losses.get("loss_objective", 0.0)
                lr = optimizer.param_groups[0]["lr"]
                seq_len = int(batch["images"].shape[1]) if torch.is_tensor(batch.get("images")) else 0
                _sampler = getattr(train_loader, "sampler", None)
                mix_weights = None
                manip_ratio = None
                if _sampler is not None:
                    if hasattr(_sampler, "get_p_manip"):
                        manip_ratio = float(_sampler.get_p_manip(global_step))
                    if hasattr(_sampler, "get_dataset_weights"):
                        mix_weights = _sampler.get_dataset_weights(global_step)
                        if manip_ratio is None and "manip" in mix_weights:
                            manip_ratio = float(mix_weights["manip"])
                postfix = (
                    f"{_dim('loss')} {_yellow(f'{current_loss:.4f}')}"
                    f"  {_dim('lr')} {_green(f'{lr:.2e}')}"
                    f"  {_dim('seq')} {seq_len}"
                    f"  {_dim('ep')} {epoch + 1}"
                )
                if manip_ratio is not None:
                    postfix += f"  {_dim('pManipLong')} {_magenta(f'{manip_ratio:.3f}')}"
                step_progress.set_postfix_str(postfix, refresh=False)
                tb_metrics = dict(scalar_losses)
                tb_metrics.update(tensorboard_input_metrics(batch))
                tb_metrics.update(per_mode_loss_metrics(batch, scalar_losses))
                tb_metrics["lr"] = lr
                tb_metrics["epoch"] = float(epoch + 1)
                # Push the current step into every dataset that exposes a
                # set_global_step hook. With ConcatDataset wrapping Manip +
                # externals, ``train_loader.dataset`` itself has no such hook;
                # walk its .datasets list and call the hook on each leaf.
                _propagate_global_step(train_loader.dataset, global_step)
                # Curriculum mixture sampler (if active) also tracks step so
                # its weight schedule advances. Cheap no-op when not active.
                if getattr(train_loader, "sampler", None) is not None and hasattr(
                    train_loader.sampler, "set_global_step"
                ):
                    train_loader.sampler.set_global_step(global_step)
                # Keep validation in sync with the training curriculum so
                # val mode-mix tracks training rather than freezing at initial.
                if val_loader is not None:
                    _propagate_global_step(val_loader.dataset, global_step)
                # Surface the active S/W/M mode-mix weights (Manip-internal).
                manip_ds = _find_dataset_with_attr(train_loader.dataset, "_compute_mode_weights")
                if manip_ds is not None:
                    schedule_weights = manip_ds._compute_mode_weights(global_step)
                    for mode_key, mode_weight in schedule_weights.items():
                        tb_metrics[f"mode_weight/{mode_key}"] = float(mode_weight)
                # Surface the active per-dataset mixture weights too — this is
                # the cross-source schedule (Manip vs DL3DV vs ScanNet++ ...).
                if mix_weights is not None:
                    for src, w in mix_weights.items():
                        tb_metrics[f"mix_weight/{src}"] = float(w)
                write_tensorboard_scalars(tb_writer, "train", tb_metrics, global_step)
                if (
                    tb_writer is not None
                    and args.tensorboard_flush_every > 0
                    and global_step % args.tensorboard_flush_every == 0
                ):
                    tb_writer.flush()

                if args.log_every > 0 and global_step % args.log_every == 0 and running_count > 0:
                    averaged = {key: value / running_count for key, value in running.items()}
                    elapsed = time.time() - start_time
                    steps_in_run = max(1, global_step - initial_global_step)
                    remaining_steps = max(0, total_train_steps - global_step)
                    eta = elapsed * remaining_steps / steps_in_run
                    lr = optimizer.param_groups[0]["lr"]
                    avg_loss = averaged.get("loss_objective", 0.0)
                    if last_logged_loss is None:
                        delta_str = _dim("Δ----")
                    else:
                        delta = avg_loss - last_logged_loss
                        sign = "↓" if delta < 0 else ("↑" if delta > 0 else "·")
                        color = _green if delta < 0 else (_red if delta > 0 else _dim)
                        delta_str = color(f"{sign}{abs(delta):.4f}")
                    last_logged_loss = avg_loss
                    src = format_batch_source(batch)
                    mode = format_batch_mode(batch)
                    seq_len = int(batch["images"].shape[1]) if torch.is_tensor(batch.get("images")) else 0
                    seq_field = f"{_dim('seq')} {seq_len}"
                    if mode:
                        seq_field += f" {_magenta(mode)}"
                    ts = time.strftime("%H:%M:%S")
                    sep = _SEP()
                    parts = [
                        f"{_dim('[' + ts + ']')}",
                        f"{_bold('step')} {_cyan(f'{global_step}')}{_dim('/' + str(total_train_steps))}",
                        f"{_dim('ep')} {epoch + 1}",
                        f"{_dim('loss')} {_yellow(f'{avg_loss:.4f}')} {delta_str}",
                    ]
                    components = format_loss_components(averaged)
                    if components:
                        parts.append(components)
                    parts.extend([
                        f"{_dim('lr')} {_green(f'{lr:.2e}')}",
                        seq_field,
                        f"{_dim('pManipLong')} {_magenta(f'{manip_ratio:.3f}')}" if manip_ratio is not None else "",
                        f"{_dim('src')} {_blue(src)}",
                        f"{_dim('t')} {_fmt_duration(elapsed)}{_dim('/eta')} {_fmt_duration(eta)}",
                    ])
                    tqdm.write(sep.join(part for part in parts if part))
                    running.clear()
                    running_count = 0

                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(
                        output_dir / f"checkpoint_step_{global_step:08d}.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        args,
                        scaler=scaler,
                    )

                if (
                    val_loader is not None
                    and args.val_every > 0
                    and global_step % args.val_every == 0
                ):
                    val_metrics = run_validation(
                        model,
                        criterion,
                        val_loader,
                        args,
                        device,
                        amp_enabled,
                        amp_dtype,
                    )
                    if val_metrics:
                        ts = time.strftime("%H:%M:%S")
                        sep = _SEP()
                        val_loss = val_metrics.get("loss_objective", 0.0)
                        parts = [
                            f"{_dim('[' + ts + ']')} {_magenta(_bold('VAL'))}",
                            f"{_bold('step')} {_cyan(f'{global_step}')}",
                            f"{_dim('loss')} {_yellow(f'{val_loss:.4f}')}",
                        ]
                        components = format_loss_components(val_metrics)
                        if components:
                            parts.append(components)
                        tqdm.write(sep.join(parts))
                        write_tensorboard_scalars(tb_writer, "val", val_metrics, global_step)
                        if tb_writer is not None:
                            tb_writer.flush()
                    model.train()

        save_checkpoint(
            output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
            model,
            optimizer,
            scheduler,
            epoch + 1,
            global_step,
            args,
            scaler=scaler,
        )
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    step_progress.close()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    save_checkpoint(output_dir / "checkpoint_last.pt", model, optimizer, scheduler, args.epochs, global_step, args, scaler=scaler)
    print(f"[done] training finished at step={global_step}; checkpoints saved to {output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune LingBot-MAP on Manip RGB-D trajectories")

    # Data.
    parser.add_argument("--data_roots", nargs="+", default=DEFAULT_DATA_ROOTS)
    parser.add_argument("--scene_manifest", type=str, default=None)
    parser.add_argument("--write_manifest", type=str, default=None)
    parser.add_argument("--oss_uri_roots", type=str, default="", help="Comma-separated OSS URI roots matching --data_roots; enables faster first-run trajectory listing with ossutil.")
    parser.add_argument("--ossutil_bin", type=str, default="ossutil")
    parser.add_argument("--ossutil_config", type=str, default="", help="Explicit ossutil config file passed with -c/--config-file")
    parser.add_argument("--max_scenes", type=int, default=0, help="0 means use all discovered scenes")
    parser.add_argument("--val_fraction", type=float, default=0.02)
    # DL3DV mix-in (sampling follows base3d-clean/datasets/dl3dv.py)
    parser.add_argument(
        "--dl3dv_root",
        type=str,
        default="",
        help="Root dir containing valid_train.json / valid_test.json (e.g. /cpfs/shared/landmark/renkerui/data/dl3dv). Empty disables DL3DV.",
    )
    parser.add_argument(
        "--dl3dv_num_views",
        type=int,
        default=0,
        help="Per-sample clip length for DL3DV. 0 means follow --max_sample_frames.",
    )
    parser.add_argument(
        "--dl3dv_min_views",
        type=int,
        default=0,
        help="Per-sample minimum clip length for DL3DV (dynamic length in [min,num_views]). 0 means follow --min_sample_frames.",
    )
    parser.add_argument(
        "--dl3dv_repeat",
        type=int,
        default=1,
        help="Repeat the DL3DV dataset N times in the ConcatDataset to bias the mix ratio.",
    )
    parser.add_argument(
        "--dl3dv_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also mix DL3DV's test split into the val DataLoader.",
    )
    parser.add_argument("--dl3dv_min_interval", type=int, default=1)
    parser.add_argument("--dl3dv_max_interval", type=int, default=32)
    parser.add_argument("--dl3dv_video_prob", type=float, default=0.8)
    parser.add_argument("--dl3dv_fix_interval_prob", type=float, default=0.6)
    parser.add_argument("--dl3dv_block_shuffle", type=int, default=16)
    # ScanNet++ v2 mix-in (sampling follows base3d-clean/datasets/scannet.py,
    # NOT scannetpp.py — explicitly requested).
    parser.add_argument(
        "--scannetpp_root",
        type=str,
        default="",
        help="Root dir containing per-scene subdirs and valid.json (e.g. /shared/smartbot/renkerui/data/scannetppv2). Empty disables ScanNet++.",
    )
    parser.add_argument(
        "--scannetpp_num_views",
        type=int,
        default=0,
        help="Per-sample clip length for ScanNet++. 0 means follow --max_sample_frames.",
    )
    parser.add_argument(
        "--scannetpp_min_views",
        type=int,
        default=0,
        help="Per-sample minimum clip length for ScanNet++ (dynamic length in [min,num_views]). 0 means follow --min_sample_frames.",
    )
    parser.add_argument(
        "--scannetpp_repeat",
        type=int,
        default=1,
        help="Repeat the ScanNet++ dataset N times in the ConcatDataset to bias the mix ratio.",
    )
    parser.add_argument(
        "--scannetpp_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also mix ScanNet++ into the val DataLoader (uses the same scene pool — ScanNet++ has no per-split json here).",
    )
    # Sampling defaults intentionally mirror base3d-clean/datasets/scannet.py
    # (NOT scannetpp.py): max_interval=30, video_prob=0.6, min_interval defaults
    # to 1, fix_interval_prob=0.6, block_shuffle=16.
    parser.add_argument("--scannetpp_min_interval", type=int, default=1)
    parser.add_argument("--scannetpp_max_interval", type=int, default=30)
    parser.add_argument("--scannetpp_video_prob", type=float, default=0.6)
    parser.add_argument("--scannetpp_fix_interval_prob", type=float, default=0.6)
    parser.add_argument("--scannetpp_block_shuffle", type=int, default=16)
    # TartanAir mix-in (sampling follows base3d-clean/datasets/tartanair.py).
    parser.add_argument(
        "--tartanair_root",
        type=str,
        default="",
        help="Root dir containing rgb/<scene>/<Easy|Hard>/<P***>/ and depth/... subtrees (e.g. /cpfs/shared/landmark/renkerui/data/tartanair). Empty disables TartanAir.",
    )
    parser.add_argument(
        "--tartanair_num_views",
        type=int,
        default=0,
        help="Per-sample clip length for TartanAir. 0 means follow --max_sample_frames.",
    )
    parser.add_argument(
        "--tartanair_min_views",
        type=int,
        default=0,
        help="Per-sample minimum clip length for TartanAir (dynamic length in [min,num_views]). 0 means follow --min_sample_frames.",
    )
    parser.add_argument(
        "--tartanair_repeat",
        type=int,
        default=1,
        help="Repeat the TartanAir dataset N times in the ConcatDataset to bias the mix ratio.",
    )
    parser.add_argument(
        "--tartanair_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also mix TartanAir into the val DataLoader (uses the same episode pool — TartanAir has no per-split json).",
    )
    # Sampling defaults intentionally mirror base3d-clean/datasets/tartanair.py
    # (and ONLY that file): min_interval=1, max_interval=32, video_prob=0.8,
    # fix_interval_prob=0.6, block_shuffle=16.
    parser.add_argument("--tartanair_min_interval", type=int, default=1)
    parser.add_argument("--tartanair_max_interval", type=int, default=32)
    parser.add_argument("--tartanair_video_prob", type=float, default=0.8)
    parser.add_argument("--tartanair_fix_interval_prob", type=float, default=0.6)
    parser.add_argument("--tartanair_block_shuffle", type=int, default=16)
    # DynamicReplica mix-in (sampling follows base3d-clean/datasets/dynamic_replica.py).
    parser.add_argument(
        "--dynamic_replica_root",
        type=str,
        default="",
        help="Root dir containing {train,valid,test}/<scene>/left/{rgb,depth,cam}/ subtrees (e.g. /shared/smartbot/renkerui/data/dynamic_replica). Empty disables DynamicReplica.",
    )
    parser.add_argument(
        "--dynamic_replica_num_views",
        type=int,
        default=0,
        help="Per-sample clip length for DynamicReplica. 0 means follow --max_sample_frames.",
    )
    parser.add_argument(
        "--dynamic_replica_min_views",
        type=int,
        default=0,
        help="Per-sample minimum clip length for DynamicReplica (dynamic length in [min,num_views]). 0 means follow --min_sample_frames.",
    )
    parser.add_argument(
        "--dynamic_replica_repeat",
        type=int,
        default=1,
        help="Repeat the DynamicReplica dataset N times in the ConcatDataset to bias the mix ratio.",
    )
    parser.add_argument(
        "--dynamic_replica_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also mix DynamicReplica's valid split into the val DataLoader.",
    )
    # Sampling defaults intentionally mirror base3d-clean/datasets/dynamic_replica.py:
    # max_interval=64, video_prob=1.0, fix_interval_prob=1.0, min_interval=1.
    parser.add_argument("--dynamic_replica_min_interval", type=int, default=1)
    parser.add_argument("--dynamic_replica_max_interval", type=int, default=64)
    parser.add_argument("--dynamic_replica_video_prob", type=float, default=1.0)
    parser.add_argument("--dynamic_replica_fix_interval_prob", type=float, default=1.0)
    parser.add_argument("--dynamic_replica_block_shuffle", type=int, default=16)
    # MapFree mix-in (sampling follows base3d-clean/datasets/mapfree.py).
    parser.add_argument(
        "--mapfree_root",
        type=str,
        default="",
        help="Root dir containing valid.json + <scene>/dense{0,1}/{rgb,depth,cam,sky_mask}/ subtrees (e.g. /cpfs/shared/landmark/renkerui/data/mapfree). Empty disables MapFree.",
    )
    parser.add_argument(
        "--mapfree_num_views",
        type=int,
        default=0,
        help="Per-sample clip length for MapFree. 0 means follow --max_sample_frames.",
    )
    parser.add_argument(
        "--mapfree_min_views",
        type=int,
        default=0,
        help="Per-sample minimum clip length for MapFree (dynamic length in [min,num_views]). 0 means follow --min_sample_frames.",
    )
    parser.add_argument(
        "--mapfree_repeat",
        type=int,
        default=1,
        help="Repeat the MapFree dataset N times in the ConcatDataset to bias the mix ratio.",
    )
    parser.add_argument(
        "--mapfree_val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also mix MapFree into the val DataLoader (uses the same scene pool — MapFree has no per-split json).",
    )
    # Sampling defaults intentionally mirror base3d-clean/datasets/mapfree.py:
    # min_interval=1, max_interval=64, video_prob=0.8, fix_interval_prob=0.6, block_shuffle=16.
    parser.add_argument("--mapfree_min_interval", type=int, default=1)
    parser.add_argument("--mapfree_max_interval", type=int, default=64)
    parser.add_argument("--mapfree_video_prob", type=float, default=0.8)
    parser.add_argument("--mapfree_fix_interval_prob", type=float, default=0.6)
    parser.add_argument("--mapfree_block_shuffle", type=int, default=16)
    # Step-aware mixture curriculum (Manip vs. external 3D datasets).
    # When enabled, the DataLoader uses CurriculumMixtureSampler instead of
    # the default shuffle: every batch is drawn from Manip with probability
    # p_manip(step), else uniformly from one of the enabled externals.
    parser.add_argument(
        "--mixture_curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable step-aware Manip-vs-external mixture sampling. Forces *_repeat=1.",
    )
    parser.add_argument(
        "--mixture_p_manip_start",
        type=float,
        default=0.30,
        help="Manip draw probability at step <= --mixture_warmup_start. 1 - p is split uniformly across enabled externals.",
    )
    parser.add_argument(
        "--mixture_p_manip_end",
        type=float,
        default=0.90,
        help="Manip draw probability at step >= --mixture_warmup_end.",
    )
    parser.add_argument(
        "--mixture_warmup_start",
        type=int,
        default=2000,
        help="Step at which the linear ramp begins. Before this step the schedule holds at --mixture_p_manip_start.",
    )
    parser.add_argument(
        "--mixture_warmup_end",
        type=int,
        default=70000,
        help="Step at which the linear ramp ends. After this step the schedule holds at --mixture_p_manip_end.",
    )
    parser.add_argument("--clip_len", type=int, default=8, help="Fixed sequence length used by fixed_stride sampling")
    parser.add_argument("--samples_per_scene", type=int, default=1)
    parser.add_argument("--sequence_mode", choices=["all_views", "single_view", "manip_4d_mixed"], default="all_views")
    parser.add_argument("--view_ids", type=str, default="", help="Comma-separated numeric view ids, e.g. 0,1,2,3")
    parser.add_argument("--camera_names", type=str, default="", help="Comma-separated Manip camera names")
    parser.add_argument("--sample_strategy", choices=["fixed_stride", "random_interval"], default="fixed_stride")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--random_stride_min", type=int, default=6)
    parser.add_argument("--random_stride_max", type=int, default=64)
    parser.add_argument("--random_interval_start", choices=["first", "random"], default="first")
    parser.add_argument("--max_sample_frames", type=int, default=8, help="0 disables random-interval sequence length cap")
    parser.add_argument("--min_sample_frames", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--preprocess_mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Use <=0 to auto-scale uint16 raw depth or uint8 depth images")
    parser.add_argument("--min_depth", type=float, default=1e-6)
    parser.add_argument("--max_depth", type=float, default=0.0, help="0 disables max-depth filtering")
    parser.add_argument("--use_mask", action="store_true", help="Also require Manip mask > 0 for valid pixels")
    parser.add_argument("--invert_cam_extrinsics", action="store_true")
    # manip_4d_mixed (W/S/M) sampling configuration
    parser.add_argument("--wrist_camera_prefix", type=str, default="realsense",
                        help="Camera-name prefix that marks wrist (moving) cameras for manip_4d_mixed")
    parser.add_argument("--static_camera_prefix", type=str, default="surround",
                        help="Camera-name prefix that marks static (surround) cameras for manip_4d_mixed")
    parser.add_argument("--m_stride_min", type=int, default=2,
                        help="Mode M (4D grid): min frame stride between sampled timestamps (~0.07s @ 30fps)")
    parser.add_argument("--m_stride_max", type=int, default=8,
                        help="Mode M (4D grid): max frame stride between sampled timestamps (~0.27s @ 30fps)")
    parser.add_argument("--s_views_min", type=int, default=4,
                        help="Mode S (static snapshot): min number of static cameras to sample at one timestamp")
    parser.add_argument("--s_views_max", type=int, default=8,
                        help="Mode S (static snapshot): max number of static cameras to sample at one timestamp")
    parser.add_argument("--m_num_views", type=int, default=4,
                        help="(Deprecated, kept for back-compat) Use --m_views_min/--m_views_max instead")
    parser.add_argument("--m_views_min", type=int, default=3,
                        help="Mode M (4D grid): min number of static cameras to sample per clip")
    parser.add_argument("--m_views_max", type=int, default=6,
                        help="Mode M (4D grid): max number of static cameras to sample per clip")
    parser.add_argument("--m_num_times", type=int, default=0,
                        help="Mode M (4D grid): optional upper cap on number of timestamps. "
                             "0 means no cap (T computed dynamically so V*T lands in "
                             "[min_sample_frames, max_sample_frames])")
    parser.add_argument("--mode_weights_initial", type=str, default="S=0.70,W=0.20,M=0.10",
                        help="Initial mode-mix weights at warmup_start. Format: 'S=..,W=..,M=..'")
    parser.add_argument("--mode_weights_final", type=str, default="S=0.30,W=0.40,M=0.30",
                        help="Final mode-mix weights at warmup_end. Format: 'S=..,W=..,M=..'")
    parser.add_argument("--mode_warmup_start", type=int, default=2000,
                        help="Step at which we start ramping the mode-mix weights")
    parser.add_argument("--mode_warmup_end", type=int, default=8000,
                        help="Step at which the mode-mix weights reach their final values")
    parser.add_argument("--t_stride_min", type=int, default=15,
                        help="Mode T (Manip_long5 trajectory): min frame stride per step "
                             "(~0.5s @ 30fps). Mode T is auto-selected for scenes whose path "
                             "contains --long5_root_marker.")
    parser.add_argument("--t_stride_max", type=int, default=60,
                        help="Mode T (Manip_long5 trajectory): max frame stride per step "
                             "(~2.0s @ 30fps).")
    parser.add_argument("--long5_root_marker", type=str, default="Manip_long5",
                        help="Substring used to detect Manip_long5 scenes by path. Such scenes "
                             "always sample with mode T (single surround_cam, random start, "
                             "per-step random stride) and bypass the S/W/M curriculum. Set empty "
                             "to disable long5 routing.")
    parser.add_argument("--color_jitter_strength", type=float, default=0.0,
                        help="RGB-only ColorJitter strength: brightness/contrast/saturation = strength, "
                             "hue = strength*0.25 (capped at 0.5). 0 disables. Same params shared across "
                             "all frames in a clip; depth/mask/intrinsics/extrinsics are untouched.")
    parser.add_argument("--color_jitter_prob", type=float, default=0.5,
                        help="Per-clip probability of applying ColorJitter when strength > 0.")
    parser.add_argument(
        "--normalize_scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LingBot-MAP anchor-frame scale normalization from the paper.",
    )
    parser.add_argument(
        "--canonicalize_first_frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-express extrinsics and world_points so frame 0's c2w is identity, "
             "matching VGGT's training-time normalization. Required when fine-tuning "
             "the lingbot-map.pt checkpoint on datasets that ship absolute world poses; "
             "without it the camera head pays a large gratuitous loss on the global "
             "rigid offset between the dataset's world frame and the first camera.",
    )

    # Model.
    parser.add_argument("--model_path", type=str, default="/cpfs/shared/aigc/guowenqi/lingbot-map/lingbot-map.pt")
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--use_sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_3d_rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=4)
    parser.add_argument("--num_frame_per_block", type=int, default=1)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--depth_frames_chunk_size", type=int, default=1, help="Frames per DPT depth-head chunk; smaller saves memory but is slower")
    parser.add_argument("--no_depth_activation_checkpoint", action="store_true", help="Disable activation checkpointing in the DPT depth head")
    parser.add_argument("--camera_num_iterations", type=int, default=4)
    parser.add_argument("--no_gradient_checkpoint", action="store_true")
    parser.add_argument(
        "--freeze_dino_patch_embed",
        action="store_true",
        help="Freeze the DINOv2 image encoder under aggregator.patch_embed.",
    )
    parser.add_argument("--freeze_aggregator", action="store_true")
    parser.add_argument("--freeze_camera", action="store_true")
    parser.add_argument("--freeze_depth", action="store_true")
    parser.add_argument("--freeze_point", action="store_true")

    # Loss, mirroring VGGT defaults where possible.
    parser.add_argument("--camera_weight", type=float, default=5.0)
    parser.add_argument("--depth_weight", type=float, default=1.0)
    parser.add_argument("--relative_pose_weight", type=float, default=1.0)
    parser.add_argument("--relative_trans_weight", type=float, default=1.0)
    parser.add_argument("--relative_pose_window", type=int, default=64)
    parser.add_argument("--point_weight", type=float, default=0.0, help="Deprecated; point-map loss is disabled for LingBot-MAP fine-tuning.")
    parser.add_argument("--camera_loss_type", choices=["l1", "l2"], default="l1")
    parser.add_argument("--camera_gamma", type=float, default=0.6)
    parser.add_argument("--weight_trans", type=float, default=1.0)
    parser.add_argument("--weight_rot", type=float, default=1.0)
    parser.add_argument("--weight_focal", type=float, default=0.5)
    parser.add_argument("--depth_gradient_loss_fn", type=str, default="grad")
    parser.add_argument("--point_gradient_loss_fn", type=str, default="normal", help="Deprecated; point-map loss is disabled.")
    parser.add_argument("--loss_gamma", type=float, default=1.0)
    parser.add_argument("--loss_alpha", type=float, default=0.2)
    parser.add_argument("--valid_range", type=float, default=0.98)
    parser.add_argument("--min_valid_pixels", type=int, default=100)

    # Optim/training.
    parser.add_argument("--output_dir", type=str, default="/cpfs/shared/aigc/guowenqi/lingbot-map/runs/manip_train")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--max_steps", type=int, default=150000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit_train_batches", type=int, default=0)
    parser.add_argument("--limit_val_batches", type=int, default=20)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Used when --warmup_steps <= 0")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Override --warmup_ratio when > 0")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--empty_cache_every", type=int, default=1, help="Call torch.cuda.empty_cache every N optimizer steps; 0 disables it")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--allow_tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cudnn_benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--print_input_every", type=int, default=10)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard_dir", type=str, default="", help="Defaults to OUTPUT_DIR/tensorboard")
    parser.add_argument("--tensorboard_flush_secs", type=int, default=30)
    parser.add_argument("--tensorboard_flush_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--val_every", type=int, default=1000)

    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
