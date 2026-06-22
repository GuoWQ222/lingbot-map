#!/usr/bin/env python3
"""Shared model-native geometry loaders for LingBot-MAP eval adapters.

Adapters define the model-specific image geometry. This module applies that
geometry consistently to RGB, depth, mask, intrinsics, and derived world points.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parents[1]
for _path in (_REPO_DIR, _REPO_DIR / "scripts" / "train", _REPO_DIR / "scripts" / "eval", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF

import eval as E
import train as T

GeometryFn = Callable[[int, int], Dict[str, int]]


def load_frame_with_model_geometry(
    dataset: object,
    scene_dir: Path,
    entry: T.FrameEntry,
    geometry_fn: GeometryFn,
    *,
    rgb_resample: Image.Resampling,
    jitter_params: Optional[Tuple] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = Image.open(entry.rgb_path)
    if rgb.mode == "RGBA":
        background = Image.new("RGBA", rgb.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(background, rgb)
    rgb = rgb.convert("RGB")

    width, height = rgb.size
    geometry = geometry_fn(width, height)
    intrinsics_raw, extrinsics = dataset._load_camera_for_entry(scene_dir, entry)  # type: ignore[attr-defined]
    intrinsics = T.preprocess_intrinsics(intrinsics_raw, width, height, geometry)
    if not torch.is_tensor(extrinsics):
        extrinsics = torch.from_numpy(np.asarray(extrinsics, dtype=np.float32))

    rgb = T.apply_preprocess_to_image(
        rgb,
        geometry,
        resample=rgb_resample,
        fill=(255, 255, 255),
    )
    if jitter_params is not None:
        rgb = dataset._apply_color_jitter(rgb, jitter_params)  # type: ignore[attr-defined]
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
        depth_scale = float(getattr(dataset, "depth_scale"))
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

    valid = np.isfinite(depth_np) & (depth_np > float(getattr(dataset, "min_depth")))
    max_depth = float(getattr(dataset, "max_depth"))
    if max_depth > 0:
        valid &= depth_np < max_depth
    if bool(getattr(dataset, "use_mask")):
        valid &= mask_np > 0

    depth = torch.from_numpy(depth_np).float()
    point_mask = torch.from_numpy(valid.astype(np.bool_))
    extrinsics_t = cast(torch.Tensor, extrinsics).float()
    world_points = T.depth_to_world_points(depth, intrinsics.float(), extrinsics_t)
    world_points = torch.where(point_mask[..., None], world_points, torch.zeros_like(world_points))

    return image_tensor, depth, point_mask, intrinsics.float(), extrinsics_t, world_points.float()


class ModelGeometryEvalDataset(E.EvalLinspaceDataset):  # type: ignore[misc]
    """Eval dataset whose RGB-D/GT tensors share one model-native geometry."""

    rgb_resample: Image.Resampling = Image.Resampling.BICUBIC

    def model_preprocess_geometry(self, width: int, height: int) -> Dict[str, int]:
        raise NotImplementedError

    def _load_one(
        self,
        scene_dir: Path,
        entry: T.FrameEntry,
        jitter_params: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return load_frame_with_model_geometry(
            self,
            scene_dir,
            entry,
            self.model_preprocess_geometry,
            rgb_resample=self.rgb_resample,
            jitter_params=jitter_params,
        )

    def __getitem__(self, index: int) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        start = index % len(self._eval_index_map)
        for attempt in range(min(16, len(self._eval_index_map))):
            scene_dir, track_spec = self._eval_index_map[(start + attempt) % len(self._eval_index_map)]
            try:
                entries = self._entries_for_scene(scene_dir)
                selected, mode_label, camera_name = self._sample_entries_for_eval(
                    entries, scene_dir, track_spec
                )
                jitter_params = self._sample_color_jitter_params()
                loaded = [self._load_one(scene_dir, entry, jitter_params) for entry in selected]
                images, depths, masks, intrinsics, extrinsics, world_points = zip(*loaded)
                scene_label = scene_dir.name
                if camera_name:
                    scene_label = f"{scene_label}__{camera_name}"
                return {
                    "images": torch.stack(list(images), dim=0),
                    "depths": torch.stack(list(depths), dim=0),
                    "point_masks": torch.stack(list(masks), dim=0),
                    "intrinsics": torch.stack(list(intrinsics), dim=0),
                    "extrinsics": torch.stack(list(extrinsics), dim=0),
                    "world_points": torch.stack(list(world_points), dim=0),
                    "frame_ids": torch.tensor([entry.frame_id for entry in selected], dtype=torch.long),
                    "view_ids": torch.tensor([entry.view_id for entry in selected], dtype=torch.long),
                    "rgb_paths": [str(entry.rgb_path) for entry in selected],
                    "scene": scene_label,
                    "sample_mode": mode_label,
                }
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load a valid model-geometry eval sample near index {index}: {last_error}")
