"""ASE RGB-D adapter emitting LingBot-MAP training samples.

Sampling and preprocessing mirror base3d-clean/datasets/ase_20251008.py:
sample views from a local temporal window, remove the Aria RGB vignette,
convert ray depth in millimeters to OpenCV z-depth in meters, then shuffle.
"""
from __future__ import annotations

import concurrent.futures
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image, ImageOps
from projectaria_tools.core import calibration
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.projects import ase
from torch.utils.data import Dataset
from tqdm import tqdm


def _ray_depth_to_z_depth(depthmap: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    height, width = depthmap.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    return depthmap / np.sqrt(x**2 + y**2 + 1.0)


def _preprocess_depth(depthmap: np.ndarray, geometry: Dict[str, int]) -> torch.Tensor:
    depth = torch.from_numpy(depthmap.astype(np.float32))[None, None]
    depth = F.interpolate(
        depth,
        size=(geometry["new_height"], geometry["new_width"]),
        mode="nearest",
    )[0, 0]
    top = geometry["crop_top"]
    left = geometry["crop_left"]
    depth = depth[
        top : top + geometry["crop_height"],
        left : left + geometry["crop_width"],
    ]
    return F.pad(
        depth,
        (
            geometry["pad_left"],
            geometry["pad_right"],
            geometry["pad_top"],
            geometry["pad_bottom"],
        ),
        value=0.0,
    )


def _scene_frame_count(root: str, scene: str) -> Optional[int]:
    scene_dir = osp.join(root, scene)
    modality_dirs = [osp.join(scene_dir, name) for name in ("rgb", "depth", "cam")]
    if not all(osp.isdir(path) for path in modality_dirs):
        return None
    counts = [len(os.listdir(path)) for path in modality_dirs]
    if len(set(counts)) != 1 or counts[0] == 0:
        return None
    return counts[0]


class ASETrajectoryDataset(Dataset):
    """ASE adapter using the same schema as LingBot-MAP's external datasets."""

    def __init__(
        self,
        root: str,
        *,
        split: str = "train",
        num_views: int,
        min_views: Optional[int] = None,
        image_size: int,
        patch_size: int,
        preprocess_mode: str = "crop",
        min_depth: float = 1e-6,
        max_depth: float = 0.0,
        samples_per_scene: int = 1,
        color_jitter_strength: float = 0.0,
        color_jitter_prob: float = 0.0,
        max_distance: int = 32,
        cache_path: Optional[str] = None,
        io_max_workers: int = 8,
        scan_max_workers: int = 16,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        from train import (
            apply_preprocess_to_image,
            compute_preprocess_geometry,
            depth_to_world_points,
            preprocess_intrinsics,
        )

        self._apply_preprocess_to_image = apply_preprocess_to_image
        self._compute_preprocess_geometry = compute_preprocess_geometry
        self._depth_to_world_points = depth_to_world_points
        self._preprocess_intrinsics = preprocess_intrinsics

        if num_views <= 0:
            raise ValueError(f"num_views must be > 0, got {num_views}")
        self.root = root.rstrip("/")
        self.split = split
        self.num_views = int(num_views)
        self.min_views = int(min_views) if min_views is not None and min_views > 0 else self.num_views
        if not (1 <= self.min_views <= self.num_views):
            raise ValueError(
                f"min_views must satisfy 1 <= min_views <= num_views; got "
                f"min_views={self.min_views}, num_views={self.num_views}"
            )
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.preprocess_mode = preprocess_mode
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.samples_per_scene = max(1, int(samples_per_scene))
        self.max_distance = int(max_distance)
        self.cache_path = cache_path.strip() if cache_path else osp.join(self.root, "ase_cache.npy")
        self.io_max_workers = int(io_max_workers)
        self.scan_max_workers = int(scan_max_workers)

        if color_jitter_strength > 0 and color_jitter_prob > 0:
            self._color_jitter = TF.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=min(color_jitter_strength * 0.5, 0.5),
            )
        else:
            self._color_jitter = None
        self.color_jitter_prob = float(color_jitter_prob)

        self._load_or_build_cache(verbose)
        self.vignette_mask = self._build_vignette_mask()
        print(
            f"[ase] {split}: {len(self.sequences)} scenes, "
            f"{sum(self.num_imgs.values())} frames total, "
            f"num_views=[{self.min_views}, {self.num_views}], "
            f"max_distance={self.max_distance}, "
            f"image_size={self.image_size}"
        )

    def _load_or_build_cache(self, verbose: bool) -> None:
        cache_path = self.cache_path
        if osp.exists(cache_path):
            cache = np.load(cache_path, allow_pickle=True).item()
            self.sequences = list(cache["sequences"])
            self.num_imgs = {str(k): int(v) for k, v in cache["num_imgs"].items()}
            if verbose:
                print(f"[ase] loaded cache: {cache_path} ({len(self.sequences)} scenes)")
            return

        scenes = sorted(
            scene for scene in os.listdir(self.root) if osp.isdir(osp.join(self.root, scene))
        )
        self.num_imgs: Dict[str, int] = {}
        if self.scan_max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.scan_max_workers) as ex:
                counts = ex.map(lambda scene: _scene_frame_count(self.root, scene), scenes)
                iterator = zip(scenes, counts)
                if verbose:
                    iterator = tqdm(iterator, total=len(scenes), desc="[ase] scan")
                for scene, count in iterator:
                    if count is not None:
                        self.num_imgs[scene] = count
        else:
            for scene in tqdm(scenes, desc="[ase] scan", disable=not verbose):
                count = _scene_frame_count(self.root, scene)
                if count is not None:
                    self.num_imgs[scene] = count
        self.sequences = sorted(self.num_imgs)
        try:
            np.save(cache_path, {"sequences": self.sequences, "num_imgs": self.num_imgs})
            if verbose:
                print(f"[ase] wrote cache: {cache_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ase] WARNING: failed to write cache: {exc}")

    def _build_vignette_mask(self) -> np.ndarray:
        device = ase.get_ase_rgb_calibration()
        focal_length = device.get_focal_lengths()[0]
        pinhole = calibration.get_linear_camera_calibration(
            512, 512, focal_length, "camera-rgb", device.get_transform_device_camera()
        )
        vignette = ImageOps.invert(Image.open(osp.join(self.root, "vignette.png")).convert("L"))
        vignette_np = np.asarray(vignette, dtype=np.float32)[:, :, None] / 255.0
        rectified = calibration.distort_by_calibration(
            vignette_np, pinhole, device, InterpolationMethod.BILINEAR
        )
        rectified = np.rot90(rectified, k=3).astype(np.float32)
        if rectified.ndim == 2:
            rectified = rectified[..., None]
        return rectified

    def __len__(self) -> int:
        return len(self.sequences) * self.samples_per_scene

    def _sample_indices(self, num_imgs: int, rng: np.random.Generator) -> List[int]:
        target_views = self.num_views
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        indices = [int(rng.integers(0, num_imgs))]
        max_distance = int(self.max_distance / 8 * target_views)
        start = max(0, indices[0] - max_distance)
        end = min(num_imgs - 1, start + max_distance)
        start = max(0, end - max_distance)
        valid = np.arange(start, end + 1)
        indices.extend(
            int(idx)
            for idx in rng.choice(valid, target_views - 1, replace=len(valid) < target_views - 1)
        )
        return indices

    def _sample_color_jitter_params(self, rng: np.random.Generator):
        if self._color_jitter is None or rng.random() >= self.color_jitter_prob:
            return None
        return TF.ColorJitter.get_params(
            self._color_jitter.brightness,
            self._color_jitter.contrast,
            self._color_jitter.saturation,
            self._color_jitter.hue,
        )

    @staticmethod
    def _apply_color_jitter(image: Image.Image, params) -> Image.Image:
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

    def _load_view(self, scene: str, frame_idx: int, jitter_params):
        stem = f"{frame_idx:07d}"
        scene_dir = osp.join(self.root, scene)
        rgb_path = osp.join(scene_dir, "rgb", stem + ".jpg")
        depth_path = osp.join(scene_dir, "depth", stem + ".npy")
        cam_path = osp.join(scene_dir, "cam", stem + ".npz")

        with np.load(cam_path) as cam:
            K_raw = np.asarray(cam["intrinsics"], dtype=np.float32)
            c2w = np.asarray(cam["pose"], dtype=np.float64)

        raw_rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        if raw_rgb.shape[:2] != self.vignette_mask.shape[:2]:
            raise ValueError(
                f"RGB/vignette shape mismatch: {raw_rgb.shape[:2]} vs {self.vignette_mask.shape[:2]}"
            )
        rgb = np.clip(
            raw_rgb.astype(np.float32) / (self.vignette_mask + 1e-6), 0, 255
        ).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb).convert("RGB")
        width, height = rgb_pil.size

        depthmap = np.load(depth_path).astype(np.float32)
        # ASE stores invalid ray depths as float32 max, which is finite but
        # would otherwise produce enormous world-space coordinates.
        depthmap[depthmap >= np.finfo(np.float32).max / 2] = 0.0
        depthmap /= 1000.0
        depthmap = _ray_depth_to_z_depth(depthmap, K_raw)
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)

        geometry = self._compute_preprocess_geometry(
            width, height, self.image_size, self.patch_size, self.preprocess_mode
        )
        rgb_pre = self._apply_preprocess_to_image(
            rgb_pil, geometry, resample=Image.Resampling.BICUBIC, fill=(255, 255, 255)
        )
        if jitter_params is not None:
            rgb_pre = self._apply_color_jitter(rgb_pre, jitter_params)
        image = TF.ToTensor()(rgb_pre)

        depth = _preprocess_depth(depthmap, geometry)
        depth_np = depth.numpy()
        intrinsics = self._preprocess_intrinsics(K_raw, width, height, geometry).float()
        w2c = np.linalg.inv(c2w)
        extrinsics = torch.from_numpy(w2c[:3, :4].astype(np.float32))

        valid = np.isfinite(depth_np) & (depth_np > self.min_depth)
        if self.max_depth > 0:
            valid &= depth_np < self.max_depth
        point_mask = torch.from_numpy(valid)
        world_points = self._depth_to_world_points(depth, intrinsics, extrinsics)
        world_points = torch.where(point_mask[..., None], world_points, torch.zeros_like(world_points))
        return image, depth.float(), point_mask, intrinsics, extrinsics, world_points.float()

    def __getitem__(self, index: int) -> Dict[str, object]:
        scene = self.sequences[index % len(self.sequences)]
        rng = np.random.default_rng(int(np.random.randint(0, 2**32, dtype=np.uint32)))
        sampled = self._sample_indices(self.num_imgs[scene], rng)
        jitter_params = self._sample_color_jitter_params(rng)
        if self.io_max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.io_max_workers, len(sampled))
            ) as ex:
                loaded = list(ex.map(lambda idx: self._load_view(scene, idx, jitter_params), sampled))
        else:
            loaded = [self._load_view(scene, idx, jitter_params) for idx in sampled]

        perm = torch.from_numpy(rng.permutation(len(sampled))).long()
        return {
            "images": torch.stack([item[0] for item in loaded])[perm],
            "depths": torch.stack([item[1] for item in loaded])[perm],
            "point_masks": torch.stack([item[2] for item in loaded])[perm],
            "intrinsics": torch.stack([item[3] for item in loaded])[perm],
            "extrinsics": torch.stack([item[4] for item in loaded])[perm],
            "world_points": torch.stack([item[5] for item in loaded])[perm],
            "frame_ids": torch.tensor(sampled, dtype=torch.long)[perm],
            "view_ids": torch.zeros(len(sampled), dtype=torch.long)[perm],
            "scene": scene,
            "sample_mode": "ase",
        }
