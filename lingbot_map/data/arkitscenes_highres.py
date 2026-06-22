"""ARKitScenes high-resolution RGB-D adapter for LingBot-MAP.

This module mirrors the scene layout and sampling policy from
``base3d-clean/datasets/arkitscenes_highres.py`` while emitting LingBot-MAP's
training sample schema. It is intentionally independent from
``ManipTrajectoryDataset``: ARKitScenesHighRes uses base3d's sequential
``get_seq_from_start_id`` sampling, not Manip_long W/T/S/M sampling.

Expected layout::

    ROOT/
      Training/<scene>/scene_metadata.npz
      Training/<scene>/vga_wide/<scene>_<timestamp>.jpg
      Training/<scene>/highres_depth/<scene>_<timestamp>.png
      Validation/<scene>/...

The metadata npz must contain ``images`` (.png names), ``intrinsics`` (N, 6 as
width, height, fx, fy, cx, cy), and ``trajectories`` (N, 4, 4 camera-to-world).
"""
from __future__ import annotations

import concurrent.futures
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset

from .dl3dv import _get_seq_from_start_id


class ARKitScenesHighResTrajectoryDataset(Dataset):
    """ARKitScenesHighRes dataset with base3d-clean sampling semantics."""

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
        # Defaults match base3d-clean/datasets/arkitscenes_highres.py:
        # get_seq_from_start_id(..., max_interval=32, block_shuffle=16), with
        # BaseMultiViewDataset defaults for min_interval/video/fix probabilities.
        min_interval: int = 1,
        max_interval: int = 32,
        video_prob: float = 0.5,
        fix_interval_prob: float = 0.5,
        block_shuffle: Optional[int] = 16,
        io_max_workers: int = 8,
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
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        self.root = root.rstrip("/")
        self.split = "Training" if split == "train" else "Validation"
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

        self.min_interval = int(min_interval)
        self.max_interval = int(max_interval)
        self.video_prob = float(video_prob)
        self.fix_interval_prob = float(fix_interval_prob)
        self.block_shuffle = block_shuffle
        self.io_max_workers = int(io_max_workers)

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

        self.scenes: List[str] = []
        self.scene_frames: List[Dict[str, np.ndarray]] = []
        self.start_refs: List[Tuple[int, int]] = []
        self._load_metadata()
        if not self.start_refs:
            raise RuntimeError(
                f"No valid ARKitScenesHighRes samples under {osp.join(self.root, self.split)}"
            )
        if verbose:
            print(
                f"[arkitscenes_highres] {split}: {len(self.scenes)} scenes, "
                f"{sum(len(s['rgb_paths']) for s in self.scene_frames)} frames total, "
                f"{len(self.start_refs)} start positions, "
                f"num_views=[{self.min_views}, {self.num_views}], "
                f"image_size={self.image_size}"
            )

    def _load_metadata(self) -> None:
        split_dir = osp.join(self.root, self.split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(f"ARKitScenesHighRes split dir not found: {split_dir}")
        all_scenes = sorted(
            d for d in os.listdir(split_dir) if osp.isdir(osp.join(split_dir, d))
        )
        cut_off = max(self.num_views // 8, 4)
        cut_off = max(cut_off, min(self.min_views, self.num_views))
        for scene in all_scenes:
            scene_dir = osp.join(split_dir, scene)
            meta_path = osp.join(scene_dir, "scene_metadata.npz")
            if not osp.isfile(meta_path):
                continue
            try:
                with np.load(meta_path) as data:
                    imgs_with_indices = sorted(enumerate(data["images"]), key=lambda x: str(x[1]))
                    indices = [int(i) for i, _ in imgs_with_indices]
                    imgs = [str(name) for _, name in imgs_with_indices]
                    if len(imgs) < cut_off:
                        continue
                    if any(not name.startswith(scene + "_") or not name.endswith(".png") for name in imgs):
                        continue
                    intrins_raw = np.asarray(data["intrinsics"])[indices]
                    traj = np.asarray(data["trajectories"])[indices]
            except Exception:
                continue

            K = np.repeat(np.eye(3, dtype=np.float32)[None], len(imgs), axis=0)
            K[:, 0, 0] = intrins_raw[:, 2]
            K[:, 1, 1] = intrins_raw[:, 3]
            K[:, 0, 2] = intrins_raw[:, 4]
            K[:, 1, 2] = intrins_raw[:, 5]

            rgb_paths = np.array(
                [osp.join(scene_dir, "vga_wide", name.replace(".png", ".jpg")) for name in imgs]
            )
            depth_paths = np.array([osp.join(scene_dir, "highres_depth", name) for name in imgs])
            if any(not osp.isfile(p) for p in rgb_paths) or any(not osp.isfile(p) for p in depth_paths):
                continue

            scene_idx = len(self.scenes)
            self.scenes.append(scene)
            self.scene_frames.append(
                dict(
                    rgb_paths=rgb_paths,
                    depth_paths=depth_paths,
                    intrinsics=K.astype(np.float32),
                    c2w=traj.astype(np.float64),
                )
            )
            for start in range(0, len(imgs) - cut_off + 1):
                self.start_refs.append((scene_idx, start))

    def __len__(self) -> int:
        return len(self.start_refs) * self.samples_per_scene

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

    def _sample_indices(self, scene_idx: int, start_id: int, rng: np.random.Generator) -> Tuple[List[int], bool]:
        num_imgs = len(self.scene_frames[scene_idx]["rgb_paths"])
        ids_all = list(range(num_imgs))
        target_views = self.num_views
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        pos, is_video = _get_seq_from_start_id(
            target_views,
            start_id,
            ids_all,
            rng,
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=self.fix_interval_prob,
            block_shuffle=self.block_shuffle,
        )
        return [ids_all[p] for p in pos], is_video

    def _load_view(self, scene_idx: int, frame_idx: int, jitter_params):
        meta = self.scene_frames[scene_idx]
        rgb_path = str(meta["rgb_paths"][frame_idx])
        depth_path = str(meta["depth_paths"][frame_idx])
        K_raw = np.asarray(meta["intrinsics"][frame_idx], dtype=np.float32)
        c2w_4x4 = np.asarray(meta["c2w"][frame_idx], dtype=np.float64)

        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size

        depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depthmap is None:
            raise IOError(f"Could not load depth map: {depth_path}")
        depthmap = depthmap.astype(np.float32) / 1000.0
        depthmap[~np.isfinite(depthmap)] = 0.0
        if depthmap.shape[:2] != (height, width):
            depthmap = cv2.resize(depthmap, (width, height), interpolation=cv2.INTER_NEAREST)

        geometry = self._compute_preprocess_geometry(
            width, height, self.image_size, self.patch_size, self.preprocess_mode
        )
        rgb_pre = self._apply_preprocess_to_image(
            rgb_pil, geometry, resample=Image.Resampling.BICUBIC, fill=(255, 255, 255)
        )
        if jitter_params is not None:
            rgb_pre = self._apply_color_jitter(rgb_pre, jitter_params)
        image_tensor = TF.ToTensor()(rgb_pre)

        depth_pil = Image.fromarray(depthmap.astype(np.float32)).convert("F")
        depth_pil = self._apply_preprocess_to_image(
            depth_pil, geometry, resample=Image.Resampling.NEAREST, fill=0
        )
        depth_np = np.asarray(depth_pil, dtype=np.float32).copy()
        depth = torch.from_numpy(depth_np)

        intrinsics = self._preprocess_intrinsics(K_raw, width, height, geometry).float()
        w2c_4x4 = np.linalg.inv(c2w_4x4)
        extrinsics = torch.from_numpy(w2c_4x4[:3, :4].astype(np.float32))

        valid = np.isfinite(depth_np) & (depth_np > self.min_depth)
        if self.max_depth > 0:
            valid &= depth_np < self.max_depth
        point_mask = torch.from_numpy(valid.astype(np.bool_))
        world_points = self._depth_to_world_points(depth, intrinsics, extrinsics)
        world_points = torch.where(point_mask[..., None], world_points, torch.zeros_like(world_points))
        return image_tensor, depth.float(), point_mask, intrinsics, extrinsics.float(), world_points.float()

    def __getitem__(self, index: int) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        n = len(self.start_refs)
        base_index = index % n
        seed_entropy = int(np.random.randint(0, 2**32, dtype=np.uint32))
        rng = np.random.default_rng(seed_entropy)

        for attempt in range(min(16, n)):
            scene_idx, start_id = self.start_refs[(base_index + attempt) % n]
            try:
                sampled, is_video = self._sample_indices(scene_idx, start_id, rng)
                if is_video:
                    sampled = sorted(sampled)
                jitter_params = self._sample_color_jitter_params(rng)

                def _job(frame_idx: int):
                    return self._load_view(scene_idx, frame_idx, jitter_params)

                if self.io_max_workers > 1 and len(sampled) > 1:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(self.io_max_workers, len(sampled))
                    ) as ex:
                        loaded = list(ex.map(_job, sampled, chunksize=1))
                else:
                    loaded = [_job(frame_idx) for frame_idx in sampled]

                images = torch.stack([item[0] for item in loaded], dim=0)
                depths = torch.stack([item[1] for item in loaded], dim=0)
                masks = torch.stack([item[2] for item in loaded], dim=0)
                intrinsics = torch.stack([item[3] for item in loaded], dim=0)
                extrinsics = torch.stack([item[4] for item in loaded], dim=0)
                world_points = torch.stack([item[5] for item in loaded], dim=0)
                frame_ids = torch.tensor(sampled, dtype=torch.long)
                view_ids = torch.zeros(len(sampled), dtype=torch.long)

                if is_video:
                    perm = torch.arange(len(sampled), dtype=torch.long)
                else:
                    perm = torch.from_numpy(rng.permutation(len(sampled))).long()
                return {
                    "images": images[perm],
                    "depths": depths[perm],
                    "point_masks": masks[perm],
                    "intrinsics": intrinsics[perm],
                    "extrinsics": extrinsics[perm],
                    "world_points": world_points[perm],
                    "frame_ids": frame_ids[perm],
                    "view_ids": view_ids[perm],
                    "scene": self.scenes[scene_idx],
                    "sample_mode": "arkitscenes_highres",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to load a valid ARKitScenesHighRes sample near index {index}: {last_error}"
        )
