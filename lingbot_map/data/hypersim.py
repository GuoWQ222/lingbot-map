"""Hypersim RGBD dataset adapter for LingBot-MAP.

Sampling mirrors ``base3d-clean/datasets/hypersim.py``:

    pos, ordered_video = get_seq_from_start_id(
        num_views,
        start_id,
        all_image_ids,
        rng,
        max_interval=4,
        block_shuffle=16,
    )

The omitted arguments keep BaseMultiViewDataset defaults
``min_interval=1``, ``video_prob=0.5`` and ``fix_interval_prob=0.5``. This
adapter does not use Manip_long trajectory sampling.
"""
from __future__ import annotations

import concurrent.futures
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from .dl3dv import _get_seq_from_start_id


def _read_pose_file(pose_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (OpenCV w2c, K) from the six-line Hypersim pose file."""
    with open(pose_path, "r", encoding="utf-8") as handle:
        rows = [[float(value) for value in line.split()] for line in handle if line.strip()]
    if len(rows) < 6 or any(len(row) != 4 for row in rows[:3]) or any(
        len(row) != 3 for row in rows[3:6]
    ):
        raise ValueError(f"Invalid Hypersim pose file: {pose_path}")
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = np.asarray(rows[:3], dtype=np.float64)
    if not np.isfinite(w2c).all():
        raise ValueError(f"Non-finite Hypersim camera pose: {pose_path}")
    K = np.asarray(rows[3:6], dtype=np.float32)
    if not np.isfinite(K).all():
        raise ValueError(f"Non-finite Hypersim intrinsics: {pose_path}")
    return w2c, K


def _load_scene_metadata(root: str, scene: str) -> Optional[np.ndarray]:
    scene_dir = osp.join(root, scene)
    rgb_dir = osp.join(scene_dir, "rgb")
    depth_dir = osp.join(scene_dir, "depth")
    pose_dir = osp.join(scene_dir, "pose")
    if not all(osp.isdir(path) for path in (rgb_dir, depth_dir, pose_dir)):
        return None

    rgb_names = {
        osp.splitext(name)[0]
        for name in os.listdir(rgb_dir)
        if name.lower().endswith(".jpg")
    }
    depth_names = {
        osp.splitext(name)[0]
        for name in os.listdir(depth_dir)
        if name.lower().endswith(".png")
    }
    pose_names = {
        osp.splitext(name)[0]
        for name in os.listdir(pose_dir)
        if name.lower().endswith(".txt")
    }
    if not (rgb_names == depth_names == pose_names):
        return None
    basenames = sorted(rgb_names)
    if not basenames:
        return None
    return np.asarray(basenames)


def _load_one(args: Tuple[str, str]):
    root, scene = args
    return scene, _load_scene_metadata(root, scene)


class HypersimTrajectoryDataset(Dataset):
    """Hypersim adapter emitting LingBot-MAP sample dicts."""

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
        min_interval: int = 1,
        max_interval: int = 4,
        video_prob: float = 0.5,
        fix_interval_prob: float = 0.5,
        block_shuffle: Optional[int] = 16,
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
        effective_min_views = int(num_views) if min_views is None else int(min_views)
        if not (1 <= effective_min_views <= int(num_views)):
            raise ValueError(
                f"min_views must satisfy 1 <= min_views <= num_views; got "
                f"min_views={effective_min_views}, num_views={num_views}"
            )

        self.root = root.rstrip("/")
        self.split = split
        self.num_views = int(num_views)
        self.min_views = effective_min_views
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

        candidate_scenes = sorted(
            scene
            for scene in os.listdir(self.root)
            if scene.startswith("ai") and osp.isdir(osp.join(self.root, scene))
        )
        if not candidate_scenes:
            raise FileNotFoundError(f"No ai* directories found under {self.root}")
        if verbose:
            print(f"[hypersim] discovered {len(candidate_scenes)} scene directories under {self.root}")

        cache_path = osp.join(self.root, "hypersim_lingbot_valid_cache.npy")
        if osp.exists(cache_path):
            try:
                cache = np.load(cache_path, allow_pickle=True).item()
                self.scene_frames: Dict[str, Dict[str, np.ndarray]] = cache["scene_frames"]
                self.num_imgs: Dict[str, int] = cache["num_imgs"]
                if not all(
                    isinstance(meta, dict) and "basenames" in meta
                    for meta in self.scene_frames.values()
                ):
                    raise ValueError("cache does not contain Hypersim basenames")
                self.sequences = list(self.scene_frames.keys())
                if verbose:
                    print(f"[hypersim] loaded cache: {cache_path} ({len(self.sequences)} scenes)")
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[hypersim] cache load failed ({exc}); rebuilding")
                self._build_cache(candidate_scenes, cache_path, verbose)
        else:
            self._build_cache(candidate_scenes, cache_path, verbose)

        keep = [s for s in self.sequences if self.num_imgs[s] >= self.min_views]
        if len(keep) != len(self.sequences):
            dropped = len(self.sequences) - len(keep)
            self.sequences = keep
            if verbose:
                print(
                    f"[hypersim] dropped {dropped} scenes with < {self.min_views} frames; "
                    f"{len(self.sequences)} scenes remain"
                )

        if not self.sequences:
            raise RuntimeError(
                f"No valid Hypersim scenes with at least {self.min_views} frames under {self.root}"
            )

        print(
            f"[hypersim] {split}: {len(self.sequences)} scenes, "
            f"{sum(self.num_imgs[s] for s in self.sequences)} matched frames, "
            f"samples_per_scene={self.samples_per_scene}, "
            f"num_views=[{self.min_views}, {self.num_views}]"
        )

    def _build_cache(self, scenes: List[str], cache_path: str, verbose: bool) -> None:
        self.scene_frames = {}
        self.num_imgs = {}
        jobs = [(self.root, scene) for scene in scenes]
        if self.scan_max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.scan_max_workers) as executor:
                iterator = executor.map(_load_one, jobs, chunksize=1)
                if verbose:
                    iterator = tqdm(iterator, total=len(jobs), desc="[hypersim] scan")
                for scene, basenames in iterator:
                    if basenames is None:
                        continue
                    self.scene_frames[scene] = {"basenames": basenames}
                    self.num_imgs[scene] = int(len(basenames))
        else:
            iterator = tqdm(jobs, desc="[hypersim] scan", disable=not verbose)
            for job in iterator:
                scene, basenames = _load_one(job)
                if basenames is None:
                    continue
                self.scene_frames[scene] = {"basenames": basenames}
                self.num_imgs[scene] = int(len(basenames))
        self.sequences = list(self.scene_frames.keys())
        try:
            np.save(cache_path, dict(scene_frames=self.scene_frames, num_imgs=self.num_imgs))
            if verbose:
                print(f"[hypersim] wrote cache: {cache_path} ({len(self.sequences)} scenes)")
        except Exception as exc:  # noqa: BLE001
            print(f"[hypersim] WARNING: failed to write cache to {cache_path}: {exc}")

    def __len__(self) -> int:
        return len(self.sequences) * self.samples_per_scene

    def _sample_color_jitter_params(self, rng: np.random.Generator):
        if self._color_jitter is None or rng.random() >= self.color_jitter_prob:
            return None
        return self._color_jitter.get_params(
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

    def _sample_indices(self, num_imgs: int, rng: np.random.Generator) -> Tuple[List[int], bool]:
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        else:
            target_views = self.num_views
        target_views = min(target_views, num_imgs)
        image_indices = list(range(num_imgs))
        max_start = max(0, num_imgs - target_views)
        start_id = int(rng.integers(0, max_start + 1))
        pos, ordered_video = _get_seq_from_start_id(
            target_views,
            start_id,
            image_indices,
            rng,
            min_interval=self.min_interval,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=self.fix_interval_prob,
            block_shuffle=self.block_shuffle,
        )
        return [image_indices[p] for p in pos], ordered_video

    def _load_view(
        self,
        rgb_path: str,
        depth_path: str,
        K_raw: np.ndarray,
        w2c_4x4: np.ndarray,
        jitter_params,
    ):
        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size

        depth_np = np.asarray(Image.open(depth_path), dtype=np.float32) / 1000.0
        depth_np[~np.isfinite(depth_np)] = 0.0
        depth_h, depth_w = depth_np.shape[:2]
        if (height, width) != (depth_h, depth_w):
            depth_pil_align = Image.fromarray(depth_np.astype(np.float32)).convert("F")
            depth_pil_align = depth_pil_align.resize((width, height), Image.Resampling.NEAREST)
            depth_np = np.asarray(depth_pil_align, dtype=np.float32).copy()

        geometry = self._compute_preprocess_geometry(
            width, height, self.image_size, self.patch_size, self.preprocess_mode
        )
        rgb_pre = self._apply_preprocess_to_image(
            rgb_pil, geometry, resample=Image.Resampling.BICUBIC, fill=(255, 255, 255)
        )
        if jitter_params is not None:
            rgb_pre = self._apply_color_jitter(rgb_pre, jitter_params)
        image_tensor = TF.ToTensor()(rgb_pre)

        depth_pil = Image.fromarray(depth_np.astype(np.float32)).convert("F")
        depth_pil = self._apply_preprocess_to_image(
            depth_pil, geometry, resample=Image.Resampling.NEAREST, fill=0
        )
        depth_np = np.asarray(depth_pil, dtype=np.float32).copy()
        depth = torch.from_numpy(depth_np)

        K = np.asarray(K_raw, dtype=np.float32)
        if K.shape == (4, 4):
            K = K[:3, :3]
        intrinsics = self._preprocess_intrinsics(K, width, height, geometry).float()
        extrinsics = torch.from_numpy(w2c_4x4[:3, :4].astype(np.float32))

        valid = np.isfinite(depth_np) & (depth_np > self.min_depth)
        if self.max_depth > 0:
            valid &= depth_np < self.max_depth
        point_mask = torch.from_numpy(valid.astype(np.bool_))

        world_points = self._depth_to_world_points(depth, intrinsics, extrinsics)
        world_points = torch.where(
            point_mask[..., None], world_points, torch.zeros_like(world_points)
        )
        return image_tensor, depth.float(), point_mask, intrinsics.float(), extrinsics.float(), world_points.float()

    def __getitem__(self, index: int) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        n = len(self.sequences)
        start_scene = index % n
        seed_entropy = int(np.random.randint(0, 2**32, dtype=np.uint32))
        rng = np.random.default_rng(seed_entropy)

        for attempt in range(min(16, n)):
            scene_id = self.sequences[(start_scene + attempt) % n]
            try:
                basenames: np.ndarray = self.scene_frames[scene_id]["basenames"]
                sampled, ordered_video = self._sample_indices(int(self.num_imgs[scene_id]), rng)
                if ordered_video:
                    sampled = sorted(sampled)
                jitter_params = self._sample_color_jitter_params(rng)
                scene_dir = osp.join(self.root, scene_id)

                def _job(view_idx: int):
                    basename = str(basenames[view_idx])
                    rgb_p = osp.join(scene_dir, "rgb", basename + ".jpg")
                    depth_p = osp.join(scene_dir, "depth", basename + ".png")
                    pose_p = osp.join(scene_dir, "pose", basename + ".txt")
                    w2c, K = _read_pose_file(pose_p)
                    return self._load_view(rgb_p, depth_p, K, w2c, jitter_params)

                if self.io_max_workers > 1:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(self.io_max_workers, len(sampled))
                    ) as executor:
                        loaded = list(executor.map(_job, sampled, chunksize=1))
                else:
                    loaded = [_job(v) for v in sampled]

                images = torch.stack([item[0] for item in loaded], dim=0)
                depths = torch.stack([item[1] for item in loaded], dim=0)
                masks = torch.stack([item[2] for item in loaded], dim=0)
                intrinsics = torch.stack([item[3] for item in loaded], dim=0)
                extrinsics = torch.stack([item[4] for item in loaded], dim=0)
                world_points = torch.stack([item[5] for item in loaded], dim=0)
                frame_ids = torch.tensor(sampled, dtype=torch.long)
                view_ids = torch.zeros(len(sampled), dtype=torch.long)

                if ordered_video:
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
                    "scene": scene_id,
                    "sample_mode": "hypersim",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to load a valid Hypersim sample near index {index}: {last_error}"
        )
