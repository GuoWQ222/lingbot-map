"""DL3DV multi-view RGBD dataset adapter for LingBot-MAP.

This module mirrors the data-loading + sampling strategy of
``base3d-clean/datasets/dl3dv.py`` (Hossein/renkerui's pipeline) but emits
samples in LingBot-MAP's :func:`train.collate_rgbd_sequences` schema so the
dataset can sit side-by-side with :class:`ManipTrajectoryDataset`.

Pipeline (per sample):
    1. Pick a scene from ``valid_train.json`` / ``valid_test.json``.
    2. Sample ``num_views`` frames using ``get_seq_from_start_id`` semantics
       (min_interval=1, max_interval=32, video_prob=0.8, fix_interval_prob=0.6,
       block_shuffle=16) — copied verbatim from base3d-clean.
    3. For each frame:
         - load RGB from ``images_4/`` (the /4-downsampled images),
         - load depth from ``dense/depth/`` and clean it
           (sky -> -1, MVS outlier -> 0, drop > 98%-percentile),
         - convert the Blender c2w in ``transforms.json`` to OpenCV w2c,
         - apply LingBot-MAP's standard preprocess (resize + crop/pad + intrinsic
           rescale) so the geometry matches ManipTrajectoryDataset's output.
    4. Stack and return.

Cache:
    First instantiation per (ROOT, mode) scans transforms.json for every scene
    and writes ``dl3dv_{mode}_valid_cache.npy`` next to the JSON splits. This
    cache file is the same format base3d-clean uses, so either side can read
    the other's cache.
"""
from __future__ import annotations

import concurrent.futures
import itertools
import json
import os
import os.path as osp
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Sampling helpers (copied verbatim from
# base3d-clean/datasets/base/base_multiview_dataset.py to avoid a cross-repo
# import). Behavior must stay byte-identical to that implementation.
# ---------------------------------------------------------------------------

def _blockwise_shuffle(x: List[int], rng: np.random.Generator, block_shuffle: Optional[int]) -> List[int]:
    if block_shuffle is None:
        return rng.permutation(x).tolist()
    assert block_shuffle > 0
    blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
    shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
    return [item for block in shuffled_blocks for item in block]


def _get_seq_from_start_id(
    num_views: int,
    id_ref: int,
    ids_all: List[int],
    rng: np.random.Generator,
    min_interval: int = 1,
    max_interval: int = 25,
    video_prob: float = 0.5,
    fix_interval_prob: float = 0.5,
    block_shuffle: Optional[int] = None,
) -> Tuple[List[int], bool]:
    assert min_interval > 0
    assert min_interval <= max_interval
    assert id_ref in ids_all
    pos_ref = ids_all.index(id_ref)
    all_possible_pos = np.arange(pos_ref, len(ids_all))
    remaining_sum = len(ids_all) - 1 - pos_ref

    if remaining_sum >= num_views - 1:
        if remaining_sum == num_views - 1:
            return [pos_ref + i for i in range(num_views)], True
        max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
        intervals = [
            rng.choice(range(min_interval, max_interval + 1))
            for _ in range(num_views - 1)
        ]
        if rng.random() < video_prob:
            if rng.random() < fix_interval_prob:
                fixed_interval = rng.choice(
                    range(1, min(remaining_sum // (num_views - 1) + 1, max_interval + 1))
                )
                intervals = [fixed_interval for _ in range(num_views - 1)]
            is_video = True
        else:
            is_video = False
        pos = list(itertools.accumulate([pos_ref] + intervals))
        pos = [p for p in pos if p < len(ids_all)]
        pos_candidates = [p for p in all_possible_pos if p not in pos]
        pos = pos + rng.choice(pos_candidates, num_views - len(pos), replace=False).tolist()
        pos = sorted(pos) if is_video else _blockwise_shuffle(pos, rng, block_shuffle)
    else:
        uniq_num = remaining_sum
        new_pos_ref = rng.choice(np.arange(pos_ref + 1))
        new_remaining_sum = len(ids_all) - 1 - new_pos_ref
        new_max_interval = min(max_interval, new_remaining_sum // max(uniq_num - 1, 1))
        new_intervals = [
            rng.choice(range(1, new_max_interval + 1)) for _ in range(max(uniq_num - 1, 0))
        ]
        revisit_random = rng.random()
        video_random = rng.random()
        if rng.random() < fix_interval_prob and video_random < video_prob:
            fixed_interval = rng.choice(range(1, max(new_max_interval, 1) + 1))
            new_intervals = [fixed_interval for _ in range(max(uniq_num - 1, 0))]
        pos = list(itertools.accumulate([new_pos_ref] + new_intervals))
        is_video = False
        if revisit_random < 0.5 or video_prob == 1.0:
            is_video = video_random < video_prob
            pos = (
                _blockwise_shuffle(pos, rng, block_shuffle)
                if not is_video
                else pos
            )
            num_full_repeat = num_views // max(uniq_num, 1)
            pos = pos * num_full_repeat + pos[: num_views - len(pos) * num_full_repeat]
        elif revisit_random < 0.9:
            pos = rng.choice(pos, num_views, replace=True).tolist()
        else:
            pos = sorted(rng.choice(pos, num_views, replace=True).tolist())
    assert len(pos) == num_views
    return pos, is_video


# ---------------------------------------------------------------------------
# Scene metadata
# ---------------------------------------------------------------------------

_BLENDER2OPENCV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)


def _convert_intrinsics(meta: dict) -> np.ndarray:
    """Build the OpenCV K matrix at images_4 resolution (full /4)."""
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = float(meta["fl_x"]) / 4.0
    K[1, 1] = float(meta["fl_y"]) / 4.0
    K[0, 2] = float(meta["cx"]) / 4.0
    K[1, 2] = float(meta["cy"]) / 4.0
    return K


def _blender2opencv_c2w(pose_4x4: Sequence[Sequence[float]]) -> np.ndarray:
    return (np.asarray(pose_4x4, dtype=np.float64) @ _BLENDER2OPENCV).astype(np.float64)


def _load_metadata(scene_path: str) -> Tuple[List[dict], str]:
    json_path = osp.join(scene_path, "transforms.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    K = _convert_intrinsics(data)
    scene_frames: List[dict] = []
    for frame in data["frames"]:
        scene_frames.append(
            dict(
                file_path=osp.join(scene_path, frame["file_path"]).replace("images", "images_4"),
                intrinsics=K.tolist(),
                extrinsics=_blender2opencv_c2w(frame["transform_matrix"]).tolist(),
            )
        )
    scene_id = scene_path.split("/")[-2] + "_" + scene_path.split("/")[-1]
    return scene_frames, scene_id


def _check_valid(rgb_paths: Sequence[str]) -> bool:
    for rgb_p in rgb_paths:
        if not osp.exists(rgb_p):
            return False
        depth_p = rgb_p.replace("images_4", "dense/depth").replace(".png", ".npy")
        if not osp.exists(depth_p):
            return False
        outlier_p = rgb_p.replace("images_4", "dense/outlier_mask")
        if not osp.exists(outlier_p):
            return False
        sky_p = rgb_p.replace("images_4", "dense/sky_mask")
        if not osp.exists(sky_p):
            return False
    return True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DL3DVTrajectoryDataset(Dataset):
    """DL3DV multi-view dataset emitting LingBot-MAP sample dicts.

    Output dict (one per ``__getitem__``)::

        {
            "images":       (T, 3, H, W) float32 in [0, 1],
            "depths":       (T, H, W) float32,
            "point_masks":  (T, H, W) bool,
            "intrinsics":   (T, 3, 3) float32 (target image pixel coords),
            "extrinsics":   (T, 3, 4) float32 OpenCV w2c,
            "world_points": (T, H, W, 3) float32,
            "frame_ids":    (T,) long,
            "view_ids":     (T,) long (= 0; DL3DV is single-camera),
            "scene":        str,
            "sample_mode":  "dl3dv",
        }
    """

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
        # Sampling knobs (defaults match base3d-clean/datasets/dl3dv.py)
        min_interval: int = 1,
        max_interval: int = 32,
        video_prob: float = 0.8,
        fix_interval_prob: float = 0.6,
        block_shuffle: Optional[int] = 16,
        io_max_workers: int = 8,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        # Lazy-import lingbot helpers to keep this module importable in isolation
        # for unit testing (helpers live in train.py).
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
        # min_views == num_views (default) keeps the original fixed-length
        # behavior; min_views < num_views activates per-call random length.
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

        # Color jitter is sampled per clip (matches ManipTrajectoryDataset behavior)
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

        # Load the JSON split + build / load scene_frames cache
        json_name = f"valid_{split}.json"
        json_path = osp.join(self.root, json_name)
        if not osp.exists(json_path):
            raise FileNotFoundError(
                f"DL3DV split file not found: {json_path}. "
                f"Generate it first (see /cpfs/user/guowenqi/base3d-clean instructions)."
            )
        with open(json_path, "r") as f:
            self.valid_sequence: List[str] = json.load(f)

        cache_path = osp.join(self.root, f"dl3dv_{split}_valid_cache.npy")
        if osp.exists(cache_path):
            cache = np.load(cache_path, allow_pickle=True).item()
            self.scene_frames: Dict[str, Dict[str, np.ndarray]] = cache["scene_frames"]
            self.num_imgs: Dict[str, int] = cache["num_imgs"]
            self.sequences = list(self.scene_frames.keys())
            if verbose:
                print(f"[dl3dv] loaded cache: {cache_path} ({len(self.sequences)} scenes)")
        else:
            self.scene_frames = {}
            self.num_imgs = {}
            iterator = self.valid_sequence
            if verbose:
                iterator = tqdm(iterator, desc=f"[dl3dv] building {split} cache")
            for scene_rel in iterator:
                scene_frames, scene_id = _load_metadata(osp.join(self.root, scene_rel))
                rgb_paths = np.array([fr["file_path"] for fr in scene_frames])
                intrinsics = np.array([fr["intrinsics"] for fr in scene_frames])
                extrinsics = np.array([fr["extrinsics"] for fr in scene_frames])
                if not _check_valid(rgb_paths):
                    continue
                self.scene_frames[scene_id] = dict(
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    rgb_paths=rgb_paths,
                )
                self.num_imgs[scene_id] = len(rgb_paths)
            self.sequences = list(self.scene_frames.keys())
            try:
                np.save(cache_path, dict(scene_frames=self.scene_frames, num_imgs=self.num_imgs))
                if verbose:
                    print(f"[dl3dv] wrote cache: {cache_path} ({len(self.sequences)} scenes)")
            except Exception as exc:  # noqa: BLE001
                print(f"[dl3dv] WARNING: failed to write cache to {cache_path}: {exc}")

        if not self.sequences:
            raise RuntimeError(f"No valid DL3DV scenes found under {self.root} (split={split})")

        print(
            f"[dl3dv] {split}: {len(self.sequences)} scenes, "
            f"{sum(self.num_imgs.values())} frames total, "
            f"num_views={self.num_views}, image_size={self.image_size}"
        )

    # ---- length / __getitem__ -------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences) * self.samples_per_scene

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

    def _sample_indices(self, num_imgs: int, rng: np.random.Generator) -> Tuple[List[int], bool]:
        # Per-call random target length in [min_views, num_views]. Identical
        # to the fixed behavior when min_views == num_views.
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        else:
            target_views = self.num_views
        # Avoid `replace=True` duplicate-frame degradation: if the scene has
        # fewer frames than `target_views`, shrink target to fit instead of
        # padding with duplicate frames.
        if num_imgs < target_views:
            target_views = num_imgs
        image_indices = list(range(num_imgs))
        if num_imgs <= target_views:
            # Scene length == target — return the full trajectory in order
            # (the base3d-clean `rng.choice(replace=True)` path duplicated
            # frames, which is bad training signal; we skip that).
            return image_indices, True
        start_id = int(rng.integers(0, num_imgs - target_views))
        pos, is_video = _get_seq_from_start_id(
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
        return [image_indices[p] for p in pos], is_video

    def _load_view(
        self,
        rgb_path: str,
        intrinsic_raw_4x4_or_3x3: np.ndarray,
        c2w_4x4: np.ndarray,
        jitter_params,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Load + clean + preprocess one frame; return tensors in lingbot schema."""
        # ---- Paths
        dir_path = osp.dirname(osp.dirname(rgb_path))
        file_name = osp.splitext(osp.basename(rgb_path))[0]
        depth_path = osp.join(dir_path, "dense", "depth", file_name + ".npy")
        sky_path = osp.join(dir_path, "dense", "sky_mask", file_name + ".png")
        outlier_path = osp.join(dir_path, "dense", "outlier_mask", file_name + ".png")

        # ---- RGB
        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size  # (W, H)

        # ---- Depth (clean BEFORE resize, like base3d-clean)
        depthmap = np.load(depth_path).astype(np.float32)
        depthmap[~np.isfinite(depthmap)] = 0.0
        sky_mask = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED) >= 127
        outlier_mask = cv2.imread(outlier_path, cv2.IMREAD_UNCHANGED)
        # Sky -> -1 (then we'll treat as invalid post-clip); MVS outlier -> 0
        depthmap[sky_mask] = -1.0
        depthmap[outlier_mask >= 127] = 0.0
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        pos_d = depthmap[depthmap > 0]
        if pos_d.size > 0:
            threshold = float(np.percentile(pos_d, 98))
            depthmap[depthmap > threshold] = 0.0
        # Resize depth to RGB native res (depth comes from MVS at slightly different scale)
        depthmap = cv2.resize(depthmap, (width, height), interpolation=cv2.INTER_NEAREST)

        # ---- Geometry: lingbot preprocess (resize + crop/pad)
        geometry = self._compute_preprocess_geometry(
            width, height, self.image_size, self.patch_size, self.preprocess_mode
        )

        # RGB preprocess
        rgb_pre = self._apply_preprocess_to_image(
            rgb_pil, geometry, resample=Image.Resampling.BICUBIC, fill=(255, 255, 255)
        )
        if jitter_params is not None:
            rgb_pre = self._apply_color_jitter(rgb_pre, jitter_params)
        image_tensor = TF.ToTensor()(rgb_pre)  # (3, H, W) float in [0,1]

        # Depth preprocess via PIL (float32 -> 'F' mode supports nearest)
        depth_pil = Image.fromarray(depthmap.astype(np.float32), mode="F")
        depth_pil = self._apply_preprocess_to_image(
            depth_pil, geometry, resample=Image.Resampling.NEAREST, fill=0
        )
        depth_np = np.asarray(depth_pil, dtype=np.float32).copy()
        depth = torch.from_numpy(depth_np)

        # Intrinsics: input matrix is at original RGB res (images_4); preprocess_intrinsics rescales.
        K_raw = np.asarray(intrinsic_raw_4x4_or_3x3, dtype=np.float32)
        if K_raw.shape == (4, 4):
            K_raw = K_raw[:3, :3]
        intrinsics = self._preprocess_intrinsics(K_raw, width, height, geometry).float()

        # Extrinsics: c2w (OpenCV) -> w2c (OpenCV)
        c2w_4x4 = np.asarray(c2w_4x4, dtype=np.float64)
        w2c_4x4 = np.linalg.inv(c2w_4x4)
        extrinsics = torch.from_numpy(w2c_4x4[:3, :4].astype(np.float32))

        # Valid mask: positive depth in finite range (sky was set to -1, outliers to 0)
        valid = np.isfinite(depth_np) & (depth_np > self.min_depth)
        if self.max_depth > 0:
            valid &= depth_np < self.max_depth
        point_mask = torch.from_numpy(valid.astype(np.bool_))

        world_points = self._depth_to_world_points(depth, intrinsics, extrinsics)
        world_points = torch.where(
            point_mask[..., None], world_points, torch.zeros_like(world_points)
        )

        return image_tensor, depth.float(), point_mask, intrinsics.float(), extrinsics.float(), world_points.float(), rgb_path

    def __getitem__(self, index: int) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        n = len(self.sequences)
        start_scene = index % n
        # Worker-local rng — fresh per call so train-time randomness comes from python+numpy default streams.
        # Match base3d-clean's behavior: numpy Generator with non-deterministic seed.
        seed_entropy = int(np.random.randint(0, 2**32, dtype=np.uint32))
        rng = np.random.default_rng(seed_entropy)

        for attempt in range(min(16, n)):
            scene_id = self.sequences[(start_scene + attempt) % n]
            try:
                meta = self.scene_frames[scene_id]
                rgb_paths: np.ndarray = meta["rgb_paths"]
                intrinsics_arr: np.ndarray = meta["intrinsics"]
                extrinsics_arr: np.ndarray = meta["extrinsics"]
                num_imgs = int(self.num_imgs[scene_id])

                sampled, is_video = self._sample_indices(num_imgs, rng)
                if is_video:
                    sampled = sorted(sampled)
                jitter_params = self._sample_color_jitter_params(rng)

                def _job(view_idx: int):
                    return self._load_view(
                        rgb_paths[view_idx],
                        intrinsics_arr[view_idx],
                        extrinsics_arr[view_idx],
                        jitter_params,
                    )

                if self.io_max_workers > 1 and len(sampled) > 1:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(self.io_max_workers, len(sampled))
                    ) as ex:
                        loaded = list(ex.map(_job, sampled, chunksize=1))
                else:
                    loaded = [_job(v) for v in sampled]

                images = torch.stack([l[0] for l in loaded], dim=0)
                depths = torch.stack([l[1] for l in loaded], dim=0)
                masks = torch.stack([l[2] for l in loaded], dim=0)
                intrinsics_t = torch.stack([l[3] for l in loaded], dim=0)
                extrinsics_t = torch.stack([l[4] for l in loaded], dim=0)
                world_points = torch.stack([l[5] for l in loaded], dim=0)
                frame_ids = torch.tensor(sampled, dtype=torch.long)
                view_ids = torch.zeros(len(sampled), dtype=torch.long)

                # Preserve chronological order for video samples; keep unordered
                # multi-view regularization for non-video samples.
                if is_video:
                    perm = torch.arange(len(sampled), dtype=torch.long)
                else:
                    perm = torch.from_numpy(rng.permutation(len(sampled))).long()
                return {
                    "images": images[perm],
                    "depths": depths[perm],
                    "point_masks": masks[perm],
                    "intrinsics": intrinsics_t[perm],
                    "extrinsics": extrinsics_t[perm],
                    "world_points": world_points[perm],
                    "frame_ids": frame_ids[perm],
                    "view_ids": view_ids[perm],
                    "scene": scene_id,
                    "sample_mode": "dl3dv",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load a valid DL3DV sample near index {index}: {last_error}")
