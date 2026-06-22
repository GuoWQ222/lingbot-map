"""ADT RGBD dataset adapter for LingBot-MAP.

Sampling intentionally mirrors ``base3d-clean/datasets/adt.py``:
for each sample, pick one random reference frame, sample the remaining views from
its local temporal window, then shuffle the resulting views. This adapter does
not use Manip_long trajectory sampling.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def _resize_depth(depthmap: np.ndarray, height: int, width: int) -> torch.Tensor:
    depth = torch.from_numpy(depthmap.astype(np.float32))[None, None]
    return F.interpolate(depth, size=(height, width), mode="nearest")[0, 0]


def _preprocess_depth(depthmap: np.ndarray, geometry: Dict[str, int]) -> torch.Tensor:
    depth = _resize_depth(depthmap, geometry["new_height"], geometry["new_width"])
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


def _rotate_camera_clockwise(
    image: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """Rotate preprocessed pixels clockwise while preserving world-space rays."""
    height, width = depth.shape
    image = torch.rot90(image, k=-1, dims=(-2, -1)).contiguous()
    depth = torch.rot90(depth, k=-1, dims=(-2, -1)).contiguous()

    rotated_K = torch.eye(3, dtype=intrinsics.dtype)
    rotated_K[0, 0] = intrinsics[1, 1]
    rotated_K[1, 1] = intrinsics[0, 0]
    rotated_K[0, 2] = height - 1 - intrinsics[1, 2]
    rotated_K[1, 2] = intrinsics[0, 2]

    new_camera_to_old_camera = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotated_c2w = np.asarray(c2w, dtype=np.float64) @ new_camera_to_old_camera
    return image, depth, rotated_K, rotated_c2w


def _convert_intrinsics(frame: dict) -> np.ndarray:
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = float(frame["fx"])
    intrinsics[1, 1] = float(frame["fy"])
    intrinsics[0, 2] = float(frame["cx"])
    intrinsics[1, 2] = float(frame["cy"])
    return intrinsics


def _load_metadata(root: str, scene: str) -> Tuple[List[dict], str]:
    scene_root = osp.join(root, scene)
    json_path = osp.join(scene_root, "transforms.json")
    with open(json_path, "r") as f:
        data = json.load(f)

    frames: List[dict] = []
    for frame in data["frames"]:
        rgb_path = osp.join(scene_root, frame["image_path"])
        depth_path = osp.join(scene_root, frame["depth_path"])
        if not osp.exists(rgb_path) or not osp.exists(depth_path):
            continue
        frames.append(
            dict(
                file_path=rgb_path,
                depth_path=depth_path,
                intrinsics=_convert_intrinsics(frame).tolist(),
                extrinsics=frame["transform_matrix"],
            )
        )
    return frames, scene.split("/")[-1]


class ADTTrajectoryDataset(Dataset):
    """ADT adapter emitting LingBot-MAP sample dicts.

    The frame sampling is a direct port of ``ADT_Multi._get_views`` from
    ``base3d-clean/datasets/adt.py``. Image/depth preprocessing follows the
    standard LingBot-MAP external-dataset adapters so the output schema matches
    ``collate_rgbd_sequences``.
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
        max_distance: int = 128,
        rotate_clockwise: bool = True,
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
        if split not in ("train", "test", "valid", "val"):
            raise ValueError(f"Unsupported ADT split {split!r}; ADT uses the same scene pool for all splits")

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
        self.rotate_clockwise = bool(rotate_clockwise)
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

        cache_path = osp.join(self.root, "adt_cache.npy")
        if osp.exists(cache_path):
            cache = np.load(cache_path, allow_pickle=True).item()
            self.scene_frames: Dict[str, Dict[str, np.ndarray]] = cache["scene_frames"]
            self.num_imgs: Dict[str, int] = cache["num_imgs"]
            self.sequences = list(self.scene_frames.keys())
            if verbose:
                print(f"[adt] loaded cache: {cache_path} ({len(self.sequences)} scenes)")
        else:
            self.scene_frames = {}
            self.num_imgs = {}
            scenes = [
                scene
                for scene in os.listdir(self.root)
                if osp.isdir(osp.join(self.root, scene))
            ]
            iterator: Sequence[str] = scenes
            if verbose:
                iterator = tqdm(scenes, desc="[adt] building cache")
            for scene in iterator:
                try:
                    frames, scene_id = _load_metadata(self.root, scene)
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f"[adt] skip scene {scene}: {exc}")
                    continue
                if not frames:
                    continue
                self.scene_frames[scene_id] = dict(
                    intrinsics=np.array([fr["intrinsics"] for fr in frames]),
                    extrinsics=np.array([fr["extrinsics"] for fr in frames]),
                    rgb_paths=np.array([fr["file_path"] for fr in frames]),
                    depth_paths=np.array([fr["depth_path"] for fr in frames]),
                )
                self.num_imgs[scene_id] = len(frames)
            self.sequences = list(self.scene_frames.keys())
            try:
                np.save(cache_path, dict(scene_frames=self.scene_frames, num_imgs=self.num_imgs))
                if verbose:
                    print(f"[adt] wrote cache: {cache_path} ({len(self.sequences)} scenes)")
            except Exception as exc:  # noqa: BLE001
                print(f"[adt] WARNING: failed to write cache to {cache_path}: {exc}")

        if not self.sequences:
            raise RuntimeError(f"No valid ADT scenes found under {self.root}")

        print(
            f"[adt] {split}: {len(self.sequences)} scenes, "
            f"{sum(self.num_imgs.values())} frames total, "
            f"num_views=[{self.min_views}, {self.num_views}], "
            f"max_distance={self.max_distance}, "
            f"image_size={self.image_size}, rotate_clockwise={self.rotate_clockwise}"
        )

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

    def _sample_indices(self, num_imgs: int, rng: np.random.Generator) -> List[int]:
        # Direct port of base3d-clean/datasets/adt.py::ADT_Multi._get_views.
        target_views = self.num_views
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        idxs = [int(rng.integers(0, num_imgs))]
        max_distance = int(self.max_distance / 8 * target_views)
        start_idx = max(0, idxs[-1] - max_distance)
        end_idx = min(num_imgs - 1, start_idx + max_distance)
        start_idx = max(0, end_idx - max_distance)
        valid_indices = np.arange(start_idx, end_idx + 1)
        should_replace = len(valid_indices) < target_views - 1
        idxs.extend(
            list(rng.choice(valid_indices, target_views - 1, replace=should_replace))
        )
        return [int(i) for i in idxs]

    def _load_view(
        self,
        rgb_path: str,
        depth_path: str,
        intrinsic_raw: np.ndarray,
        c2w_4x4: np.ndarray,
        jitter_params,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size

        depthmap = np.load(depth_path).astype(np.float32)
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        if depthmap.shape[:2] != (height, width):
            depthmap = _resize_depth(depthmap, height, width).numpy()

        K_raw = np.asarray(intrinsic_raw, dtype=np.float32)
        c2w_4x4 = np.asarray(c2w_4x4, dtype=np.float64)

        geometry = self._compute_preprocess_geometry(
            width, height, self.image_size, self.patch_size, self.preprocess_mode
        )

        rgb_pre = self._apply_preprocess_to_image(
            rgb_pil, geometry, resample=Image.Resampling.BICUBIC, fill=(255, 255, 255)
        )
        if jitter_params is not None:
            rgb_pre = self._apply_color_jitter(rgb_pre, jitter_params)
        image_tensor = TF.ToTensor()(rgb_pre)

        depth = _preprocess_depth(depthmap, geometry)

        if K_raw.shape == (4, 4):
            K_raw = K_raw[:3, :3]
        intrinsics = self._preprocess_intrinsics(K_raw, width, height, geometry).float()
        if self.rotate_clockwise:
            image_tensor, depth, intrinsics, c2w_4x4 = _rotate_camera_clockwise(
                image_tensor, depth, intrinsics, c2w_4x4
            )
        depth_np = depth.numpy()

        w2c_4x4 = np.linalg.inv(c2w_4x4)
        extrinsics = torch.from_numpy(w2c_4x4[:3, :4].astype(np.float32))

        valid = np.isfinite(depth_np) & (depth_np > self.min_depth)
        if self.max_depth > 0:
            valid &= depth_np < self.max_depth
        point_mask = torch.from_numpy(valid.astype(np.bool_))

        world_points = self._depth_to_world_points(depth, intrinsics, extrinsics)
        world_points = torch.where(point_mask[..., None], world_points, torch.zeros_like(world_points))

        return image_tensor, depth.float(), point_mask, intrinsics.float(), extrinsics.float(), world_points.float(), rgb_path

    def __getitem__(self, index: int) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        n = len(self.sequences)
        start_scene = index % n
        seed_entropy = int(np.random.randint(0, 2**32, dtype=np.uint32))
        rng = np.random.default_rng(seed_entropy)

        for attempt in range(min(16, n)):
            scene_id = self.sequences[(start_scene + attempt) % n]
            try:
                meta = self.scene_frames[scene_id]
                rgb_paths: np.ndarray = meta["rgb_paths"]
                depth_paths: np.ndarray = meta["depth_paths"]
                intrinsics_arr: np.ndarray = meta["intrinsics"]
                extrinsics_arr: np.ndarray = meta["extrinsics"]
                num_imgs = int(self.num_imgs[scene_id])

                sampled = self._sample_indices(num_imgs, rng)
                jitter_params = self._sample_color_jitter_params(rng)

                def _job(view_idx: int):
                    return self._load_view(
                        str(rgb_paths[view_idx]),
                        str(depth_paths[view_idx]),
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
                    "sample_mode": "adt",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load a valid ADT sample near index {index}: {last_error}")
