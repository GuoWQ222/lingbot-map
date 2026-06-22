"""CO3D RGBD dataset adapter for LingBot-MAP.

Sampling is intentionally delegated to
``base3d-clean/datasets/co3d_20251006.py``. That keeps CO3D's original random
view sampling and avoids reusing the Manip_long trajectory sampler.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as TF

from lingbot_map.data.oss_stage_cache import OssStageCache


def _load_base3d_co3d(base3d_root: str):
    root = os.path.abspath(os.path.expanduser(base3d_root))
    if root not in sys.path:
        sys.path.insert(0, root)
    module = importlib.import_module("datasets.co3d_20251006")
    return module.Co3d_Multi


def _co3d_mask_bg_value(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "rand":
            return "rand"
    return value


class _Co3dInitLock:
    def __init__(self, path: Path, *, stale_seconds: int = 60 * 60) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.acquired = False

    def __enter__(self) -> "_Co3dInitLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                self.path.mkdir()
                self.acquired = True
                (self.path / "pid").write_text(str(os.getpid()))
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        shutil.rmtree(self.path, ignore_errors=True)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(0.2)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.acquired:
            shutil.rmtree(self.path, ignore_errors=True)


class Co3dTrajectoryDataset(Dataset):
    """Expose CO3D samples in LingBot-MAP's external dataset schema."""

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
        mask_bg=True,
        base3d_root: str = "/cpfs/user/guowenqi/base3d-clean",
        oss_stage_cache: Optional[OssStageCache] = None,
        oss_uri_root: str = "",
        verbose: bool = False,
    ) -> None:
        super().__init__()
        from train import depth_to_world_points

        self._depth_to_world_points = depth_to_world_points
        del preprocess_mode  # CO3D crop/resize is owned by the base3d dataset.

        if num_views <= 0:
            raise ValueError(f"num_views must be > 0, got {num_views}")
        effective_min_views = int(num_views) if min_views is None else int(min_views)
        if not (1 <= effective_min_views <= int(num_views)):
            raise ValueError(
                f"min_views must satisfy 1 <= min_views <= num_views; got "
                f"min_views={effective_min_views}, num_views={num_views}"
            )

        self.root = os.path.abspath(os.path.expanduser(root))
        self.split = split
        self.num_views = int(num_views)
        self.min_views = effective_min_views
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.samples_per_scene = max(1, int(samples_per_scene))
        self.color_jitter_prob = float(color_jitter_prob)
        self.oss_stage_cache = oss_stage_cache
        self.oss_uri_root = str(oss_uri_root or "").rstrip("/")
        self._staged_root = (
            oss_stage_cache.stage_root / "CO3Dv2" if oss_stage_cache is not None else None
        )

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )

        if color_jitter_strength > 0 and color_jitter_prob > 0:
            self._color_jitter = TF.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=min(color_jitter_strength * 0.5, 0.5),
            )
        else:
            self._color_jitter = None

        Co3dMulti = _load_base3d_co3d(base3d_root)

        def _build_dataset(init_root: str):
            return Co3dMulti(
                ROOT=init_root,
                split=self.split,
                num_views=self.num_views,
                resolution=self.image_size,
                transform="ImgToTensor",
                mask_bg=_co3d_mask_bg_value(mask_bg),
                seed=None,
            )

        init_root = self.root
        try:
            self.dataset = _build_dataset(init_root)
        except Exception as exc:  # noqa: BLE001
            if (
                self.oss_stage_cache is None
                or not self.oss_stage_cache.enable_fallback_from_exception(exc, source_path=self.root)
            ):
                raise
            init_root = self._stage_metadata_root()
            with self._init_cache_lock():
                self.dataset = _build_dataset(init_root)
        if len(self.dataset) == 0:
            raise RuntimeError(f"CO3D split {self.split!r} is empty under {init_root}")
        if verbose:
            print(
                f"[co3d] adapter split={self.split} scenes={len(self.dataset)} "
                f"views=[{self.min_views}, {self.num_views}] root={init_root}"
            )

    def __len__(self) -> int:
        return len(self.dataset) * self.samples_per_scene

    def _sample_nviews(self) -> int:
        if self.min_views == self.num_views:
            return self.num_views
        return int(np.random.randint(self.min_views, self.num_views + 1))

    def _co3d_oss_uri(self, scene_name: str) -> str:
        return self.oss_uri_root + "/" + scene_name.strip("/") + "/"

    def _stage_metadata_root(self) -> str:
        if self.oss_stage_cache is None or self._staged_root is None or not self.oss_uri_root:
            return self.root
        staged = self.oss_stage_cache.stage_dir(
            dataset="co3d_meta",
            relative_key="co3d_anno",
            oss_uri=self.oss_uri_root + "/co3d_anno/",
            dest_root=self._staged_root,
            count_entry=False,
        )
        return str(staged.parent)

    def _init_cache_lock(self):
        if self._staged_root is None:
            class _NoopLock:
                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc, tb):
                    return None

            return _NoopLock()
        lock_path = self._staged_root / ".locks" / f"co3d_init_{self.split}.lock"
        return _Co3dInitLock(lock_path)

    def _stage_scene(self, scene_idx: int) -> str:
        if self.oss_stage_cache is None or self._staged_root is None or not self.oss_uri_root:
            return self.root
        scene_name = self.dataset.scenes[scene_idx]
        source_path = Path(self.root) / scene_name
        staged = self.oss_stage_cache.resolve_dir(
            source_path=source_path,
            dataset="co3d",
            relative_key=scene_name,
            oss_uri=self._co3d_oss_uri(scene_name),
            dest_root=self._staged_root,
        )
        return str(staged.parent.parent)

    def _load_views_with_fallback(self, scene_idx: int, nviews: int):
        original_root = self.dataset.ROOT
        read_root = self._stage_scene(scene_idx)
        try:
            self.dataset.ROOT = read_root
            return self.dataset[(scene_idx, 0, nviews)]
        except Exception as exc:  # noqa: BLE001
            if (
                self.oss_stage_cache is None
                or not self.oss_stage_cache.enable_fallback_from_exception(
                    exc,
                    source_path=Path(self.root) / self.dataset.scenes[scene_idx],
                )
            ):
                raise
            read_root = self._stage_scene(scene_idx)
            self.dataset.ROOT = read_root
            return self.dataset[(scene_idx, 0, nviews)]
        finally:
            self.dataset.ROOT = original_root

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        n_scenes = len(self.dataset)
        last_error: Optional[Exception] = None
        for attempt in range(min(16, n_scenes)):
            scene_idx = (int(index) + attempt) % n_scenes
            nviews = self._sample_nviews()
            try:
                views = self._load_views_with_fallback(scene_idx, nviews)
                return self._convert_views(scene_idx, views)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(f"Failed to load CO3D sample near index {index}: {last_error}")

    def _convert_views(self, scene_idx: int, views) -> Dict[str, torch.Tensor]:
        images = []
        depths = []
        point_masks = []
        intrinsics = []
        extrinsics = []
        world_points = []
        labels = []

        for view in views:
            image = view["img"]
            if not torch.is_tensor(image):
                image = TF.ToTensor()(image)
            image = image.float()
            if self._color_jitter is not None and np.random.random() < self.color_jitter_prob:
                image = self._color_jitter(image).clamp_(0.0, 1.0)

            depth_np = np.asarray(view["depthmap"], dtype=np.float32)
            depth_np = np.where(np.isfinite(depth_np), depth_np, 0.0)
            depth = torch.from_numpy(depth_np.copy()).float()

            K_np = np.asarray(view["camera_intrinsics"], dtype=np.float32)[:3, :3]
            K = torch.from_numpy(K_np.copy()).float()

            c2w_np = np.asarray(view["camera_pose"], dtype=np.float32)
            if c2w_np.shape == (3, 4):
                c2w_full = np.eye(4, dtype=np.float32)
                c2w_full[:3, :4] = c2w_np
                c2w_np = c2w_full
            w2c_np = np.linalg.inv(c2w_np).astype(np.float32)[:3, :4]
            extr = torch.from_numpy(w2c_np.copy()).float()

            valid = torch.isfinite(depth) & (depth > self.min_depth)
            if self.max_depth > 0.0:
                valid &= depth < self.max_depth
            points = self._depth_to_world_points(depth, K, extr)
            points = torch.where(valid[..., None], points, torch.zeros_like(points))

            images.append(image)
            depths.append(depth)
            point_masks.append(valid.bool())
            intrinsics.append(K)
            extrinsics.append(extr)
            world_points.append(points)
            labels.append(str(view.get("label", "")))

        if not images:
            raise RuntimeError(f"CO3D scene {scene_idx} produced no views")

        scene = next((label for label in labels if label), f"co3d/{scene_idx}")
        n = len(images)
        return {
            "images": torch.stack(images, dim=0),
            "depths": torch.stack(depths, dim=0),
            "point_masks": torch.stack(point_masks, dim=0),
            "intrinsics": torch.stack(intrinsics, dim=0),
            "extrinsics": torch.stack(extrinsics, dim=0),
            "world_points": torch.stack(world_points, dim=0),
            "frame_ids": torch.arange(n, dtype=torch.long),
            "view_ids": torch.arange(n, dtype=torch.long),
            "scene": scene,
            "sample_mode": "co3d",
        }
