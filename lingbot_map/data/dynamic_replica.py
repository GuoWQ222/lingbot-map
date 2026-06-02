"""DynamicReplica multi-view RGBD dataset adapter for LingBot-MAP.

This module mirrors the data-loading + sampling strategy of
``base3d-clean/datasets/dynamic_replica.py`` but emits samples in LingBot-MAP's
:func:`train.collate_rgbd_sequences` schema so the dataset can sit alongside
:class:`ManipTrajectoryDataset`, :class:`DL3DVTrajectoryDataset`,
:class:`ScanNetppTrajectoryDataset` and :class:`TartanAirTrajectoryDataset`.

Sampling parameters are taken from base3d-clean/datasets/dynamic_replica.py:
``max_interval=64, video_prob=1.0, fix_interval_prob=1.0`` (always video,
always fixed interval). ``min_interval=1`` and ``block_shuffle=16`` follow
the convention used by the sibling adapters.

Directory layout (matches /shared/smartbot/renkerui/data/dynamic_replica)::

    ROOT/
      {split}/                          # train | valid | test
        <scene>/
          left/                         # we only use the left camera
            rgb/<timestamp>.png         # 1280x720 RGB
            depth/<timestamp>.npy       # float32 meters, (720, 1280)
            cam/<timestamp>.npz         # keys: 'intrinsics' (3,3), 'pose' (4,4) c2w
            mask/                       # NOT USED by this adapter
          right/                        # NOT USED by this adapter

Cache:
    First instantiation per (ROOT, split) scans every (scene, basename) pair
    and writes ``dynamic_replica_lingbot_<split>_cache.npy`` next to ROOT for
    fast subsequent loads.

Output dict (one per ``__getitem__``) — same schema as the sibling adapters::

    {
        "images":       (T, 3, H, W) float32 in [0, 1],
        "depths":       (T, H, W) float32 meters,
        "point_masks":  (T, H, W) bool,
        "intrinsics":   (T, 3, 3) float32 (target image pixel coords),
        "extrinsics":   (T, 3, 4) float32 OpenCV w2c,
        "world_points": (T, H, W, 3) float32,
        "frame_ids":    (T,) long,
        "view_ids":     (T,) long (= 0; left camera only),
        "scene":        str,
        "sample_mode":  "dynamic_replica",
    }
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

# Re-use the byte-identical port of base3d-clean's get_seq_from_start_id that
# already lives next to us (it was copied verbatim from
# base3d-clean/datasets/base/base_multiview_dataset.py).
from .dl3dv import _get_seq_from_start_id


# ---------------------------------------------------------------------------
# Per-scene metadata loading
# ---------------------------------------------------------------------------

def _load_scene_metadata(
    root: str, split: str, scene: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build (rgb_paths, depth_paths, intrinsics_arr, c2w_arr) for one scene.

    The base3d-clean dataloader sorts basenames by float value
    (``key=lambda x: float(x)``); we replicate that exactly so frame ordering
    matches. Returns ``None`` if the scene is missing required subfolders.
    """
    scene_dir = osp.join(root, split, scene, "left")
    rgb_dir = osp.join(scene_dir, "rgb")
    depth_dir = osp.join(scene_dir, "depth")
    cam_dir = osp.join(scene_dir, "cam")

    if not (osp.isdir(rgb_dir) and osp.isdir(depth_dir) and osp.isdir(cam_dir)):
        return None

    try:
        basenames = sorted(
            (f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")),
            key=lambda x: float(x),
        )
    except (OSError, ValueError):
        return None
    if not basenames:
        return None

    rgb_paths: List[str] = []
    depth_paths: List[str] = []
    Ks: List[np.ndarray] = []
    c2ws: List[np.ndarray] = []
    for b in basenames:
        rgb_p = osp.join(rgb_dir, b + ".png")
        depth_p = osp.join(depth_dir, b + ".npy")
        cam_p = osp.join(cam_dir, b + ".npz")
        if not (osp.isfile(depth_p) and osp.isfile(cam_p)):
            # Skip frames with missing companion files (mirrors base3d-clean's
            # implicit assumption — it would fail at __getitem__ time).
            continue
        try:
            with np.load(cam_p) as cam:
                Ks.append(np.asarray(cam["intrinsics"], dtype=np.float64))
                c2ws.append(np.asarray(cam["pose"], dtype=np.float64))
        except Exception:
            continue
        rgb_paths.append(rgb_p)
        depth_paths.append(depth_p)

    if not rgb_paths:
        return None

    return (
        np.array(rgb_paths),
        np.array(depth_paths),
        np.stack(Ks, axis=0),
        np.stack(c2ws, axis=0),
    )


def _load_one(args: Tuple[str, str, str]):
    root, split, scene = args
    out = _load_scene_metadata(root, split, scene)
    return scene, out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DynamicReplicaTrajectoryDataset(Dataset):
    """DynamicReplica multi-view dataset emitting LingBot-MAP sample dicts.

    Sampling parameters intentionally match
    ``base3d-clean/datasets/dynamic_replica.py`` (and ONLY that file).
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
        # Sampling knobs — DEFAULTS COPIED VERBATIM FROM
        # base3d-clean/datasets/dynamic_replica.py (do NOT change).
        min_interval: int = 1,
        max_interval: int = 64,
        video_prob: float = 1.0,
        fix_interval_prob: float = 1.0,
        block_shuffle: Optional[int] = 16,
        scan_max_workers: int = 16,
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
        self.scan_max_workers = int(scan_max_workers)

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

        split_dir = osp.join(self.root, self.split)
        if not osp.isdir(split_dir):
            raise FileNotFoundError(
                f"DynamicReplica split dir not found: {split_dir}"
            )

        # ----- Load (or rebuild) the per-scene cache -----
        cache_path = osp.join(
            self.root, f"dynamic_replica_lingbot_{self.split}_cache.npy"
        )
        if osp.exists(cache_path):
            try:
                cache = np.load(cache_path, allow_pickle=True).item()
                self.scene_frames: Dict[str, Dict[str, np.ndarray]] = cache["scene_frames"]
                self.num_imgs: Dict[str, int] = cache["num_imgs"]
                self.sequences = list(self.scene_frames.keys())
                if verbose:
                    print(
                        f"[dynamic_replica] loaded cache: {cache_path} "
                        f"({len(self.sequences)} scenes)"
                    )
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(
                        f"[dynamic_replica] cache load failed ({exc}); rebuilding"
                    )
                self._build_cache(cache_path, verbose)
        else:
            self._build_cache(cache_path, verbose)

        if not self.sequences:
            raise RuntimeError(
                f"No valid DynamicReplica scenes found under {split_dir}"
            )

        # Pre-filter: scenes must have at least min_views frames so we can
        # always sample without falling back to duplicates/padding. Scenes
        # with fewer frames than num_views are still kept and get a dynamic
        # clip length clamped to their available frame count.
        keep = [s for s in self.sequences if self.num_imgs[s] >= self.min_views]
        if len(keep) != len(self.sequences):
            dropped = len(self.sequences) - len(keep)
            self.sequences = keep
            if verbose:
                print(
                    f"[dynamic_replica] dropped {dropped} scenes with < "
                    f"{self.min_views} frames; {len(self.sequences)} remain"
                )

        if not self.sequences:
            raise RuntimeError(
                f"All DynamicReplica scenes have fewer than min_views={self.min_views} frames"
            )

        print(
            f"[dynamic_replica] {split}: {len(self.sequences)} scenes, "
            f"{sum(self.num_imgs[s] for s in self.sequences)} frames total, "
            f"num_views=[{self.min_views}, {self.num_views}], image_size={self.image_size}"
        )

    # ---------------------------------------------------------------------
    # Cache build
    # ---------------------------------------------------------------------

    def _build_cache(self, cache_path: str, verbose: bool) -> None:
        self.scene_frames = {}
        self.num_imgs = {}

        split_dir = osp.join(self.root, self.split)
        scenes = sorted(
            s for s in os.listdir(split_dir) if osp.isdir(osp.join(split_dir, s))
        )
        if not scenes:
            self.sequences = []
            return

        jobs = [(self.root, self.split, s) for s in scenes]

        def _consume(scene: str, out):
            if out is None:
                return
            rgb_paths, depth_paths, Ks, c2ws = out
            self.scene_frames[scene] = dict(
                rgb_paths=rgb_paths,
                depth_paths=depth_paths,
                intrinsics=Ks,
                extrinsics=c2ws,
            )
            self.num_imgs[scene] = int(len(rgb_paths))

        if self.scan_max_workers > 1 and len(jobs) > 1:
            ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.scan_max_workers, len(jobs))
            )
            results_iter = ex.map(_load_one, jobs, chunksize=1)
            results = tqdm(
                results_iter,
                total=len(jobs),
                desc="[dynamic_replica] scan",
                disable=not verbose,
            )
            try:
                for scene, out in results:
                    _consume(scene, out)
            finally:
                ex.shutdown(wait=True)
        else:
            for job in tqdm(
                jobs, desc="[dynamic_replica] scan", disable=not verbose
            ):
                scene, out = _load_one(job)
                _consume(scene, out)

        self.sequences = list(self.scene_frames.keys())
        try:
            np.save(
                cache_path,
                dict(scene_frames=self.scene_frames, num_imgs=self.num_imgs),
            )
            if verbose:
                print(
                    f"[dynamic_replica] wrote cache: {cache_path} "
                    f"({len(self.sequences)} scenes)"
                )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[dynamic_replica] WARNING: failed to write cache to {cache_path}: {exc}"
            )

    # ---------------------------------------------------------------------
    # Sampling / loading
    # ---------------------------------------------------------------------

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
        """Replicates base3d-clean/datasets/dynamic_replica.py's _get_views pipeline.

        base3d-clean picks ``start_id`` deterministically (one per valid offset).
        Here, each ``__getitem__`` index already maps to one scene, so we
        randomize ``start_id`` within the same valid range.
        ``_get_seq_from_start_id`` is then called with the EXACT
        DynamicReplica knobs (max_interval=64, video_prob=1.0,
        fix_interval_prob=1.0).

        Each call draws a target_views in ``[min_views, num_views]`` so the
        clip length varies per sample, then clamps to the scene's frame count
        so short scenes still produce a valid clip without duplicates.
        """
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        else:
            target_views = self.num_views
        if num_imgs < target_views:
            target_views = num_imgs
        image_indices = list(range(num_imgs))
        if num_imgs <= target_views:
            return image_indices, True
        max_start = max(0, num_imgs - target_views)
        start_id = int(rng.integers(0, max_start + 1))
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
        depth_path: str,
        K_raw: np.ndarray,
        c2w_4x4: np.ndarray,
        jitter_params,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Load + clean + preprocess one frame; return tensors in lingbot schema."""
        # ---- RGB
        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size  # (W, H) — DynamicReplica is 1280x720

        # ---- Depth: float32 .npy in meters; mask invalids (matches base3d-clean)
        depthmap = np.load(depth_path).astype(np.float32)
        depthmap[~np.isfinite(depthmap)] = 0.0

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

        # Depth preprocess via PIL ('F' mode keeps float32 + supports nearest)
        depth_pil = Image.fromarray(depthmap.astype(np.float32), mode="F")
        depth_pil = self._apply_preprocess_to_image(
            depth_pil, geometry, resample=Image.Resampling.NEAREST, fill=0
        )
        depth_np = np.asarray(depth_pil, dtype=np.float32).copy()
        depth = torch.from_numpy(depth_np)

        # Intrinsics: input matrix is at native (1280x720); preprocess_intrinsics rescales.
        K = np.asarray(K_raw, dtype=np.float32)
        if K.shape == (4, 4):
            K = K[:3, :3]
        intrinsics = self._preprocess_intrinsics(K, width, height, geometry).float()

        # Extrinsics: cam.npz['pose'] is c2w (right-handed, OpenCV-style); lingbot wants OpenCV w2c.
        c2w_4x4 = np.asarray(c2w_4x4, dtype=np.float64)
        w2c_4x4 = np.linalg.inv(c2w_4x4)
        extrinsics = torch.from_numpy(w2c_4x4[:3, :4].astype(np.float32))

        # Valid mask: positive depth within optional max-depth bound.
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
        # Worker-local rng — fresh per call; matches dl3dv.py / scannetpp.py / tartanair.py.
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

                sampled, is_video = self._sample_indices(num_imgs, rng)
                if is_video:
                    sampled = sorted(sampled)
                jitter_params = self._sample_color_jitter_params(rng)

                # Sequential I/O — DynamicReplica frames are large (1280x720
                # float32 depth = ~3.5 MB raw .npy each); a thread pool tends
                # to thrash CPFS reads. Each DataLoader worker already gives
                # us per-sample parallelism.
                loaded = [
                    self._load_view(
                        rgb_paths[v],
                        depth_paths[v],
                        intrinsics_arr[v],
                        extrinsics_arr[v],
                        jitter_params,
                    )
                    for v in sampled
                ]

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
                    "sample_mode": "dynamic_replica",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to load a valid DynamicReplica sample near index {index}: {last_error}"
        )
