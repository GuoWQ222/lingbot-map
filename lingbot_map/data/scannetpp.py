"""ScanNet++ v2 multi-view RGBD dataset adapter for LingBot-MAP.

This module mirrors the data-loading + sampling strategy of
``base3d-clean/datasets/scannet.py`` (NOT scannetpp.py) but emits samples in
LingBot-MAP's :func:`train.collate_rgbd_sequences` schema so the dataset can sit
side-by-side with :class:`ManipTrajectoryDataset` and :class:`DL3DVTrajectoryDataset`.

Sampling strategy (intentionally copied from ``scannet.py``, NOT ``scannetpp.py``)::

    pos, ordered_video = get_seq_from_start_id(
        num_views,
        start_id,
        all_image_ids,
        rng,
        max_interval=30,    # scannet.py default; scannetpp.py used 32
        video_prob=0.6,     # scannet.py; scannetpp.py used 0.8
        fix_interval_prob=0.6,
        block_shuffle=16,
    )
    # min_interval defaults to 1 (scannet.py omits it; scannetpp.py used 2)

Directory layout expected (``/shared/smartbot/renkerui/data/scannetppv2``)::

    ROOT/
      valid.json                # list of scene IDs (or valid_new.json from prior run)
      00777c41d4/
        scene_metadata.npz      # keys: trajectories (N,4,4) c2w OpenCV,
                                #       intrinsics (N,3,3),
                                #       images (N,) like "DSC00850.JPG"
        images/DSC00850.jpg     # lower-case .jpg on disk (.JPG in npz is fine,
                                # we strip the extension via osp.splitext)
        depth/DSC00850.png      # uint16 mm depth (divide by 1000 -> meters)

Output dict (one per ``__getitem__``) — same schema as
:class:`DL3DVTrajectoryDataset`::

    {
        "images":       (T, 3, H, W) float32 in [0, 1],
        "depths":       (T, H, W) float32 meters,
        "point_masks":  (T, H, W) bool,
        "intrinsics":   (T, 3, 3) float32 (target image pixel coords),
        "extrinsics":   (T, 3, 4) float32 OpenCV w2c,
        "world_points": (T, H, W, 3) float32,
        "frame_ids":    (T,) long,
        "view_ids":     (T,) long (= 0; ScanNet++ is single-camera),
        "scene":        str,
        "sample_mode":  "scannetpp",
    }
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import cv2
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
    root: str, scene: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load (rgb_paths, intrinsics, extrinsics) for one scene.

    Mirrors the metadata-discovery loop in
    ``base3d-clean/datasets/scannetpp.py`` but tailored to lingbot's schema.
    Returns ``None`` on any failure or if the scene is missing required files.
    """
    scene_dir = osp.join(root, scene)
    meta_path = osp.join(scene_dir, "scene_metadata.npz")
    if not osp.isfile(meta_path):
        return None
    try:
        meta = np.load(meta_path, allow_pickle=True)
        Ks = np.asarray(meta["intrinsics"])          # (N, 3, 3)
        traj = np.asarray(meta["trajectories"])       # (N, 4, 4) c2w OpenCV
        imgs = np.asarray(meta["images"])             # (N,) strings (e.g. "DSC00850.JPG")
    except Exception:
        return None
    if Ks.shape[0] == 0 or Ks.shape[0] != traj.shape[0] or Ks.shape[0] != imgs.shape[0]:
        return None

    images_dir = osp.join(scene_dir, "images")
    depth_dir = osp.join(scene_dir, "depth")
    if not osp.isdir(images_dir) or not osp.isdir(depth_dir):
        return None

    # The npz stores names like "DSC00850.JPG" but disk files are
    # ".jpg" (lower-case). Strip extension via osp.splitext to match the
    # base3d-clean dataloader behavior.
    rgb_paths = np.array(
        [osp.join(images_dir, osp.splitext(name)[0] + ".jpg") for name in imgs]
    )
    return rgb_paths, Ks.astype(np.float64), traj.astype(np.float64)


def _load_one(args: Tuple[str, str]):
    root, scene = args
    out = _load_scene_metadata(root, scene)
    return scene, out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ScanNetppTrajectoryDataset(Dataset):
    """ScanNet++ v2 multi-view dataset emitting LingBot-MAP sample dicts.

    Sampling parameters intentionally match ``base3d-clean/datasets/scannet.py``
    (NOT ``scannetpp.py``). ``split`` is accepted for API parity with the DL3DV
    adapter but is informational only — ScanNet++ in this layout has no
    per-split json. We use ``valid_new.json`` if present, else ``valid.json``.
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
        # Sampling knobs — DEFAULTS COPIED FROM base3d-clean/datasets/scannet.py
        # (do NOT change to scannetpp.py's values).
        min_interval: int = 1,
        max_interval: int = 30,
        video_prob: float = 0.6,
        fix_interval_prob: float = 0.6,
        block_shuffle: Optional[int] = 16,
        io_max_workers: int = 8,
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
        self.io_max_workers = int(io_max_workers)
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

        # ----- Scene list: prefer valid_new.json (verified), fall back to valid.json -----
        valid_new = osp.join(self.root, "valid_new.json")
        valid_raw = osp.join(self.root, "valid.json")
        if osp.exists(valid_new):
            with open(valid_new, "r") as f:
                candidate_scenes: List[str] = json.load(f)
            if verbose:
                print(f"[scannetpp] loaded {len(candidate_scenes)} scenes from {valid_new}")
        elif osp.exists(valid_raw):
            with open(valid_raw, "r") as f:
                candidate_scenes = json.load(f)
            if verbose:
                print(f"[scannetpp] loaded {len(candidate_scenes)} candidate scenes from {valid_raw}")
        else:
            raise FileNotFoundError(
                f"Neither valid_new.json nor valid.json found under {self.root}. "
                "Generate valid.json first (a JSON list of scene-id directory names)."
            )

        # ----- Load (or rebuild) the per-scene cache -----
        cache_path = osp.join(self.root, f"scannetpp_valid_cache.npy")
        if osp.exists(cache_path):
            try:
                cache = np.load(cache_path, allow_pickle=True).item()
                self.scene_frames: Dict[str, Dict[str, np.ndarray]] = cache["scene_frames"]
                self.num_imgs: Dict[str, int] = cache["num_imgs"]
                self.sequences = list(self.scene_frames.keys())
                if verbose:
                    print(f"[scannetpp] loaded cache: {cache_path} ({len(self.sequences)} scenes)")
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[scannetpp] cache load failed ({exc}); rebuilding")
                self._build_cache(candidate_scenes, cache_path, verbose)
        else:
            self._build_cache(candidate_scenes, cache_path, verbose)

        if not self.sequences:
            raise RuntimeError(
                f"No valid ScanNet++ scenes found under {self.root}"
            )

        # Pre-filter: scenes must have at least num_views frames (matches scannet.py cut_off)
        keep = [s for s in self.sequences if self.num_imgs[s] >= self.min_views]
        if len(keep) != len(self.sequences):
            dropped = len(self.sequences) - len(keep)
            self.sequences = keep
            if verbose:
                print(
                    f"[scannetpp] dropped {dropped} scenes with < {self.min_views} frames; "
                    f"{len(self.sequences)} scenes remain"
                )

        if not self.sequences:
            raise RuntimeError(
                f"All ScanNet++ scenes have fewer than num_views={self.num_views} frames"
            )

        print(
            f"[scannetpp] {split}: {len(self.sequences)} scenes, "
            f"{sum(self.num_imgs[s] for s in self.sequences)} frames total, "
            f"num_views={self.num_views}, image_size={self.image_size}"
        )

    # ---------------------------------------------------------------------
    # Cache build
    # ---------------------------------------------------------------------

    def _build_cache(self, candidate_scenes: List[str], cache_path: str, verbose: bool) -> None:
        self.scene_frames = {}
        self.num_imgs = {}

        # Parallel metadata scan: per-scene reads a small .npz, so I/O-bound;
        # threadpool keeps things simple and avoids fork issues.
        jobs = [(self.root, scene) for scene in candidate_scenes]
        iterator = jobs
        if self.scan_max_workers > 1 and len(jobs) > 1:
            ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.scan_max_workers, len(jobs))
            )
            results_iter = ex.map(_load_one, jobs, chunksize=1)
            results = tqdm(
                results_iter, total=len(jobs), desc="[scannetpp] scan", disable=not verbose
            )
            try:
                for scene, out in results:
                    if out is None:
                        continue
                    rgb_paths, Ks, traj = out
                    self.scene_frames[scene] = dict(
                        intrinsics=Ks,
                        extrinsics=traj,
                        rgb_paths=rgb_paths,
                    )
                    self.num_imgs[scene] = int(len(rgb_paths))
            finally:
                ex.shutdown(wait=True)
        else:
            for job in tqdm(iterator, desc="[scannetpp] scan", disable=not verbose):
                scene, out = _load_one(job)
                if out is None:
                    continue
                rgb_paths, Ks, traj = out
                self.scene_frames[scene] = dict(
                    intrinsics=Ks,
                    extrinsics=traj,
                    rgb_paths=rgb_paths,
                )
                self.num_imgs[scene] = int(len(rgb_paths))

        self.sequences = list(self.scene_frames.keys())
        try:
            np.save(
                cache_path,
                dict(scene_frames=self.scene_frames, num_imgs=self.num_imgs),
            )
            if verbose:
                print(f"[scannetpp] wrote cache: {cache_path} ({len(self.sequences)} scenes)")
        except Exception as exc:  # noqa: BLE001
            print(f"[scannetpp] WARNING: failed to write cache to {cache_path}: {exc}")

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
        """Replicates scannet.py's _get_views sampling pipeline.

        Per-call random target length in [min_views, num_views] (= fixed
        when min_views == num_views, matching the original scannet.py
        behavior). Downstream ``get_seq_from_start_id`` uses scannet.py's
        sampling arguments (max_interval=30, video_prob=0.6,
        fix_interval_prob=0.6, block_shuffle=16).
        """
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        else:
            target_views = self.num_views
        target_views = min(target_views, num_imgs)
        image_indices = list(range(num_imgs))
        # __init__ filters out scenes with num_imgs < min_views, so the
        # remaining scenes guarantee num_imgs >= min_views = lower bound on
        # target_views. The clamp above ensures we never ask for more.
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
        """Load + preprocess one frame; return tensors in lingbot schema."""
        # ---- RGB
        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size  # (W, H)

        # ---- Depth: uint16 mm -> float32 m (same as base3d-clean/datasets/scannet.py)
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise IOError(f"Failed to read depth: {depth_path}")
        depthmap = depth_raw.astype(np.float32) / 1000.0
        depthmap[~np.isfinite(depthmap)] = 0.0
        # Resize to RGB native res if shapes differ (defensive; ScanNet++
        # depth is already at the same resolution as images_4-style RGB).
        if depthmap.shape[:2] != (height, width):
            depthmap = cv2.resize(
                depthmap, (width, height), interpolation=cv2.INTER_NEAREST
            )

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

        # Intrinsics: input matrix is at original RGB res; preprocess_intrinsics rescales.
        K = np.asarray(K_raw, dtype=np.float32)
        if K.shape == (4, 4):
            K = K[:3, :3]
        intrinsics = self._preprocess_intrinsics(K, width, height, geometry).float()

        # Extrinsics: scene_metadata stores OpenCV c2w; lingbot wants OpenCV w2c.
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
        # Worker-local rng — fresh per call; matches dl3dv.py behavior.
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
                    rgb_p = rgb_paths[view_idx]
                    # depth path: same basename, depth/ instead of images/, .png instead of .jpg
                    depth_p = rgb_p.replace("/images/", "/depth/")
                    if depth_p.endswith(".jpg"):
                        depth_p = depth_p[:-4] + ".png"
                    elif depth_p.endswith(".JPG"):
                        depth_p = depth_p[:-4] + ".png"
                    return self._load_view(
                        rgb_p,
                        depth_p,
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
                    "sample_mode": "scannetpp",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to load a valid ScanNet++ sample near index {index}: {last_error}"
        )
