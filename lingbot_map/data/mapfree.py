"""MapFree multi-view RGBD dataset adapter for LingBot-MAP.

This module mirrors the data-loading + sampling strategy of
``base3d-clean/datasets/mapfree.py`` but emits samples in LingBot-MAP's
:func:`train.collate_rgbd_sequences` schema so the dataset can sit alongside
:class:`ManipTrajectoryDataset`, :class:`DL3DVTrajectoryDataset`,
:class:`ScanNetppTrajectoryDataset`, :class:`TartanAirTrajectoryDataset`, and
:class:`DynamicReplicaTrajectoryDataset`.

Sampling parameters are taken VERBATIM from base3d-clean/datasets/mapfree.py
(``min_interval=1, max_interval=64, video_prob=0.8, fix_interval_prob=0.6,
block_shuffle=16``). The downstream sampling helper is the same
``_get_seq_from_start_id`` already ported next to us.

Directory layout (matches /cpfs/shared/landmark/renkerui/data/mapfree)::

    ROOT/
      valid.json                                 # list of "<scene>/dense{0,1}"
      mapfree_train_valid_cache.npy              # base3d-clean per-scene path cache
      <scene>/dense{0,1}/
          rgb/frame_*.jpg
          depth/frame_*.npy           # float32, (H, W), meters
          cam/frame_*.npz             # {intrinsic: (3,3), pose: (4,4) c2w}
          sky_mask/frame_*.jpg        # uint8 (>=127 -> sky)

Per-frame depth cleaning matches base3d-clean/datasets/mapfree.py:246-254
exactly: sky pixels are set to invalid, depths > 400 m are zeroed, NaN/Inf
are zeroed, and depths above the per-frame 90th percentile of positive values
are clipped (this last step is computed at load time and cannot be cached).

Cache:
    First instantiation per ROOT loads ``mapfree_train_valid_cache.npy``
    (built by either base3d-clean or our /tmp/rebuild_mapfree_cache.py),
    then walks every cam .npz to pre-extract per-frame intrinsics + c2w
    poses, and writes ``mapfree_lingbot_cache.npy`` next to ROOT. Subsequent
    instantiations skip the .npz walk entirely.

Output dict (one per ``__getitem__``) — same schema as
:class:`TartanAirTrajectoryDataset`::

    {
        "images":       (T, 3, H, W) float32 in [0, 1],
        "depths":       (T, H, W) float32 meters,
        "point_masks":  (T, H, W) bool,
        "intrinsics":   (T, 3, 3) float32 (target image pixel coords),
        "extrinsics":   (T, 3, 4) float32 OpenCV w2c,
        "world_points": (T, H, W, 3) float32,
        "frame_ids":    (T,) long,
        "view_ids":     (T,) long (= 0; mapfree has only one camera per scene),
        "scene":        str,
        "sample_mode":  "mapfree",
    }
"""
from __future__ import annotations

import concurrent.futures
import json
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
# Per-frame metadata loading
# ---------------------------------------------------------------------------

# Far-depth clip threshold (meters). base3d-clean/datasets/mapfree.py uses 400.
_MAPFREE_FAR_CLIP = 400.0
# Per-frame quantile used for the long-tail clip (matches base3d-clean).
_MAPFREE_DEPTH_QUANTILE = 90.0
# Sky threshold on the JPG (matches base3d-clean: ``>= 127`` -> sky).
_MAPFREE_SKY_THRESHOLD = 127


def _load_cam_npz(cam_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Read one cam .npz; return (K, c2w) or None if unreadable."""
    try:
        data = np.load(cam_path)
        K = np.asarray(data["intrinsic"], dtype=np.float32)
        pose = np.asarray(data["pose"], dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    if K.shape != (3, 3) or pose.shape != (4, 4):
        return None
    return K, pose


def _load_one_scene_cams(args: Tuple[str, List[str]]):
    scene_id, cam_paths = args
    Ks: List[np.ndarray] = []
    Ps: List[np.ndarray] = []
    for cp in cam_paths:
        out = _load_cam_npz(cp)
        if out is None:
            return scene_id, None
        Ks.append(out[0])
        Ps.append(out[1])
    return scene_id, (np.stack(Ks, axis=0), np.stack(Ps, axis=0))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MapfreeTrajectoryDataset(Dataset):
    """MapFree multi-view dataset emitting LingBot-MAP sample dicts.

    Sampling parameters intentionally match
    ``base3d-clean/datasets/mapfree.py`` (and ONLY that file). ``split`` is
    accepted for API parity with the other adapters but is informational only
    — base3d-clean uses ``valid.json`` as the entire training universe.
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
        # base3d-clean/datasets/mapfree.py (do NOT change to other datasets).
        min_interval: int = 1,
        max_interval: int = 64,
        video_prob: float = 0.8,
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

        # ----- Load (or rebuild) the lingbot-specific cache (paths + per-frame K + c2w) -----
        cache_path = osp.join(self.root, "mapfree_lingbot_cache.npy")
        if osp.exists(cache_path):
            try:
                cache = np.load(cache_path, allow_pickle=True).item()
                self.scene_frames: Dict[str, Dict[str, np.ndarray]] = cache["scene_frames"]
                self.num_imgs: Dict[str, int] = cache["num_imgs"]
                self.sequences = list(self.scene_frames.keys())
                if verbose:
                    print(
                        f"[mapfree] loaded cache: {cache_path} "
                        f"({len(self.sequences)} scenes)"
                    )
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[mapfree] cache load failed ({exc}); rebuilding")
                self._build_cache(cache_path, verbose)
        else:
            self._build_cache(cache_path, verbose)

        if not self.sequences:
            raise RuntimeError(
                f"No valid MapFree scenes found under {self.root}"
            )

        # Pre-filter: scenes must have at least min_views frames so we can
        # always sample without falling back to duplicates/padding. Scenes
        # with frame count between min_views and num_views still produce a
        # valid clip clamped to their available frames.
        keep = [s for s in self.sequences if self.num_imgs[s] >= self.min_views]
        if len(keep) != len(self.sequences):
            dropped = len(self.sequences) - len(keep)
            self.sequences = keep
            if verbose:
                print(
                    f"[mapfree] dropped {dropped} scenes with < "
                    f"{self.min_views} frames; {len(self.sequences)} remain"
                )

        if not self.sequences:
            raise RuntimeError(
                f"All MapFree scenes have fewer than min_views={self.min_views} frames"
            )

        print(
            f"[mapfree] {split}: {len(self.sequences)} scenes, "
            f"{sum(self.num_imgs[s] for s in self.sequences)} frames total, "
            f"num_views=[{self.min_views}, {self.num_views}], image_size={self.image_size}"
        )

    # ---------------------------------------------------------------------
    # Cache build
    # ---------------------------------------------------------------------

    def _build_cache(self, cache_path: str, verbose: bool) -> None:
        """Build mapfree_lingbot_cache.npy (paths + per-frame K + c2w arrays).

        Step 1: load mapfree_train_valid_cache.npy (built by base3d-clean or
                /tmp/rebuild_mapfree_cache.py) for the validated path lists.
                If absent, fall back to scanning valid.json + per-scene listdir
                (the same logic as base3d-clean/datasets/mapfree.py:60-93).
        Step 2: walk every cam .npz to pre-extract intrinsics + c2w arrays.
        """
        path_cache = osp.join(self.root, "mapfree_train_valid_cache.npy")
        if osp.exists(path_cache):
            try:
                pc = np.load(path_cache, allow_pickle=True).item()
                base_scene_frames = pc["scene_frames"]
                base_num_imgs = pc["num_imgs"]
                if verbose:
                    print(
                        f"[mapfree] reusing path cache {path_cache} "
                        f"({len(base_scene_frames)} scenes)"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"[mapfree] failed to load {path_cache}: {exc}; falling back to scan")
                base_scene_frames, base_num_imgs = self._scan_valid_json(verbose)
        else:
            base_scene_frames, base_num_imgs = self._scan_valid_json(verbose)

        # ----- Step 2: pre-extract intrinsics + c2w from every cam .npz -----
        self.scene_frames = {}
        self.num_imgs = {}
        jobs = [
            (sid, list(meta["cam_paths"])) for sid, meta in base_scene_frames.items()
        ]

        def _consume(scene_id: str, out, src_meta):
            if out is None:
                return
            Ks, Ps = out
            self.scene_frames[scene_id] = dict(
                rgb_paths=np.asarray(src_meta["rgb_paths"]),
                depth_paths=np.asarray(src_meta["depth_paths"]),
                sky_mask_paths=np.asarray(src_meta["sky_mask_paths"]),
                intrinsics=Ks.astype(np.float32),
                extrinsics=Ps.astype(np.float64),
            )
            self.num_imgs[scene_id] = int(len(src_meta["rgb_paths"]))

        if self.scan_max_workers > 1 and len(jobs) > 1:
            ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.scan_max_workers, len(jobs))
            )
            results_iter = ex.map(_load_one_scene_cams, jobs, chunksize=1)
            results = tqdm(
                results_iter, total=len(jobs), desc="[mapfree] cam scan", disable=not verbose
            )
            try:
                for scene_id, out in results:
                    _consume(scene_id, out, base_scene_frames[scene_id])
            finally:
                ex.shutdown(wait=True)
        else:
            for job in tqdm(jobs, desc="[mapfree] cam scan", disable=not verbose):
                scene_id, out = _load_one_scene_cams(job)
                _consume(scene_id, out, base_scene_frames[scene_id])

        self.sequences = list(self.scene_frames.keys())
        try:
            np.save(
                cache_path,
                dict(scene_frames=self.scene_frames, num_imgs=self.num_imgs),
            )
            if verbose:
                print(
                    f"[mapfree] wrote cache: {cache_path} "
                    f"({len(self.sequences)} scenes)"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[mapfree] WARNING: failed to write cache to {cache_path}: {exc}")

    def _scan_valid_json(self, verbose: bool) -> Tuple[Dict, Dict]:
        """Fallback path enumeration when mapfree_train_valid_cache.npy is missing.

        Mirrors base3d-clean/datasets/mapfree.py:60-93 exactly: load valid.json,
        list each rgb dir, derive cam/depth/sky paths by suffix substitution,
        and accept the scene only if all four lists align.
        """
        valid_json = osp.join(self.root, "valid.json")
        with open(valid_json) as f:
            scenes = json.load(f)
        if verbose:
            print(f"[mapfree] scanning {len(scenes)} scenes from valid.json")

        scene_frames: Dict[str, Dict[str, np.ndarray]] = {}
        num_imgs: Dict[str, int] = {}
        for rel in tqdm(scenes, desc="[mapfree] valid.json", disable=not verbose):
            scene_path = osp.join(self.root, rel)
            rgb_d = osp.join(scene_path, "rgb")
            depth_d = osp.join(scene_path, "depth")
            cam_d = osp.join(scene_path, "cam")
            sky_d = osp.join(scene_path, "sky_mask")
            try:
                fns = sorted(os.listdir(rgb_d))
            except OSError:
                continue
            rgb_paths, depth_paths, cam_paths, sky_paths = [], [], [], []
            ok = True
            for fn in fns:
                rgb_paths.append(osp.join(rgb_d, fn))
                depth_paths.append(osp.join(depth_d, fn.replace(".jpg", ".npy")))
                cam_paths.append(osp.join(cam_d, fn.replace(".jpg", ".npz")))
                sky_paths.append(osp.join(sky_d, fn))
                # Same per-file existence + ID check as base3d-clean's check_valid.
                rgb_id = osp.splitext(osp.basename(rgb_paths[-1]))[0]
                cam_id = osp.splitext(osp.basename(cam_paths[-1]))[0]
                if rgb_id != cam_id:
                    ok = False
                    break
                if not (
                    osp.exists(rgb_paths[-1])
                    and osp.exists(cam_paths[-1])
                    and osp.exists(depth_paths[-1])
                    and osp.exists(sky_paths[-1])
                ):
                    ok = False
                    break
            if not ok or not rgb_paths:
                continue
            scene_id = rel.replace("/", "_")
            scene_frames[scene_id] = dict(
                rgb_paths=np.asarray(rgb_paths),
                depth_paths=np.asarray(depth_paths),
                cam_paths=np.asarray(cam_paths),
                sky_mask_paths=np.asarray(sky_paths),
            )
            num_imgs[scene_id] = len(rgb_paths)
        return scene_frames, num_imgs

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
        """Replicates base3d-clean/datasets/mapfree.py:_get_views sampling.

        Each call draws a target_views in ``[min_views, num_views]`` so the
        clip length varies per sample, then clamps to the scene's frame count
        so short scenes still produce a valid clip without duplicates. Picks
        ``start_id`` uniformly in ``[0, num_imgs - target_views]`` and fans
        out via ``_get_seq_from_start_id`` with the EXACT mapfree knobs
        (max_interval=64, video_prob=0.8, fix_interval_prob=0.6, block_shuffle=16).
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
        start_id = int(rng.integers(0, num_imgs - target_views + 1))
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
        sky_path: str,
        K_3x3: np.ndarray,
        c2w_4x4: np.ndarray,
        jitter_params,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Load + clean + preprocess one frame; return tensors in lingbot schema.

        Depth cleaning order matches base3d-clean/datasets/mapfree.py:246-254:
            1. sky pixels (mask >= 127) -> -1 (treated as invalid)
            2. depth > 400 -> 0
            3. NaN/Inf -> 0
            4. clip values above per-frame 90th percentile of positive depths to 0
        """
        # ---- RGB
        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size  # MapFree: (540, 720)

        # ---- Sky mask (uint8 jpg, >=127 = sky)
        sky_pil = Image.open(sky_path)
        sky_np = np.asarray(sky_pil)
        if sky_np.ndim == 3:
            sky_np = sky_np[..., 0]
        sky_mask = sky_np >= _MAPFREE_SKY_THRESHOLD

        # ---- Depth: float32 .npy in meters
        depthmap = np.load(depth_path).astype(np.float32)
        # Sanity: depth and sky must agree on (H, W)
        if depthmap.shape != sky_mask.shape:
            raise ValueError(
                f"depth/sky shape mismatch: {depthmap.shape} vs {sky_mask.shape} ({depth_path})"
            )
        # 1. sky -> invalid (use 0; mask stage below treats it as invalid).
        depthmap[sky_mask] = 0.0
        # 2. far clip
        depthmap[depthmap > _MAPFREE_FAR_CLIP] = 0.0
        # 3. NaN / Inf
        depthmap = np.nan_to_num(depthmap, nan=0.0, posinf=0.0, neginf=0.0)
        # 4. per-frame long-tail clip via 90th percentile of positive depths.
        positive = depthmap[depthmap > 0]
        if positive.size > 0:
            threshold = float(np.percentile(positive, _MAPFREE_DEPTH_QUANTILE))
            depthmap[depthmap > threshold] = 0.0

        # ---- Geometry: lingbot preprocess (resize + crop/pad)
        geometry = self._compute_preprocess_geometry(
            width, height, self.image_size, self.patch_size, self.preprocess_mode
        )

        rgb_pre = self._apply_preprocess_to_image(
            rgb_pil, geometry, resample=Image.Resampling.BICUBIC, fill=(255, 255, 255)
        )
        if jitter_params is not None:
            rgb_pre = self._apply_color_jitter(rgb_pre, jitter_params)
        image_tensor = TF.ToTensor()(rgb_pre)  # (3, H, W) float in [0,1]

        depth_pil = Image.fromarray(depthmap.astype(np.float32), mode="F")
        depth_pil = self._apply_preprocess_to_image(
            depth_pil, geometry, resample=Image.Resampling.NEAREST, fill=0
        )
        depth_np = np.asarray(depth_pil, dtype=np.float32).copy()
        depth = torch.from_numpy(depth_np)

        intrinsics = self._preprocess_intrinsics(
            np.asarray(K_3x3, dtype=np.float32), width, height, geometry
        ).float()

        # Extrinsics: c2w -> w2c.
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
        # Worker-local rng — fresh per call; matches dl3dv.py / scannetpp.py / tartanair.py behavior.
        seed_entropy = int(np.random.randint(0, 2**32, dtype=np.uint32))
        rng = np.random.default_rng(seed_entropy)

        for attempt in range(min(16, n)):
            scene_id = self.sequences[(start_scene + attempt) % n]
            try:
                meta = self.scene_frames[scene_id]
                rgb_paths: np.ndarray = meta["rgb_paths"]
                depth_paths: np.ndarray = meta["depth_paths"]
                sky_paths: np.ndarray = meta["sky_mask_paths"]
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
                        depth_paths[view_idx],
                        sky_paths[view_idx],
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
                    "sample_mode": "mapfree",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to load a valid MapFree sample near index {index}: {last_error}"
        )
