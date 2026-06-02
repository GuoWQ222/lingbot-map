"""TartanAir multi-view RGBD dataset adapter for LingBot-MAP.

This module mirrors the data-loading + sampling strategy of
``base3d-clean/datasets/tartanair.py`` but emits samples in LingBot-MAP's
:func:`train.collate_rgbd_sequences` schema so the dataset can sit alongside
:class:`ManipTrajectoryDataset`, :class:`DL3DVTrajectoryDataset`, and
:class:`ScanNetppTrajectoryDataset`.

Sampling parameters are taken VERBATIM from base3d-clean/datasets/tartanair.py
(``min_interval=1, max_interval=32, video_prob=0.8, fix_interval_prob=0.6,
block_shuffle=16``). The downstream sampling helper is the same
``_get_seq_from_start_id`` already ported from
``base3d-clean/datasets/base/base_multiview_dataset.py``.

Directory layout (matches /cpfs/shared/landmark/renkerui/data/tartanair)::

    ROOT/
      rgb/<scene>/<Easy|Hard>/<P***>/
          image_left/<basename>_left.png
          pose_left.txt              # x y z qx qy qz qw, one row per frame
      depth/<scene>/<Easy|Hard>/<P***>/
          depth_left/<basename>_left_depth.npy

Camera intrinsics are constant (TartanAir official, pinhole, no distortion):
``fx = fy = 320, cx = 320, cy = 240`` at 640x480.

The pose file stores the raw simulator state ``(x, y, z, qx, qy, qz, qw)`` in
NED-style world coordinates. We use the SAME ``xyzqxqyqxqw_to_c2w`` conversion
as base3d-clean to land in the right-handed camera convention expected by the
rest of the pipeline (note the ``z, x, y`` translation reorder + matching
quaternion reorder).

Cache:
    First instantiation per ROOT scans every (scene, difficulty, episode)
    triplet and writes ``tartanair_lingbot_cache.npy`` next to ROOT for fast
    subsequent loads.

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
        "view_ids":     (T,) long (= 0; we use the left camera only),
        "scene":        str,
        "sample_mode":  "tartanair",
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
# Pose conversion (verbatim from base3d-clean/datasets/tartanair.py)
# ---------------------------------------------------------------------------

def _xyzqxqyqxqw_to_c2w(xyzqxqyqxqw: np.ndarray) -> np.ndarray:
    """Convert a TartanAir 7-vector pose to a 4x4 c2w matrix.

    Copied verbatim from ``base3d-clean/datasets/tartanair.py`` so the
    coordinate convention matches the rest of the pipeline byte-for-byte.
    The reorder ``z, x, y = xyzqxqyqxqw[:3]`` (and matching quaternion swap)
    converts the raw NED-style world coordinates into the right-handed
    camera convention used downstream.
    """
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    # NOTE: convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ]
    )
    c2w[:3, 3] = np.array([x, y, z])
    return c2w


# ---------------------------------------------------------------------------
# Per-episode metadata loading
# ---------------------------------------------------------------------------

# Fixed camera intrinsics (TartanAir official: pinhole, 640x480, no distortion)
_K_BASE = np.array(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
_TARTANAIR_BASE_W = 640
_TARTANAIR_BASE_H = 480
# Far-depth clip threshold (meters). base3d-clean/datasets/tartanair.py uses 80.
_TARTANAIR_FAR_CLIP = 80.0


def _load_episode_metadata(
    root: str, scene: str, diff: str, name: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build (rgb_paths, depth_paths, c2w_arr) for one episode.

    Returns ``None`` if the episode is missing / inconsistent.
    """
    img_dir = osp.join(root, "rgb", scene, diff, name, "image_left")
    depth_dir = osp.join(root, "depth", scene, diff, name, "depth_left")
    pose_txt = osp.join(root, "rgb", scene, diff, name, "pose_left.txt")

    if not (osp.isdir(img_dir) and osp.isdir(depth_dir) and osp.isfile(pose_txt)):
        return None

    try:
        # Match base3d-clean: list image basenames, sorted lexicographically
        basenames = sorted(
            f.split("_")[0] for f in os.listdir(img_dir) if f.endswith("_left.png")
        )
    except OSError:
        return None
    if not basenames:
        return None

    try:
        caminfo = np.loadtxt(pose_txt)
    except Exception:  # noqa: BLE001
        return None
    if caminfo.ndim != 2 or caminfo.shape[1] != 7:
        return None

    # Same alignment policy as base3d-clean: truncate to the shorter of the two
    minlen = min(caminfo.shape[0], len(basenames))
    if minlen == 0:
        return None
    caminfo = caminfo[:minlen]
    basenames = basenames[:minlen]

    rgb_paths = np.array(
        [osp.join(img_dir, f"{b}_left.png") for b in basenames]
    )
    depth_paths = np.array(
        [osp.join(depth_dir, f"{b}_left_depth.npy") for b in basenames]
    )
    c2w_arr = np.stack([_xyzqxqyqxqw_to_c2w(caminfo[i]) for i in range(minlen)], axis=0)
    return rgb_paths, depth_paths, c2w_arr.astype(np.float64)


def _load_one(args: Tuple[str, str, str, str]):
    root, scene, diff, name = args
    out = _load_episode_metadata(root, scene, diff, name)
    return (scene, diff, name), out


def _enumerate_episodes(root: str) -> List[Tuple[str, str, str]]:
    """Mirror base3d-clean's episode discovery: rgb/<scene>/<Easy|Hard>/<P***>."""
    rgb_root = osp.join(root, "rgb")
    if not osp.isdir(rgb_root):
        raise FileNotFoundError(f"TartanAir rgb root not found: {rgb_root}")
    episodes: List[Tuple[str, str, str]] = []
    for scene in os.listdir(rgb_root):
        scene_dir = osp.join(rgb_root, scene)
        if not osp.isdir(scene_dir):
            continue
        for diff in ("Easy", "Hard"):
            ddir = osp.join(scene_dir, diff)
            if not osp.isdir(ddir):
                continue
            for name in os.listdir(ddir):
                if osp.isdir(osp.join(ddir, name)):
                    episodes.append((scene, diff, name))
    episodes.sort()
    return episodes


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TartanAirTrajectoryDataset(Dataset):
    """TartanAir multi-view dataset emitting LingBot-MAP sample dicts.

    Sampling parameters intentionally match
    ``base3d-clean/datasets/tartanair.py`` (and ONLY that file). ``split`` is
    accepted for API parity with the DL3DV / ScanNet++ adapters but is
    informational only — TartanAir has no per-split json in the source layout,
    and base3d-clean uses every episode for training as well.
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
        # base3d-clean/datasets/tartanair.py (do NOT change to other datasets).
        min_interval: int = 1,
        max_interval: int = 32,
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

        # ----- Load (or rebuild) the per-episode cache -----
        cache_path = osp.join(self.root, "tartanair_lingbot_cache.npy")
        if osp.exists(cache_path):
            try:
                cache = np.load(cache_path, allow_pickle=True).item()
                self.scene_frames: Dict[str, Dict[str, np.ndarray]] = cache["scene_frames"]
                self.num_imgs: Dict[str, int] = cache["num_imgs"]
                self.sequences = list(self.scene_frames.keys())
                if verbose:
                    print(
                        f"[tartanair] loaded cache: {cache_path} "
                        f"({len(self.sequences)} episodes)"
                    )
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[tartanair] cache load failed ({exc}); rebuilding")
                self._build_cache(cache_path, verbose)
        else:
            self._build_cache(cache_path, verbose)

        if not self.sequences:
            raise RuntimeError(
                f"No valid TartanAir episodes found under {self.root}"
            )

        # Pre-filter: episodes must have at least num_views frames
        # (mirrors base3d-clean's cut_off check at tartanair.py:89-91).
        keep = [s for s in self.sequences if self.num_imgs[s] >= self.min_views]
        if len(keep) != len(self.sequences):
            dropped = len(self.sequences) - len(keep)
            self.sequences = keep
            if verbose:
                print(
                    f"[tartanair] dropped {dropped} episodes with < "
                    f"{self.min_views} frames; {len(self.sequences)} remain"
                )

        if not self.sequences:
            raise RuntimeError(
                f"All TartanAir episodes have fewer than min_views={self.min_views} frames"
            )

        print(
            f"[tartanair] {split}: {len(self.sequences)} episodes, "
            f"{sum(self.num_imgs[s] for s in self.sequences)} frames total, "
            f"num_views={self.num_views}, image_size={self.image_size}"
        )

    # ---------------------------------------------------------------------
    # Cache build
    # ---------------------------------------------------------------------

    def _build_cache(self, cache_path: str, verbose: bool) -> None:
        self.scene_frames = {}
        self.num_imgs = {}

        episodes = _enumerate_episodes(self.root)
        if not episodes:
            self.sequences = []
            return

        jobs = [(self.root, s, d, n) for (s, d, n) in episodes]

        def _consume(scene_id: str, out):
            if out is None:
                return
            rgb_paths, depth_paths, c2w_arr = out
            self.scene_frames[scene_id] = dict(
                rgb_paths=rgb_paths,
                depth_paths=depth_paths,
                extrinsics=c2w_arr,
            )
            self.num_imgs[scene_id] = int(len(rgb_paths))

        if self.scan_max_workers > 1 and len(jobs) > 1:
            ex = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.scan_max_workers, len(jobs))
            )
            results_iter = ex.map(_load_one, jobs, chunksize=1)
            results = tqdm(
                results_iter, total=len(jobs), desc="[tartanair] scan", disable=not verbose
            )
            try:
                for (scene, diff, name), out in results:
                    _consume(f"{scene}/{diff}/{name}", out)
            finally:
                ex.shutdown(wait=True)
        else:
            for job in tqdm(jobs, desc="[tartanair] scan", disable=not verbose):
                (scene, diff, name), out = _load_one(job)
                _consume(f"{scene}/{diff}/{name}", out)

        self.sequences = list(self.scene_frames.keys())
        try:
            np.save(
                cache_path,
                dict(scene_frames=self.scene_frames, num_imgs=self.num_imgs),
            )
            if verbose:
                print(
                    f"[tartanair] wrote cache: {cache_path} "
                    f"({len(self.sequences)} episodes)"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[tartanair] WARNING: failed to write cache to {cache_path}: {exc}")

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
        """Replicates base3d-clean/datasets/tartanair.py's _get_views pipeline.

        Per-call random target length in [min_views, num_views] (= fixed when
        min_views == num_views). ``_get_seq_from_start_id`` is then called
        with the EXACT TartanAir knobs (max_interval=32, video_prob=0.8,
        fix_interval_prob=0.6, block_shuffle=16).
        """
        if self.min_views < self.num_views:
            target_views = int(rng.integers(self.min_views, self.num_views + 1))
        else:
            target_views = self.num_views
        target_views = min(target_views, num_imgs)
        image_indices = list(range(num_imgs))
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
        c2w_4x4: np.ndarray,
        jitter_params,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """Load + clean + preprocess one frame; return tensors in lingbot schema."""
        # ---- RGB
        rgb_pil = Image.open(rgb_path).convert("RGB")
        width, height = rgb_pil.size  # (W, H) — TartanAir is 640x480

        # ---- Depth: float32 .npy in meters; clip far depth (matches base3d-clean)
        depthmap = np.load(depth_path).astype(np.float32)
        depthmap[~np.isfinite(depthmap)] = 0.0
        # base3d-clean sets values >80 to -1; we set them to 0 instead, which
        # the standard `depth > min_depth` mask treats identically as invalid.
        depthmap[depthmap > _TARTANAIR_FAR_CLIP] = 0.0

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

        # Intrinsics: fixed at (640x480) base; preprocess_intrinsics rescales.
        intrinsics = self._preprocess_intrinsics(
            _K_BASE, width, height, geometry
        ).float()

        # Extrinsics: c2w (right-handed, OpenCV-style after the xyzqxqyqxqw_to_c2w
        # conversion above) -> w2c.
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
        # Worker-local rng — fresh per call; matches dl3dv.py / scannetpp.py behavior.
        seed_entropy = int(np.random.randint(0, 2**32, dtype=np.uint32))
        rng = np.random.default_rng(seed_entropy)

        for attempt in range(min(16, n)):
            scene_id = self.sequences[(start_scene + attempt) % n]
            try:
                meta = self.scene_frames[scene_id]
                rgb_paths: np.ndarray = meta["rgb_paths"]
                depth_paths: np.ndarray = meta["depth_paths"]
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
                    "sample_mode": "tartanair",
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(
            f"Failed to load a valid TartanAir sample near index {index}: {last_error}"
        )
