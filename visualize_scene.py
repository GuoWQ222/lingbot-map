"""Visualize one scene's reprojected point cloud with viser.

Supports six datasets via ``--dataset``:

* ``manip``    — uses the Manip_long3/4/5 trajectory layout used by training:
                  per-trajectory camera dirs contain ``images/`` and
                  ``depth_real/`` plus ``*_pose.txt`` intrinsics/poses.
                  Sampling mirrors the current training script defaults:
                  train/val split by seed, Long3/4 mode W over ``realsense_*``
                  random-interval trajectories, and Long5 mode T over
                  ``surround_cam_*`` random-interval trajectories.

* ``dl3dv``     — uses :mod:`lingbot_map.data.dl3dv` conventions:
                  read ``valid_{split}.json`` + per-scene ``transforms.json``;
                  RGB at ``images_4/`` (raw /4); depth ``dense/depth/*.npy``
                  cleaned by ``dense/{sky_mask,outlier_mask}/`` + 98%-clip;
                  poses are Blender c2w → post-multiplied by ``BLENDER2OPENCV``.
                  World is Y-up.

* ``scannetpp`` — uses :mod:`lingbot_map.data.scannetpp` conventions:
                  read ``valid_new.json`` or ``valid.json`` + per-scene
                  ``scene_metadata.npz`` (``intrinsics`` / ``trajectories`` /
                  ``images``); RGB at ``images/<basename>.jpg``; depth at
                  ``depth/<basename>.png`` (uint16 mm → /1000); poses are
                  already OpenCV c2w. World is Z-up (verified empirically
                  across 8 scenes — camera-up vectors align with +Z by >0.999).

* ``tartanair`` — uses :mod:`lingbot_map.data.tartanair` conventions:
                  enumerate ``rgb/<scene>/<Easy|Hard>/<P***>/`` episodes;
                  RGB at ``image_left/<basename>_left.png``; depth at
                  ``depth_left/<basename>_left_depth.npy`` (float32 m, far
                  values >80 m clipped to 0); poses parsed from
                  ``pose_left.txt`` (one ``(x, y, z, qx, qy, qz, qw)`` per
                  row) via the same ``xyzqxqyqxqw_to_c2w`` conversion used
                  by base3d-clean — the resulting world is NED-style with
                  +Y pointing down, so the default viser up axis is ``-y``.
                  Intrinsics are fixed (fx=fy=320, cx=320, cy=240, 640x480).

* ``dynamic_replica`` — uses :mod:`lingbot_map.data.dynamic_replica` conventions:
                  enumerate ``{split}/<scene>/left/`` directories;
                  RGB at ``rgb/<basename>.png`` (basenames are float
                  timestamps sorted by ``float()``); depth at
                  ``depth/<basename>.npy`` (float32 m); per-frame
                  ``cam/<basename>.npz`` with keys ``pose`` (4x4 c2w
                  OpenCV) and ``intrinsics`` (3x3). World is Y-up
                  (verified empirically: camera-up world Y component
                  ≈ +0.95 across early frames of one scene).

* ``mapfree``   — uses :mod:`lingbot_map.data.mapfree` conventions:
                  read ``valid.json`` (entries like ``"<scene>/dense{0,1}"``);
                  RGB at ``rgb/<basename>.jpg``; depth at
                  ``depth/<basename>.npy`` (float32 m). Per-frame cleanup
                  matches base3d-clean: sky pixels (``sky_mask/*.jpg`` ≥127)
                  → 0, depth >400 m → 0, NaN/Inf → 0, then drop depths above
                  the per-frame 90th percentile of positive values. Per-frame
                  ``cam/<basename>.npz`` with keys ``intrinsic`` (3x3) and
                  ``pose`` (4x4 c2w OpenCV). World up is ``-y`` (verified
                  empirically: camera-up world Y component is consistently
                  negative — −0.84 to −0.997 across 4 sampled scenes).

Usage::

    # Manip trajectory data
    python visualize_scene.py --dataset manip --seed 7
    python visualize_scene.py --dataset manip --scene /oss-guowenqi/Manip_long5/data/2026-05-14_06_06_08_509223

    # DL3DV (default)
    python visualize_scene.py
    python visualize_scene.py --dataset dl3dv --seed 7

    # ScanNet++ v2
    python visualize_scene.py --dataset scannetpp
    python visualize_scene.py --dataset scannetpp --scene 00777c41d4 --num_frames 30

    # TartanAir
    python visualize_scene.py --dataset tartanair
    python visualize_scene.py --dataset tartanair --scene abandonedfactory/Easy/P000

    # DynamicReplica
    python visualize_scene.py --dataset dynamic_replica
    python visualize_scene.py --dataset dynamic_replica --split train --scene 009850-3_obj

    # MapFree
    python visualize_scene.py --dataset mapfree
    python visualize_scene.py --dataset mapfree --scene s00000/dense0

Inspect the result in your browser at http://localhost:<port>.
If running on a remote machine, forward the port::

    ssh -L 8080:localhost:8080 user@remote

then open http://localhost:8080 locally.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import List, Tuple

import cv2
import numpy as np
import viser
from PIL import Image
from scipy.spatial.transform import Rotation


BLENDER2OPENCV = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)

MANIP_DEFAULT_ROOTS = (
    "/oss-guowenqi/Manip_long3/data",
    "/oss-guowenqi/Manip_long4/data",
    "/oss-guowenqi/Manip_long5/data",
)
MANIP_DEFAULT_MANIFEST = "/cpfs/user/guowenqi/lingbot-map/runs/manip_long_train_64gpu/manip_trajectory_manifest.txt"
MANIP_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIP_CAMERA_NAMES = (
    "realsense_left",
    "realsense_right",
    "surround_cam_0",
    "surround_cam_1",
    "surround_cam_2",
    "surround_cam_3",
    "surround_cam_4",
    "surround_cam_5",
)
OPENCV_TO_GENMANIP_CAMERA_ROTATION = np.array(
    [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float32,
)


# ---------------------------------------------------------------------------
# Per-frame loaders (one per dataset; both return (rgb, depth_m, c2w_opencv, label))
# ---------------------------------------------------------------------------

def _load_dl3dv_frame(scene_path: str, file_rel_path: str, transform_matrix):
    """Mirror the per-frame logic used by the dl3dv dataloader."""
    rgb_path = os.path.join(scene_path, file_rel_path).replace("images", "images_4")
    depth_path = rgb_path.replace("images_4", "dense/depth").replace(".png", ".npy")
    sky_path = rgb_path.replace("images_4", "dense/sky_mask")
    outlier_path = rgb_path.replace("images_4", "dense/outlier_mask")

    rgb = np.array(Image.open(rgb_path).convert("RGB"))  # (H, W, 3) uint8
    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0

    sky_mask = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED) >= 127
    outlier_mask = cv2.imread(outlier_path, cv2.IMREAD_UNCHANGED) >= 127
    depth[sky_mask] = 0.0          # mark sky invalid (we don't need -1 sentinel for viz)
    depth[outlier_mask] = 0.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    pos = depth[depth > 0]
    if pos.size > 0:
        thresh = float(np.percentile(pos, 98))
        depth[depth > thresh] = 0.0

    # Depth comes from MVS at a slightly different resolution; resize to RGB.
    H, W = rgb.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    # Blender c2w (Y-up, Z-back) -> OpenCV c2w (Y-down, Z-forward)
    c2w_opencv = (np.asarray(transform_matrix, dtype=np.float64) @ BLENDER2OPENCV).astype(np.float64)
    return rgb, depth, c2w_opencv, rgb_path


def _load_scannetpp_frame(scene_dir: str, image_npz_name: str, c2w_opencv: np.ndarray):
    """Mirror the per-frame logic used by the scannetpp dataloader.

    ``image_npz_name`` is taken from the npz ``images`` array (e.g.
    ``DSC00850.JPG``); we strip the extension and re-suffix with lowercase
    ``.jpg`` / ``.png`` to match what's on disk. ``c2w_opencv`` is already in
    OpenCV convention — no Blender conversion needed.
    """
    basename = os.path.splitext(image_npz_name)[0]
    rgb_path = os.path.join(scene_dir, "images", basename + ".jpg")
    depth_path = os.path.join(scene_dir, "depth", basename + ".png")

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise IOError(f"Failed to read depth: {depth_path}")
    depth = depth_raw.astype(np.float32) / 1000.0  # uint16 mm -> float32 m
    depth[~np.isfinite(depth)] = 0.0

    H, W = rgb.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    return rgb, depth, np.asarray(c2w_opencv, dtype=np.float64), rgb_path


# TartanAir: pose_left.txt rows store (x, y, z, qx, qy, qz, qw) in NED-style
# world coordinates. The conversion below is copied verbatim from
# ``base3d-clean/datasets/tartanair.py`` — note the ``z, x, y`` reorder that
# turns the raw NED translation/quaternion into the right-handed convention
# the rest of the pipeline expects.
def _xyzqxqyqxqw_to_c2w(xyzqxqyqxqw: np.ndarray) -> np.ndarray:
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ]
    )
    c2w[:3, 3] = np.array([x, y, z])
    return c2w


def _load_tartanair_frame(
    img_dir: str, depth_dir: str, basename: str, c2w_opencv: np.ndarray
):
    """Mirror the per-frame logic used by the tartanair dataloader."""
    rgb_path = os.path.join(img_dir, f"{basename}_left.png")
    depth_path = os.path.join(depth_dir, f"{basename}_left_depth.npy")

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    # base3d-clean clips depth > 80m as invalid (originally to -1, equivalent
    # for the ``depth > 0`` validity check we use here).
    depth[depth > 80.0] = 0.0

    H, W = rgb.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    return rgb, depth, np.asarray(c2w_opencv, dtype=np.float64), rgb_path


def _load_dynamic_replica_frame(scene_dir: str, basename: str):
    """Mirror the per-frame logic used by the dynamic_replica dataloader.

    ``scene_dir`` is the ``left/`` directory of one scene. ``basename`` is the
    timestamp string (e.g. ``"0.0333..."``) without extension. Per-frame
    ``cam/<basename>.npz`` carries both intrinsics and the OpenCV c2w pose, so
    we read both here (intrinsics is stored on the adapter; we only return the
    pose so the (rgb, depth, c2w, label) signature stays consistent with the
    other loaders).
    """
    rgb_path = os.path.join(scene_dir, "rgb", basename + ".png")
    depth_path = os.path.join(scene_dir, "depth", basename + ".npy")
    cam_path = os.path.join(scene_dir, "cam", basename + ".npz")

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0

    with np.load(cam_path) as cam:
        c2w_opencv = np.asarray(cam["pose"], dtype=np.float64)

    H, W = rgb.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    return rgb, depth, c2w_opencv, rgb_path


def _load_mapfree_frame(rgb_dir: str, depth_dir: str, sky_dir: str, cam_dir: str, basename: str):
    """Mirror the per-frame logic used by the mapfree dataloader.

    Cleaning matches base3d-clean/datasets/mapfree.py:246-254 exactly:
    sky pixels (mask >= 127) → 0, depth > 400 m → 0, NaN/Inf → 0, then
    drop depths above the per-frame 90th percentile of positive values.
    Per-frame ``cam/<basename>.npz`` carries both intrinsics (varies
    slightly per frame) and the OpenCV c2w pose; we only return the pose
    here because :class:`MapfreeAdapter` uses frame-0 K for the unproject
    and frustum FOV (same approach as ``dynamic_replica``).
    """
    rgb_path = os.path.join(rgb_dir, basename + ".jpg")
    depth_path = os.path.join(depth_dir, basename + ".npy")
    sky_path = os.path.join(sky_dir, basename + ".jpg")
    cam_path = os.path.join(cam_dir, basename + ".npz")

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth = np.load(depth_path).astype(np.float32)

    sky_mask = cv2.imread(sky_path, cv2.IMREAD_UNCHANGED)
    if sky_mask is None:
        raise IOError(f"Failed to read sky mask: {sky_path}")
    if sky_mask.ndim == 3:
        sky_mask = sky_mask[..., 0]
    sky_mask = sky_mask >= 127

    depth[sky_mask] = 0.0
    depth[depth > 400.0] = 0.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    pos = depth[depth > 0]
    if pos.size > 0:
        thresh = float(np.percentile(pos, 90))
        depth[depth > thresh] = 0.0

    H, W = rgb.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    with np.load(cam_path) as cam:
        c2w_opencv = np.asarray(cam["pose"], dtype=np.float64)

    return rgb, depth, c2w_opencv, rgb_path




def _manip_camera_sort_key(camera_name: str) -> Tuple[int, object]:
    if camera_name in MANIP_CAMERA_NAMES:
        return (0, MANIP_CAMERA_NAMES.index(camera_name))
    return (1, camera_name)


def _parse_manip_frame_stem(stem: str) -> int:
    if stem.isdigit():
        return int(stem)
    for token in reversed(stem.replace("-", "_").split("_")):
        if token.isdigit():
            return int(token)
    raise ValueError(f"Cannot parse Manip frame id from {stem}")


def _quat_wxyz_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 0:
        raise ValueError("Invalid zero-norm quaternion")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _manip_camera2env_pose_to_opencv_w2c(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    camera_to_env = np.eye(4, dtype=np.float32)
    camera_to_env[:3, :3] = _quat_wxyz_to_rotation_matrix(quat_wxyz)
    camera_to_env[:3, 3] = np.asarray(position, dtype=np.float32)

    opencv_to_genmanip = np.eye(4, dtype=np.float32)
    opencv_to_genmanip[:3, :3] = OPENCV_TO_GENMANIP_CAMERA_ROTATION
    opencv_camera_to_env = camera_to_env @ opencv_to_genmanip
    return np.linalg.inv(opencv_camera_to_env).astype(np.float32)


def _read_manip_camera_pose_file(pose_path: str) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
    intrinsic_rows: List[List[float]] = []
    extrinsics_by_frame: Dict[int, np.ndarray] = {}
    with open(pose_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                parts = stripped[1:].strip().split()
                if len(parts) == 3:
                    try:
                        intrinsic_rows.append([float(value) for value in parts])
                    except ValueError:
                        pass
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            frame_id = int(float(parts[0]))
            position = np.asarray([float(value) for value in parts[1:4]], dtype=np.float32)
            quat_wxyz = np.asarray([float(value) for value in parts[4:8]], dtype=np.float32)
            extrinsics_by_frame[frame_id] = _manip_camera2env_pose_to_opencv_w2c(position, quat_wxyz)
    if len(intrinsic_rows) < 3:
        raise ValueError(f"No 3x3 intrinsics found in {pose_path}")
    if not extrinsics_by_frame:
        raise ValueError(f"No per-frame poses found in {pose_path}")
    return np.asarray(intrinsic_rows[:3], dtype=np.float32), extrinsics_by_frame


def _load_manip_frame(rgb_path: str, depth_path: str, w2c_opencv: np.ndarray):
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise IOError(f"Failed to read depth: {depth_path}")
    depth_dtype = depth_raw.dtype
    if depth_raw.ndim == 3:
        depth_raw = depth_raw.astype(np.float32).mean(axis=2)
    if depth_dtype == np.uint16:
        depth_scale = 10000.0
    elif np.issubdtype(depth_dtype, np.integer):
        depth_scale = float(np.iinfo(depth_dtype).max)
    else:
        depth_scale = 1.0
    depth = depth_raw.astype(np.float32) / depth_scale
    depth[~np.isfinite(depth)] = 0.0

    H, W = rgb.shape[:2]
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    w2c_44 = np.eye(4, dtype=np.float64)
    w2c_44[:3, :4] = np.asarray(w2c_opencv, dtype=np.float64)[:3, :4]
    c2w_opencv = np.linalg.inv(w2c_44)
    return rgb, depth, c2w_opencv, rgb_path

# ---------------------------------------------------------------------------
# Scene adapters: hide dataset-specific scene-list & metadata fetching behind
# one tiny common interface.
# ---------------------------------------------------------------------------

class _SceneAdapter:
    """Common interface used by main().

    Attributes set by subclass __init__:
        scene_label   (str)        : human-readable scene id
        K             (3,3 float)  : shared intrinsics for unproject + FOV
        num_total     (int)        : total frames in scene
        default_up    (str)        : viser up axis hint ('+x','-x','+y','-y','+z','-z')
    """
    scene_label: str = ""
    K: np.ndarray = np.eye(3, dtype=np.float32)
    num_total: int = 0
    default_up: str = "+y"

    def get_frame(self, fid: int):
        raise NotImplementedError



class ManipAdapter(_SceneAdapter):
    DEFAULT_ROOTS = MANIP_DEFAULT_ROOTS
    default_up = "+z"

    def __init__(
        self,
        root: str,
        split: str,
        seed: int,
        scene_override: str,
        camera_filter: str = "",
        manifest: str = MANIP_DEFAULT_MANIFEST,
        val_fraction: float = 0.02,
        max_sample_frames: int = 64,
        min_sample_frames: int = 16,
        random_stride_min: int = 10,
        random_stride_max: int = 60,
        t_stride_min: int = 15,
        t_stride_max: int = 60,
        wrist_camera_prefix: str = "realsense",
        static_camera_prefix: str = "surround",
        long5_root_marker: str = "Manip_long5",
    ):
        py_rng = random.Random(seed)
        self.roots = self._parse_roots(root)
        self.split = split
        self.camera_filter = camera_filter
        self.manifest = manifest or MANIP_DEFAULT_MANIFEST
        self.val_fraction = float(val_fraction)
        self.max_sample_frames = max(1, int(max_sample_frames))
        self.min_sample_frames = max(1, int(min_sample_frames))
        self.random_stride_min = max(1, int(random_stride_min))
        self.random_stride_max = max(self.random_stride_min, int(random_stride_max))
        self.t_stride_min = max(1, int(t_stride_min))
        self.t_stride_max = max(self.t_stride_min, int(t_stride_max))
        self.wrist_camera_prefix = wrist_camera_prefix.lower()
        self.static_camera_prefix = static_camera_prefix.lower()
        self.long5_root_marker = long5_root_marker or ""

        if scene_override:
            self.scene_dir = self._resolve_scene(scene_override)
            print(f"[scene] (manip) using provided trajectory: {self.scene_dir}")
        else:
            scenes = self._discover_scenes()
            if not scenes:
                raise FileNotFoundError(f"No Manip trajectories found under {self.roots}")
            train_scenes, val_scenes = self._split_scenes(scenes, self.val_fraction, seed)
            split_key = split.lower()
            if split_key in {"val", "valid", "validation", "test"}:
                pool = val_scenes
                pool_name = "val"
            else:
                pool = train_scenes
                pool_name = "train"
            if not pool:
                raise RuntimeError(f"Manip {pool_name} split is empty (total scenes={len(scenes)})")
            idx = py_rng.randrange(len(pool))
            self.scene_dir = pool[idx]
            print(
                f"[scene] (manip/train.sh) split={pool_name} val_fraction={self.val_fraction:g} "
                f"seed={seed} picked #{idx} / {len(pool)}:"
            )
            print(f"        {self.scene_dir}")

        cameras = self._discover_cameras(self.scene_dir)
        if not cameras:
            raise RuntimeError(f"No valid Manip camera dirs in {self.scene_dir}")
        cam_name, cam_dir, pose_path, mode_label = self._choose_camera(cameras, py_rng)
        self.camera_name = cam_name
        self.camera_dir = cam_dir
        self.pose_path = pose_path
        self.sample_mode = mode_label
        self.K, self.w2c_by_frame = _read_manip_camera_pose_file(pose_path)

        image_dir = os.path.join(cam_dir, "images")
        depth_dir = os.path.join(cam_dir, "depth_real")
        frames = []
        for name in sorted(os.listdir(image_dir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in MANIP_IMAGE_SUFFIXES:
                continue
            try:
                frame_id = _parse_manip_frame_stem(stem)
            except ValueError:
                continue
            rgb_path = os.path.join(image_dir, name)
            depth_path = os.path.join(depth_dir, stem + ".png")
            if frame_id not in self.w2c_by_frame or not os.path.isfile(depth_path):
                continue
            frames.append((frame_id, rgb_path, depth_path))
        if not frames:
            raise RuntimeError(f"No usable RGB/depth/pose frames for {cam_name} in {self.scene_dir}")
        self.frames = sorted(frames, key=lambda item: item[0])
        self.num_total = len(self.frames)
        self.sampled_positions = self._sample_train_positions(py_rng)
        self.scene_label = f"{os.path.basename(self.scene_dir)}/{cam_name}/{self.sample_mode}"

        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        sampled_frame_ids = [self.frames[i][0] for i in self.sampled_positions]
        print(f"[K] (manip) {cam_name}: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
        print(f"[camera] (manip/train.sh) mode={self.sample_mode} selected {cam_name}; frames={self.num_total}; pose={pose_path}")
        print(
            f"[sampling] (manip/train.sh) selected {len(self.sampled_positions)} / {self.num_total} "
            f"frames with ids: {sampled_frame_ids}"
        )
        if camera_filter:
            print(f"[sampling] (manip/train.sh) --manip_camera acts like training CAMERA_NAMES filter: {camera_filter}")

    @staticmethod
    def _parse_roots(root: str) -> List[str]:
        if not root:
            return list(MANIP_DEFAULT_ROOTS)
        parts = []
        for chunk in root.replace(":", ",").split(","):
            item = chunk.strip()
            if item:
                parts.append(item)
        return parts or list(MANIP_DEFAULT_ROOTS)

    @staticmethod
    def _split_scenes(scenes: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
        scenes = list(scenes)
        rng = random.Random(seed)
        rng.shuffle(scenes)
        if val_fraction <= 0 or len(scenes) < 2:
            return scenes, []
        val_count = max(1, int(round(len(scenes) * val_fraction)))
        return scenes[val_count:], scenes[:val_count]

    def _resolve_scene(self, scene_override: str) -> str:
        if os.path.isdir(scene_override):
            return scene_override
        for root in self.roots:
            candidate = os.path.join(root, scene_override.strip("/"))
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(f"Manip scene not found: {scene_override}")

    def _discover_scenes(self) -> List[str]:
        if os.path.isfile(self.manifest):
            with open(self.manifest, "r", encoding="utf-8") as handle:
                scenes = [line.strip() for line in handle if line.strip()]
            scenes = [p for p in scenes if any(p.startswith(root.rstrip("/") + "/") for root in self.roots)]
            if scenes:
                print(f"[scene] (manip) loaded {len(scenes)} trajectories from {self.manifest}")
                return scenes
        scenes = []
        for root in self.roots:
            if not os.path.isdir(root):
                print(f"[warn] Manip root missing: {root}")
                continue
            for name in sorted(os.listdir(root)):
                if not name or name.startswith(".") or "claim" in name.lower():
                    continue
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    scenes.append(path)
        return scenes

    @staticmethod
    def _discover_cameras(scene_dir: str) -> List[Tuple[str, str, str]]:
        out = []
        for name in sorted(os.listdir(scene_dir), key=_manip_camera_sort_key):
            cam_dir = os.path.join(scene_dir, name)
            if not os.path.isdir(cam_dir):
                continue
            if not os.path.isdir(os.path.join(cam_dir, "images")):
                continue
            if not os.path.isdir(os.path.join(cam_dir, "depth_real")):
                continue
            pose_path = os.path.join(cam_dir, f"{name}_pose.txt")
            if not os.path.isfile(pose_path):
                pose_path = os.path.join(scene_dir, f"{name}_pose.txt")
            if os.path.isfile(pose_path):
                out.append((name, cam_dir, pose_path))
        return out

    def _is_long5_scene(self) -> bool:
        return bool(self.long5_root_marker) and self.long5_root_marker.lower() in self.scene_dir.lower()

    def _choose_camera(self, cameras: List[Tuple[str, str, str]], rng: random.Random) -> Tuple[str, str, str, str]:
        if self.camera_filter:
            cameras = [item for item in cameras if item[0] == self.camera_filter]
            if not cameras:
                raise ValueError(f"--manip_camera={self.camera_filter!r} not found in {self.scene_dir}")

        wrist = [item for item in cameras if item[0].lower().startswith(self.wrist_camera_prefix)]
        static = [item for item in cameras if item[0].lower().startswith(self.static_camera_prefix)]
        if self._is_long5_scene():
            # train.sh: Manip_long5 bypasses S/W/M and always uses mode T.
            candidates = static or wrist or cameras
            mode_label = "T"
        else:
            # train.sh defaults MODE_WEIGHTS_INITIAL/FINAL to S=0,W=1,M=0,
            # so Long3/4 use mode W. Mode W prefers wrist/realsense cameras.
            candidates = wrist or static or cameras
            mode_label = "W"
        if not candidates:
            raise RuntimeError("No candidate Manip cameras after train.sh-style filtering")
        return (*rng.choice(candidates), mode_label)

    def _sample_train_positions(self, rng: random.Random) -> List[int]:
        if self.sample_mode == "T":
            stride_min, stride_max = self.t_stride_min, self.t_stride_max
        else:
            stride_min, stride_max = self.random_stride_min, self.random_stride_max
        n = len(self.frames)
        start = rng.randint(0, n - 1) if n > 1 else 0

        def walk(direction: int) -> List[int]:
            selected = [start]
            idx = start
            while len(selected) < self.max_sample_frames:
                step = rng.randint(stride_min, stride_max)
                nxt = idx + direction * step
                if nxt < 0 or nxt >= n:
                    break
                selected.append(nxt)
                idx = nxt
            return selected

        selected = walk(+1)
        if len(selected) < self.min_sample_frames:
            backward = walk(-1)
            backward.reverse()
            if len(backward) > len(selected):
                selected = backward
        if len(selected) < self.min_sample_frames and n >= self.min_sample_frames:
            indices = np.linspace(0, n - 1, self.min_sample_frames)
            selected = [int(round(idx)) for idx in indices]
        return selected

    def sample_frame_indices(self, num_frames: int) -> List[int]:
        if num_frames <= 0 or num_frames >= len(self.sampled_positions):
            return list(self.sampled_positions)
        indices = np.round(np.linspace(0, len(self.sampled_positions) - 1, num_frames)).astype(int)
        return [self.sampled_positions[int(i)] for i in indices]

    def get_frame(self, fid: int):
        order = list(range(fid, len(self.frames))) + list(range(fid - 1, -1, -1))
        last_error: Exception | None = None
        for idx in order:
            frame_id, rgb_path, depth_path = self.frames[idx]
            try:
                return _load_manip_frame(rgb_path, depth_path, self.w2c_by_frame[frame_id])
            except Exception as exc:  # skip rare corrupt mounted frames
                last_error = exc
                continue
        raise RuntimeError(f"No readable Manip frame near index {fid}: {last_error}")


class DL3DVAdapter(_SceneAdapter):
    DEFAULT_ROOT = "/cpfs/shared/landmark/renkerui/data/dl3dv"
    default_up = "+y"  # Blender Y-up world is preserved by post-multiplying c2w by BLENDER2OPENCV.

    def __init__(self, root: str, split: str, seed: int, scene_override: str):
        rng = np.random.default_rng(seed)
        if scene_override:
            self.scene_rel = scene_override
            print(f"[scene] using provided path: {self.scene_rel}")
        else:
            split_json = os.path.join(root, f"valid_{split}.json")
            with open(split_json) as f:
                scenes: List[str] = json.load(f)
            idx = int(rng.integers(0, len(scenes)))
            self.scene_rel = scenes[idx]
            print(f"[scene] (dl3dv) seed={seed} picked #{idx} / {len(scenes)} from {split_json}:")
            print(f"        {self.scene_rel}")

        self.scene_path = os.path.join(root, self.scene_rel)
        with open(os.path.join(self.scene_path, "transforms.json")) as f:
            self.meta = json.load(f)

        # transforms.json stores at full 4K; images_4/ is /4 downsampled.
        fx = float(self.meta["fl_x"]) / 4.0
        fy = float(self.meta["fl_y"]) / 4.0
        cx = float(self.meta["cx"]) / 4.0
        cy = float(self.meta["cy"]) / 4.0
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        print(f"[K] (dl3dv) images_4 intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

        self.num_total = len(self.meta["frames"])
        self.scene_label = self.scene_rel

    def get_frame(self, fid: int):
        fr = self.meta["frames"][fid]
        return _load_dl3dv_frame(self.scene_path, fr["file_path"], fr["transform_matrix"])


class ScanNetppAdapter(_SceneAdapter):
    DEFAULT_ROOT = "/shared/smartbot/renkerui/data/scannetppv2"
    # Empirically validated: median camera-up vector (R @ [0,-1,0]) aligns with
    # +Z within 0.999 across multiple ScanNet++ v2 scenes — see the inline
    # verification in the chat that produced this file.
    default_up = "+z"

    def __init__(self, root: str, split: str, seed: int, scene_override: str):
        rng = np.random.default_rng(seed)
        if scene_override:
            self.scene_rel = scene_override
            print(f"[scene] using provided path: {self.scene_rel}")
        else:
            valid_new = os.path.join(root, "valid_new.json")
            valid_raw = os.path.join(root, "valid.json")
            list_path = valid_new if os.path.exists(valid_new) else valid_raw
            if not os.path.exists(list_path):
                raise FileNotFoundError(
                    f"Neither valid_new.json nor valid.json under {root}. "
                    "Generate valid.json first (a JSON list of scene-id directories)."
                )
            with open(list_path) as f:
                scenes: List[str] = json.load(f)
            # ScanNet++ in this layout has no per-split json; --split is informational.
            idx = int(rng.integers(0, len(scenes)))
            self.scene_rel = scenes[idx]
            print(f"[scene] (scannetpp) seed={seed} split={split} picked #{idx} / {len(scenes)} from {list_path}:")
            print(f"        {self.scene_rel}")

        self.scene_dir = os.path.join(root, self.scene_rel)
        npz_path = os.path.join(self.scene_dir, "scene_metadata.npz")
        meta = np.load(npz_path, allow_pickle=True)
        self.intrinsics_arr = np.asarray(meta["intrinsics"], dtype=np.float64)    # (N, 3, 3)
        self.extrinsics_arr = np.asarray(meta["trajectories"], dtype=np.float64)  # (N, 4, 4) c2w opencv
        self.images_arr = np.asarray(meta["images"])                              # (N,) string

        # scene_meta.json says shared_intrinsics=True, but be defensive and use frame-0 K.
        self.K = self.intrinsics_arr[0].astype(np.float32)
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        print(f"[K] (scannetpp) shared intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

        self.num_total = int(len(self.images_arr))
        self.scene_label = self.scene_rel

    def get_frame(self, fid: int):
        return _load_scannetpp_frame(
            self.scene_dir, str(self.images_arr[fid]), self.extrinsics_arr[fid]
        )


class TartanAirAdapter(_SceneAdapter):
    DEFAULT_ROOT = "/cpfs/shared/landmark/renkerui/data/tartanair"
    # After ``_xyzqxqyqxqw_to_c2w``'s NED reorder, the world frame is
    # x=East, y=Down, z=North. So the up direction is -y.
    default_up = "-y"
    # Fixed TartanAir intrinsics (pinhole, no distortion, 640x480).
    BASE_K = np.array([[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    def __init__(self, root: str, split: str, seed: int, scene_override: str):
        rng = np.random.default_rng(seed)
        rgb_root = os.path.join(root, "rgb")
        if not os.path.isdir(rgb_root):
            raise FileNotFoundError(f"TartanAir rgb root not found: {rgb_root}")

        if scene_override:
            parts = scene_override.strip("/").split("/")
            if len(parts) != 3:
                raise ValueError(
                    f"--scene for tartanair must be 'scene/diff/episode' (e.g. "
                    f"'abandonedfactory/Easy/P000'); got {scene_override!r}"
                )
            self.scene_name, self.diff, self.episode = parts
            print(f"[scene] using provided path: {self.scene_name}/{self.diff}/{self.episode}")
        else:
            episodes: List[Tuple[str, str, str]] = []
            for s in sorted(os.listdir(rgb_root)):
                sdir = os.path.join(rgb_root, s)
                if not os.path.isdir(sdir):
                    continue
                for d in ("Easy", "Hard"):
                    ddir = os.path.join(sdir, d)
                    if not os.path.isdir(ddir):
                        continue
                    for n in sorted(os.listdir(ddir)):
                        if os.path.isdir(os.path.join(ddir, n)):
                            episodes.append((s, d, n))
            if not episodes:
                raise FileNotFoundError(f"No TartanAir episodes under {rgb_root}")
            idx = int(rng.integers(0, len(episodes)))
            self.scene_name, self.diff, self.episode = episodes[idx]
            print(
                f"[scene] (tartanair) seed={seed} split={split} picked #{idx} / "
                f"{len(episodes)}: {self.scene_name}/{self.diff}/{self.episode}"
            )

        self.img_dir = os.path.join(
            root, "rgb", self.scene_name, self.diff, self.episode, "image_left"
        )
        self.depth_dir = os.path.join(
            root, "depth", self.scene_name, self.diff, self.episode, "depth_left"
        )
        pose_txt = os.path.join(
            root, "rgb", self.scene_name, self.diff, self.episode, "pose_left.txt"
        )
        for p in (self.img_dir, self.depth_dir):
            if not os.path.isdir(p):
                raise FileNotFoundError(p)
        if not os.path.isfile(pose_txt):
            raise FileNotFoundError(pose_txt)

        # Match base3d-clean: list image basenames lexicographically by the part
        # before the first '_' (frame ids like '000000', '000001', ...).
        basenames = sorted(
            f.split("_")[0] for f in os.listdir(self.img_dir) if f.endswith("_left.png")
        )
        caminfo = np.loadtxt(pose_txt)
        if caminfo.ndim != 2 or caminfo.shape[1] != 7:
            raise ValueError(
                f"pose_left.txt must be (N, 7); got shape {caminfo.shape} at {pose_txt}"
            )
        # Truncate to whichever is shorter (same alignment policy as base3d-clean).
        minlen = min(caminfo.shape[0], len(basenames))
        if minlen == 0:
            raise RuntimeError(f"Empty episode: {self.scene_name}/{self.diff}/{self.episode}")
        self.basenames: List[str] = basenames[:minlen]
        self.extrinsics_arr = np.stack(
            [_xyzqxqyqxqw_to_c2w(caminfo[i]) for i in range(minlen)], axis=0
        )

        self.K = self.BASE_K.copy()
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        print(f"[K] (tartanair) fixed intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

        self.num_total = minlen
        self.scene_label = f"{self.scene_name}/{self.diff}/{self.episode}"

    def get_frame(self, fid: int):
        return _load_tartanair_frame(
            self.img_dir, self.depth_dir, self.basenames[fid], self.extrinsics_arr[fid]
        )


class DynamicReplicaAdapter(_SceneAdapter):
    DEFAULT_ROOT = "/shared/smartbot/renkerui/data/dynamic_replica"
    # Empirically validated on scene 009850-3_obj early frames: camera-up world
    # Y component ≈ +0.95, so world up is +Y (matches the underlying Replica
    # synthetic world convention).
    default_up = "+y"

    def __init__(self, root: str, split: str, seed: int, scene_override: str):
        rng = np.random.default_rng(seed)
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"DynamicReplica split dir not found: {split_dir} "
                f"(expected one of train/valid/test)"
            )

        if scene_override:
            self.scene_id = scene_override
            print(f"[scene] using provided path: {self.scene_id}")
        else:
            scenes = sorted(
                s for s in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, s))
            )
            if not scenes:
                raise FileNotFoundError(f"No DynamicReplica scenes under {split_dir}")
            idx = int(rng.integers(0, len(scenes)))
            self.scene_id = scenes[idx]
            print(
                f"[scene] (dynamic_replica) seed={seed} split={split} "
                f"picked #{idx} / {len(scenes)}: {self.scene_id}"
            )

        self.scene_dir = os.path.join(split_dir, self.scene_id, "left")
        rgb_dir = os.path.join(self.scene_dir, "rgb")
        cam_dir = os.path.join(self.scene_dir, "cam")
        depth_dir = os.path.join(self.scene_dir, "depth")
        for p in (rgb_dir, cam_dir, depth_dir):
            if not os.path.isdir(p):
                raise FileNotFoundError(p)

        # Match base3d-clean: list basenames sorted by float value.
        self.basenames: List[str] = sorted(
            (f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")),
            key=lambda x: float(x),
        )
        if not self.basenames:
            raise RuntimeError(f"No .png frames in {rgb_dir}")

        # Frame-0 intrinsics are used for unprojection + frustum FOV. The
        # dynamic_replica scenes I checked store the same K across frames, but
        # be defensive and only rely on frame-0.
        with np.load(os.path.join(cam_dir, self.basenames[0] + ".npz")) as cam:
            self.K = np.asarray(cam["intrinsics"], dtype=np.float32)
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        print(
            f"[K] (dynamic_replica) frame-0 intrinsics: "
            f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
        )

        self.num_total = len(self.basenames)
        self.scene_label = f"{split}/{self.scene_id}"

    def get_frame(self, fid: int):
        return _load_dynamic_replica_frame(self.scene_dir, self.basenames[fid])


class MapfreeAdapter(_SceneAdapter):
    DEFAULT_ROOT = "/cpfs/shared/landmark/renkerui/data/mapfree"
    # Empirically validated across 4 scenes (s00000/dense0, s00050/dense0,
    # s00200/dense1, s00400/dense0): mean camera-up world Y component
    # ranges -0.84 to -0.997 — i.e. world up is consistently -y.
    default_up = "-y"

    def __init__(self, root: str, split: str, seed: int, scene_override: str):
        rng = np.random.default_rng(seed)
        valid_json = os.path.join(root, "valid.json")
        if not os.path.exists(valid_json):
            raise FileNotFoundError(
                f"valid.json not found at {valid_json}. MapFree expects a JSON "
                f"list of scene-relative paths like 's00000/dense0'."
            )

        if scene_override:
            self.scene_rel = scene_override
            print(f"[scene] using provided path: {self.scene_rel}")
        else:
            with open(valid_json) as f:
                scenes: List[str] = json.load(f)
            # MapFree has no per-split json; --split is informational.
            idx = int(rng.integers(0, len(scenes)))
            self.scene_rel = scenes[idx]
            print(
                f"[scene] (mapfree) seed={seed} split={split} "
                f"picked #{idx} / {len(scenes)} from {valid_json}:"
            )
            print(f"        {self.scene_rel}")

        self.scene_dir = os.path.join(root, self.scene_rel)
        self.rgb_dir = os.path.join(self.scene_dir, "rgb")
        self.depth_dir = os.path.join(self.scene_dir, "depth")
        self.cam_dir = os.path.join(self.scene_dir, "cam")
        self.sky_dir = os.path.join(self.scene_dir, "sky_mask")
        for p in (self.rgb_dir, self.depth_dir, self.cam_dir, self.sky_dir):
            if not os.path.isdir(p):
                raise FileNotFoundError(p)

        # Match base3d-clean: list rgb basenames sorted lexicographically.
        self.basenames: List[str] = sorted(
            f[:-4] for f in os.listdir(self.rgb_dir) if f.endswith(".jpg")
        )
        if not self.basenames:
            raise RuntimeError(f"No .jpg frames in {self.rgb_dir}")

        # MapFree intrinsics vary slightly per frame (~1% in fx, sub-pixel in
        # cx/cy across the 580 frames I sampled). Use frame-0 K for both
        # unprojection and frustum FOV — same approach as dynamic_replica.
        with np.load(os.path.join(self.cam_dir, self.basenames[0] + ".npz")) as cam:
            self.K = np.asarray(cam["intrinsic"], dtype=np.float32)
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        print(
            f"[K] (mapfree) frame-0 intrinsics: "
            f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
        )

        self.num_total = len(self.basenames)
        self.scene_label = self.scene_rel

    def get_frame(self, fid: int):
        return _load_mapfree_frame(
            self.rgb_dir, self.depth_dir, self.sky_dir, self.cam_dir, self.basenames[fid]
        )


# ---------------------------------------------------------------------------
# Common visualization plumbing
# ---------------------------------------------------------------------------

def unproject(
    rgb: np.ndarray, depth: np.ndarray, K: np.ndarray, c2w: np.ndarray,
    max_points: int, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject valid depth pixels to world coordinates with their RGB colors."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = (depth > 0) & np.isfinite(depth)
    ys, xs = np.where(valid)
    if len(ys) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    if max_points > 0 and len(ys) > max_points:
        sel = rng.choice(len(ys), size=max_points, replace=False)
        ys, xs = ys[sel], xs[sel]
    zs = depth[ys, xs]

    x_cam = (xs - cx) / fx * zs
    y_cam = (ys - cy) / fy * zs
    pts_cam = np.stack([x_cam, y_cam, zs, np.ones_like(zs)], axis=1)  # (N, 4)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    colors = rgb[ys, xs]
    return pts_world.astype(np.float32), colors.astype(np.uint8)


def rotmat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> (w, x, y, z) quaternion for viser."""
    xyzw = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)


def build_adapter(args: argparse.Namespace) -> _SceneAdapter:
    if args.dataset == "manip":
        root = args.root or args.manip_roots
        return ManipAdapter(
            root=root,
            split=args.split,
            seed=args.seed,
            scene_override=args.scene,
            camera_filter=args.manip_camera,
            manifest=args.manip_manifest,
            val_fraction=args.manip_val_fraction,
            max_sample_frames=args.manip_max_sample_frames,
            min_sample_frames=args.manip_min_sample_frames,
            random_stride_min=args.manip_random_stride_min,
            random_stride_max=args.manip_random_stride_max,
            t_stride_min=args.manip_t_stride_min,
            t_stride_max=args.manip_t_stride_max,
            wrist_camera_prefix=args.manip_wrist_camera_prefix,
            static_camera_prefix=args.manip_static_camera_prefix,
            long5_root_marker=args.manip_long5_root_marker,
        )
    elif args.dataset == "dl3dv":
        root = args.root or DL3DVAdapter.DEFAULT_ROOT
        return DL3DVAdapter(root, args.split, args.seed, args.scene)
    elif args.dataset == "scannetpp":
        root = args.root or ScanNetppAdapter.DEFAULT_ROOT
        return ScanNetppAdapter(root, args.split, args.seed, args.scene)
    elif args.dataset == "tartanair":
        root = args.root or TartanAirAdapter.DEFAULT_ROOT
        return TartanAirAdapter(root, args.split, args.seed, args.scene)
    elif args.dataset == "dynamic_replica":
        root = args.root or DynamicReplicaAdapter.DEFAULT_ROOT
        return DynamicReplicaAdapter(root, args.split, args.seed, args.scene)
    elif args.dataset == "mapfree":
        root = args.root or MapfreeAdapter.DEFAULT_ROOT
        return MapfreeAdapter(root, args.split, args.seed, args.scene)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["manip", "dl3dv", "scannetpp", "tartanair", "dynamic_replica", "mapfree"], default="dl3dv",
                    help="Which dataset's conventions to use for loading.")
    ap.add_argument("--root", default="",
                    help="Dataset root. Empty -> use the dataset's built-in DEFAULT_ROOT. For manip, comma-separated roots are accepted.")
    ap.add_argument("--manip_roots", default=",".join(MANIP_DEFAULT_ROOTS),
                    help="Comma-separated Manip roots used when --dataset manip and --root is empty.")
    ap.add_argument("--manip_camera", default="",
                    help="Optional Manip camera-name filter, matching train.sh CAMERA_NAMES semantics for one camera.")
    ap.add_argument("--manip_manifest", default=MANIP_DEFAULT_MANIFEST,
                    help="Manip trajectory manifest. Set this to train.sh's SCENE_MANIFEST for an exact scene universe.")
    ap.add_argument("--manip_val_fraction", type=float, default=0.02,
                    help="Manip train/val split fraction; default matches train.sh.")
    ap.add_argument("--manip_max_sample_frames", type=int, default=64,
                    help="Manip W/T max sampled frames; default matches train.sh MAX_SAMPLE_FRAMES.")
    ap.add_argument("--manip_min_sample_frames", type=int, default=16,
                    help="Manip W/T minimum fallback frames; default matches train.sh MIN_SAMPLE_FRAMES.")
    ap.add_argument("--manip_random_stride_min", type=int, default=10,
                    help="Manip Long3/4 mode-W random stride min; default matches train.sh RANDOM_STRIDE_MIN.")
    ap.add_argument("--manip_random_stride_max", type=int, default=60,
                    help="Manip Long3/4 mode-W random stride max; default matches train.sh RANDOM_STRIDE_MAX.")
    ap.add_argument("--manip_t_stride_min", type=int, default=15,
                    help="Manip Long5 mode-T random stride min; default matches train.sh T_STRIDE_MIN.")
    ap.add_argument("--manip_t_stride_max", type=int, default=60,
                    help="Manip Long5 mode-T random stride max; default matches train.sh T_STRIDE_MAX.")
    ap.add_argument("--manip_wrist_camera_prefix", default="realsense",
                    help="Manip mode-W wrist camera prefix; default matches train.sh WRIST_CAMERA_PREFIX.")
    ap.add_argument("--manip_static_camera_prefix", default="surround",
                    help="Manip mode-T/static camera prefix; default matches train.sh STATIC_CAMERA_PREFIX.")
    ap.add_argument("--manip_long5_root_marker", default="Manip_long5",
                    help="Substring used to route Manip_long5 scenes to mode T; default matches train.sh.")
    ap.add_argument("--split", default="train",
                    help="DL3DV: chooses valid_{split}.json. ScanNet++/TartanAir/MapFree: informational only. "
                         "DynamicReplica: chooses the {train,valid,test}/ subdir.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scene", default="",
                    help="Optional: bypass random pick and use this exact scene id "
                         "(Manip: full trajectory path or trajectory dir name under one root; "
                         "DL3DV: 'bucket/scene'; ScanNet++: 'scene_id'; "
                         "TartanAir: 'scene/diff/episode' e.g. 'abandonedfactory/Easy/P000'; "
                         "DynamicReplica: 'scene_id' e.g. '009850-3_obj'; "
                         "MapFree: 'scene/dense{0,1}' e.g. 's00000/dense0').")
    ap.add_argument("--num_frames", type=int, default=64,
                    help="Number of frames to unproject. Manip first samples a train.sh-style clip, then caps/downsamples to this count; other datasets spread evenly across the trajectory.")
    ap.add_argument("--max_points_per_frame", type=int, default=50000,
                    help="Subsample points per frame (0 = keep all).")
    ap.add_argument("--point_size", type=float, default=0.005)
    ap.add_argument("--frustum_scale", type=float, default=0.03)
    ap.add_argument("--show_frustums", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--up_direction", default="",
                    help="viser up axis. Empty -> use adapter default "
                         "('+z' for manip, '+y' for dl3dv, '+z' for scannetpp, '-y' for tartanair, "
                         "'+y' for dynamic_replica, '-y' for mapfree).")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- Pick a scene & build the dataset adapter ------------------------------
    adapter = build_adapter(args)
    K = adapter.K
    fx, fy = float(K[0, 0]), float(K[1, 1])

    # ---- Select frames ---------------------------------------------------------
    n_total = adapter.num_total
    frame_idxs = adapter.sample_frame_indices(args.num_frames)
    n_pick = len(frame_idxs)
    print(f"[frames] total={n_total}, picking {n_pick}: {frame_idxs}")

    # ---- Load + unproject ------------------------------------------------------
    all_pts, all_cols = [], []
    frustums = []
    for fid in frame_idxs:
        rgb, depth, c2w, rgb_path = adapter.get_frame(fid)
        pts, colors = unproject(rgb, depth, K, c2w, args.max_points_per_frame, rng)
        n_valid_total = int(((depth > 0) & np.isfinite(depth)).sum())
        H, W = depth.shape
        z_pos = depth[depth > 0]
        print(
            f"[frame {fid:>4}]  valid_px={n_valid_total:>6} / {H*W:>6} ({100*n_valid_total/(H*W):.1f}%)"
            f"  used_pts={len(pts):>6}  z=[{(z_pos.min() if z_pos.size else 0):.2f}, "
            f"{(z_pos.max() if z_pos.size else 0):.2f}]"
        )
        all_pts.append(pts)
        all_cols.append(colors)
        # Build a downsampled thumbnail for the frustum tile
        thumb = cv2.resize(rgb, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        frustums.append((fid, c2w, thumb, H, W))

    if not all_pts:
        raise RuntimeError("No frames produced any valid points.")
    points = np.concatenate(all_pts, axis=0)
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    centroid = points.mean(axis=0)
    print(f"\n[cloud] total points: {len(points):,}")
    print(f"        bbox_min  : {bbox_min}")
    print(f"        bbox_max  : {bbox_max}")
    print(f"        centroid  : {centroid}")
    print(f"        size      : {bbox_max - bbox_min}")

    # ---- Launch viser ----------------------------------------------------------
    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    up_axis = args.up_direction or adapter.default_up
    server.scene.set_up_direction(up_axis)
    print(f"[viser] up direction: {up_axis} (dataset default: {adapter.default_up})")

    # Re-center every point and every camera around the global centroid so the
    # scene sits near viser's origin.
    all_points_centered = [p - centroid for p in all_pts]

    # Per-frame point cloud + camera frustum, so the GUI can toggle each independently.
    pc_handles: list = []
    frustum_handles: list = []
    cam_positions: list = []   # (3,) world position (centroid-subtracted) per frame
    cam_forwards: list = []    # (3,) unit "forward" direction in world per frame
    cam_ups: list = []         # (3,) unit "up" direction in world per frame
    fov_y = 2.0 * float(np.arctan(0.5 * frustums[0][3] / fy))  # all frames share K
    aspect = frustums[0][4] / frustums[0][3]
    for i, ((fid, c2w, thumb, H, W), pts_c, cols) in enumerate(
        zip(frustums, all_points_centered, all_cols)
    ):
        pc = server.scene.add_point_cloud(
            f"/scene/points/frame_{i:02d}_id{fid:04d}",
            points=pts_c,
            colors=cols,
            point_size=args.point_size,
            point_shape="circle",
        )
        pc_handles.append(pc)

        R = c2w[:3, :3]
        t = c2w[:3, 3] - centroid
        wxyz = rotmat_to_wxyz(R)
        cam_positions.append(t.astype(np.float32))
        # OpenCV camera convention: +Z is forward, +Y is image-down (so -Y is up).
        cam_forwards.append((R @ np.array([0.0, 0.0, 1.0])).astype(np.float32))
        cam_ups.append((R @ np.array([0.0, -1.0, 0.0])).astype(np.float32))
        fr = server.scene.add_camera_frustum(
            f"/scene/cameras/frame_{i:02d}_id{fid:04d}",
            fov=fov_y,
            aspect=aspect,
            scale=args.frustum_scale,
            color=(255, 153, 25),  # orange (RGB uint8) — "non-current" color
            wxyz=wxyz,
            position=t.astype(np.float32),
            image=thumb,
        )
        frustum_handles.append(fr)

    axes_handle = server.scene.add_frame(
        "/scene/world_axes", show_axes=True, axes_length=0.5, axes_radius=0.01
    )

    # ---------- GUI controls ----------
    N = len(frustums)
    info_md = (
        f"**Dataset** `{args.dataset}`\n\n"
        f"**Scene** `{adapter.scene_label}`\n\n"
        f"**Frames in clip:** {N}    **Total frames:** {n_total}\n\n"
        f"**Total points:** {sum(len(p) for p in all_pts):,}\n\n"
        f"Bbox size: {bbox_max - bbox_min} m\n\n"
        f"World up: `{up_axis}`"
    )
    server.gui.add_markdown(info_md)

    with server.gui.add_folder("Display"):
        mode_dropdown = server.gui.add_dropdown(
            "Mode",
            options=("All frames", "Up to current", "Current only"),
            initial_value="All frames",
            hint="How many of the per-frame point clouds to show",
        )
        current_slider = server.gui.add_slider(
            "Current frame",
            min=0,
            max=max(N - 1, 0),
            step=1,
            initial_value=0,
            hint="Highlighted frame (and frame index used by the Mode dropdown)",
        )
        prev_frame_btn = server.gui.add_button("Previous frame")
        next_frame_btn = server.gui.add_button("Next frame")
        point_size_slider = server.gui.add_slider(
            "Point size (m)",
            min=0.001,
            max=0.10,
            step=0.001,
            initial_value=float(args.point_size),
        )

    with server.gui.add_folder("Cameras"):
        show_frustums_cb = server.gui.add_checkbox(
            "Show frustums",
            initial_value=bool(args.show_frustums),
        )
        frustum_scale_slider = server.gui.add_slider(
            "Frustum scale (m)",
            min=0.01,
            max=1.00,
            step=0.01,
            initial_value=float(args.frustum_scale),
        )
        follow_current_cb = server.gui.add_checkbox(
            "Look at current frame",
            initial_value=False,
            hint="Move viser viewer's target to the current camera position whenever the slider moves.",
        )

    with server.gui.add_folder("Scene"):
        show_axes_cb = server.gui.add_checkbox("Show world axes", initial_value=True)

    HIGHLIGHT_COLOR = (50, 220, 90)   # green for the "current" frustum
    NORMAL_COLOR    = (255, 153, 25)  # orange for the rest

    def apply_visibility():
        mode = mode_dropdown.value
        cur = int(current_slider.value)
        show_fr = bool(show_frustums_cb.value)
        for i, (pc, fr) in enumerate(zip(pc_handles, frustum_handles)):
            if mode == "All frames":
                pc.visible = True
                fr.visible = show_fr
            elif mode == "Up to current":
                pc.visible = i <= cur
                fr.visible = show_fr and (i <= cur)
            elif mode == "Current only":
                pc.visible = i == cur
                fr.visible = show_fr and (i == cur)
            # Highlight the current frustum regardless of mode
            fr.color = HIGHLIGHT_COLOR if i == cur else NORMAL_COLOR

    def apply_point_size():
        size = float(point_size_slider.value)
        for pc in pc_handles:
            pc.point_size = size

    def apply_frustum_scale():
        scale = float(frustum_scale_slider.value)
        for fr in frustum_handles:
            fr.scale = scale

    def apply_axes_visibility():
        axes_handle.visible = bool(show_axes_cb.value)

    def maybe_follow_current(_unused=None):
        if not follow_current_cb.value:
            return
        cur = int(current_slider.value)
        target = cam_positions[cur]
        for client in server.get_clients().values():
            client.camera.look_at = tuple(float(v) for v in target)

    def set_current_frame(i: int):
        current_slider.value = int(np.clip(i, 0, max(N - 1, 0)))
        apply_visibility()
        maybe_follow_current()

    @mode_dropdown.on_update
    def _(_event):
        apply_visibility()

    @current_slider.on_update
    def _(_event):
        apply_visibility()
        maybe_follow_current()

    @prev_frame_btn.on_click
    def _(_event):
        set_current_frame(int(current_slider.value) - 1)

    @next_frame_btn.on_click
    def _(_event):
        set_current_frame(int(current_slider.value) + 1)

    @point_size_slider.on_update
    def _(_event):
        apply_point_size()

    @show_frustums_cb.on_update
    def _(_event):
        apply_visibility()

    @frustum_scale_slider.on_update
    def _(_event):
        apply_frustum_scale()

    @show_axes_cb.on_update
    def _(_event):
        apply_axes_visibility()

    @follow_current_cb.on_update
    def _(_event):
        maybe_follow_current()

    # ---------- Click-to-teleport on camera frustums ----------
    def _teleport_to(i: int, client):
        pos    = cam_positions[i]
        fwd    = cam_forwards[i]
        up_vec = cam_ups[i]
        target = pos + fwd * 2.0    # look 2 m ahead of the camera
        client.camera.position = tuple(float(v) for v in pos)
        client.camera.look_at  = tuple(float(v) for v in target)
        client.camera.up_direction = tuple(float(v) for v in up_vec)

    def _make_click_handler(i: int):
        def _handler(event):
            set_current_frame(i)
            _teleport_to(i, event.client)
        return _handler

    for i, fr in enumerate(frustum_handles):
        fr.on_click(_make_click_handler(i))

    apply_visibility()
    apply_point_size()
    apply_frustum_scale()
    apply_axes_visibility()

    print(f"\n[viser] http://localhost:{args.port}")
    print("        (if remote: ssh -L {p}:localhost:{p} user@host)".format(p=args.port))
    print("        GUI controls: Mode dropdown, Current frame slider, Point size, Frustum scale.")
    print("        Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[viser] shutting down")


if __name__ == "__main__":
    main()
