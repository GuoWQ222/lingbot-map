#!/usr/bin/env python3
"""Visualize one Manip_long6 surround_cam_moving trajectory sampled by the training DataLoader."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parents[1]
for _path in (_REPO_DIR, _REPO_DIR / "scripts" / "train", _REPO_DIR / "scripts" / "eval", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TypeVar

import numpy as np
import torch
import matplotlib.cm as cm
import viser
import viser.transforms as tf
from torch.utils.data import DataLoader, Subset

T = TypeVar("T")

from train import (
    ManipTrajectoryDataset,
    canonicalize_to_first_frame,
    collate_rgbd_sequences,
    discover_trajectory_dirs,
    normalize_scene_batch,
    split_scenes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one Manip_long6 surround_cam_moving training trajectory and visualize "
            "its RGB-D point cloud."
        )
    )
    parser.add_argument(
        "--root",
        default="/oss-guowenqi/Manip_long6/data",
        help="Manip_long6 RGB-D export root. Defaults to DATA_ROOT_LONG6 in train.sh.",
    )
    parser.add_argument(
        "--oss-uri-root",
        default="",
        help="Optional OSS URI root for trajectory discovery, matching train.py's --oss_uri_roots.",
    )
    parser.add_argument("--ossutil-bin", default="/cpfs/user/guowenqi/ossutil/ossutil")
    parser.add_argument("--ossutil-config", default="/cpfs/user/guowenqi/ossutil/.ossutilconfig")
    parser.add_argument("--scene-manifest", default="/cpfs/user/guowenqi/lingbot-map-copy/outputs/runs/manip_long_train/manip_long6_trajectory_manifest.txt", help="Optional training manifest path.")
    parser.add_argument("--write-manifest", default="", help="Optional path to write discovered scenes.")
    parser.add_argument("--scene", default="", help="Trajectory directory name to sample. Overrides --index.")
    parser.add_argument("--index", type=int, default=0, help="Training split sample index.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for trajectory sampling.")
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means use all discovered scenes.")
    parser.add_argument("--val-fraction", type=float, default=0.02)

    parser.add_argument("--clip-len", type=int, default=64)
    parser.add_argument("--max-sample-frames", type=int, default=64)
    parser.add_argument("--min-sample-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=280)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=0.0)
    parser.add_argument("--depth-scale", type=float, default=0.0)
    parser.add_argument(
        "--use-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use exported masks if present. Default is false because Manip_long6 surround_cam_moving exports do not include masks.",
    )

    parser.add_argument(
        "--camera-names",
        default="surround_cam_moving",
        help="Comma-separated Manip_long6 camera dirs to allow.",
    )
    parser.add_argument("--moving-stride-min", type=int, default=2)
    parser.add_argument("--moving-stride-max", type=int, default=6)
    parser.add_argument("--wrist-camera-prefix", default="realsense")
    parser.add_argument("--static-camera-prefix", default="surround")
    parser.add_argument("--moving-camera-prefix", default="surround_cam_moving")
    parser.add_argument("--fixed-camera-prefix", default="surround_cam_fixed")
    parser.add_argument("--long6-root-marker", default="Manip_long6")
    parser.add_argument("--color-jitter-strength", type=float, default=0.2)
    parser.add_argument("--color-jitter-prob", type=float, default=0.5)

    parser.add_argument(
        "--training-transform",
        action="store_true",
        help="Apply train.sh's first-frame canonicalization and anchor-scale normalization.",
    )
    parser.add_argument(
        "--uniform-sample",
        action="store_true",
        help=(
            "Sample the selected camera uniformly over the full trajectory, matching "
            "the checkpoint prediction visualizer's frame selection."
        ),
    )
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--max-points-per-frame", type=int, default=0)
    parser.add_argument("--point-size", type=float, default=0.006)
    parser.add_argument("--camera-scale", type=float, default=0.08)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--recenter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center points and cameras around the valid point-cloud centroid.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and summarize the DataLoader batch without starting viser.",
    )
    return parser.parse_args()


def _parse_camera_names(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _filter_long6_scenes(scenes: List[Path], args: argparse.Namespace) -> List[Path]:
    marker = args.long6_root_marker.lower()
    root_path = Path(args.root).expanduser()
    root_text = str(root_path)
    filtered: List[Path] = []
    for scene in scenes:
        scene_text = str(scene)
        if marker and marker not in scene_text.lower():
            continue
        if root_text and not scene_text.startswith(root_text):
            continue
        filtered.append(scene)
    return filtered


def uniform_sample_entries(entries: Sequence[T], target_count: int) -> List[T]:
    if target_count <= 0 or len(entries) <= target_count:
        return list(entries)
    indices = np.rint(np.linspace(0, len(entries) - 1, target_count)).astype(np.int64)
    unique_indices = np.unique(indices)
    if len(unique_indices) != target_count:
        unique_indices = np.linspace(0, len(entries) - 1, target_count, dtype=np.int64)
    return [entries[int(idx)] for idx in unique_indices.tolist()]


def _select_sample_index(scenes: Sequence[Path], args: argparse.Namespace) -> int:
    if args.scene:
        scene_names = [scene.name for scene in scenes]
        scene_paths = [str(scene) for scene in scenes]
        if args.scene in scene_names:
            return scene_names.index(args.scene)
        if args.scene in scene_paths:
            return scene_paths.index(args.scene)
        raise ValueError(
            f"Scene {args.scene!r} is not in the Manip_long6 training split "
            f"({len(scenes)} scenes)."
        )
    return args.index % len(scenes)


def _discover_train_scenes(args: argparse.Namespace) -> List[Path]:
    oss_roots = [args.oss_uri_root] if args.oss_uri_root else []
    scenes = discover_trajectory_dirs(
        [args.root],
        max_scenes=0,
        manifest=args.scene_manifest or None,
        write_manifest=args.write_manifest or None,
        oss_uri_roots=oss_roots or None,
        ossutil_bin=args.ossutil_bin,
        ossutil_config=args.ossutil_config,
    )
    scenes = _filter_long6_scenes(scenes, args)
    if not scenes and args.scene_manifest:
        print("[data][warn] manifest had no matching Manip_long6 scenes; scanning --root instead")
        scenes = discover_trajectory_dirs(
            [args.root],
            max_scenes=0,
            manifest=None,
            write_manifest=args.write_manifest or None,
            oss_uri_roots=oss_roots or None,
            ossutil_bin=args.ossutil_bin,
            ossutil_config=args.ossutil_config,
        )
        scenes = _filter_long6_scenes(scenes, args)
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]
    if not scenes:
        raise RuntimeError(f"No Manip_long6 trajectories were discovered under {args.root}")
    train_scenes, _ = split_scenes(scenes, args.val_fraction, args.seed)
    if not train_scenes:
        raise RuntimeError("Training split is empty after val split")
    return train_scenes


def make_dataloader(args: argparse.Namespace) -> Tuple[DataLoader, ManipTrajectoryDataset, int, List[Path]]:
    train_scenes = _discover_train_scenes(args)
    camera_names = _parse_camera_names(args.camera_names)
    if not camera_names:
        raise ValueError("--camera-names must contain at least one camera")

    dataset = ManipTrajectoryDataset(
        train_scenes,
        clip_len=args.clip_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        sequence_mode="manip_4d_mixed",
        view_ids=None,
        camera_names=camera_names,
        sample_strategy="random_interval",
        frame_stride=1,
        random_stride_min=2,
        random_stride_max=8,
        w_stride_min=2,
        w_stride_max=8,
        random_interval_start="first",
        max_sample_frames=args.max_sample_frames,
        min_sample_frames=args.min_sample_frames,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_mask=args.use_mask,
        invert_cam_extrinsics=False,
        samples_per_scene=1,
        wrist_camera_prefix=args.wrist_camera_prefix,
        static_camera_prefix=args.static_camera_prefix,
        moving_stride_min=args.moving_stride_min,
        moving_stride_max=args.moving_stride_max,
        fixed_stride_min=4,
        fixed_stride_max=16,
        long6_root_marker=args.long6_root_marker,
        long6_mode_weights={"W": 0.0, "T": 1.0, "F": 0.0, "M": 0.0},
        moving_camera_prefix=args.moving_camera_prefix,
        fixed_camera_prefix=args.fixed_camera_prefix,
        color_jitter_strength=args.color_jitter_strength,
        color_jitter_prob=args.color_jitter_prob,
    )

    sample_index = _select_sample_index(train_scenes, args)

    loader = DataLoader(
        Subset(dataset, [sample_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_rgbd_sequences,
    )
    return loader, dataset, sample_index, train_scenes


def prepare_uniform_batch(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_scenes = _discover_train_scenes(args)
    camera_names = _parse_camera_names(args.camera_names)
    if not camera_names:
        raise ValueError("--camera-names must contain at least one camera")
    camera_name = camera_names[0]
    sample_index = _select_sample_index(train_scenes, args)
    scene_dir = train_scenes[sample_index]

    dataset = ManipTrajectoryDataset(
        [scene_dir],
        clip_len=args.clip_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        sequence_mode="manip_4d_mixed",
        view_ids=None,
        camera_names=[camera_name],
        sample_strategy="random_interval",
        frame_stride=1,
        random_stride_min=2,
        random_stride_max=8,
        w_stride_min=2,
        w_stride_max=8,
        random_interval_start="first",
        max_sample_frames=args.max_sample_frames,
        min_sample_frames=args.min_sample_frames,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_mask=args.use_mask,
        invert_cam_extrinsics=False,
        samples_per_scene=1,
        wrist_camera_prefix=args.wrist_camera_prefix,
        static_camera_prefix=args.static_camera_prefix,
        moving_stride_min=args.moving_stride_min,
        moving_stride_max=args.moving_stride_max,
        fixed_stride_min=4,
        fixed_stride_max=16,
        long6_root_marker=args.long6_root_marker,
        long6_mode_weights={"W": 0.0, "T": 1.0, "F": 0.0, "M": 0.0},
        moving_camera_prefix=args.moving_camera_prefix,
        fixed_camera_prefix=args.fixed_camera_prefix,
        color_jitter_strength=args.color_jitter_strength,
        color_jitter_prob=0.0,
    )
    entries = dataset._entries_for_scene(scene_dir)
    camera_entries = [entry for entry in entries if entry.camera_name == camera_name]
    camera_entries.sort(key=lambda entry: entry.frame_id)
    if not camera_entries:
        raise RuntimeError(f"No entries found for camera {camera_name!r} in {scene_dir}")

    target_count = args.max_sample_frames if args.max_sample_frames > 0 else args.clip_len
    selected = uniform_sample_entries(camera_entries, target_count)
    loaded = [dataset._load_one(scene_dir, entry, jitter_params=None) for entry in selected]
    images, depths, masks, intrinsics, extrinsics, world_points = zip(*loaded)
    sample = {
        "images": torch.stack(list(images), dim=0),
        "depths": torch.stack(list(depths), dim=0),
        "point_masks": torch.stack(list(masks), dim=0),
        "intrinsics": torch.stack(list(intrinsics), dim=0),
        "extrinsics": torch.stack(list(extrinsics), dim=0),
        "world_points": torch.stack(list(world_points), dim=0),
        "frame_ids": torch.tensor([entry.frame_id for entry in selected], dtype=torch.long),
        "view_ids": torch.tensor([entry.view_id for entry in selected], dtype=torch.long),
        "scene": scene_dir.name,
        "sample_mode": f"{camera_name}_uniform_{len(selected)}",
    }
    batch = collate_rgbd_sequences([sample])
    if args.training_transform:
        batch = canonicalize_to_first_frame(batch)
        batch = normalize_scene_batch(batch, num_anchor_frames=min(args.num_scale_frames, len(selected)))

    length = int(batch["sequence_lengths"][0])
    valid_points = int(batch["point_masks"][0, :length].sum())
    suffix = " + canonicalize/normalize" if args.training_transform else ""
    print(
        f"[sample] uniform dataset_index={sample_index} scene={scene_dir.name} frames={len(selected)} "
        f"total_camera_frames={len(camera_entries)} valid_points={valid_points:,}{suffix}"
    )
    print(f"[sample] scene_dir={scene_dir}")
    print(f"[sample] camera_names={[camera_name]} use_mask={args.use_mask}")
    print(f"[sample] frame_ids={sample['frame_ids'].tolist()}")
    print(f"[sample] view_ids={sample['view_ids'].tolist()}")
    return batch


def prepare_batch(args: argparse.Namespace) -> Dict[str, object]:
    if args.uniform_sample:
        return prepare_uniform_batch(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loader, _, sample_index, train_scenes = make_dataloader(args)
    batch = next(iter(loader))
    if args.training_transform:
        batch = canonicalize_to_first_frame(batch)
        batch = normalize_scene_batch(batch, num_anchor_frames=args.num_scale_frames)

    length = int(batch["sequence_lengths"][0])
    scene = batch["scene"][0]
    frame_ids = batch["frame_ids"][0, :length].tolist()
    view_ids = batch["view_ids"][0, :length].tolist()
    valid_points = int(batch["point_masks"][0, :length].sum())
    suffix = " + canonicalize/normalize" if args.training_transform else ""
    scene_dir = train_scenes[sample_index % len(train_scenes)]
    print(
        f"[sample] dataset_index={sample_index} scene={scene} frames={length} "
        f"valid_points={valid_points:,}{suffix}"
    )
    print(f"[sample] scene_dir={scene_dir}")
    print(f"[sample] camera_names={_parse_camera_names(args.camera_names)} use_mask={args.use_mask}")
    print(f"[sample] frame_ids={frame_ids}")
    print(f"[sample] view_ids={view_ids}")
    return batch


def invert_w2c(extrinsics: np.ndarray) -> np.ndarray:
    count = extrinsics.shape[0]
    w2c = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
    w2c[:, :3, :4] = extrinsics
    return np.linalg.inv(w2c)


def extract_frame_clouds(
    batch: Dict[str, object],
    max_points_per_frame: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[int]]:
    length = int(batch["sequence_lengths"][0])
    images = batch["images"][0, :length].cpu().numpy().transpose(0, 2, 3, 1)
    points = batch["world_points"][0, :length].cpu().numpy()
    masks = batch["point_masks"][0, :length].cpu().numpy().astype(bool)
    intrinsics = batch["intrinsics"][0, :length].cpu().numpy()
    extrinsics = batch["extrinsics"][0, :length].cpu().numpy()
    frame_ids = batch["frame_ids"][0, :length].cpu().numpy().astype(int).tolist()
    c2w = invert_w2c(extrinsics)

    rng = np.random.default_rng(seed)
    frame_points: List[np.ndarray] = []
    frame_colors: List[np.ndarray] = []
    for frame_idx in range(length):
        valid = masks[frame_idx] & np.isfinite(points[frame_idx]).all(axis=-1)
        valid_points = points[frame_idx][valid].astype(np.float32)
        valid_colors = np.clip(images[frame_idx][valid] * 255.0, 0, 255).astype(np.uint8)
        if max_points_per_frame > 0 and len(valid_points) > max_points_per_frame:
            selected = rng.choice(len(valid_points), max_points_per_frame, replace=False)
            valid_points = valid_points[selected]
            valid_colors = valid_colors[selected]
        frame_points.append(valid_points)
        frame_colors.append(valid_colors)
    return frame_points, frame_colors, c2w, intrinsics, images, frame_ids


def visible_frame_indices(mode: str, current_frame: int, frame_count: int) -> List[int]:
    if frame_count <= 0:
        return []
    frame = max(0, min(int(current_frame), frame_count - 1))
    if mode == "Current frame":
        return [frame]
    if mode == "0 to current frame":
        return list(range(frame + 1))
    return list(range(frame_count))


def visible_camera_flags(
    mode: str, current_frame: int, frame_count: int, show_cameras: bool
) -> List[bool]:
    if not show_cameras:
        return [False] * max(frame_count, 0)
    visible = set(visible_frame_indices(mode, current_frame, frame_count))
    return [idx in visible for idx in range(max(frame_count, 0))]


def set_camera_handle_scale(handles: Sequence[object], scale: float) -> None:
    for handle in handles:
        setattr(handle, "scale", float(scale))

_POSE_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _float_tuple(values: object, size: int, label: str) -> Tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != size:
        raise ValueError(f"{label} must contain {size} numbers")
    return tuple(float(value) for value in values)


def format_camera_pose(position: Sequence[float], wxyz: Sequence[float]) -> str:
    values = [float(value) for value in position] + [float(value) for value in wxyz]
    if len(values) != 7:
        raise ValueError("camera pose must contain x y z qw qx qy qz")
    return " ".join(f"{value:.6f}" for value in values)


def parse_camera_pose(text: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("pose input is empty")

    if stripped.startswith("{"):
        data = json.loads(stripped)
        position = _float_tuple(data.get("position"), 3, "position")
        wxyz = _float_tuple(data.get("wxyz"), 4, "wxyz")
    else:
        numbers = [float(match.group(0)) for match in _POSE_NUMBER_PATTERN.finditer(stripped)]
        if len(numbers) != 7:
            raise ValueError("expected JSON pose or 7 numbers: x y z qw qx qy qz")
        position = tuple(numbers[:3])
        wxyz = tuple(numbers[3:])

    norm = float(np.linalg.norm(np.asarray(wxyz, dtype=np.float64)))
    if norm <= 1e-12:
        raise ValueError("wxyz quaternion must be non-zero")
    wxyz = tuple(float(value) / norm for value in wxyz)
    return position, wxyz


def set_client_camera_pose(client: object, pose_text: str) -> str:
    position, wxyz = parse_camera_pose(pose_text)
    client.camera.position = np.asarray(position, dtype=np.float64)
    client.camera.wxyz = np.asarray(wxyz, dtype=np.float64)
    return format_camera_pose(position, wxyz)


def install_view_pose_controls(server: viser.ViserServer) -> None:
    latest_client: Dict[str, object] = {"client": None}
    with server.gui.add_folder("View pose"):
        gui_current_pose = server.gui.add_text(
            "Current view pose",
            format_camera_pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            multiline=True,
            disabled=True,
        )
        gui_target_pose = server.gui.add_text(
            "Target view pose",
            "",
            multiline=True,
            hint='Paste x y z qw qx qy qz, or JSON with position/wxyz.',
        )
        gui_jump_pose = server.gui.add_button("Jump to pose")
        gui_pose_status = server.gui.add_text("Pose status", "Waiting for a viewer client", disabled=True)

    def _update_current_pose(camera: viser.CameraHandle) -> None:
        latest_client["client"] = camera.client
        pose_text = format_camera_pose(camera.position, camera.wxyz)
        gui_current_pose.value = pose_text
        if not gui_target_pose.value.strip():
            gui_target_pose.value = pose_text
        gui_pose_status.value = "Ready"

    @server.on_client_connect
    def _on_client_connect(client: viser.ClientHandle) -> None:
        latest_client["client"] = client

        @client.camera.on_update
        def _on_camera_update(camera: viser.CameraHandle) -> None:
            _update_current_pose(camera)

    @gui_jump_pose.on_click
    def _jump_to_pose(event) -> None:
        client = event.client or latest_client["client"]
        if client is None:
            gui_pose_status.value = "No viewer client connected"
            return
        try:
            pose_text = set_client_camera_pose(client, gui_target_pose.value)
        except Exception as exc:  # noqa: BLE001 - show parse errors directly in the GUI.
            gui_pose_status.value = f"Invalid pose: {exc}"
            return
        gui_current_pose.value = pose_text
        gui_target_pose.value = pose_text
        gui_pose_status.value = "Jumped to pose"


def run_viser(args: argparse.Namespace, batch: Dict[str, object]) -> None:
    frame_points, frame_colors, c2w, intrinsics, images, frame_ids = extract_frame_clouds(
        batch, args.max_points_per_frame, args.seed
    )
    nonempty = [points for points in frame_points if len(points)]
    if not nonempty:
        raise RuntimeError("The sampled trajectory has no valid depth points.")

    centroid = np.median(np.concatenate(nonempty, axis=0), axis=0) if args.recenter else np.zeros(3)
    frame_points = [points - centroid for points in frame_points]
    c2w[:, :3, 3] -= centroid

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    scene_name = str(batch["scene"][0])
    server.scene.add_grid("/ground", width=10.0, height=10.0)

    visible_modes = ["All frames", "Current frame", "0 to current frame"]
    with server.gui.add_folder("Manip_long6 surround_cam_moving trajectory"):
        gui_visible_points = server.gui.add_dropdown(
            "Visible points", options=visible_modes, initial_value="All frames"
        )
        gui_frame_index = server.gui.add_slider(
            "Current frame index",
            min=0,
            max=len(frame_points) - 1,
            step=1,
            initial_value=0,
        )
        gui_prev_frame = server.gui.add_button("Prev frame")
        gui_next_frame = server.gui.add_button("Next frame")
        gui_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.03, step=0.001, initial_value=args.point_size
        )
        gui_camera_size = server.gui.add_slider(
            "Camera size", min=0.005, max=0.3, step=0.005, initial_value=args.camera_scale
        )

    install_view_pose_controls(server)

    cloud = server.scene.add_point_cloud(
        "/trajectory/points",
        points=np.concatenate(frame_points, axis=0),
        colors=np.concatenate(frame_colors, axis=0),
        point_size=args.point_size,
        point_shape="circle",
    )

    camera_handles = []
    for frame_idx, (pose, K, image, frame_id) in enumerate(
        zip(c2w, intrinsics, images, frame_ids)
    ):
        rotation = tf.SO3.from_matrix(pose[:3, :3]).wxyz
        height, width = image.shape[:2]
        fov = float(2.0 * np.arctan2(height / 2.0, K[1, 1]))
        norm_idx = frame_idx / (len(c2w) - 1) if len(c2w) > 1 else 0.0
        color = cm.get_cmap("gist_rainbow")(norm_idx)[:3]
        handle = server.scene.add_camera_frustum(
            f"/trajectory/cameras/{frame_idx:03d}_{frame_id}",
            fov=fov,
            aspect=width / height,
            scale=args.camera_scale,
            wxyz=rotation,
            position=pose[:3, 3],
            color=color,
            line_width=2.0,
        )

        @handle.on_click
        def _focus(event, pose=pose) -> None:
            center = pose[:3, 3]
            forward = pose[:3, 2]
            up = -pose[:3, 1]
            event.client.camera.position = tuple(center - 0.15 * forward)
            event.client.camera.look_at = tuple(center + forward)
            event.client.camera.up_direction = tuple(up)

        camera_handles.append(handle)

    def _set_cloud_for_visible_points() -> None:
        indices = visible_frame_indices(
            str(gui_visible_points.value), int(gui_frame_index.value), len(frame_points)
        )
        if not indices:
            return
        cloud.points = np.concatenate([frame_points[idx] for idx in indices], axis=0)
        cloud.colors = np.concatenate([frame_colors[idx] for idx in indices], axis=0)

    def _set_cameras_for_visible_points() -> None:
        flags = visible_camera_flags(
            str(gui_visible_points.value),
            int(gui_frame_index.value),
            len(camera_handles),
            bool(gui_cameras.value),
        )
        for handle, visible in zip(camera_handles, flags):
            handle.visible = visible

    def _set_visible_frames() -> None:
        _set_cloud_for_visible_points()
        _set_cameras_for_visible_points()

    @gui_visible_points.on_update
    def _update_cloud(_) -> None:
        _set_visible_frames()

    @gui_frame_index.on_update
    def _update_frame_index(_) -> None:
        _set_visible_frames()

    @gui_prev_frame.on_click
    def _prev_frame(_) -> None:
        gui_frame_index.value = max(0, int(gui_frame_index.value) - 1)

    @gui_next_frame.on_click
    def _next_frame(_) -> None:
        gui_frame_index.value = min(len(frame_points) - 1, int(gui_frame_index.value) + 1)

    @gui_cameras.on_update
    def _update_cameras(_) -> None:
        _set_cameras_for_visible_points()

    @gui_point_size.on_update
    def _update_point_size(_) -> None:
        cloud.point_size = gui_point_size.value

    @gui_camera_size.on_update
    def _update_camera_size(_) -> None:
        set_camera_handle_scale(camera_handles, gui_camera_size.value)

    shown_points = sum(len(points) for points in frame_points)
    print(f"[viser] scene={scene_name} displayed_points={shown_points:,}")
    print(f"[viser] open http://localhost:{args.port}")
    while True:
        time.sleep(0.1)


def main() -> None:
    args = parse_args()
    batch = prepare_batch(args)
    if not args.dry_run:
        run_viser(args, batch)


if __name__ == "__main__":
    main()
