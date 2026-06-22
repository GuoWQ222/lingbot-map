#!/usr/bin/env python3
"""Visualize one CO3D trajectory sampled by the training DataLoader."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parents[1]
for _path in (_REPO_DIR, _REPO_DIR / "scripts" / "train", _REPO_DIR / "scripts" / "eval", _SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import viser
import viser.transforms as tf
from torch.utils.data import DataLoader, Subset

from lingbot_map.data.co3d import Co3dTrajectoryDataset
from train import canonicalize_to_first_frame, collate_rgbd_sequences, normalize_scene_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample one CO3D training trajectory and visualize its RGB-D point cloud."
    )
    parser.add_argument(
        "--root",
        default="/oss-guowenqi/CO3Dv2",
        help="CO3D root. Defaults to CO3D_ROOT in train.sh.",
    )
    parser.add_argument("--scene", default="", help="Scene ID to sample. Overrides --index.")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for trajectory sampling.")
    parser.add_argument("--num-views", type=int, default=64)
    parser.add_argument("--min-views", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=280)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth", type=float, default=0.0)
    parser.add_argument("--mask-bg", choices=("true", "false", "rand"), default="rand")
    parser.add_argument(
        "--base3d-root",
        default="/cpfs/user/guowenqi/base3d-clean",
        help="Repo root containing datasets/co3d_20251006.py.",
    )
    parser.add_argument(
        "--training-transform",
        action="store_true",
        help="Apply train.sh's first-frame canonicalization and anchor-scale normalization.",
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


def make_dataloader(args: argparse.Namespace) -> Tuple[DataLoader, Co3dTrajectoryDataset, int]:
    # These defaults mirror train.sh and build_dataloaders() in train.py.
    dataset = Co3dTrajectoryDataset(
        root=args.root,
        split="train",
        num_views=args.num_views,
        min_views=args.min_views,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        samples_per_scene=1,
        color_jitter_strength=0.2,
        color_jitter_prob=0.5,
        mask_bg=args.mask_bg,
        base3d_root=args.base3d_root,
        verbose=True,
    )

    scenes = list(dataset.dataset.scenes)
    if args.scene:
        if args.scene not in scenes:
            raise ValueError(
                f"Scene {args.scene!r} is not in the CO3D training dataset "
                f"({len(scenes)} scenes)."
            )
        sample_index = scenes.index(args.scene)
    else:
        sample_index = args.index % len(dataset.dataset)

    loader = DataLoader(
        Subset(dataset, [sample_index]),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_rgbd_sequences,
    )
    return loader, dataset, sample_index


def prepare_batch(args: argparse.Namespace) -> Dict[str, object]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    loader, _, sample_index = make_dataloader(args)
    batch = next(iter(loader))
    if args.training_transform:
        batch = canonicalize_to_first_frame(batch)
        batch = normalize_scene_batch(batch, num_anchor_frames=args.num_scale_frames)

    length = int(batch["sequence_lengths"][0])
    scene = batch["scene"][0]
    frame_ids = batch["frame_ids"][0, :length].tolist()
    valid_points = int(batch["point_masks"][0, :length].sum())
    suffix = " + canonicalize/normalize" if args.training_transform else ""
    print(
        f"[sample] dataset_index={sample_index} scene={scene} frames={length} "
        f"valid_points={valid_points:,}{suffix}"
    )
    print(f"[sample] frame_ids={frame_ids}")
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

    options = ["All"] + [f"{idx}: frame {frame_id}" for idx, frame_id in enumerate(frame_ids)]
    with server.gui.add_folder("CO3D trajectory"):
        gui_frame = server.gui.add_dropdown("Visible points", options=options, initial_value="All")
        gui_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.03, step=0.001, initial_value=args.point_size
        )

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
        image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        handle = server.scene.add_camera_frustum(
            f"/trajectory/cameras/{frame_idx:03d}_{frame_id}",
            fov=fov,
            aspect=width / height,
            scale=args.camera_scale,
            wxyz=rotation,
            position=pose[:3, 3],
            image=image_u8,
            line_width=1.0,
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

    @gui_frame.on_update
    def _update_cloud(_) -> None:
        if gui_frame.value == "All":
            indices = range(len(frame_points))
        else:
            indices = [int(gui_frame.value.split(":", 1)[0])]
        cloud.points = np.concatenate([frame_points[idx] for idx in indices], axis=0)
        cloud.colors = np.concatenate([frame_colors[idx] for idx in indices], axis=0)

    @gui_cameras.on_update
    def _update_cameras(_) -> None:
        for handle in camera_handles:
            handle.visible = gui_cameras.value

    @gui_point_size.on_update
    def _update_point_size(_) -> None:
        cloud.point_size = gui_point_size.value

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
