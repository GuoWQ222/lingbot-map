#!/usr/bin/env python3
"""Visualize one eval_vggt data batch as a GT RGB-D point cloud with Viser."""

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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import viser
import viser.transforms as tf

import eval as E
import eval_vggt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the same eval batch used by eval_vggt.py and visualize the GT "
            "point cloud from depth back-projection."
        )
    )
    parser.add_argument(
        "--train-args-json",
        default="/cpfs/user/guowenqi/lingbot-map-copy/outputs/runs/"
        "manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/args.json",
    )
    parser.add_argument("--output-dir", default="/tmp/eval_vggt_gt_batch_vis")
    parser.add_argument("--vggt-repo", default="/cpfs/user/guowenqi/vggt")
    parser.add_argument("--model-weights", default="/cpfs/user/guowenqi/vggt/model.pt")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--max-scenes-eval", type=int, default=1)
    parser.add_argument("--eval-mode", choices=["left_moving_tracks", "wrist_track", "manip_track", "random_static_track"], default="left_moving_tracks")
    parser.add_argument("--eval-num-frames", type=int, default=64)
    parser.add_argument("--eval-wrist-camera-name", default="realsense_left")
    parser.add_argument("--eval-surround-camera-name", default="surround_cam_moving")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--vggt-preprocess-mode", choices=["crop", "pad"], default="crop")
    parser.add_argument("--batch-index", type=int, default=0, help="0=realsense_left, 1=surround_cam_moving for left_moving_tracks.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-points-per-frame", type=int, default=30000)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--camera-scale", type=float, default=0.08)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--recenter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    eval_parser = eval_vggt.build_argparser()
    eval_args = eval_parser.parse_args(
        [
            "--train_args_json",
            args.train_args_json,
            "--output_dir",
            args.output_dir,
            "--vggt_repo",
            args.vggt_repo,
            "--model_weights",
            args.model_weights,
            "--split",
            args.split,
            "--max_scenes_eval",
            str(args.max_scenes_eval),
            "--num_workers",
            str(args.num_workers),
            "--device",
            "cpu",
            "--eval_strategy",
            args.eval_mode,
            "--eval_num_frames",
            str(args.eval_num_frames),
            "--eval_wrist_camera_name",
            args.eval_wrist_camera_name,
            "--eval_surround_camera_name",
            args.eval_surround_camera_name,
            "--eval_seed",
            str(args.eval_seed),
            "--image_size",
            str(args.image_size),
            "--vggt_preprocess_mode",
            args.vggt_preprocess_mode,
        ]
    )
    return eval_vggt.args_from_run_json(eval_args)


def load_batch(args: argparse.Namespace) -> Dict[str, object]:
    eval_args = build_eval_args(args)
    loader, scenes = E.build_eval_loader(eval_args, eval_mode=args.eval_mode)
    if len(loader) <= 0:
        raise RuntimeError("eval_vggt loader produced no batches")
    target = args.batch_index % len(loader)
    for batch_idx, batch in enumerate(loader):
        if batch_idx == target:
            scene = batch["scene"][0] if isinstance(batch["scene"], list) else str(batch["scene"])
            sample_mode = batch["sample_mode"][0] if isinstance(batch["sample_mode"], list) else str(batch["sample_mode"])
            print(
                f"[batch] split={args.split} scenes={len(scenes)} batches={len(loader)} "
                f"batch_index={target} scene={scene} sample_mode={sample_mode}"
            )
            print(
                f"[batch] input_preprocess={eval_args.input_preprocess} "
                f"image_size={eval_args.image_size} patch_size={eval_args.patch_size} "
                f"preprocess_mode={eval_args.preprocess_mode}"
            )
            print(
                f"[batch] images={tuple(batch['images'].shape)} depths={tuple(batch['depths'].shape)} "
                f"world_points={tuple(batch['world_points'].shape)}"
            )
            return batch
    raise RuntimeError(f"batch_index={target} not found")


def invert_w2c(extrinsics: np.ndarray) -> np.ndarray:
    count = extrinsics.shape[0]
    w2c = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
    w2c[:, :3, :4] = extrinsics
    return np.linalg.inv(w2c)


def frame_clouds(
    batch: Dict[str, object],
    max_points_per_frame: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[int]]:
    images_t = batch["images"][0]
    length = int(images_t.shape[0])
    images = images_t.cpu().numpy().transpose(0, 2, 3, 1)
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


def visible_indices(mode: str, current: int, count: int) -> List[int]:
    current = max(0, min(int(current), max(count - 1, 0)))
    if mode == "Current frame":
        return [current]
    if mode == "0 to current frame":
        return list(range(current + 1))
    return list(range(count))


def run_viser(args: argparse.Namespace, batch: Dict[str, object]) -> None:
    frame_points, frame_colors, c2w, intrinsics, images, frame_ids = frame_clouds(
        batch, args.max_points_per_frame, args.eval_seed
    )
    nonempty = [pts for pts in frame_points if len(pts)]
    if not nonempty:
        raise RuntimeError("The selected eval batch has no valid GT depth points")

    centroid = np.median(np.concatenate(nonempty, axis=0), axis=0) if args.recenter else np.zeros(3)
    frame_points = [pts - centroid for pts in frame_points]
    c2w[:, :3, 3] -= centroid

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    scene = batch["scene"][0] if isinstance(batch["scene"], list) else str(batch["scene"])
    sample_mode = batch["sample_mode"][0] if isinstance(batch["sample_mode"], list) else str(batch["sample_mode"])
    server.scene.add_grid("/ground", width=10.0, height=10.0)

    modes = ["All frames", "Current frame", "0 to current frame"]
    with server.gui.add_folder("eval_vggt GT batch"):
        gui_visible = server.gui.add_dropdown("Visible points", options=modes, initial_value="All frames")
        gui_frame = server.gui.add_slider("Current frame index", min=0, max=len(frame_points) - 1, step=1, initial_value=0)
        gui_prev = server.gui.add_button("Prev frame")
        gui_next = server.gui.add_button("Next frame")
        gui_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
        gui_point_size = server.gui.add_slider("Point size", min=0.001, max=0.03, step=0.001, initial_value=args.point_size)

    cloud = server.scene.add_point_cloud(
        "/gt/points",
        points=np.concatenate(frame_points, axis=0),
        colors=np.concatenate(frame_colors, axis=0),
        point_size=args.point_size,
        point_shape="circle",
    )

    camera_handles = []
    for frame_idx, (pose, K, image, frame_id) in enumerate(zip(c2w, intrinsics, images, frame_ids)):
        rotation = tf.SO3.from_matrix(pose[:3, :3]).wxyz
        height, width = image.shape[:2]
        fov = float(2.0 * np.arctan2(height / 2.0, K[1, 1]))
        image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        handle = server.scene.add_camera_frustum(
            f"/gt/cameras/{frame_idx:03d}_{frame_id}",
            fov=fov,
            aspect=width / height,
            scale=args.camera_scale,
            wxyz=rotation,
            position=pose[:3, 3],
            image=image_u8,
            line_width=1.0,
        )
        camera_handles.append(handle)

    def update_scene() -> None:
        indices = visible_indices(str(gui_visible.value), int(gui_frame.value), len(frame_points))
        cloud.points = np.concatenate([frame_points[idx] for idx in indices], axis=0)
        cloud.colors = np.concatenate([frame_colors[idx] for idx in indices], axis=0)
        visible = set(indices)
        for idx, handle in enumerate(camera_handles):
            handle.visible = bool(gui_cameras.value) and idx in visible

    @gui_visible.on_update
    def _(_) -> None:
        update_scene()

    @gui_frame.on_update
    def _(_) -> None:
        update_scene()

    @gui_cameras.on_update
    def _(_) -> None:
        update_scene()

    @gui_point_size.on_update
    def _(_) -> None:
        cloud.point_size = gui_point_size.value

    @gui_prev.on_click
    def _(_) -> None:
        gui_frame.value = max(0, int(gui_frame.value) - 1)

    @gui_next.on_click
    def _(_) -> None:
        gui_frame.value = min(len(frame_points) - 1, int(gui_frame.value) + 1)

    update_scene()
    print(f"[viser] scene={scene} sample_mode={sample_mode}")
    print(f"[viser] displayed_points={sum(len(pts) for pts in frame_points):,} cameras={len(camera_handles)}")
    print(f"[viser] open http://localhost:{args.port}")
    while True:
        time.sleep(0.1)


def main() -> None:
    args = parse_args()
    batch = load_batch(args)
    if not args.dry_run:
        run_viser(args, batch)


if __name__ == "__main__":
    main()
