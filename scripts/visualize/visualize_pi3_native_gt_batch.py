#!/usr/bin/env python3
"""Visualize one Pi3-native eval batch as a GT point cloud in viser."""

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
from typing import Dict, List, Tuple, cast

import numpy as np
import torch

import train as T


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load one eval_pi3 batch and visualize the back-projected GT depth point cloud."
    )
    parser.add_argument("--train_args_json", required=True)
    parser.add_argument(
        "--eval_mode",
        choices=("left_moving_tracks", "manip_track", "wrist_track", "random_static_track", "train_default"),
        default="left_moving_tracks",
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--split", choices=("val", "train", "all"), default="val")
    parser.add_argument("--max_scenes_eval", type=int, default=1)
    parser.add_argument("--eval_num_frames", type=int, default=16)
    parser.add_argument("--eval_wrist_camera_name", default="realsense_left")
    parser.add_argument("--eval_surround_camera_name", default="surround_cam_moving")
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pi3_repo", default="/cpfs/user/guowenqi/Pi3")
    parser.add_argument("--pi3_native_width", type=int, default=518)
    parser.add_argument("--max_points_per_frame", type=int, default=50000)
    parser.add_argument("--point_stride", type=int, default=2)
    parser.add_argument("--point_size", type=float, default=0.006)
    parser.add_argument("--camera_scale", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--recenter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Center points and cameras around the valid point-cloud centroid.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def invert_w2c(extrinsics: np.ndarray) -> np.ndarray:
    count = extrinsics.shape[0]
    w2c = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
    w2c[:, :3, :4] = extrinsics
    return np.linalg.inv(w2c)


def _sequence_length(batch: Dict[str, object]) -> int:
    if "sequence_lengths" in batch:
        return int(cast(torch.Tensor, batch["sequence_lengths"])[0])
    return int(cast(torch.Tensor, batch["images"]).shape[1])


def _frame_ids(batch: Dict[str, object], length: int) -> List[int]:
    if "frame_ids" in batch:
        return cast(torch.Tensor, batch["frame_ids"])[0, :length].cpu().numpy().astype(int).tolist()
    if "frame_indices" in batch:
        return cast(torch.Tensor, batch["frame_indices"])[0, :length].cpu().numpy().astype(int).tolist()
    return list(range(length))


def extract_frame_clouds(
    batch: Dict[str, object],
    max_points_per_frame: int,
    point_stride: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Back-project batch depths with batch intrinsics/extrinsics into world points."""
    length = _sequence_length(batch)
    stride = max(1, int(point_stride))

    images_t = cast(torch.Tensor, batch["images"])[0, :length].detach().cpu()
    depths_t = cast(torch.Tensor, batch["depths"])[0, :length].detach().cpu()
    masks_t = cast(torch.Tensor, batch["point_masks"])[0, :length].detach().cpu().bool()
    intrinsics_t = cast(torch.Tensor, batch["intrinsics"])[0, :length].detach().cpu()
    extrinsics_t = cast(torch.Tensor, batch["extrinsics"])[0, :length].detach().cpu()

    images = images_t.numpy().transpose(0, 2, 3, 1)
    intrinsics = intrinsics_t.numpy()
    extrinsics = extrinsics_t.numpy()
    c2w = invert_w2c(extrinsics)
    frame_ids = _frame_ids(batch, length)
    rng = np.random.default_rng(seed)

    frame_points: List[np.ndarray] = []
    frame_colors: List[np.ndarray] = []
    for frame_idx in range(length):
        world = T.depth_to_world_points(
            depths_t[frame_idx].float(),
            intrinsics_t[frame_idx].float(),
            extrinsics_t[frame_idx].float(),
        )
        valid = masks_t[frame_idx] & torch.isfinite(world).all(dim=-1)

        world_np = world[::stride, ::stride].numpy()
        valid_np = valid[::stride, ::stride].numpy().astype(bool)
        image_np = images[frame_idx, ::stride, ::stride]

        valid_points = world_np[valid_np].astype(np.float32)
        valid_colors = np.clip(image_np[valid_np] * 255.0, 0, 255).astype(np.uint8)
        if max_points_per_frame > 0 and len(valid_points) > max_points_per_frame:
            selected = rng.choice(len(valid_points), max_points_per_frame, replace=False)
            valid_points = valid_points[selected]
            valid_colors = valid_colors[selected]
        frame_points.append(valid_points)
        frame_colors.append(valid_colors)

    return frame_points, frame_colors, c2w, intrinsics, images, frame_ids


def make_eval_args(args: argparse.Namespace) -> argparse.Namespace:
    import eval_pi3

    eval_cli = [
        "--train_args_json",
        args.train_args_json,
        "--pi3_repo",
        args.pi3_repo,
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
        "--pi3_input_mode",
        "native",
        "--pi3_native_width",
        str(args.pi3_native_width),
    ]
    return eval_pi3.coerce_pi3_args_from_json(eval_pi3.build_argparser().parse_args(eval_cli))


def prepare_batch(args: argparse.Namespace) -> Dict[str, object]:
    import eval_pi3

    eval_args = make_eval_args(args)
    loader, scenes = eval_pi3.build_pi3_eval_loader(eval_args, eval_mode=args.eval_mode)
    if len(loader) == 0:
        raise RuntimeError("The eval loader is empty.")

    sample_index = args.sample_index % len(loader)
    iterator = iter(loader)
    batch = None
    for _ in range(sample_index + 1):
        batch = next(iterator)
    assert batch is not None

    length = _sequence_length(batch)
    scene = cast(List[str], batch["scene"])[0] if isinstance(batch.get("scene"), list) else str(batch.get("scene", ""))
    frame_ids = _frame_ids(batch, length)
    valid_points = int(cast(torch.Tensor, batch["point_masks"])[0, :length].sum())
    mode = cast(List[str], batch["sample_mode"])[0] if isinstance(batch.get("sample_mode"), list) else str(batch.get("sample_mode", ""))
    print(
        f"[sample] scenes={len(scenes)} sample_index={sample_index} scene={scene} mode={mode} "
        f"frames={length} valid_points={valid_points:,}"
    )
    print(f"[sample] frame_ids={frame_ids}")
    return batch


def run_viser(args: argparse.Namespace, batch: Dict[str, object]) -> None:
    import viser
    import viser.transforms as tf

    frame_points, frame_colors, c2w, intrinsics, images, frame_ids = extract_frame_clouds(
        batch, args.max_points_per_frame, args.point_stride, args.seed
    )
    nonempty = [points for points in frame_points if len(points)]
    if not nonempty:
        raise RuntimeError("The sampled batch has no valid depth points.")

    centroid = np.median(np.concatenate(nonempty, axis=0), axis=0) if args.recenter else np.zeros(3)
    frame_points = [points - centroid for points in frame_points]
    c2w[:, :3, 3] -= centroid

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_grid("/ground", width=10.0, height=10.0)

    options = ["All"] + [f"{idx}: frame {frame_id}" for idx, frame_id in enumerate(frame_ids)]
    with server.gui.add_folder("Pi3 native GT batch"):
        gui_frame = server.gui.add_dropdown("Visible points", options=options, initial_value="All")
        gui_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.03, step=0.001, initial_value=args.point_size
        )

    cloud = server.scene.add_point_cloud(
        "/gt/points",
        points=np.concatenate(frame_points, axis=0),
        colors=np.concatenate(frame_colors, axis=0),
        point_size=args.point_size,
        point_shape="circle",
    )

    camera_handles = []
    for frame_idx, (pose, k_mat, image, frame_id) in enumerate(zip(c2w, intrinsics, images, frame_ids)):
        rotation = tf.SO3.from_matrix(pose[:3, :3]).wxyz
        height, width = image.shape[:2]
        fov = float(2.0 * np.arctan2(height / 2.0, k_mat[1, 1]))
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
    print(f"[viser] displayed_points={shown_points:,}")
    print(f"[viser] open http://localhost:{args.port}")
    while True:
        time.sleep(0.1)


def main() -> None:
    args = parse_args()
    batch = prepare_batch(args)
    if args.dry_run:
        frame_points, _, _, _, images, _ = extract_frame_clouds(
            batch, args.max_points_per_frame, args.point_stride, args.seed
        )
        print(
            f"[dry-run] images_shape={tuple(cast(torch.Tensor, batch['images']).shape)} "
            f"viz_image_shape={images.shape} displayed_points={sum(len(points) for points in frame_points):,}"
        )
        return
    run_viser(args, batch)


if __name__ == "__main__":
    main()
