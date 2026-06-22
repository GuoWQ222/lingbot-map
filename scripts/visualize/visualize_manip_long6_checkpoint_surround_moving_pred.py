#!/usr/bin/env python3
"""Visualize checkpoint predictions on the first Manip_long6 val trajectory."""

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
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
import torch
import matplotlib.cm as cm
import viser
import viser.transforms as tf

import train as train_mod
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from train import (
    ManipTrajectoryDataset,
    canonicalize_to_first_frame,
    collate_rgbd_sequences,
    discover_trajectory_dirs,
    normalize_scene_batch,
    split_scenes,
)


T = TypeVar("T")


def uniform_sample_entries(entries: Sequence[T], target_count: int) -> List[T]:
    if target_count <= 0 or len(entries) <= target_count:
        return list(entries)
    indices = np.rint(np.linspace(0, len(entries) - 1, target_count)).astype(np.int64)
    unique_indices = np.unique(indices)
    if len(unique_indices) != target_count:
        unique_indices = np.linspace(0, len(entries) - 1, target_count, dtype=np.int64)
    return [entries[int(idx)] for idx in unique_indices.tolist()]


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


def first_pose_max_abs_diff(c2w: np.ndarray) -> float:
    if len(c2w) == 0:
        return float("inf")
    return float(np.max(np.abs(c2w[0].astype(np.float64) - np.eye(4, dtype=np.float64))))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3).astype(np.float64)
    transformed = (transform[:3, :3] @ flat.T).T + transform[:3, 3]
    return transformed.reshape(points.shape).astype(points.dtype, copy=False)


def apply_first_frame_canonicalization(
    pred_points: np.ndarray,
    pred_c2w: np.ndarray,
    policy: str,
    atol: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    diff = first_pose_max_abs_diff(pred_c2w)
    if policy == "on":
        do_canonicalize = True
    elif policy == "off":
        do_canonicalize = False
    elif policy == "auto":
        do_canonicalize = diff > atol
    else:
        raise ValueError(f"Unknown canonicalization policy: {policy}")

    if not do_canonicalize:
        print(f"[pred] first-frame canonicalization skipped: policy={policy}, first_pose_max_abs_diff={diff:.6g}")
        return pred_points, pred_c2w, False, diff

    transform = np.linalg.inv(pred_c2w[0].astype(np.float64))
    pred_points = transform_points(pred_points, transform)
    pred_c2w = np.einsum("ij,tjk->tik", transform, pred_c2w.astype(np.float64))
    print(f"[pred] first-frame canonicalization enabled: policy={policy}, first_pose_max_abs_diff={diff:.6g}")
    return pred_points, pred_c2w, True, diff


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


def split_camera_visibility_flags(
    mode: str,
    current_frame: int,
    frame_count: int,
    show_cameras: bool,
    show_pred: bool,
    show_gt: bool,
) -> Tuple[List[bool], List[bool]]:
    visible = visible_camera_flags(mode, current_frame, frame_count, show_cameras)
    pred_flags = [flag and show_pred for flag in visible]
    gt_flags = [flag and show_gt for flag in visible]
    return pred_flags, gt_flags


def camera_focus_from_pose(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = pose[:3, 3]
    forward = pose[:3, 2]
    up = -pose[:3, 1]
    return center - 0.15 * forward, center + forward, up


def unproject_depth_with_c2w_pose(depth: torch.Tensor, pose_enc: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, height, width, _ = depth.shape
    dtype = depth.dtype
    device = depth.device

    c2w_3x4, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(height, width), build_intrinsics=True
    )
    c2w_3x4 = c2w_3x4.to(device=device, dtype=dtype)
    intrinsics = intrinsics.to(device=device, dtype=dtype)

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
    camera_dirs = torch.einsum("bsij,hwj->bshwi", torch.inverse(intrinsics), pixel_coords)
    camera_points = camera_dirs * depth

    rotation = c2w_3x4[..., :3, :3]
    translation = c2w_3x4[..., :3, 3]
    return torch.einsum("bsij,bshwj->bshwi", rotation, camera_points) + translation[:, :, None, None, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one LingBot-MAP checkpoint and visualize predicted point clouds with Viser."
    )
    parser.add_argument(
        "--checkpoint",
        default="/cpfs/user/guowenqi/lingbot-map-copy/outputs/runs/manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/checkpoint_step_00010000.pt",
    )
    parser.add_argument(
        "--train-args-json",
        default="/cpfs/user/guowenqi/lingbot-map-copy/outputs/runs/manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/args.json",
    )
    parser.add_argument("--root", default="/oss-guowenqi/Manip_long6/data")
    parser.add_argument("--scene-dir", default="", help="Optional exact Manip trajectory directory to visualize; bypasses val split discovery.")
    parser.add_argument(
        "--scene-manifest",
        default="/cpfs/user/guowenqi/lingbot-map-copy/outputs/runs/manip_long_train_16gpu_dsw_280_d24_all3d_no_cudnn_bench_from_scratch/manip_long6_trajectory_manifest.txt",
    )
    parser.add_argument("--camera-name", default="surround_cam_moving")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=280)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--preprocess-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument("--use-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--depth-frames-chunk-size", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--max-points-per-frame", type=int, default=50000)
    parser.add_argument("--point-size", type=float, default=0.006)
    parser.add_argument("--camera-scale", type=float, default=0.08)
    parser.add_argument(
        "--canonicalize",
        choices=("auto", "on", "off"),
        default="auto",
        help="First-frame canonicalization policy for predicted geometry before visualization.",
    )
    parser.add_argument(
        "--canonicalize-atol",
        type=float,
        default=1e-2,
        help="Auto-canonicalization skips predicted poses whose first C2W is within this max-abs diff of identity.",
    )
    parser.add_argument("--no-recenter", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_train_args(args: argparse.Namespace) -> argparse.Namespace:
    train_args = train_mod.build_argparser().parse_args([])
    with open(args.train_args_json, "r", encoding="utf-8") as f:
        saved = json.load(f)
    for key, value in saved.items():
        setattr(train_args, key, value)
    train_args.model_path = ""
    train_args.image_size = args.image_size
    train_args.patch_size = args.patch_size
    train_args.clip_len = max(int(args.num_frames), int(getattr(train_args, "clip_len", args.num_frames)))
    train_args.max_sample_frames = max(
        int(args.num_frames), int(getattr(train_args, "max_sample_frames", args.num_frames))
    )
    train_args.num_scale_frames = args.num_scale_frames
    train_args.depth_frames_chunk_size = args.depth_frames_chunk_size
    train_args.no_gradient_checkpoint = True
    train_args.no_depth_activation_checkpoint = True
    train_args.camera_activation_checkpoint = False
    return train_args


def discover_first_val_scene(args: argparse.Namespace) -> Path:
    scenes = discover_trajectory_dirs(
        [args.root],
        max_scenes=0,
        manifest=args.scene_manifest,
        write_manifest=None,
        oss_uri_roots=None,
        ossutil_bin="/cpfs/user/guowenqi/ossutil/ossutil",
        ossutil_config="/cpfs/user/guowenqi/ossutil/.ossutilconfig",
    )
    scenes = [scene for scene in scenes if "Manip_long6" in str(scene)]
    _, val_scenes = split_scenes(scenes, args.val_fraction, args.seed)
    if not val_scenes:
        raise RuntimeError("Validation split is empty.")
    return val_scenes[0]


def make_dataset(scene_dir: Path, args: argparse.Namespace) -> ManipTrajectoryDataset:
    return ManipTrajectoryDataset(
        [scene_dir],
        clip_len=max(args.num_frames, 1),
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        sequence_mode="manip_4d_mixed",
        camera_names=[args.camera_name],
        sample_strategy="random_interval",
        frame_stride=1,
        random_stride_min=2,
        random_stride_max=8,
        max_sample_frames=args.num_frames,
        min_sample_frames=min(args.num_frames, 24),
        depth_scale=0.0,
        min_depth=1e-6,
        max_depth=0.0,
        use_mask=args.use_mask,
        invert_cam_extrinsics=False,
        samples_per_scene=1,
        wrist_camera_prefix="realsense",
        static_camera_prefix="surround",
        moving_stride_min=2,
        moving_stride_max=6,
        fixed_stride_min=4,
        fixed_stride_max=16,
        long6_root_marker="Manip_long6",
        long6_mode_weights={"W": 0.0, "T": 1.0, "F": 0.0},
        moving_camera_prefix=args.camera_name,
        fixed_camera_prefix="surround_cam_fixed",
        color_jitter_strength=0.0,
        color_jitter_prob=0.0,
    )


def prepare_uniform_batch(args: argparse.Namespace) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    scene_dir = Path(args.scene_dir).expanduser() if str(getattr(args, "scene_dir", "")).strip() else discover_first_val_scene(args)
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"scene_dir not found: {scene_dir}")
    dataset = make_dataset(scene_dir, args)
    entries = dataset._entries_for_scene(scene_dir)
    camera_entries = [entry for entry in entries if entry.camera_name == args.camera_name]
    camera_entries.sort(key=lambda entry: entry.frame_id)
    if not camera_entries:
        raise RuntimeError(f"No entries found for camera {args.camera_name!r} in {scene_dir}")

    selected = uniform_sample_entries(camera_entries, args.num_frames)
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
        "sample_mode": f"{args.camera_name}_uniform_{len(selected)}",
    }
    batch = collate_rgbd_sequences([sample])
    batch = canonicalize_to_first_frame(batch)
    batch = normalize_scene_batch(batch, num_anchor_frames=min(args.num_scale_frames, len(selected)))
    print(f"[sample] scene={scene_dir.name}")
    print(f"[sample] camera={args.camera_name} frames={len(selected)} total_camera_frames={len(camera_entries)}")
    print(f"[sample] frame_ids={sample['frame_ids'].tolist()}")
    return batch


def to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def build_checkpoint_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    train_args = load_train_args(args)
    model = train_mod.build_model(train_args, device)
    train_mod.load_state_dict_flexible(model, args.checkpoint, strict=False, map_location="cpu")
    model.to(device)
    model.eval()
    return model


def run_inference(
    args: argparse.Namespace,
    batch: Dict[str, object],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    model = build_checkpoint_model(args, device)
    batch_device = to_device(batch, device)
    images = batch_device["images"]
    assert torch.is_tensor(images)
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            predictions = model(
                images,
                num_frame_for_scale=min(args.num_scale_frames, int(images.shape[1])),
                num_frame_per_block=1,
                depth_frames_chunk_size=args.depth_frames_chunk_size,
                causal_inference=True,
            )
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()

    if "world_points" not in predictions:
        if "depth" not in predictions or "pose_enc" not in predictions:
            raise RuntimeError(f"Cannot build predicted point cloud from keys: {sorted(predictions.keys())}")
        predictions["world_points"] = unproject_depth_with_c2w_pose(
            predictions["depth"].float(), predictions["pose_enc"].float()
        )
        print("[pred] world_points not in checkpoint output; reconstructed from depth + c2w pose_enc")
    print(f"[pred] keys={sorted(predictions.keys())}")
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in predictions.items()}


def invert_w2c(extrinsics: np.ndarray) -> np.ndarray:
    count = extrinsics.shape[0]
    w2c = np.broadcast_to(np.eye(4, dtype=np.float64), (count, 4, 4)).copy()
    w2c[:, :3, :4] = extrinsics
    return np.linalg.inv(w2c)


def sample_clouds(
    points: np.ndarray,
    colors_rgb: np.ndarray,
    masks: np.ndarray,
    max_points_per_frame: int,
    seed: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    rng = np.random.default_rng(seed)
    frame_points: List[np.ndarray] = []
    frame_colors: List[np.ndarray] = []
    for frame_idx in range(points.shape[0]):
        valid = masks[frame_idx] & np.isfinite(points[frame_idx]).all(axis=-1)
        valid_points = points[frame_idx][valid].astype(np.float32)
        valid_colors = colors_rgb[frame_idx][valid].astype(np.uint8)
        if max_points_per_frame > 0 and len(valid_points) > max_points_per_frame:
            selected = rng.choice(len(valid_points), max_points_per_frame, replace=False)
            valid_points = valid_points[selected]
            valid_colors = valid_colors[selected]
        frame_points.append(valid_points)
        frame_colors.append(valid_colors)
    return frame_points, frame_colors


def prediction_camera_matrices(
    predictions: Dict[str, torch.Tensor],
    length: int,
    image_hw: Tuple[int, int],
    centroid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if "pose_enc" not in predictions:
        raise RuntimeError("Checkpoint visualization requires predictions['pose_enc'] to show predicted cameras.")
    pose_enc = predictions["pose_enc"][:, :length].float()
    c2w_3x4, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=image_hw, build_intrinsics=True
    )
    c2w_3x4_np = c2w_3x4[0].cpu().numpy()
    pred_c2w = np.broadcast_to(
        np.eye(4, dtype=np.float64), (length, 4, 4)
    ).copy()
    pred_c2w[:, :3, :4] = c2w_3x4_np
    if centroid is not None:
        pred_c2w[:, :3, 3] -= centroid
    return pred_c2w, intrinsics[0].cpu().numpy()


def make_cloud_data(
    batch: Dict[str, object],
    predictions: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[int],
]:
    length = int(batch["sequence_lengths"][0])
    images = batch["images"][0, :length].cpu().numpy().transpose(0, 2, 3, 1)
    image_colors = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    gt_points = batch["world_points"][0, :length].cpu().numpy()
    gt_masks = batch["point_masks"][0, :length].cpu().numpy().astype(bool)

    pred_points = predictions["world_points"][0, :length].float().numpy()
    pred_c2w, pred_intrinsics = prediction_camera_matrices(
        predictions,
        length,
        image_hw=(images.shape[1], images.shape[2]),
    )
    pred_points, pred_c2w, _, _ = apply_first_frame_canonicalization(
        pred_points, pred_c2w, str(args.canonicalize), float(args.canonicalize_atol)
    )

    pred_masks = np.isfinite(pred_points).all(axis=-1)
    if "depth" in predictions:
        pred_depth = predictions["depth"][0, :length].float().numpy()
        if pred_depth.ndim == 4:
            pred_depth = pred_depth[..., 0]
        pred_masks &= np.isfinite(pred_depth) & (pred_depth > 1e-6)

    gt_frame_points, gt_frame_colors = sample_clouds(
        gt_points,
        np.broadcast_to(np.array([170, 170, 170], dtype=np.uint8), image_colors.shape),
        gt_masks,
        args.max_points_per_frame,
        args.seed,
    )
    pred_frame_points, pred_frame_colors = sample_clouds(
        pred_points,
        image_colors,
        pred_masks,
        args.max_points_per_frame,
        args.seed + 17,
    )

    nonempty_gt = [pts for pts in gt_frame_points if len(pts)]
    centroid = np.median(np.concatenate(nonempty_gt, axis=0), axis=0) if nonempty_gt and not args.no_recenter else np.zeros(3)
    gt_frame_points = [pts - centroid for pts in gt_frame_points]
    pred_frame_points = [pts - centroid for pts in pred_frame_points]

    gt_c2w = invert_w2c(batch["extrinsics"][0, :length].cpu().numpy())
    gt_c2w[:, :3, 3] -= centroid
    gt_intrinsics = batch["intrinsics"][0, :length].cpu().numpy()
    pred_c2w[:, :3, 3] -= centroid
    frame_ids = batch["frame_ids"][0, :length].cpu().numpy().astype(int).tolist()
    return (
        gt_frame_points,
        gt_frame_colors,
        pred_frame_points,
        pred_frame_colors,
        gt_c2w,
        gt_intrinsics,
        pred_c2w,
        pred_intrinsics,
        frame_ids,
    )

def concat_frames(points_by_frame: List[np.ndarray], colors_by_frame: List[np.ndarray], indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    points = [points_by_frame[idx] for idx in indices if len(points_by_frame[idx])]
    colors = [colors_by_frame[idx] for idx in indices if len(colors_by_frame[idx])]
    if not points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.concatenate(points, axis=0), np.concatenate(colors, axis=0)


def add_camera_frustums(
    server: viser.ViserServer,
    prefix: str,
    poses: np.ndarray,
    intrinsics: np.ndarray,
    images: np.ndarray,
    frame_ids: Sequence[int],
    camera_scale: float,
) -> List[object]:
    handles: List[object] = []
    for frame_idx, (pose, K, image, frame_id) in enumerate(zip(poses, intrinsics, images, frame_ids)):
        rotation = tf.SO3.from_matrix(pose[:3, :3]).wxyz
        height, width = image.shape[:2]
        fov = float(2.0 * np.arctan2(height / 2.0, K[1, 1]))
        norm_idx = frame_idx / (len(poses) - 1) if len(poses) > 1 else 0.0
        color = cm.get_cmap("gist_rainbow")(norm_idx)[:3]
        handle = server.scene.add_camera_frustum(
            f"{prefix}/{frame_idx:03d}_{frame_id}",
            fov=fov,
            aspect=width / height,
            scale=camera_scale,
            wxyz=rotation,
            position=pose[:3, 3],
            color=color,
            line_width=2.0,
        )

        @handle.on_click
        def _focus(event, pose=pose) -> None:
            position, look_at, up = camera_focus_from_pose(pose)
            event.client.camera.position = tuple(position)
            event.client.camera.look_at = tuple(look_at)
            event.client.camera.up_direction = tuple(up)

        handles.append(handle)
    return handles


def run_viser(args: argparse.Namespace, batch: Dict[str, object], predictions: Dict[str, torch.Tensor]) -> None:
    (
        gt_points,
        gt_colors,
        pred_points,
        pred_colors,
        gt_c2w,
        gt_intrinsics,
        pred_c2w,
        pred_intrinsics,
        frame_ids,
    ) = make_cloud_data(batch, predictions, args)
    length = len(frame_ids)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.add_grid("/ground", width=10.0, height=10.0)

    all_indices = list(range(length))
    gt_initial_points, gt_initial_colors = concat_frames(gt_points, gt_colors, all_indices)
    pred_initial_points, pred_initial_colors = concat_frames(pred_points, pred_colors, all_indices)

    gt_cloud = server.scene.add_point_cloud(
        "/gt/points",
        points=gt_initial_points,
        colors=gt_initial_colors,
        point_size=args.point_size,
        point_shape="circle",
    )
    pred_cloud = server.scene.add_point_cloud(
        "/pred/points",
        points=pred_initial_points,
        colors=pred_initial_colors,
        point_size=args.point_size,
        point_shape="circle",
    )

    images = batch["images"][0, :length].cpu().numpy().transpose(0, 2, 3, 1)
    gt_camera_handles = add_camera_frustums(
        server, "/gt/cameras", gt_c2w, gt_intrinsics, images, frame_ids, args.camera_scale
    )
    pred_camera_handles = add_camera_frustums(
        server, "/pred/cameras", pred_c2w, pred_intrinsics, images, frame_ids, args.camera_scale
    )

    visible_modes = ["All frames", "Current frame", "0 to current frame"]
    with server.gui.add_folder("checkpoint prediction"):
        gui_visible_points = server.gui.add_dropdown("Visible points", options=visible_modes, initial_value="All frames")
        gui_frame_index = server.gui.add_slider("Current frame index", min=0, max=length - 1, step=1, initial_value=0)
        gui_prev_frame = server.gui.add_button("Prev frame")
        gui_next_frame = server.gui.add_button("Next frame")
        gui_show_pred = server.gui.add_checkbox("Show pred", initial_value=True)
        gui_show_gt = server.gui.add_checkbox("Show GT", initial_value=False)
        gui_show_cameras = server.gui.add_checkbox("Show cameras", initial_value=True)
        gui_point_size = server.gui.add_slider("Point size", min=0.001, max=0.03, step=0.001, initial_value=args.point_size)
        gui_camera_size = server.gui.add_slider("Camera size", min=0.005, max=0.3, step=0.005, initial_value=args.camera_scale)

    install_view_pose_controls(server)

    gt_cloud.visible = bool(gui_show_gt.value)
    pred_cloud.visible = bool(gui_show_pred.value)

    def _set_clouds() -> None:
        indices = visible_frame_indices(str(gui_visible_points.value), int(gui_frame_index.value), length)
        gt_cloud.points, gt_cloud.colors = concat_frames(gt_points, gt_colors, indices)
        pred_cloud.points, pred_cloud.colors = concat_frames(pred_points, pred_colors, indices)

    def _set_cameras() -> None:
        pred_flags, gt_flags = split_camera_visibility_flags(
            str(gui_visible_points.value),
            int(gui_frame_index.value),
            length,
            bool(gui_show_cameras.value),
            bool(gui_show_pred.value),
            bool(gui_show_gt.value),
        )
        for handle, visible in zip(pred_camera_handles, pred_flags):
            handle.visible = visible
        for handle, visible in zip(gt_camera_handles, gt_flags):
            handle.visible = visible

    def _set_visible_frames() -> None:
        _set_clouds()
        _set_cameras()

    _set_cameras()

    @gui_visible_points.on_update
    def _update_visible_points(_) -> None:
        _set_visible_frames()

    @gui_frame_index.on_update
    def _update_frame_index(_) -> None:
        _set_visible_frames()

    @gui_prev_frame.on_click
    def _prev_frame(_) -> None:
        gui_frame_index.value = max(0, int(gui_frame_index.value) - 1)

    @gui_next_frame.on_click
    def _next_frame(_) -> None:
        gui_frame_index.value = min(length - 1, int(gui_frame_index.value) + 1)

    @gui_show_gt.on_update
    def _show_gt(_) -> None:
        gt_cloud.visible = bool(gui_show_gt.value)
        _set_cameras()

    @gui_show_pred.on_update
    def _show_pred(_) -> None:
        pred_cloud.visible = bool(gui_show_pred.value)
        _set_cameras()

    @gui_show_cameras.on_update
    def _show_cameras(_) -> None:
        _set_cameras()

    @gui_point_size.on_update
    def _point_size(_) -> None:
        gt_cloud.point_size = gui_point_size.value
        pred_cloud.point_size = gui_point_size.value

    @gui_camera_size.on_update
    def _camera_size(_) -> None:
        set_camera_handle_scale(gt_camera_handles, gui_camera_size.value)
        set_camera_handle_scale(pred_camera_handles, gui_camera_size.value)

    print(f"[viser] pred_points={sum(len(p) for p in pred_points):,} gt_points={sum(len(p) for p in gt_points):,}")
    print(f"[viser] pred_cameras={len(pred_camera_handles):,} gt_cameras={len(gt_camera_handles):,}")
    print(f"[viser] open http://localhost:{args.port}")
    while True:
        time.sleep(0.1)

def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")
    device = torch.device(args.device)
    batch = prepare_uniform_batch(args)
    predictions = run_inference(args, batch, device)
    if args.dry_run:
        pred_shape = tuple(predictions["world_points"].shape)
        make_cloud_data(batch, predictions, args)
        print(f"[dry-run] pred_world_points_shape={pred_shape}")
        return
    run_viser(args, batch, predictions)


if __name__ == "__main__":
    main()
