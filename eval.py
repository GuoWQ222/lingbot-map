"""Evaluate a LingBot-MAP checkpoint on the held-out Manip validation split.

Mirrors the training-time validation loop (anchor-scale normalization, streaming
causal inference, train.py's ManipTrajectoryDataset) so the numbers stay
comparable to what the trainer logs, while reporting interpretable depth and
camera metrics on top of the raw losses.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train as T
from lingbot_map.utils.rotation import mat_to_quat
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

EPS = 1e-8


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


TRAIN_ARG_KEYS_FROM_JSON = (
    # Model architecture / inference settings
    "image_size",
    "patch_size",
    "enable_3d_rope",
    "max_frame_num",
    "kv_cache_sliding_window",
    "num_scale_frames",
    "num_frame_per_block",
    "depth_frames_chunk_size",
    "use_sdpa",
    "camera_num_iterations",
    "no_gradient_checkpoint",
    "no_depth_activation_checkpoint",
    "strict_load",
    # Dataset / sampling
    "data_roots",
    "scene_manifest",
    "oss_uri_roots",
    "ossutil_bin",
    "ossutil_config",
    "max_scenes",
    "val_fraction",
    "seed",
    "clip_len",
    "preprocess_mode",
    "sequence_mode",
    "view_ids",
    "camera_names",
    "sample_strategy",
    "frame_stride",
    "random_stride_min",
    "random_stride_max",
    "random_interval_start",
    "max_sample_frames",
    "min_sample_frames",
    "depth_scale",
    "min_depth",
    "max_depth",
    "use_mask",
    "invert_cam_extrinsics",
    "samples_per_scene",
    "wrist_camera_prefix",
    "static_camera_prefix",
    "m_stride_min",
    "m_stride_max",
    "s_views_min",
    "s_views_max",
    "m_num_views",
    "m_num_times",
    "m_views_min",
    "m_views_max",
    "t_stride_min",
    "t_stride_max",
    "long5_root_marker",
    "mode_weights_initial",
    "mode_weights_final",
    "mode_warmup_start",
    "mode_warmup_end",
    "color_jitter_strength",
    "color_jitter_prob",
    "min_valid_pixels",
    "normalize_scene",
    "canonicalize_first_frame",
    "amp",
    "amp_dtype",
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate LingBot-MAP on Manip val split")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint .pt (training save_checkpoint format).")
    p.add_argument("--train_args_json", type=str, default="",
                   help="args.json from the training run; used to recreate model/dataset config. "
                        "Defaults to <checkpoint_parent>/args.json.")
    p.add_argument("--output_dir", type=str, default="",
                   help="Where to save metrics JSON / per-scene CSV. "
                        "Defaults to <checkpoint_parent>/eval/<checkpoint_stem>.")
    p.add_argument("--split", choices=["val", "train", "all"], default="val",
                   help="Which slice of the manifest to evaluate.")
    p.add_argument("--max_scenes_eval", type=int, default=0,
                   help="Debug cap on number of scenes evaluated. 0 = all.")
    p.add_argument("--eval_shard_count", type=int, default=1,
                   help="Split eval scenes into this many deterministic shards.")
    p.add_argument("--eval_shard_index", type=int, default=0,
                   help="Evaluate only scenes whose deterministic shard index matches this value.")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--per_scene_csv", action="store_true",
                   help="Also write a per-scene CSV alongside the summary JSON.")
    p.add_argument("--save_predictions", action="store_true",
                   help="Save per-frame depth predictions as .npz files (one per scene).")
    p.add_argument("--print_every", type=int, default=5)
    p.add_argument("--eval_strategy", choices=["train_default", "manip_track", "wrist_track", "random_static_track", "both"],
                   default="manip_track",
                   help="Sampling strategy. "
                        "manip_track = Manip_long3/4 use a realsense single-camera track; "
                        "Manip_long5 uses a surround_cam single-camera track. "
                        "train_default = reuse manip_4d_mixed S/W/M curriculum. "
                        "wrist_track = take eval_num_frames evenly-spaced frames from realsense_left. "
                        "random_static_track = take eval_num_frames evenly-spaced timestamps; at each ts "
                        "randomly pick one of the 6 surround_cam cameras. "
                        "both = run wrist_track and random_static_track sequentially per scene.")
    p.add_argument("--eval_num_frames", type=int, default=64,
                   help="Frames per eval clip. Default 64 = train max_sample_frames. "
                        "Sampling uses np.linspace(0, T-1, eval_num_frames) so every scene contributes "
                        "exactly this many frames regardless of trajectory length.")
    p.add_argument("--eval_wrist_camera_name", type=str, default="realsense_left",
                   help="Which realsense camera to use for wrist_track. Matches FrameEntry.camera_name. "
                        "Default = realsense_left.")
    p.add_argument("--eval_surround_camera_name", type=str, default="surround_cam_0",
                   help="Optional single surround camera for Manip_long5 in manip_track. "
                        "Default = surround_cam_0. Empty = evaluate all 6 surround cameras.")
    p.add_argument("--eval_seed", type=int, default=42,
                   help="Seed for random_static_track's per-timestamp camera choice. "
                        "Each (scene, mode, eval_seed) triple is reproducible across runs.")
    p.add_argument("--depth_align", choices=["none", "median", "lsq", "pi3_scale", "pi3_scale_shift"],
                   default="pi3_scale_shift",
                   help="Depth alignment before metrics. pi3_scale_shift (default) mirrors Pi3 "
                        "videodepth: one sequence-level scale+shift optimized with L1 loss. "
                        "pi3_scale mirrors Pi3 scale-only sequence alignment. Legacy modes are "
                        "per-frame: median, lsq, or none.")
    p.add_argument("--image_size", type=int, default=0,
                   help="Override the image_size from train_args_json. 0 (default) keeps the value "
                        "from args.json. Useful when evaluating a checkpoint at a resolution different "
                        "from training (pos_embed will be interpolated).")
    p.add_argument("--depth_frames_chunk_size", type=int, default=0,
                   help="Override depth_frames_chunk_size from train_args_json. 0 keeps the training value. "
                        "Larger values can speed eval slightly at the cost of more GPU memory.")
    p.add_argument("--pointcloud_metrics", action="store_true",
                   help="Compute LoGeR-style point cloud ACC / Completeness / CD from predicted and GT point clouds.")
    p.add_argument("--pointcloud_max_points", type=int, default=100000,
                   help="Deterministically subsample each predicted/GT cloud to at most this many valid points "
                        "before nearest-neighbor metrics. <=0 keeps all valid points.")
    p.add_argument("--pointcloud_align", choices=["none", "scale_center", "umeyama", "icp", "pi3_icp"], default="pi3_icp",
                   help="Point cloud alignment before nearest-neighbor metrics. umeyama estimates a Sim(3) "
                        "from corresponding valid pixels; icp applies CUT3R-style center/scale initialization "
                        "followed by rigid point-to-point ICP; scale_center only aligns center+scale; "
                        "none compares raw anchor-normalized coordinates; pi3_icp mirrors Pi3 mv_recon with Umeyama followed by rigid ICP refinement.")
    p.add_argument("--pointcloud_icp_threshold", type=float, default=0.1,
                   help="Nearest-neighbor distance threshold for pointcloud_align=icp or pi3_icp. CUT3R uses 0.1 "
                        "for non-DTU multi-view reconstruction scenes.")
    p.add_argument("--pointcloud_icp_max_iterations", type=int, default=30,
                   help="Maximum rigid point-to-point ICP iterations for pointcloud_align=icp or pi3_icp.")
    p.add_argument("--geometry_normalization", choices=["native", "vggt_independent", "none"],
                   default="none",
                   help="Geometry frame/scale used for metrics. native keeps the training-time "
                        "first-frame canonicalization + anchor-frame scale normalization. "
                        "vggt_independent applies VGGT-style first-frame canonicalization and "
                        "all-valid-point scale normalization independently to GT and prediction "
                        "before computing metrics. none leaves GT geometry untouched.")
    p.add_argument("--camera_align", choices=["sim3", "none"], default="sim3",
                   help="Camera metric alignment after geometry normalization. sim3 mirrors the "
                        "old CUT3R-style evo path (align=True, correct_scale=True); none compares "
                        "the normalized trajectories directly.")
    return p


def coerce_args_from_json(eval_args: argparse.Namespace) -> argparse.Namespace:
    """Take eval_args + the training run's args.json and produce a fully-populated
    argparse.Namespace that the train.py helpers can accept verbatim."""
    if not eval_args.train_args_json:
        eval_args.train_args_json = str(Path(eval_args.checkpoint).parent / "args.json")
    args_json_path = Path(eval_args.train_args_json)
    if not args_json_path.is_file():
        raise FileNotFoundError(f"args.json not found: {args_json_path}")
    with open(args_json_path, "r", encoding="utf-8") as f:
        train_args_dict = json.load(f)

    merged = dict(train_args_dict)
    for key in TRAIN_ARG_KEYS_FROM_JSON:
        if key not in train_args_dict:
            # Fill in any defaults present on argparse; benign if absent.
            continue
    # Hand over the known training-time values to a Namespace train.py expects.
    ns = argparse.Namespace(**merged)
    ns.checkpoint = eval_args.checkpoint
    ns.train_args_json = eval_args.train_args_json
    ns.output_dir = eval_args.output_dir
    ns.split = eval_args.split
    ns.max_scenes_eval = eval_args.max_scenes_eval
    ns.num_workers = eval_args.num_workers
    ns.device = eval_args.device
    ns.per_scene_csv = eval_args.per_scene_csv
    ns.save_predictions = eval_args.save_predictions
    ns.print_every = eval_args.print_every
    ns.eval_strategy = eval_args.eval_strategy
    ns.eval_num_frames = eval_args.eval_num_frames
    ns.eval_wrist_camera_name = eval_args.eval_wrist_camera_name
    ns.eval_surround_camera_name = eval_args.eval_surround_camera_name
    ns.eval_seed = eval_args.eval_seed
    ns.depth_align = eval_args.depth_align
    ns.pointcloud_metrics = bool(eval_args.pointcloud_metrics)
    ns.pointcloud_max_points = int(eval_args.pointcloud_max_points)
    ns.pointcloud_align = str(eval_args.pointcloud_align)
    ns.pointcloud_icp_threshold = float(eval_args.pointcloud_icp_threshold)
    ns.pointcloud_icp_max_iterations = int(eval_args.pointcloud_icp_max_iterations)
    ns.geometry_normalization = str(eval_args.geometry_normalization)
    ns.camera_align = str(eval_args.camera_align)
    if int(eval_args.image_size) > 0:
        ns.image_size = int(eval_args.image_size)
    if int(eval_args.depth_frames_chunk_size) > 0:
        ns.depth_frames_chunk_size = int(eval_args.depth_frames_chunk_size)
    # train.build_model checks args.model_path to load a pretrained init; we want
    # the eval-time checkpoint to be the source of truth, so set this to empty
    # and call load_state_dict_flexible manually with --checkpoint.
    ns.model_path = ""
    ns.cpu = (eval_args.device == "cpu")
    # Force batch_size=1 like training.
    ns.batch_size = 1
    ns.write_manifest = ns.__dict__.get("write_manifest", None) or None
    return ns


# -----------------------------------------------------------------------------
# Dataset / dataloader
# -----------------------------------------------------------------------------


def build_eval_loader(args: argparse.Namespace, eval_mode: str = "default") -> Tuple[DataLoader, List[Path]]:
    """Build a deterministic eval DataLoader over the requested split.

    eval_mode:
      - "default"               : reuse the trained manip_4d_mixed S/W/M curriculum
      - "manip_track"           : Long3/4 realsense track; Long5 surround track
      - "wrist_track"           : 48 linspace-sampled frames from realsense_left (or args.eval_wrist_camera_name)
      - "random_static_track"   : 48 linspace-sampled timestamps; at each ts a random surround camera is picked
    """
    scenes = T.discover_trajectory_dirs(
        args.data_roots,
        max_scenes=args.max_scenes,
        manifest=args.scene_manifest,
        write_manifest=None,
        oss_uri_roots=T.parse_str_list(args.oss_uri_roots),
        ossutil_bin=args.ossutil_bin,
        ossutil_config=args.ossutil_config,
    )
    if not scenes:
        raise RuntimeError("No Manip trajectories were discovered")

    train_scenes, val_scenes = T.split_scenes(scenes, args.val_fraction, args.seed)
    if args.split == "val":
        eval_scenes = val_scenes
    elif args.split == "train":
        eval_scenes = train_scenes
    else:
        eval_scenes = list(scenes)

    if args.max_scenes_eval > 0:
        eval_scenes = eval_scenes[: args.max_scenes_eval]
    shard_count = max(1, int(getattr(args, "eval_shard_count", 1)))
    shard_index = int(getattr(args, "eval_shard_index", 0))
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"eval_shard_index must be in [0, {shard_count}), got {shard_index}")
    if shard_count > 1:
        total_eval_scenes = len(eval_scenes)
        eval_scenes = eval_scenes[shard_index::shard_count]
        print(f"[eval] shard {shard_index}/{shard_count}: {len(eval_scenes)} of {total_eval_scenes} scenes")
    if not eval_scenes:
        raise RuntimeError(f"split={args.split} produced no scenes")

    view_ids = T.parse_int_list(args.view_ids)
    camera_names = T.parse_str_list(args.camera_names)
    common_kwargs = dict(
        clip_len=args.clip_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        sequence_mode=args.sequence_mode,
        view_ids=view_ids,
        camera_names=camera_names,
        sample_strategy=args.sample_strategy,
        frame_stride=args.frame_stride,
        random_stride_min=args.random_stride_min,
        random_stride_max=args.random_stride_max,
        random_interval_start=args.random_interval_start,
        max_sample_frames=args.max_sample_frames,
        min_sample_frames=args.min_sample_frames,
        depth_scale=args.depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        use_mask=args.use_mask,
        invert_cam_extrinsics=args.invert_cam_extrinsics,
        samples_per_scene=args.samples_per_scene,
        wrist_camera_prefix=args.wrist_camera_prefix,
        static_camera_prefix=args.static_camera_prefix,
        m_stride_min=args.m_stride_min,
        m_stride_max=args.m_stride_max,
        s_views_min=args.s_views_min,
        s_views_max=args.s_views_max,
        m_num_views=args.m_num_views,
        m_num_times=args.m_num_times,
        m_views_min=args.m_views_min,
        m_views_max=args.m_views_max,
        t_stride_min=getattr(args, "t_stride_min", 15),
        t_stride_max=getattr(args, "t_stride_max", 60),
        long5_root_marker=getattr(args, "long5_root_marker", "Manip_long5"),
        mode_weights_initial=T.parse_mode_weights(args.mode_weights_initial),
        mode_weights_final=T.parse_mode_weights(args.mode_weights_final),
        mode_warmup_start=0,
        mode_warmup_end=0,
        color_jitter_strength=0.0,
        color_jitter_prob=0.0,
    )

    if eval_mode == "default":
        dataset = T.ManipTrajectoryDataset(eval_scenes, **common_kwargs)
    elif eval_mode in {"manip_track", "wrist_track", "random_static_track"}:
        dataset = EvalLinspaceDataset(
            eval_scenes,
            eval_mode=eval_mode,
            eval_num_frames=int(args.eval_num_frames),
            eval_wrist_camera_name=str(args.eval_wrist_camera_name),
            eval_surround_camera_name=str(getattr(args, "eval_surround_camera_name", "")),
            eval_seed=int(args.eval_seed),
            **common_kwargs,
        )
    else:
        raise ValueError(f"Unknown eval_mode: {eval_mode}")
    # Force the curriculum onto the final-stage weights so eval uses the
    # post-warmup S/W/M mix (only relevant for the default manip_4d_mixed mode).
    dataset.set_global_step(max(int(args.mode_warmup_end), 1) * 10)

    num_shards = max(1, int(getattr(args, "eval_num_shards", 1)))
    shard_index = int(getattr(args, "eval_shard_index", 0))
    if num_shards > 1:
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(f"eval_shard_index={shard_index} must be in [0, {num_shards})")
        indices = list(range(shard_index, len(dataset), num_shards))
        if not indices:
            raise RuntimeError(
                f"Eval shard {shard_index}/{num_shards} is empty for dataset length {len(dataset)}"
            )
        dataset = Subset(dataset, indices)

    nw = max(0, int(args.num_workers))
    loader_kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available() and args.device != "cpu",
        drop_last=False,
        collate_fn=T.collate_rgbd_sequences,
    )
    if nw > 0:
        # Keep workers alive across the whole eval (especially when both modes
        # are run sequentially) and prefetch aggressively so OSS-bound data
        # loading overlaps with model forward.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, eval_scenes


# -----------------------------------------------------------------------------
# Eval-only fixed-stride dataset
# -----------------------------------------------------------------------------


class EvalLinspaceDataset(T.ManipTrajectoryDataset):  # type: ignore[misc]
    """Subclass of ManipTrajectoryDataset that overrides _sample_entries to do
    deterministic eval-only linspace sampling.

    Eval modes (each yielding one ``eval_num_frames``-length clip):

      - ``manip_track``: for Manip_long3/4, take a single realsense camera;
        for Manip_long5, take one surround camera track (default
        ``surround_cam_0``). Passing an empty ``eval_surround_camera_name``
        keeps the old all-6-surround-camera behavior. Every track uses the
        same ``np.linspace(0, T-1, N)`` sampling over that camera's frames.

      - ``wrist_track``: take a single realsense camera (default ``realsense_left``)
        and sample ``eval_num_frames`` indices via ``np.linspace(0, T-1, N)`` over
        that camera's frames. The realsense moves with the arm so this is a real
        moving-camera trajectory.

      - ``random_static_track``: pool the 6 surround cameras' frame_ids (they share
        the same timestamp grid). Pick ``eval_num_frames`` evenly-spaced timestamps
        via linspace. At each chosen timestamp, randomly pick one of the surround
        cameras that has a frame at that timestamp. Random pick is seeded by
        (scene_name, eval_seed) so the choices stay reproducible across runs.
    """

    def __init__(
        self,
        scene_dirs,
        *,
        eval_mode: str,
        eval_num_frames: int,
        eval_wrist_camera_name: str,
        eval_surround_camera_name: str,
        eval_seed: int,
        **kwargs,
    ) -> None:
        super().__init__(scene_dirs, **kwargs)
        if eval_mode not in {"manip_track", "wrist_track", "random_static_track"}:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")
        self.eval_mode = eval_mode
        self.eval_num_frames = max(1, int(eval_num_frames))
        self.eval_wrist_camera_name = str(eval_wrist_camera_name)
        self.eval_surround_camera_name = str(eval_surround_camera_name)
        self.eval_seed = int(eval_seed)
        # Eval uses an explicit index map: Long3/4 contribute one clip. Long5
        # contributes one clip by default, or one per surround camera when the
        # camera name is intentionally left empty.
        self.samples_per_scene = 1
        self._long5_surround_slots = 1 if self.eval_surround_camera_name.strip() else 6
        self._eval_index_map = []
        for scene_dir in self.scene_dirs:
            if self.eval_mode == "manip_track" and self._is_long5_scene(scene_dir):
                self._eval_index_map.extend((scene_dir, slot) for slot in range(self._long5_surround_slots))
            else:
                self._eval_index_map.append((scene_dir, None))

    def __len__(self) -> int:
        return len(self._eval_index_map)

    def __getitem__(self, index: int) -> Dict[str, object]:
        last_error: Optional[Exception] = None
        start = index % len(self._eval_index_map)
        for attempt in range(min(16, len(self._eval_index_map))):
            scene_dir, surround_slot = self._eval_index_map[(start + attempt) % len(self._eval_index_map)]
            try:
                entries = self._entries_for_scene(scene_dir)
                selected, mode_label, camera_name = self._sample_entries_for_eval(
                    entries, scene_dir, surround_slot
                )
                jitter_params = self._sample_color_jitter_params()
                loaded = [self._load_one(scene_dir, entry, jitter_params) for entry in selected]
                images, depths, masks, intrinsics, extrinsics, world_points = zip(*loaded)
                scene_label = scene_dir.name
                if camera_name:
                    scene_label = f"{scene_label}__{camera_name}"
                return {
                    "images": torch.stack(list(images), dim=0),
                    "depths": torch.stack(list(depths), dim=0),
                    "point_masks": torch.stack(list(masks), dim=0),
                    "intrinsics": torch.stack(list(intrinsics), dim=0),
                    "extrinsics": torch.stack(list(extrinsics), dim=0),
                    "world_points": torch.stack(list(world_points), dim=0),
                    "frame_ids": torch.tensor([entry.frame_id for entry in selected], dtype=torch.long),
                    "view_ids": torch.tensor([entry.view_id for entry in selected], dtype=torch.long),
                    "scene": scene_label,
                    "sample_mode": mode_label,
                }
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"Failed to load a valid eval sample near index {index}: {last_error}")

    @staticmethod
    def _stable_seed(scene_dir, salt: int) -> int:
        from hashlib import sha1
        digest = sha1(f"{Path(scene_dir).name}|{salt}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big", signed=False)

    @staticmethod
    def _linspace_indices(total: int, n: int) -> List[int]:
        """Return n evenly-spaced indices in [0, total-1]; if total < n we just
        replay the available frames so the model still sees a clip of length n."""
        if total <= 0:
            raise RuntimeError("EvalLinspaceDataset: empty trajectory")
        if total == 1:
            return [0] * n
        if total >= n:
            return [int(round(i)) for i in np.linspace(0, total - 1, n)]
        # Trajectory shorter than requested clip: linspace over what we have, then
        # pad by repeating the last index so the clip still has n frames.
        base = [int(round(i)) for i in np.linspace(0, total - 1, total)]
        while len(base) < n:
            base.append(base[-1])
        return base

    def _sample_entries(self, entries):  # type: ignore[override]
        scene_dir = entries[0].rgb_path.parent.parent.parent if entries else None
        selected, mode_label, _ = self._sample_entries_for_eval(entries, scene_dir, None)
        return selected, mode_label

    def _sample_entries_for_eval(self, entries, scene_dir, surround_slot):
        if not entries:
            raise RuntimeError("No RGB-D entries for linspace sampling")
        by_view, by_vt, wrist_views, static_views = self._classify_cameras(entries)
        if scene_dir is None:
            scene_dir = entries[0].rgb_path.parent.parent.parent
        rng = random.Random(self._stable_seed(scene_dir, self.eval_seed))

        if self.eval_mode == "manip_track":
            if self._is_long5_scene(scene_dir):
                selected = self._sample_surround_track(by_view, static_views, entries, surround_slot)
                camera_name = selected[0].camera_name or f"view{selected[0].view_id}"
                return selected, f"T*:{camera_name}", camera_name
            selected = self._sample_wrist_track(by_view, wrist_views, entries)
            camera_name = selected[0].camera_name or f"view{selected[0].view_id}"
            return selected, f"W*:{camera_name}", camera_name
        if self.eval_mode == "wrist_track":
            selected = self._sample_wrist_track(by_view, wrist_views, entries)
            camera_name = selected[0].camera_name or f"view{selected[0].view_id}"
            return selected, f"W*:{camera_name}", camera_name
        if self.eval_mode == "random_static_track":
            return self._sample_random_static_track(by_view, by_vt, static_views, rng), "S*", ""
        raise RuntimeError(f"Unhandled eval_mode={self.eval_mode}")

    def _sample_wrist_track(self, by_view, wrist_views, entries):
        # Find the view whose camera_name matches eval_wrist_camera_name; fall
        # back to the first wrist view if the requested name is missing for
        # this scene (shouldn't happen with realsense_left/right but be safe).
        target_name = self.eval_wrist_camera_name.lower()
        chosen_view = None
        for entry in entries:
            if entry.camera_name and entry.camera_name.lower() == target_name:
                chosen_view = entry.view_id
                break
        if chosen_view is None:
            if not wrist_views:
                raise RuntimeError("wrist_track: no wrist cameras found")
            chosen_view = sorted(wrist_views)[0]

        camera_entries = by_view.get(chosen_view, [])
        if not camera_entries:
            raise RuntimeError(f"wrist_track: view {chosen_view} has no entries")
        camera_entries = sorted(camera_entries, key=lambda e: e.frame_id)
        idxs = self._linspace_indices(len(camera_entries), self.eval_num_frames)
        return [camera_entries[i] for i in idxs]

    def _sample_surround_track(self, by_view, static_views, entries, surround_slot=None):
        if not static_views:
            raise RuntimeError("manip_track: no surround cameras found")

        target_name = self.eval_surround_camera_name.lower().strip()
        chosen_view = None
        if target_name:
            for entry in entries:
                if entry.camera_name and entry.camera_name.lower() == target_name:
                    chosen_view = entry.view_id
                    break
        if chosen_view is None:
            sorted_views = sorted(static_views)
            slot = 0 if surround_slot is None else int(surround_slot)
            if slot >= len(sorted_views):
                raise RuntimeError(
                    f"manip_track: requested surround slot {slot}, only {len(sorted_views)} available"
                )
            chosen_view = sorted_views[slot]

        camera_entries = by_view.get(chosen_view, [])
        if not camera_entries:
            raise RuntimeError(f"manip_track: surround view {chosen_view} has no entries")
        camera_entries = sorted(camera_entries, key=lambda e: e.frame_id)
        idxs = self._linspace_indices(len(camera_entries), self.eval_num_frames)
        return [camera_entries[i] for i in idxs]

    def _sample_random_static_track(self, by_view, by_vt, static_views, rng):
        if not static_views:
            raise RuntimeError("random_static_track: no static cameras")
        # Pool timestamps available in at least one surround camera.
        ts_to_views: Dict[int, List[int]] = {}
        for view_id in static_views:
            for entry in by_view.get(view_id, []):
                ts_to_views.setdefault(entry.frame_id, []).append(view_id)
        if not ts_to_views:
            raise RuntimeError("random_static_track: no timestamps")

        sorted_ts = sorted(ts_to_views.keys())
        idxs = self._linspace_indices(len(sorted_ts), self.eval_num_frames)
        selected: List[T.FrameEntry] = []
        for i in idxs:
            timestamp = sorted_ts[i]
            chosen_view = rng.choice(ts_to_views[timestamp])
            selected.append(by_vt[(chosen_view, timestamp)])
        return selected


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def build_eval_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=max(args.max_frame_num, args.clip_len, args.max_sample_frames),
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
        enable_point=False,
        enable_local_point=False,
        use_gradient_checkpoint=False,
    )
    if getattr(model, "depth_head", None) is not None:
        depth_head = model.depth_head
        if hasattr(depth_head, "use_activation_checkpoint"):
            depth_head.use_activation_checkpoint = False

    T.load_state_dict_flexible(model, args.checkpoint, strict=False, map_location="cpu")
    return model.to(device).eval()


def _mean_valid_point_scale(points: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """VGGT-style scale: mean distance of all valid points after canonicalization."""
    mask = valid_mask.to(device=points.device, dtype=torch.bool)
    finite = torch.isfinite(points).all(dim=-1)
    valid = mask & finite
    dist = points.float().norm(dim=-1)
    valid_f = valid.float()
    count = valid_f.sum(dim=(1, 2, 3))
    total = (dist * valid_f).sum(dim=(1, 2, 3))

    all_f = finite.float()
    all_count = all_f.sum(dim=(1, 2, 3))
    all_total = (dist * all_f).sum(dim=(1, 2, 3))
    fallback = torch.where(all_count > 0, all_total / all_count.clamp(min=1.0), torch.ones_like(all_count))
    scale = torch.where(count > 0, total / count.clamp(min=1.0), fallback)
    return scale.clamp(min=1e-6, max=1e6)


def vggt_normalize_gt_batch(batch: Dict[str, object]) -> Dict[str, object]:
    """Canonicalize GT to frame 0 and scale by all valid canonicalized GT points."""
    canon = T.canonicalize_to_first_frame(batch)
    extrinsics = cast(torch.Tensor, canon["extrinsics"])
    depths = cast(torch.Tensor, canon["depths"])
    world_points = cast(torch.Tensor, canon["world_points"])
    point_masks = cast(torch.Tensor, canon["point_masks"])

    scale = _mean_valid_point_scale(world_points, point_masks)
    new_extrinsics = extrinsics.clone()
    new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / scale[:, None, None]

    out = dict(canon)
    out["extrinsics"] = T.check_and_fix_inf_nan(new_extrinsics, "vggt_gt_extrinsics", hard_max=None)
    out["world_points"] = T.check_and_fix_inf_nan(world_points / scale[:, None, None, None, None], "vggt_gt_world_points", hard_max=None)
    out["depths"] = T.check_and_fix_inf_nan(depths / scale[:, None, None, None], "vggt_gt_depths", hard_max=None)
    out["vggt_gt_scale"] = scale
    return out


def _transform_world_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    ones = torch.ones_like(points[..., :1])
    points_h = torch.cat([points, ones], dim=-1)
    transformed = torch.einsum("bij,bshwj->bshwi", transform.to(dtype=points.dtype), points_h)
    return transformed[..., :3]


def _c2w_to_pose_encoding(c2w: torch.Tensor, template_pose: torch.Tensor) -> torch.Tensor:
    quat = mat_to_quat(c2w[..., :3, :3])
    return torch.cat([c2w[..., :3, 3], quat, template_pose[..., 7:9]], dim=-1).to(dtype=template_pose.dtype)


def vggt_normalize_predictions(
    predictions: Dict[str, torch.Tensor],
    valid_mask: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Apply VGGT-style canonicalization and scale normalization to predictions.

    The prediction uses its own first predicted camera and its own all-valid
    predicted point scale. The GT valid mask only defines the evaluation support.
    """
    if "pose_enc" not in predictions:
        return predictions

    pose = predictions["pose_enc"].float()
    pred_depth = predictions.get("depth")
    pred_world_points = predictions.get("world_points")
    if pred_world_points is None:
        if pred_depth is None:
            return predictions
        pred_world_points = unproject_depth_to_world_from_pose(pred_depth.float(), pose)
    else:
        pred_world_points = pred_world_points.float()

    c2w = T.pose_encoding_to_c2w_matrix(pose)
    first_w2c = T.inverse_se3(c2w[:, 0])
    canon_c2w = torch.matmul(first_w2c.unsqueeze(1), c2w)
    canon_world_points = _transform_world_points(pred_world_points, first_w2c)
    scale = _mean_valid_point_scale(canon_world_points, valid_mask)

    norm_c2w = canon_c2w.clone()
    norm_c2w[:, :, :3, 3] = norm_c2w[:, :, :3, 3] / scale[:, None, None]

    out = dict(predictions)
    out["pose_enc"] = T.check_and_fix_inf_nan(_c2w_to_pose_encoding(norm_c2w, pose), "vggt_pred_pose_enc", hard_max=None)
    out["world_points"] = T.check_and_fix_inf_nan(canon_world_points / scale[:, None, None, None, None], "vggt_pred_world_points", hard_max=None)
    if pred_depth is not None:
        out["depth"] = T.check_and_fix_inf_nan(pred_depth / scale[:, None, None, None, None], "vggt_pred_depth", hard_max=None)
    out["vggt_pred_scale"] = scale
    return out


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------


def _pi3_sequence_scale(pred_v: torch.Tensor, gt_v: torch.Tensor) -> float:
    """Pi3 scale-only sequence alignment from utils/depth.py."""
    if pred_v.numel() == 0:
        return 1.0
    s = torch.nanmean(gt_v) / torch.nanmean(pred_v).clamp(min=EPS)
    for _ in range(10):
        residuals = s * pred_v - gt_v
        weights = 1.0 / (residuals.abs() + 1e-8)
        num = torch.sum(weights * pred_v * gt_v)
        den = torch.sum(weights * pred_v.square()).clamp(min=EPS)
        s = num / den
    s = s.clamp(min=1e-3).detach()
    value = float(s.item())
    return value if math.isfinite(value) and value > 0 else 1.0


def _pi3_sequence_scale_shift(pred_v: torch.Tensor, gt_v: torch.Tensor) -> Tuple[float, float]:
    """Pi3 scale&shift alignment: optimize sum |s * pred + t - gt| per sequence."""
    if pred_v.numel() == 0:
        return 1.0, 0.0
    pred_opt = pred_v.detach()
    gt_opt = gt_v.detach()
    pred_med = torch.median(pred_opt).clamp(min=EPS)
    s_init = float((torch.median(gt_opt) / pred_med).item())
    if not math.isfinite(s_init) or s_init <= 0:
        s_init = 1.0
    with torch.enable_grad():
        s = torch.tensor([s_init], requires_grad=True, device=pred_opt.device, dtype=pred_opt.dtype)
        t = torch.tensor([0.0], requires_grad=True, device=pred_opt.device, dtype=pred_opt.dtype)
        optimizer = torch.optim.Adam([s, t], lr=1e-4)
        prev_loss: Optional[torch.Tensor] = None
        for _ in range(1000):
            optimizer.zero_grad()
            loss = torch.abs(s * pred_opt + t - gt_opt).sum()
            loss.backward()
            optimizer.step()
            if prev_loss is not None and torch.abs(prev_loss - loss.detach()) < 1e-6:
                break
            prev_loss = loss.detach()
    scale = float(s.detach().item())
    shift = float(t.detach().item())
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    if not math.isfinite(shift):
        shift = 0.0
    return scale, shift


def compute_depth_per_frame_metrics(
    pred_depth: torch.Tensor,    # [B, S, H, W, 1]
    gt_depth: torch.Tensor,      # [B, S, H, W]
    valid_mask: torch.Tensor,    # [B, S, H, W] bool
    min_gt_depth: float = EPS,
    align: str = "pi3_scale_shift",
) -> List[Dict[str, float]]:
    """Compute depth metrics for valid frames.

    Pi3 modes compute one alignment over the whole sequence/clip, then report
    the existing LingBot-MAP per-frame macro metrics. Legacy modes retain the
    old per-frame alignment behavior for comparability with older runs.
    """
    align_mode = str(align)
    pi3_sequence_align = align_mode in {"pi3_scale", "pi3_scale_shift"}

    pred_raw = pred_depth[..., 0]
    gt = gt_depth.clamp(min=min_gt_depth)
    if pi3_sequence_align:
        pred = pred_raw
        finite = torch.isfinite(pred) & torch.isfinite(gt) & (gt > min_gt_depth)
    else:
        pred = pred_raw.clamp(min=min_gt_depth)
        finite = (
            torch.isfinite(pred)
            & torch.isfinite(gt)
            & (gt > min_gt_depth)
            & (pred > min_gt_depth)
        )
    valid = valid_mask & finite
    B, S = pred.shape[:2]

    sequence_params: Dict[int, Tuple[float, float]] = {}
    if pi3_sequence_align:
        for b in range(B):
            seq_mask = valid[b]
            if int(seq_mask.sum().item()) <= 0:
                sequence_params[b] = (1.0, 0.0)
                continue
            pred_seq = pred[b][seq_mask].double()
            gt_seq = gt[b][seq_mask].double()
            if align_mode == "pi3_scale":
                sequence_params[b] = (_pi3_sequence_scale(pred_seq, gt_seq), 0.0)
            else:
                sequence_params[b] = _pi3_sequence_scale_shift(pred_seq, gt_seq)

    out: List[Dict[str, float]] = []
    for b in range(B):
        for s in range(S):
            mask_bs = valid[b, s]
            n = int(mask_bs.sum().item())
            if n <= 0:
                continue
            pred_v = pred[b, s][mask_bs].double()
            gt_v = gt[b, s][mask_bs].double()

            shift = 0.0
            if pi3_sequence_align:
                scale, shift = sequence_params.get(b, (1.0, 0.0))
                pred_v = pred_v * scale + shift
            elif align_mode == "median":
                pred_med = pred_v.median().clamp(min=EPS)
                scale = float((gt_v.median() / pred_med).item())
                if not math.isfinite(scale) or scale <= 0:
                    scale = 1.0
                pred_v = pred_v * scale
            elif align_mode == "lsq":
                num = float((pred_v * gt_v).sum().item())
                den = float((pred_v * pred_v).sum().clamp(min=EPS).item())
                scale = num / den
                if not math.isfinite(scale) or scale <= 0:
                    scale = 1.0
                pred_v = pred_v * scale
            elif align_mode == "none":
                scale = 1.0
            else:
                raise ValueError(f"Unknown align mode: {align_mode}")

            pred_for_ratio = pred_v.clamp(min=min_gt_depth)
            diff = pred_v - gt_v
            abs_diff = diff.abs()
            ratio = torch.maximum(pred_for_ratio / gt_v, gt_v / pred_for_ratio)
            log_diff = pred_for_ratio.log() - gt_v.log()
            log_diff_var = float((log_diff - log_diff.mean()).pow(2).mean().item())
            log10_diff = (pred_for_ratio.log10() - gt_v.log10()).abs()
            row = {
                "AbsRel": float((abs_diff / gt_v).mean().item()),
                "SqRel": float(((diff * diff) / gt_v).mean().item()),
                "RMSE": float(diff.pow(2).mean().sqrt().item()),
                "MAE": float(abs_diff.mean().item()),
                "log10": float(log10_diff.mean().item()),
                "delta<1.25": float((ratio < 1.25).double().mean().item()),
                "delta<1.25^2": float((ratio < 1.25 ** 2).double().mean().item()),
                "delta<1.25^3": float((ratio < 1.25 ** 3).double().mean().item()),
                "si-RMSE": float(math.sqrt(max(log_diff_var, 0.0))),
                "valid_pixels": float(n),
                "align_scale": float(scale),
            }
            if pi3_sequence_align:
                row["align_shift"] = float(shift)
            out.append(row)
    return out

def mean_over_frames(per_frame_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Per-frame macro: average each metric across the frame list, skipping
    NaN/Inf values per metric. ``valid_pixels`` is reported as the SUM (a
    population statistic), not the per-frame mean."""
    if not per_frame_metrics:
        return {}
    keys: set = set()
    for m in per_frame_metrics:
        keys.update(m.keys())
    out: Dict[str, float] = {}
    for k in sorted(keys):
        if k == "valid_pixels":
            out["valid_pixels_total"] = float(sum(m.get(k, 0.0) for m in per_frame_metrics))
            continue
        vals = [m[k] for m in per_frame_metrics if k in m and math.isfinite(m[k])]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_frames"] = float(len(per_frame_metrics))
    return out



def unproject_depth_to_world_from_pose(
    pred_depth: torch.Tensor,
    pred_pose_enc: torch.Tensor,
) -> torch.Tensor:
    """Build predicted world points from predicted depth plus absT_quaR_FoV pose."""
    depth = pred_depth[..., 0].float()
    B, S, H, W = depth.shape
    pose = pred_pose_enc.float()
    c2w = T.pose_encoding_to_c2w_matrix(pose)
    _, intrinsics = pose_encoding_to_extri_intri(
        pose, image_size_hw=(int(H), int(W)), pose_encoding_type="absT_quaR_FoV", build_intrinsics=True
    )

    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=depth.device, dtype=depth.dtype),
        torch.arange(W, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
    intrinsics_inv = torch.inverse(intrinsics.to(dtype=depth.dtype))
    camera_dirs = torch.einsum("bsij,hwj->bshwi", intrinsics_inv, pixel_coords)
    camera_points = camera_dirs * depth[..., None]
    ones = torch.ones(B, S, H, W, 1, device=depth.device, dtype=depth.dtype)
    camera_points_h = torch.cat([camera_points, ones], dim=-1)
    world_points_h = torch.einsum("bsij,bshwj->bshwi", c2w.to(dtype=depth.dtype), camera_points_h)
    return world_points_h[..., :3]


def _valid_point_pairs_to_numpy(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: torch.Tensor,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_flat = pred_points.float().reshape(-1, 3)
    gt_flat = gt_points.float().reshape(-1, 3)
    flat_valid = (
        valid_mask.reshape(-1).bool()
        & torch.isfinite(pred_flat).all(dim=-1)
        & torch.isfinite(gt_flat).all(dim=-1)
    )
    idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(1)
    if int(idx.numel()) == 0:
        empty = np.empty((0, 3), dtype=np.float64)
        return empty, empty
    max_points_i = int(max_points)
    if max_points_i > 0 and int(idx.numel()) > max_points_i:
        take = torch.linspace(0, idx.numel() - 1, max_points_i, device=idx.device).round().long()
        idx = idx[take]
    pred_np = pred_flat[idx].detach().cpu().numpy().astype(np.float64, copy=False)
    gt_np = gt_flat[idx].detach().cpu().numpy().astype(np.float64, copy=False)
    return pred_np, gt_np


def _center_scale_align_points(pred_points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    pred_center = pred_points.mean(axis=0, keepdims=True)
    gt_center = gt_points.mean(axis=0, keepdims=True)
    pred_radius = np.linalg.norm(pred_points - pred_center, axis=1).mean()
    gt_radius = np.linalg.norm(gt_points - gt_center, axis=1).mean()
    if not (math.isfinite(float(pred_radius)) and math.isfinite(float(gt_radius))) or pred_radius <= EPS:
        return pred_points
    return (pred_points - pred_center) * (gt_radius / pred_radius) + gt_center


def _umeyama_align_points(pred_points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    """Align pred -> GT with Umeyama's closed-form Sim(3) estimate.

    The correspondence is by construction: both arrays are sampled from the same
    valid pixels before KDTree nearest-neighbor metrics are evaluated.
    """
    if pred_points.shape[0] < 3 or gt_points.shape[0] < 3:
        return pred_points

    pred_center = pred_points.mean(axis=0)
    gt_center = gt_points.mean(axis=0)
    pred_centered = pred_points - pred_center
    gt_centered = gt_points - gt_center
    pred_var = float(np.mean(np.sum(pred_centered * pred_centered, axis=1)))
    if not math.isfinite(pred_var) or pred_var <= EPS:
        return pred_points

    cov = (gt_centered.T @ pred_centered) / float(pred_points.shape[0])
    try:
        U, singular_values, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return pred_points

    det = float(np.linalg.det(U @ Vt))
    sign = np.ones(3, dtype=np.float64)
    if det < 0.0:
        sign[-1] = -1.0
    rotation = U @ np.diag(sign) @ Vt
    scale = float(np.sum(singular_values * sign) / pred_var)
    if not math.isfinite(scale) or scale <= EPS:
        return pred_points
    translation = gt_center - scale * (rotation @ pred_center)
    return scale * (pred_points @ rotation.T) + translation


def _rigid_transform_from_pairs(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Closed-form rigid transform from src -> dst for paired 3D points."""
    if src_points.shape[0] < 3 or dst_points.shape[0] < 3:
        return None
    src_center = src_points.mean(axis=0)
    dst_center = dst_points.mean(axis=0)
    src_centered = src_points - src_center
    dst_centered = dst_points - dst_center
    cov = (dst_centered.T @ src_centered) / float(src_points.shape[0])
    try:
        U, _, Vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    sign = np.ones(3, dtype=np.float64)
    if float(np.linalg.det(U @ Vt)) < 0.0:
        sign[-1] = -1.0
    rotation = U @ np.diag(sign) @ Vt
    translation = dst_center - rotation @ src_center
    return rotation, translation


def _rigid_align_points(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    transform = _rigid_transform_from_pairs(src_points, dst_points)
    if transform is None:
        return src_points
    rotation, translation = transform
    return src_points @ rotation.T + translation


def _icp_align_points(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.1,
    max_iterations: int = 30,
    initial_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """CUT3R-style point-to-point ICP alignment without requiring Open3D.

    CUT3R performs scale/shift normalization before Open3D point-to-point ICP.
    Here the closest equivalent is a center/scale initialization followed by a
    rigid ICP refinement against nearest neighbors in the GT cloud. An explicit
    initial_points array can be passed to reuse the same rigid ICP refinement
    after another coarse alignment.
    """
    if pred_points.shape[0] < 3 or gt_points.shape[0] < 3:
        return pred_points
    from scipy.spatial import cKDTree as KDTree

    if initial_points is None:
        aligned = _center_scale_align_points(pred_points, gt_points).astype(np.float64, copy=True)
    else:
        aligned = initial_points.astype(np.float64, copy=True)
    gt_tree = KDTree(gt_points)
    threshold_f = float(threshold)
    max_iterations_i = max(0, int(max_iterations))
    prev_error: Optional[float] = None
    for _ in range(max_iterations_i):
        try:
            distances, indices = gt_tree.query(aligned, workers=-1)
        except TypeError:
            distances, indices = gt_tree.query(aligned)
        finite = np.isfinite(distances)
        if threshold_f > 0.0:
            finite &= distances <= threshold_f
        if int(np.count_nonzero(finite)) < 3:
            break
        src_corr = aligned[finite]
        dst_corr = gt_points[indices[finite]]
        transform = _rigid_transform_from_pairs(src_corr, dst_corr)
        if transform is None:
            break
        rotation, translation = transform
        updated = aligned @ rotation.T + translation
        delta = updated - aligned
        aligned = updated
        mean_error = float(np.mean(distances[finite]))
        step = float(np.mean(np.linalg.norm(delta, axis=1)))
        if prev_error is not None and abs(prev_error - mean_error) <= 1e-7 and step <= 1e-7:
            break
        prev_error = mean_error
    return aligned


def _pi3_icp_align_points(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.1,
    max_iterations: int = 30,
) -> np.ndarray:
    """Pi3 mv_recon alignment: Umeyama Sim(3) coarse alignment, then rigid ICP refinement."""
    if pred_points.shape[0] < 3 or gt_points.shape[0] < 3:
        return pred_points
    aligned = _umeyama_align_points(pred_points, gt_points)
    return _icp_align_points(
        pred_points,
        gt_points,
        threshold=threshold,
        max_iterations=max_iterations,
        initial_points=aligned,
    )


def _kdtree_query_distances(tree, points: np.ndarray) -> np.ndarray:
    try:
        distances, _ = tree.query(points, workers=-1)
    except TypeError:
        distances, _ = tree.query(points)
    return distances


def compute_pointcloud_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    valid_mask: torch.Tensor,
    max_points: int = 100000,
    align: str = "pi3_icp",
    icp_threshold: float = 0.1,
    icp_max_iterations: int = 30,
) -> Optional[Dict[str, float]]:
    """LoGeR-style point cloud metrics using nearest-neighbor distances.

    ACC queries each predicted point against the GT cloud; Completeness queries
    each GT point against the predicted cloud; CD is their average.
    """
    from scipy.spatial import cKDTree as KDTree

    pred_np, gt_np = _valid_point_pairs_to_numpy(pred_points, gt_points, valid_mask, max_points)
    if pred_np.shape[0] < 2 or gt_np.shape[0] < 2:
        return None
    if align == "umeyama":
        pred_np = _umeyama_align_points(pred_np, gt_np)
    elif align == "icp":
        pred_np = _icp_align_points(
            pred_np,
            gt_np,
            threshold=float(icp_threshold),
            max_iterations=int(icp_max_iterations),
        )
    elif align == "pi3_icp":
        pred_np = _pi3_icp_align_points(
            pred_np,
            gt_np,
            threshold=float(icp_threshold),
            max_iterations=int(icp_max_iterations),
        )
    elif align == "scale_center":
        pred_np = _center_scale_align_points(pred_np, gt_np)
    elif align != "none":
        raise ValueError(f"Unknown pointcloud_align mode: {align}")

    gt_tree = KDTree(gt_np)
    pred_to_gt = _kdtree_query_distances(gt_tree, pred_np)
    pred_tree = KDTree(pred_np)
    gt_to_pred = _kdtree_query_distances(pred_tree, gt_np)
    acc = float(np.mean(pred_to_gt))
    completeness = float(np.mean(gt_to_pred))
    return {
        "ACC": acc,
        "Completeness": completeness,
        "CD": float(0.5 * (acc + completeness)),
        "n_pred_points": float(pred_np.shape[0]),
        "n_gt_points": float(gt_np.shape[0]),
    }

def metric_group_from_sample_mode(sample_mode: str) -> str:
    lower = sample_mode.lower()
    if "realsense" in lower or lower.startswith("w"):
        return "realsense"
    if "surround" in lower or lower.startswith("t"):
        return "surround"
    return "other"


def summarize_metric_group(
    depth_frames: List[Dict[str, float]],
    traj_rows: List[Dict[str, float]],
    fov_frames: List[Dict[str, float]],
    pointcloud_rows: List[Dict[str, float]],
    clips_evaluated: int,
) -> Dict[str, object]:
    out: Dict[str, object] = {"clips_evaluated": int(clips_evaluated)}
    if depth_frames:
        out["depth"] = mean_over_frames(depth_frames)
    if traj_rows or fov_frames:
        cam: Dict[str, float] = {}
        if traj_rows:
            cam["ate_rmse_mean"] = float(np.mean([m["ate_rmse"] for m in traj_rows]))
            cam["ate_rmse_median"] = float(np.median([m["ate_rmse"] for m in traj_rows]))
            cam["rpe_trans_rmse_mean"] = float(np.mean([m["rpe_trans_rmse"] for m in traj_rows]))
            cam["rpe_trans_rmse_median"] = float(np.median([m["rpe_trans_rmse"] for m in traj_rows]))
            cam["rpe_rot_rmse_deg_mean"] = float(np.mean([m["rpe_rot_rmse_deg"] for m in traj_rows]))
            cam["rpe_rot_rmse_deg_median"] = float(np.median([m["rpe_rot_rmse_deg"] for m in traj_rows]))
            cam["n_sequences_for_traj"] = int(len(traj_rows))
        if fov_frames:
            fov_macro = mean_over_frames(fov_frames)
            cam["fov_h_deg_mae"] = float(fov_macro.get("fov_h_deg_mae", float("nan")))
            cam["fov_w_deg_mae"] = float(fov_macro.get("fov_w_deg_mae", float("nan")))
            cam["n_frames_for_fov"] = int(fov_macro.get("n_frames", 0))
        out["camera"] = cam
    if pointcloud_rows:
        pc: Dict[str, float] = {}
        for key in ("ACC", "Completeness", "CD"):
            vals = [m[key] for m in pointcloud_rows if key in m and math.isfinite(m[key])]
            if vals:
                pc[key] = float(np.mean(vals))
        pc["n_clouds"] = int(len(pointcloud_rows))
        pc["n_pred_points_mean"] = float(np.mean([m["n_pred_points"] for m in pointcloud_rows]))
        pc["n_gt_points_mean"] = float(np.mean([m["n_gt_points"] for m in pointcloud_rows]))
        out["pointcloud"] = pc
    return out

def _se3_3x4_to_4x4_np(ext_3x4: np.ndarray) -> np.ndarray:
    """[N, 3, 4] -> [N, 4, 4] with last row = [0, 0, 0, 1]."""
    n = ext_3x4.shape[0]
    out = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    out[:, :3, :] = ext_3x4.astype(np.float64)
    return out


def compute_camera_metrics_evo(
    pred_pose_enc: torch.Tensor,        # [S, 9]
    gt_extrinsics_w2c: torch.Tensor,    # [S, 3, 4] OpenCV W2C
    image_hw: Tuple[int, int],
    valid_frame_mask: torch.Tensor,     # [S] bool
    align_mode: str = "sim3",
) -> Optional[Dict[str, float]]:
    """Trajectory evaluation via the ``evo`` library.

    Both predicted and GT trajectories are converted to camera-to-world
    poses and passed through ``evo.main_ape.ape`` / ``evo.main_rpe.rpe`` with
    Sim(3) alignment controlled by camera_align. Returns:

      - ``ate_rmse``         : APE w.r.t. translation_part, RMSE (meters in
                               the anchor-normalized coordinate system)
      - ``rpe_trans_rmse``   : RPE w.r.t. translation_part, delta=1 frame,
                               all_pairs=True, RMSE
      - ``rpe_rot_rmse_deg`` : RPE w.r.t. rotation_angle_deg, delta=1, all_pairs

    See ``CUT3R/eval/relpose/evo_utils.py:eval_metrics`` for the reference
    implementation we mirror here.
    """
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    from evo.core import sync
    from evo.core.metrics import PoseRelation, Unit
    from evo.core.trajectory import PoseTrajectory3D

    if align_mode == "sim3":
        do_align = True
    elif align_mode == "none":
        do_align = False
    else:
        raise ValueError(f"Unknown camera_align mode: {align_mode}")

    mask = valid_frame_mask.bool()
    if int(mask.sum().item()) < 3:
        return None

    pred_ext_c2w, _ = pose_encoding_to_extri_intri(
        pred_pose_enc.unsqueeze(0).float(),
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=False,
    )  # [1, S, 3, 4]
    gt_ext_c2w = T.w2c_to_c2w_extrinsics(gt_extrinsics_w2c.unsqueeze(0).float())  # [1, S, 3, 4]

    pred_c2w_np = pred_ext_c2w[0].detach().cpu().numpy()
    gt_c2w_np = gt_ext_c2w[0].detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    pred_4x4 = _se3_3x4_to_4x4_np(pred_c2w_np)[mask_np]
    gt_4x4 = _se3_3x4_to_4x4_np(gt_c2w_np)[mask_np]

    if not (np.all(np.isfinite(pred_4x4)) and np.all(np.isfinite(gt_4x4))):
        return None
    if pred_4x4.shape[0] < 3:
        return None

    timestamps = np.arange(pred_4x4.shape[0], dtype=np.float64)
    pred_traj = PoseTrajectory3D(poses_se3=list(pred_4x4), timestamps=timestamps)
    gt_traj = PoseTrajectory3D(poses_se3=list(gt_4x4), timestamps=timestamps.copy())

    # evo aligns trajectories to a common timestamp grid; with our identical
    # synthetic timestamps this is a no-op but keeps API parity with CUT3R.
    gt_traj, pred_traj = sync.associate_trajectories(gt_traj, pred_traj)

    try:
        ate_result = main_ape.ape(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=do_align, correct_scale=do_align,
        )
        rpe_rot_result = main_rpe.rpe(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.rotation_angle_deg,
            align=do_align, correct_scale=do_align,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
        )
        rpe_trans_result = main_rpe.rpe(
            gt_traj, pred_traj,
            est_name="traj",
            pose_relation=PoseRelation.translation_part,
            align=do_align, correct_scale=do_align,
            delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True,
        )
    except Exception as exc:  # noqa: BLE001 - evo can fail on degenerate trajectories
        # Mirrors CUT3R's launch.py which catches "Degenerate covariance rank" etc.
        msg = str(exc).splitlines()[0][:120]
        print(f"[eval] evo trajectory metrics failed ({msg}); skipping scene")
        return None

    return {
        "ate_rmse": float(ate_result.stats["rmse"]),
        "rpe_trans_rmse": float(rpe_trans_result.stats["rmse"]),
        "rpe_rot_rmse_deg": float(rpe_rot_result.stats["rmse"]),
        "n_frames_used": float(pred_4x4.shape[0]),
    }


def compute_fov_per_frame_errors(
    pred_pose_enc: torch.Tensor,       # [S, 9]
    gt_intrinsics: torch.Tensor,       # [S, 3, 3]
    image_hw: Tuple[int, int],
    valid_frame_mask: torch.Tensor,    # [S] bool
) -> Optional[List[Dict[str, float]]]:
    """Per-frame |pred_fov - gt_fov| in degrees, for both axes. Independent
    of trajectory metrics so it can be averaged per-frame across all scenes."""
    mask = valid_frame_mask.bool()
    if int(mask.sum().item()) == 0:
        return None
    _, pred_intr = pose_encoding_to_extri_intri(
        pred_pose_enc.unsqueeze(0).float(),
        image_size_hw=image_hw,
        pose_encoding_type="absT_quaR_FoV",
        build_intrinsics=True,
    )
    pred_intr = pred_intr[0]  # [S, 3, 3]
    H, W = image_hw
    pred_fov_h = 2 * torch.atan((H / 2.0) / pred_intr[..., 1, 1].clamp(min=1e-6))
    pred_fov_w = 2 * torch.atan((W / 2.0) / pred_intr[..., 0, 0].clamp(min=1e-6))
    gt_fov_h = 2 * torch.atan((H / 2.0) / gt_intrinsics[..., 1, 1].clamp(min=1e-6))
    gt_fov_w = 2 * torch.atan((W / 2.0) / gt_intrinsics[..., 0, 0].clamp(min=1e-6))
    fov_h_err_deg = torch.rad2deg((pred_fov_h - gt_fov_h).abs())[mask].detach().cpu().numpy()
    fov_w_err_deg = torch.rad2deg((pred_fov_w - gt_fov_w).abs())[mask].detach().cpu().numpy()
    return [{"fov_h_deg_mae": float(h), "fov_w_deg_mae": float(w)}
            for h, w in zip(fov_h_err_deg, fov_w_err_deg)]



# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> Dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    base_output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint).parent / "eval" / Path(args.checkpoint).stem
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] base output_dir = {base_output_dir}")

    model = build_eval_model(args, device)
    print(f"[eval] checkpoint loaded: {args.checkpoint}")

    if args.eval_strategy == "train_default":
        modes = ["default"]
    elif args.eval_strategy == "manip_track":
        modes = ["manip_track"]
    elif args.eval_strategy == "wrist_track":
        modes = ["wrist_track"]
    elif args.eval_strategy == "random_static_track":
        modes = ["random_static_track"]
    elif args.eval_strategy == "both":
        modes = ["wrist_track", "random_static_track"]
    else:
        raise ValueError(f"Unknown eval_strategy: {args.eval_strategy}")

    overall: Dict[str, object] = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "eval_strategy": args.eval_strategy,
        "eval_num_frames": int(args.eval_num_frames),
        "geometry_normalization": str(getattr(args, 'geometry_normalization', 'none')),
        "camera_align": str(getattr(args, 'camera_align', 'sim3')),
        "depth_frames_chunk_size": int(args.depth_frames_chunk_size),
        "pointcloud_metrics": bool(getattr(args, "pointcloud_metrics", False)),
        "pointcloud_max_points": int(getattr(args, "pointcloud_max_points", 100000)),
        "pointcloud_align": str(getattr(args, "pointcloud_align", "pi3_icp")),
        "pointcloud_icp_threshold": float(getattr(args, "pointcloud_icp_threshold", 0.1)),
        "pointcloud_icp_max_iterations": int(getattr(args, "pointcloud_icp_max_iterations", 30)),
        "eval_wrist_camera_name": str(args.eval_wrist_camera_name),
        "eval_surround_camera_name": str(getattr(args, "eval_surround_camera_name", "")),
        "modes": {},
    }

    for mode_name in modes:
        # When running multiple modes, give each its own subdir; otherwise write
        # straight into base_output_dir to keep the single-mode case tidy.
        out_dir = base_output_dir if len(modes) == 1 else (base_output_dir / mode_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        print()
        print(f"[eval] === mode: {mode_name} ===")
        per_mode = _evaluate_one_mode(args, model, device, mode_name, out_dir)
        cast(Dict[str, object], overall["modes"])[mode_name] = per_mode

    summary_path = base_output_dir / "metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    print()
    print(f"[eval] wrote {summary_path}")
    return overall


@torch.no_grad()
def _evaluate_one_mode(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    eval_mode: str,
    output_dir: Path,
) -> Dict[str, object]:
    loader, eval_scenes = build_eval_loader(args, eval_mode=eval_mode)
    if eval_mode == "default":
        print(f"[eval] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
              f"eval_mode={eval_mode}")
    else:
        wrist_cam = args.eval_wrist_camera_name if eval_mode in {"manip_track", "wrist_track"} else "n/a"
        surround_cam = getattr(args, "eval_surround_camera_name", "") if eval_mode == "manip_track" else "n/a"
        surround_desc = surround_cam or ("all_6_surround" if eval_mode == "manip_track" else "n/a")
        print(f"[eval] split={args.split}, scenes={len(eval_scenes)}, batches={len(loader)}, "
              f"eval_mode={eval_mode}, num_frames={args.eval_num_frames}, "
              f"wrist_camera={wrist_cam}, surround_camera={surround_desc}")

    amp_enabled = bool(getattr(args, "amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "bf16") == "bf16" else torch.float16
    amp_ctx = (torch.amp.autocast(device_type=device.type, dtype=amp_dtype)
               if amp_enabled else contextlib.nullcontext())

    # Per-frame macro: accumulate one dict per frame across all scenes, average at the end.
    all_depth_frames: List[Dict[str, float]] = []
    # CUT3R-style trajectory metrics: one dict per scene, averaged across scenes (per-sequence macro).
    per_scene_traj: List[Dict[str, float]] = []
    # FoV: per-frame metric (independent of trajectory).
    all_fov_frames: List[Dict[str, float]] = []
    # LoGeR-style point cloud reconstruction metrics: one dict per clip.
    all_pointcloud_rows: List[Dict[str, float]] = []

    group_depth_frames: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_traj: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_fov_frames: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_pointcloud_rows: Dict[str, List[Dict[str, float]]] = {"realsense": [], "surround": [], "other": []}
    group_counts: Dict[str, int] = {"realsense": 0, "surround": 0, "other": 0}

    per_scene_rows: List[Dict[str, object]] = []
    skipped = 0
    evaluated = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images_t = cast(torch.Tensor, batch["images"])
        geometry_normalization = str(getattr(args, 'geometry_normalization', 'none'))
        if geometry_normalization == "native":
            if getattr(args, "canonicalize_first_frame", True):
                batch = T.canonicalize_to_first_frame(batch)
            if getattr(args, "normalize_scene", True):
                batch = T.normalize_scene_batch(
                    batch,
                    num_anchor_frames=min(args.num_scale_frames, int(images_t.shape[1])),
                )
        elif geometry_normalization == "vggt_independent":
            batch = vggt_normalize_gt_batch(batch)
        elif geometry_normalization == "none":
            batch = dict(batch)
        else:
            raise ValueError(f"Unknown geometry_normalization: {geometry_normalization}")
        batch = T.to_device(batch, device)
        images_t = cast(torch.Tensor, batch["images"])
        depths_t = cast(torch.Tensor, batch["depths"])
        point_masks_t = cast(torch.Tensor, batch["point_masks"])
        extrinsics_t = cast(torch.Tensor, batch["extrinsics"])
        intrinsics_t = cast(torch.Tensor, batch["intrinsics"])
        world_points_t = cast(torch.Tensor, batch["world_points"])

        try:
            model.clean_kv_cache()
            with amp_ctx:
                predictions = model(
                    images_t,
                    num_frame_for_scale=min(args.num_scale_frames, int(images_t.shape[1])),
                    num_frame_per_block=args.num_frame_per_block,
                    depth_frames_chunk_size=args.depth_frames_chunk_size,
                    causal_inference=True,
                )
            model.clean_kv_cache()
        except Exception as exc:
            skipped += 1
            print(f"[eval] batch {batch_idx} forward failed ({exc}); skipping")
            continue

        if geometry_normalization == "vggt_independent":
            predictions = vggt_normalize_predictions(predictions, point_masks_t.bool())

        scene_field = batch["scene"]
        scene_name = scene_field[0] if isinstance(scene_field, list) else str(scene_field)
        mode_field = batch["sample_mode"]
        sample_mode = mode_field[0] if isinstance(mode_field, list) else str(mode_field)
        metric_group = metric_group_from_sample_mode(sample_mode)
        if metric_group not in group_counts:
            group_counts[metric_group] = 0
            group_depth_frames[metric_group] = []
            group_traj[metric_group] = []
            group_fov_frames[metric_group] = []
            group_pointcloud_rows[metric_group] = []

        # Per-frame valid mask (matches train.py's compute_depth_loss criterion).
        per_frame_valid = point_masks_t.sum(dim=(-1, -2)) > int(getattr(args, "min_valid_pixels", 100))

        # Depth metrics: one dict per frame, accumulated globally.
        depth_row: Dict[str, object] = {"scene": scene_name, "sample_mode": sample_mode,
                                        "metric_group": metric_group,
                                        "n_frames": int(images_t.shape[1])}
        gt_scale_t = batch.get("vggt_gt_scale")
        if torch.is_tensor(gt_scale_t):
            depth_row["gt_geometry_scale"] = float(gt_scale_t[0].detach().cpu().item())
        pred_scale_t = predictions.get("vggt_pred_scale")
        if torch.is_tensor(pred_scale_t):
            depth_row["pred_geometry_scale"] = float(pred_scale_t[0].detach().cpu().item())
        if "depth" in predictions:
            pred_depth = predictions["depth"].float()  # [B, S, H, W, 1]
            gt_depth = depths_t.float()
            mask = point_masks_t.bool() & per_frame_valid[..., None, None]
            scene_frames = compute_depth_per_frame_metrics(
                pred_depth, gt_depth, mask, align=str(args.depth_align),
            )
            all_depth_frames.extend(scene_frames)
            group_depth_frames[metric_group].extend(scene_frames)
            if scene_frames:
                scene_macro = mean_over_frames(scene_frames)
                for k, v in scene_macro.items():
                    depth_row[f"depth_{k}"] = v
            else:
                depth_row["depth_AbsRel"] = float("nan")

        # Camera trajectory metrics (CUT3R-style: ATE / RPE-trans / RPE-rot via evo, Sim(3) aligned).
        if "pose_enc" in predictions:
            pred_pose = predictions["pose_enc"][0].float()  # [S, 9]
            image_hw = (int(images_t.shape[-2]), int(images_t.shape[-1]))
            traj_metrics = compute_camera_metrics_evo(
                pred_pose,
                extrinsics_t[0],
                image_hw=image_hw,
                valid_frame_mask=per_frame_valid[0],
                align_mode=str(getattr(args, 'camera_align', 'sim3')),
            )
            if traj_metrics is not None:
                per_scene_traj.append(traj_metrics)
                group_traj[metric_group].append(traj_metrics)
                depth_row["cam_ate_rmse"] = traj_metrics["ate_rmse"]
                depth_row["cam_rpe_trans_rmse"] = traj_metrics["rpe_trans_rmse"]
                depth_row["cam_rpe_rot_rmse_deg"] = traj_metrics["rpe_rot_rmse_deg"]
                depth_row["cam_n_frames_used"] = traj_metrics["n_frames_used"]

            # FoV: per-frame, scene-independent.
            fov_rows = compute_fov_per_frame_errors(
                pred_pose,
                intrinsics_t[0],
                image_hw=image_hw,
                valid_frame_mask=per_frame_valid[0],
            )
            if fov_rows is not None:
                all_fov_frames.extend(fov_rows)
                group_fov_frames[metric_group].extend(fov_rows)
                depth_row["cam_fov_h_deg_mae"] = float(np.mean([r["fov_h_deg_mae"] for r in fov_rows]))
                depth_row["cam_fov_w_deg_mae"] = float(np.mean([r["fov_w_deg_mae"] for r in fov_rows]))

        if bool(getattr(args, "pointcloud_metrics", False)):
            pred_world_points = predictions.get("world_points")
            if pred_world_points is None and "depth" in predictions and "pose_enc" in predictions:
                pred_world_points = unproject_depth_to_world_from_pose(
                    predictions["depth"].float(),
                    predictions["pose_enc"].float(),
                )
            if pred_world_points is not None:
                pc_valid = point_masks_t.bool() & per_frame_valid[..., None, None]
                try:
                    pc_metrics = compute_pointcloud_metrics(
                        pred_world_points.float(),
                        world_points_t.float(),
                        pc_valid,
                        max_points=int(getattr(args, "pointcloud_max_points", 100000)),
                        align=str(getattr(args, "pointcloud_align", "pi3_icp")),
                        icp_threshold=float(getattr(args, "pointcloud_icp_threshold", 0.1)),
                        icp_max_iterations=int(getattr(args, "pointcloud_icp_max_iterations", 30)),
                    )
                except Exception as exc:  # noqa: BLE001 - scipy/KDTree can fail on degenerate clouds
                    pc_metrics = None
                    print(f"[eval] pointcloud metrics failed for {scene_name} ({str(exc).splitlines()[0][:120]}); skipping")
                if pc_metrics is not None:
                    all_pointcloud_rows.append(pc_metrics)
                    group_pointcloud_rows[metric_group].append(pc_metrics)
                    for k, v in pc_metrics.items():
                        depth_row[f"pc_{k}"] = v

        if args.save_predictions and "depth" in predictions:
            pred_path = output_dir / "predictions" / f"{scene_name}.npz"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            depth_pred_np = predictions["depth"][0, ..., 0].float().detach().cpu().numpy().astype(np.float32)
            depth_conf_pred = predictions.get("depth_conf")
            depth_conf_np = (depth_conf_pred[0].float().detach().cpu().numpy().astype(np.float32)
                             if depth_conf_pred is not None else np.empty(0, dtype=np.float32))
            pose_enc_pred = predictions.get("pose_enc")
            pose_enc_np = (pose_enc_pred[0].float().detach().cpu().numpy().astype(np.float32)
                           if pose_enc_pred is not None else np.empty(0, dtype=np.float32))
            np.savez_compressed(
                pred_path,
                depth=depth_pred_np,
                depth_conf=depth_conf_np,
                pose_enc=pose_enc_np,
                gt_depth=depths_t[0].float().detach().cpu().numpy().astype(np.float32),
                point_masks=point_masks_t[0].detach().cpu().numpy().astype(np.uint8),
                extrinsics=extrinsics_t[0].float().detach().cpu().numpy().astype(np.float32),
                intrinsics=intrinsics_t[0].float().detach().cpu().numpy().astype(np.float32),
            )

        per_scene_rows.append(depth_row)
        evaluated += 1
        group_counts[metric_group] += 1
        if args.print_every > 0 and (batch_idx % args.print_every == 0 or batch_idx == len(loader)):
            running_depth = mean_over_frames(all_depth_frames)
            running_msg = (f"AbsRel={running_depth.get('AbsRel', float('nan')):.4f} "
                           f"RMSE={running_depth.get('RMSE', float('nan')):.4f} "
                           f"d1={running_depth.get('delta<1.25', float('nan')):.4f}") if running_depth else "no-depth"
            cam_msg = ""
            if per_scene_traj:
                ate_avg = float(np.mean([m["ate_rmse"] for m in per_scene_traj]))
                rpe_t_avg = float(np.mean([m["rpe_trans_rmse"] for m in per_scene_traj]))
                rpe_r_avg = float(np.mean([m["rpe_rot_rmse_deg"] for m in per_scene_traj]))
                cam_msg = f" ATE={ate_avg:.4f} RPEt={rpe_t_avg:.4f} RPEr={rpe_r_avg:.3f}deg"
            pc_msg = ""
            if all_pointcloud_rows:
                pc_acc = float(np.mean([m["ACC"] for m in all_pointcloud_rows]))
                pc_comp = float(np.mean([m["Completeness"] for m in all_pointcloud_rows]))
                pc_cd = float(np.mean([m["CD"] for m in all_pointcloud_rows]))
                pc_msg = f" PC_ACC={pc_acc:.4f} PC_Comp={pc_comp:.4f} PC_CD={pc_cd:.4f}"
            print(f"[eval] [{batch_idx}/{len(loader)}] scene={scene_name} mode={sample_mode} "
                  f"frames={int(images_t.shape[1])} {running_msg}{cam_msg}{pc_msg} skipped={skipped}")

    if evaluated == 0:
        raise RuntimeError(f"No batches evaluated for eval_mode={eval_mode}.")

    summary: Dict[str, object] = {
        "eval_mode": eval_mode,
        "scenes_evaluated": evaluated,
        "scenes_skipped": skipped,
        "aggregation": "overall plus independent camera-family groups; depth=per-frame macro; trajectory=per-sequence macro (CUT3R-style); fov=per-frame macro; pointcloud=per-clip macro (LoGeR-style NN distances)",
        "depth_align": str(args.depth_align),
        "geometry_normalization": str(getattr(args, 'geometry_normalization', 'none')),
        "camera_align": str(getattr(args, 'camera_align', 'sim3')),
        "pointcloud_metrics": bool(getattr(args, "pointcloud_metrics", False)),
        "pointcloud_max_points": int(getattr(args, "pointcloud_max_points", 100000)),
        "pointcloud_align": str(getattr(args, "pointcloud_align", "pi3_icp")),
        "pointcloud_icp_threshold": float(getattr(args, "pointcloud_icp_threshold", 0.1)),
        "pointcloud_icp_max_iterations": int(getattr(args, "pointcloud_icp_max_iterations", 30)),
        "long5_surround_tracks_per_scene": 6 if eval_mode == "manip_track" and not getattr(args, "eval_surround_camera_name", "") else 1,
    }
    overall_summary = summarize_metric_group(
        all_depth_frames, per_scene_traj, all_fov_frames, all_pointcloud_rows, evaluated
    )
    if "depth" in overall_summary:
        summary["depth"] = overall_summary["depth"]
    if "camera" in overall_summary:
        summary["camera"] = overall_summary["camera"]
    if "pointcloud" in overall_summary:
        summary["pointcloud"] = overall_summary["pointcloud"]
    summary["overall"] = overall_summary

    groups: Dict[str, object] = {}
    for group_name in ("realsense", "surround", "other"):
        if group_counts.get(group_name, 0) <= 0:
            continue
        groups[group_name] = summarize_metric_group(
            group_depth_frames[group_name],
            group_traj[group_name],
            group_fov_frames[group_name],
            group_pointcloud_rows[group_name],
            group_counts[group_name],
        )
    summary["groups"] = groups

    summary_path = output_dir / "metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"[eval] wrote {summary_path}")

    if args.per_scene_csv and per_scene_rows:
        keys = sorted({k for row in per_scene_rows for k in row.keys()})
        csv_path = output_dir / "per_scene.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in per_scene_rows:
                writer.writerow(row)
        print(f"[eval] wrote {csv_path}")

    print()
    print("=" * 60)
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Split         : {args.split} ({evaluated} scenes evaluated, {skipped} skipped)")
    if eval_mode == "default":
        print(f"Eval mode     : {eval_mode}")
    else:
        wrist_cam = args.eval_wrist_camera_name if eval_mode == "wrist_track" else "n/a"
        print(f"Eval mode     : {eval_mode}  num_frames={args.eval_num_frames}  wrist_cam={wrist_cam}")
    depth_summary = cast(Optional[Dict[str, float]], summary.get("depth"))
    if depth_summary is not None:
        print(f"Depth metrics (geometry_normalization={getattr(args, 'geometry_normalization', 'none')}, depth_align={args.depth_align}):")
        for k, v in depth_summary.items():
            print(f"  {k:18s} = {v:.6f}" if isinstance(v, float) else f"  {k:18s} = {v}")
    camera_summary = cast(Optional[Dict[str, float]], summary.get("camera"))
    if camera_summary is not None:
        print(f"Camera metrics (ATE/RPE via evo, camera_align={getattr(args, 'camera_align', 'sim3')}; FoV from intrinsics):")
        for k, v in camera_summary.items():
            print(f"  {k:22s} = {v:.6f}" if isinstance(v, float) else f"  {k:22s} = {v}")
    pointcloud_summary = cast(Optional[Dict[str, float]], summary.get("pointcloud"))
    if pointcloud_summary is not None:
        print(f"Point cloud metrics (LoGeR-style NN, align={getattr(args, 'pointcloud_align', 'pi3_icp')}):")
        for k, v in pointcloud_summary.items():
            print(f"  {k:22s} = {v:.6f}" if isinstance(v, float) else f"  {k:22s} = {v}")
    print("=" * 60)
    return summary


def main() -> None:
    eval_args = build_argparser().parse_args()
    args = coerce_args_from_json(eval_args)
    evaluate(args)


if __name__ == "__main__":
    main()
