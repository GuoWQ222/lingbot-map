#!/usr/bin/env python3
"""Split Robo3R multi-camera scene data into camera-specific subdirectories.

Robo3R stores camera-specific files directly under each modality directory:

    scene_00000000/rgb/0000_00.jpg
    scene_00000000/rgb/0000_01.jpg

This script reorganizes those files into:

    scene_00000000/rgb/camera1/0000_00.jpg
    scene_00000000/rgb/camera2/0000_01.jpg

It also handles the camera-independent per-frame data by copying it into every
camera directory:

    scene_00000000/qpos/camera1/0000.npy
    scene_00000000/qpos/camera2/0000.npy

For cam_param.npy, it writes one split parameter file per camera:

    scene_00000000/cam_param/camera1/cam_param.npy

By default the script runs in dry-run mode. Pass --apply to move files.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np


DEFAULT_ROOT = (
    "/cpfs/user/guowenqi/robo3r/20kScenes_dtc-objaverse_in-gripper/"
    "20kScenes_dtc-objaverse_in-gripper"
)
DEFAULT_MODALITIES = (
    "rgb",
    "depth",
    "mask",
    "qpos",
    "ee_pose",
    "keypoint_3d",
    "keypoint_2d",
    "cam_param.npy",
)
CAMERA_MODALITIES = {"rgb", "depth", "mask", "keypoint_2d"}
SHARED_MODALITIES = {"qpos", "ee_pose", "keypoint_3d"}
CAM_PARAM_MODALITY = "cam_param.npy"
MODALITY_ALIASES = {
    "all": "all",
    "*": "all",
    "rgb": "rgb",
    "depth": "depth",
    "mask": "mask",
    "qpos": "qpos",
    "eepose": "ee_pose",
    "ee_pose": "ee_pose",
    "ee-pose": "ee_pose",
    "keypoint_3d": "keypoint_3d",
    "keypoint-3d": "keypoint_3d",
    "keypoint3d": "keypoint_3d",
    "keypoint_2d": "keypoint_2d",
    "keypoint-2d": "keypoint_2d",
    "keypoint2d": "keypoint_2d",
    "cam_param": CAM_PARAM_MODALITY,
    "cam_param.npy": CAM_PARAM_MODALITY,
}
CAMERA_FILE_RE = re.compile(r"^(?P<frame>\d+)_(?P<camera>\d+)(?P<suffix>\.[^.]+)$")
CAMERA_DIR_RE = re.compile(r"^camera(?P<number>\d+)$")


@dataclass
class SplitStats:
    scenes_seen: int = 0
    scenes_changed: int = 0
    files_seen: int = 0
    files_matched: int = 0
    files_moved: int = 0
    files_copied: int = 0
    files_skipped: int = 0
    missing_modalities: int = 0

    def add(self, other: "SplitStats") -> None:
        self.scenes_seen += other.scenes_seen
        self.scenes_changed += other.scenes_changed
        self.files_seen += other.files_seen
        self.files_matched += other.files_matched
        self.files_moved += other.files_moved
        self.files_copied += other.files_copied
        self.files_skipped += other.files_skipped
        self.missing_modalities += other.missing_modalities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reorganize Robo3R camera-specific files from modality roots into "
            "camera1, camera2, ... subdirectories."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help="Dataset root containing scene_* directories.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        help=(
            "Scene subdirectories to split. Defaults to rgb depth mask. "
            "You can add keypoint_2d because it also uses frame_camera names."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move or copy files. Without this flag the script only reports a dry run.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them. Only takes effect with --apply.",
    )
    parser.add_argument(
        "--camera-base",
        type=int,
        default=1,
        help="Numbering base for output directory names: camera_idx 00 -> camera{base}.",
    )
    parser.add_argument(
        "--rename-frame-only",
        action="store_true",
        help=(
            "Rename files inside camera directories from 0000_00.jpg to 0000.jpg. "
            "The default preserves original filenames."
        ),
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Process at most this many scenes. Use 0 for all scenes.",
    )
    parser.add_argument(
        "--scene-glob",
        default="scene_*",
        help="Glob used to discover scene directories under --root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each planned operation.",
    )
    return parser.parse_args()


def iter_scene_dirs(root: Path, scene_glob: str, max_scenes: int) -> Iterator[Path]:
    count = 0
    for scene_dir in sorted(root.glob(scene_glob)):
        if not scene_dir.is_dir():
            continue
        yield scene_dir
        count += 1
        if max_scenes > 0 and count >= max_scenes:
            break


def iter_direct_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file():
            yield path


def destination_for(
    src: Path,
    camera_idx: int,
    camera_base: int,
    rename_frame_only: bool,
) -> Path:
    camera_dir = src.parent / f"camera{camera_idx + camera_base}"
    if rename_frame_only:
        match = CAMERA_FILE_RE.match(src.name)
        if match is None:
            raise ValueError(f"Cannot parse camera filename: {src}")
        filename = f"{match.group('frame')}{match.group('suffix')}"
    else:
        filename = src.name
    return camera_dir / filename


def split_modality(
    modality_dir: Path,
    apply: bool,
    copy: bool,
    camera_base: int,
    rename_frame_only: bool,
    verbose: bool,
) -> SplitStats:
    stats = SplitStats()
    if not modality_dir.is_dir():
        stats.missing_modalities += 1
        return stats

    for src in iter_direct_files(modality_dir):
        stats.files_seen += 1
        match = CAMERA_FILE_RE.match(src.name)
        if match is None:
            stats.files_skipped += 1
            continue

        stats.files_matched += 1
        camera_idx = int(match.group("camera"))
        dst = destination_for(
            src=src,
            camera_idx=camera_idx,
            camera_base=camera_base,
            rename_frame_only=rename_frame_only,
        )

        if dst.exists():
            if src.resolve() == dst.resolve():
                stats.files_skipped += 1
                continue
            raise FileExistsError(f"Refusing to overwrite existing file: {dst}")

        action = "copy" if copy else "move"
        if verbose:
            print(f"{action}: {src} -> {dst}")

        if not apply:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(src, dst)
            stats.files_copied += 1
        else:
            shutil.move(str(src), str(dst))
            stats.files_moved += 1

    return stats


def split_scene(
    scene_dir: Path,
    modalities: Sequence[str],
    apply: bool,
    copy: bool,
    camera_base: int,
    rename_frame_only: bool,
    verbose: bool,
) -> SplitStats:
    scene_stats = SplitStats(scenes_seen=1)
    before = scene_stats.files_moved + scene_stats.files_copied + scene_stats.files_matched

    for modality in modalities:
        modality_stats = split_modality(
            modality_dir=scene_dir / modality,
            apply=apply,
            copy=copy,
            camera_base=camera_base,
            rename_frame_only=rename_frame_only,
            verbose=verbose,
        )
        scene_stats.add(modality_stats)

    after = scene_stats.files_moved + scene_stats.files_copied + scene_stats.files_matched
    if after > before:
        scene_stats.scenes_changed = 1
    return scene_stats


def main() -> int:
    args = parse_args()
    if args.camera_base < 0:
        raise ValueError("--camera-base must be non-negative")
    if args.copy and not args.apply:
        print("Note: --copy has no effect without --apply; running dry-run.")

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")

    mode = "copy" if args.copy else "move"
    dry_run = not args.apply
    print(f"root: {root}")
    print(f"modalities: {' '.join(args.modalities)}")
    print(f"mode: {'dry-run ' + mode if dry_run else mode}")
    print(f"camera dirs: source camera 00 -> camera{args.camera_base}")
    if args.rename_frame_only:
        print("filename mode: 0000_00.ext -> camera*/0000.ext")
    else:
        print("filename mode: preserve original names")

    total = SplitStats()
    for scene_dir in iter_scene_dirs(root, args.scene_glob, args.max_scenes):
        stats = split_scene(
            scene_dir=scene_dir,
            modalities=args.modalities,
            apply=args.apply,
            copy=args.copy,
            camera_base=args.camera_base,
            rename_frame_only=args.rename_frame_only,
            verbose=args.verbose,
        )
        total.add(stats)
        if args.verbose:
            print(
                f"scene summary: {scene_dir.name} "
                f"matched={stats.files_matched} moved={stats.files_moved} "
                f"copied={stats.files_copied} skipped={stats.files_skipped}"
            )

    print("\nsummary")
    print(f"  scenes seen: {total.scenes_seen}")
    print(f"  scenes with matching files: {total.scenes_changed}")
    print(f"  files seen: {total.files_seen}")
    print(f"  files matched: {total.files_matched}")
    print(f"  files moved: {total.files_moved}")
    print(f"  files copied: {total.files_copied}")
    print(f"  files skipped: {total.files_skipped}")
    print(f"  missing modality dirs: {total.missing_modalities}")
    if dry_run:
        print("\nDry-run only. Re-run with --apply to modify the dataset.")
    return 0


@dataclass
class FullSplitStats:
    scenes_seen: int = 0
    scenes_with_camera_ids: int = 0
    scenes_with_outputs: int = 0
    camera_files_seen: int = 0
    camera_files_matched: int = 0
    shared_files_seen: int = 0
    shared_outputs_planned: int = 0
    cam_param_outputs_planned: int = 0
    camera_files_moved: int = 0
    camera_files_copied: int = 0
    shared_files_copied: int = 0
    cam_params_written: int = 0
    skipped_existing: int = 0
    skipped_unmatched: int = 0
    missing_modalities: int = 0
    missing_cam_param: int = 0

    def add(self, other: "FullSplitStats") -> None:
        self.scenes_seen += other.scenes_seen
        self.scenes_with_camera_ids += other.scenes_with_camera_ids
        self.scenes_with_outputs += other.scenes_with_outputs
        self.camera_files_seen += other.camera_files_seen
        self.camera_files_matched += other.camera_files_matched
        self.shared_files_seen += other.shared_files_seen
        self.shared_outputs_planned += other.shared_outputs_planned
        self.cam_param_outputs_planned += other.cam_param_outputs_planned
        self.camera_files_moved += other.camera_files_moved
        self.camera_files_copied += other.camera_files_copied
        self.shared_files_copied += other.shared_files_copied
        self.cam_params_written += other.cam_params_written
        self.skipped_existing += other.skipped_existing
        self.skipped_unmatched += other.skipped_unmatched
        self.missing_modalities += other.missing_modalities
        self.missing_cam_param += other.missing_cam_param


def parse_full_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split every Robo3R scene into camera-specific folders. "
            "Camera-specific modalities are moved/copied by camera id; shared "
            "per-frame modalities are copied to every camera; cam_param.npy is "
            "split per camera."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(DEFAULT_ROOT),
        help="Dataset root containing scene_* directories.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(DEFAULT_MODALITIES),
        help=(
            "Modalities to process. Default: all README fields "
            "(rgb depth mask qpos ee_pose keypoint_3d keypoint_2d cam_param.npy)."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write changes. Without this flag the script only reports a dry run.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help=(
            "Copy camera-specific files instead of moving them. Shared modalities "
            "are always copied because the same data belongs to every camera."
        ),
    )
    parser.add_argument(
        "--camera-base",
        type=int,
        default=1,
        help="Numbering base for output directory names: camera_idx 00 -> camera{base}.",
    )
    parser.add_argument(
        "--rename-frame-only",
        action="store_true",
        help=(
            "Rename camera-specific files inside camera directories from "
            "0000_00.jpg to 0000.jpg. Shared files keep their original names."
        ),
    )
    parser.add_argument(
        "--squeeze-cam-param",
        action="store_true",
        help=(
            "Write per-camera cam_param.npy as (2, 4, 4) for README layout "
            "instead of preserving the single-camera dimension as (2, 1, 4, 4)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing destination files. Use cautiously.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=0,
        help="Process at most this many scenes. Use 0 for all scenes.",
    )
    parser.add_argument(
        "--scene-glob",
        default="scene_*",
        help="Glob used to discover scene directories under --root.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each planned operation.",
    )
    return parser.parse_args()


def normalize_modalities(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    for raw_value in values:
        key = raw_value.strip().rstrip("/").lower()
        key = MODALITY_ALIASES.get(key, key)
        if key == "all":
            for modality in DEFAULT_MODALITIES:
                if modality not in normalized:
                    normalized.append(modality)
            continue
        if (
            key not in CAMERA_MODALITIES
            and key not in SHARED_MODALITIES
            and key != CAM_PARAM_MODALITY
        ):
            raise ValueError(f"Unknown Robo3R modality: {raw_value}")
        if key not in normalized:
            normalized.append(key)
    return normalized


def parse_camera_dir_name(path: Path, camera_base: int) -> Optional[int]:
    match = CAMERA_DIR_RE.match(path.name)
    if match is None:
        return None
    camera_idx = int(match.group("number")) - camera_base
    return camera_idx if camera_idx >= 0 else None


def cam_param_layout(cam_param: np.ndarray) -> tuple[str, int]:
    if cam_param.ndim != 4:
        raise ValueError(f"Unsupported cam_param shape: {cam_param.shape}")
    if cam_param.shape[0] >= 2:
        return "readme", cam_param.shape[1]
    if cam_param.shape[1] >= 2:
        return "alt", cam_param.shape[0]
    raise ValueError(f"Unsupported cam_param layout: {cam_param.shape}")


def camera_ids_from_cam_param(cam_param_path: Path) -> list[int]:
    if not cam_param_path.is_file():
        return []
    cam_param = np.load(cam_param_path, mmap_mode="r")
    _, num_cameras = cam_param_layout(cam_param)
    return list(range(num_cameras))


def discover_camera_ids(
    scene_dir: Path,
    camera_modalities: Sequence[str],
    camera_base: int,
) -> list[int]:
    camera_ids: set[int] = set(camera_ids_from_cam_param(scene_dir / "cam_param.npy"))
    for modality in camera_modalities:
        modality_dir = scene_dir / modality
        if not modality_dir.is_dir():
            continue
        for path in modality_dir.iterdir():
            if path.is_file():
                match = CAMERA_FILE_RE.match(path.name)
                if match is not None:
                    camera_ids.add(int(match.group("camera")))
            elif path.is_dir():
                camera_idx = parse_camera_dir_name(path, camera_base)
                if camera_idx is not None:
                    camera_ids.add(camera_idx)
    return sorted(camera_ids)


def same_size(src: Path, dst: Path) -> bool:
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def existing_destination_action(
    dst: Path,
    overwrite: bool,
    src: Optional[Path] = None,
) -> str:
    if not dst.exists():
        return "write"
    if overwrite:
        return "overwrite"
    if src is None or same_size(src, dst):
        return "skip"
    raise FileExistsError(f"Refusing to overwrite existing file: {dst}")


def write_or_copy_file(
    src: Path,
    dst: Path,
    do_apply: bool,
    copy_file: bool,
    overwrite: bool,
) -> str:
    action = existing_destination_action(dst, overwrite=overwrite, src=src)
    if action == "skip" or not do_apply:
        return action

    dst.parent.mkdir(parents=True, exist_ok=True)
    if action == "overwrite":
        dst.unlink()
    if copy_file:
        shutil.copy2(src, dst)
        return "copied"
    shutil.move(str(src), str(dst))
    return "moved"


def split_camera_modality_full(
    modality_dir: Path,
    do_apply: bool,
    copy_file: bool,
    camera_base: int,
    rename_frame_only: bool,
    overwrite: bool,
    verbose: bool,
) -> FullSplitStats:
    stats = FullSplitStats()
    if not modality_dir.is_dir():
        stats.missing_modalities += 1
        return stats

    for src in iter_direct_files(modality_dir):
        stats.camera_files_seen += 1
        match = CAMERA_FILE_RE.match(src.name)
        if match is None:
            stats.skipped_unmatched += 1
            continue

        stats.camera_files_matched += 1
        camera_idx = int(match.group("camera"))
        dst = destination_for(
            src=src,
            camera_idx=camera_idx,
            camera_base=camera_base,
            rename_frame_only=rename_frame_only,
        )
        action_name = "copy" if copy_file else "move"
        if verbose:
            print(f"{action_name}: {src} -> {dst}")

        result = write_or_copy_file(
            src=src,
            dst=dst,
            do_apply=do_apply,
            copy_file=copy_file,
            overwrite=overwrite,
        )
        if result == "moved":
            stats.camera_files_moved += 1
        elif result == "copied":
            stats.camera_files_copied += 1
        elif result == "skip":
            stats.skipped_existing += 1

    return stats


def shared_destination_for(src: Path, camera_idx: int, camera_base: int) -> Path:
    return src.parent / f"camera{camera_idx + camera_base}" / src.name


def split_shared_modality_full(
    modality_dir: Path,
    camera_ids: Sequence[int],
    do_apply: bool,
    camera_base: int,
    overwrite: bool,
    verbose: bool,
) -> FullSplitStats:
    stats = FullSplitStats()
    if not modality_dir.is_dir():
        stats.missing_modalities += 1
        return stats

    for src in iter_direct_files(modality_dir):
        stats.shared_files_seen += 1
        for camera_idx in camera_ids:
            dst = shared_destination_for(src, camera_idx, camera_base)
            stats.shared_outputs_planned += 1
            if verbose:
                print(f"copy shared: {src} -> {dst}")
            result = write_or_copy_file(
                src=src,
                dst=dst,
                do_apply=do_apply,
                copy_file=True,
                overwrite=overwrite,
            )
            if result == "copied":
                stats.shared_files_copied += 1
            elif result == "skip":
                stats.skipped_existing += 1

    return stats


def split_cam_param_array(
    cam_param: np.ndarray,
    camera_idx: int,
    squeeze: bool,
) -> np.ndarray:
    layout, num_cameras = cam_param_layout(cam_param)
    if camera_idx >= num_cameras:
        raise IndexError(
            f"camera_idx {camera_idx} is out of range for cam_param with {num_cameras} cameras"
        )
    if layout == "readme":
        if squeeze:
            return np.asarray(cam_param[:, camera_idx]).copy()
        return np.asarray(cam_param[:, camera_idx:camera_idx + 1]).copy()
    if squeeze:
        return np.asarray(cam_param[camera_idx]).copy()
    return np.asarray(cam_param[camera_idx:camera_idx + 1]).copy()


def split_cam_param_full(
    scene_dir: Path,
    camera_ids: Sequence[int],
    do_apply: bool,
    camera_base: int,
    squeeze: bool,
    overwrite: bool,
    verbose: bool,
) -> FullSplitStats:
    stats = FullSplitStats()
    src = scene_dir / "cam_param.npy"
    if not src.is_file():
        stats.missing_cam_param += 1
        return stats

    cam_param = np.load(src)
    for camera_idx in camera_ids:
        split = split_cam_param_array(cam_param, camera_idx, squeeze=squeeze)
        dst = scene_dir / "cam_param" / f"camera{camera_idx + camera_base}" / "cam_param.npy"
        stats.cam_param_outputs_planned += 1
        if verbose:
            print(f"write cam_param: {src}[camera={camera_idx}] -> {dst} shape={split.shape}")

        action = existing_destination_action(dst, overwrite=overwrite)
        if action == "skip":
            stats.skipped_existing += 1
            continue
        if not do_apply:
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if action == "overwrite":
            dst.unlink()
        np.save(dst, split)
        stats.cam_params_written += 1

    return stats


def split_scene_full(
    scene_dir: Path,
    camera_modalities: Sequence[str],
    shared_modalities: Sequence[str],
    include_cam_param: bool,
    do_apply: bool,
    copy_camera_files: bool,
    camera_base: int,
    rename_frame_only: bool,
    squeeze_cam_param: bool,
    overwrite: bool,
    verbose: bool,
) -> FullSplitStats:
    scene_stats = FullSplitStats(scenes_seen=1)
    camera_ids = discover_camera_ids(scene_dir, camera_modalities, camera_base)
    if camera_ids:
        scene_stats.scenes_with_camera_ids = 1
    elif shared_modalities or include_cam_param:
        print(f"Warning: no camera ids found for {scene_dir}")

    for modality in camera_modalities:
        scene_stats.add(
            split_camera_modality_full(
                modality_dir=scene_dir / modality,
                do_apply=do_apply,
                copy_file=copy_camera_files,
                camera_base=camera_base,
                rename_frame_only=rename_frame_only,
                overwrite=overwrite,
                verbose=verbose,
            )
        )

    for modality in shared_modalities:
        scene_stats.add(
            split_shared_modality_full(
                modality_dir=scene_dir / modality,
                camera_ids=camera_ids,
                do_apply=do_apply,
                camera_base=camera_base,
                overwrite=overwrite,
                verbose=verbose,
            )
        )

    if include_cam_param:
        scene_stats.add(
            split_cam_param_full(
                scene_dir=scene_dir,
                camera_ids=camera_ids,
                do_apply=do_apply,
                camera_base=camera_base,
                squeeze=squeeze_cam_param,
                overwrite=overwrite,
                verbose=verbose,
            )
        )

    planned = (
        scene_stats.camera_files_matched
        + scene_stats.shared_outputs_planned
        + scene_stats.cam_param_outputs_planned
    )
    if planned > 0:
        scene_stats.scenes_with_outputs = 1
    return scene_stats


def main() -> int:
    args = parse_full_args()
    if args.camera_base < 0:
        raise ValueError("--camera-base must be non-negative")
    if args.copy and not args.apply:
        print("Note: --copy has no effect without --apply; running dry-run.")

    modalities = normalize_modalities(args.modalities)
    camera_modalities = [item for item in modalities if item in CAMERA_MODALITIES]
    shared_modalities = [item for item in modalities if item in SHARED_MODALITIES]
    include_cam_param = CAM_PARAM_MODALITY in modalities

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root does not exist: {root}")

    camera_mode = "copy" if args.copy else "move"
    dry_run = not args.apply
    print(f"root: {root}")
    print(f"modalities: {' '.join(modalities)}")
    print(f"camera-specific modalities: {' '.join(camera_modalities) or '(none)'}")
    print(f"shared modalities: {' '.join(shared_modalities) or '(none)'}")
    print(f"cam_param split: {'yes' if include_cam_param else 'no'}")
    print(f"mode: {'dry-run ' + camera_mode if dry_run else camera_mode}")
    print("shared modality mode: copy to every camera directory")
    print(f"camera dirs: source camera 00 -> camera{args.camera_base}")
    if args.rename_frame_only:
        print("camera-specific filename mode: 0000_00.ext -> camera*/0000.ext")
    else:
        print("camera-specific filename mode: preserve original names")
    if include_cam_param:
        shape_note = "(2, 4, 4)" if args.squeeze_cam_param else "(2, 1, 4, 4)"
        print(f"cam_param output shape for README layout: {shape_note}")

    total = FullSplitStats()
    for scene_dir in iter_scene_dirs(root, args.scene_glob, args.max_scenes):
        stats = split_scene_full(
            scene_dir=scene_dir,
            camera_modalities=camera_modalities,
            shared_modalities=shared_modalities,
            include_cam_param=include_cam_param,
            do_apply=args.apply,
            copy_camera_files=args.copy,
            camera_base=args.camera_base,
            rename_frame_only=args.rename_frame_only,
            squeeze_cam_param=args.squeeze_cam_param,
            overwrite=args.overwrite,
            verbose=args.verbose,
        )
        total.add(stats)
        if args.verbose:
            print(
                f"scene summary: {scene_dir.name} "
                f"camera_matched={stats.camera_files_matched} "
                f"shared_outputs={stats.shared_outputs_planned} "
                f"cam_param_outputs={stats.cam_param_outputs_planned}"
            )

    print("\nsummary")
    print(f"  scenes seen: {total.scenes_seen}")
    print(f"  scenes with camera ids: {total.scenes_with_camera_ids}")
    print(f"  scenes with planned outputs: {total.scenes_with_outputs}")
    print(f"  camera-specific files seen: {total.camera_files_seen}")
    print(f"  camera-specific files matched: {total.camera_files_matched}")
    print(f"  shared files seen: {total.shared_files_seen}")
    print(f"  shared outputs planned: {total.shared_outputs_planned}")
    print(f"  cam_param outputs planned: {total.cam_param_outputs_planned}")
    print(f"  camera-specific files moved: {total.camera_files_moved}")
    print(f"  camera-specific files copied: {total.camera_files_copied}")
    print(f"  shared files copied: {total.shared_files_copied}")
    print(f"  cam_param files written: {total.cam_params_written}")
    print(f"  skipped existing outputs: {total.skipped_existing}")
    print(f"  skipped unmatched files: {total.skipped_unmatched}")
    print(f"  missing modality dirs: {total.missing_modalities}")
    print(f"  missing cam_param.npy: {total.missing_cam_param}")
    if dry_run:
        print("\nDry-run only. Re-run with --apply to modify the dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
