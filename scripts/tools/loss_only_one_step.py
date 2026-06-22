#!/usr/bin/env python3
"""Run one forward+loss batch per train dataset without backward.

This is a diagnostic entrypoint for ``train.sh LOSS_ONLY=1``. It intentionally
reuses train.py's parser, dataloader construction, model loading, normalization,
and VGGT-style loss so the command line stays aligned with normal training.
"""

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
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import ConcatDataset

import train as train_mod


METRIC_KEYS: Tuple[str, ...] = (
    "loss_objective",
    "loss_camera",
    "loss_T",
    "loss_R",
    "loss_FL",
    "loss_relative_pose",
    "loss_relative_rot",
    "loss_relative_trans",
    "loss_conf_depth",
    "loss_reg_depth",
    "loss_grad_depth",
)

DATASET_NAME_BY_CLASS = {
    "ManipTrajectoryDataset": "manip",
    "DL3DVTrajectoryDataset": "dl3dv",
    "ScanNetppTrajectoryDataset": "scannetpp",
    "ARKitScenesHighResTrajectoryDataset": "arkitscenes_highres",
    "HypersimTrajectoryDataset": "hypersim",
    "TartanAirTrajectoryDataset": "tartanair",
    "DynamicReplicaTrajectoryDataset": "dynamic_replica",
    "ADTTrajectoryDataset": "adt",
    "ASETrajectoryDataset": "ase",
    "Co3dTrajectoryDataset": "co3d",
}


def _dataset_name(dataset: object) -> str:
    return DATASET_NAME_BY_CLASS.get(dataset.__class__.__name__, dataset.__class__.__name__)


def _iter_named_leaves(dataset: object) -> Iterable[Tuple[str, object]]:
    if isinstance(dataset, ConcatDataset):
        for child in dataset.datasets:
            yield from _iter_named_leaves(child)
    else:
        yield _dataset_name(dataset), dataset


def _unique_leaves(dataset: object) -> List[Tuple[str, object]]:
    out: List[Tuple[str, object]] = []
    seen: set[Tuple[str, int]] = set()
    for name, leaf in _iter_named_leaves(dataset):
        key = (name, id(leaf))
        if key in seen:
            continue
        seen.add(key)
        out.append((name, leaf))
    return out


def _build_criterion(args: argparse.Namespace, device: torch.device) -> train_mod.VGGTStyleLoss:
    return train_mod.VGGTStyleLoss(
        camera_weight=args.camera_weight,
        depth_weight=args.depth_weight,
        relative_pose_weight=args.relative_pose_weight,
        camera_loss_type=args.camera_loss_type,
        camera_gamma=args.camera_gamma,
        weight_trans=args.weight_trans,
        weight_rot=args.weight_rot,
        weight_focal=args.weight_focal,
        relative_trans_weight=args.relative_trans_weight,
        relative_pose_window=args.relative_pose_window,
        depth_gradient_loss_fn=args.depth_gradient_loss_fn,
        loss_gamma=args.loss_gamma,
        loss_alpha=args.loss_alpha,
        valid_range=args.valid_range,
        min_valid_pixels=args.min_valid_pixels,
    ).to(device)


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    args.batch_size = 1
    args.num_workers = 0
    args.tensorboard = False
    args.color_jitter_strength = 0.0
    args.color_jitter_prob = 0.0
    args.limit_train_batches = max(1, int(args.limit_train_batches or 1))
    return args


def _load_one_batch(dataset: object, max_attempts: int) -> Tuple[Dict[str, object], int]:
    dataset_len = len(dataset)  # type: ignore[arg-type]
    if dataset_len <= 0:
        raise RuntimeError("dataset is empty")
    last_error: Optional[BaseException] = None
    for offset in range(min(max_attempts, dataset_len)):
        try:
            sample = dataset[offset]  # type: ignore[index]
            return train_mod.collate_rgbd_sequences([sample]), offset
        except Exception as exc:  # noqa: BLE001 - try another sample for diagnostics
            last_error = exc
            continue
    raise RuntimeError(f"failed to load a valid sample after {max_attempts} attempts: {last_error!r}")


def _run_one_dataset(
    name: str,
    dataset: object,
    model: torch.nn.Module,
    criterion: train_mod.VGGTStyleLoss,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
) -> Dict[str, object]:
    train_mod.seed_everything(args.seed)
    batch, sample_index = _load_one_batch(dataset, max_attempts=int(args.loss_only_sample_attempts))
    if args.canonicalize_first_frame:
        batch = train_mod.canonicalize_to_first_frame(batch)
    if args.normalize_scene:
        batch = train_mod.normalize_scene_batch(
            batch,
            num_anchor_frames=min(args.num_scale_frames, int(batch["images"].shape[1])),
        )
    batch = train_mod.to_device(batch, device)
    input_desc = train_mod.format_batch_input(batch)

    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            predictions = model(
                batch["images"],
                num_frame_for_scale=min(args.num_scale_frames, int(batch["images"].shape[1])),
                num_frame_per_block=args.num_frame_per_block,
                depth_frames_chunk_size=args.depth_frames_chunk_size,
                causal_inference=True,
            )
            losses = criterion(predictions, batch)
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()

    scalar_losses = train_mod.loss_to_float_dict(losses)
    objective = scalar_losses.get("loss_objective")
    if objective is None or not math.isfinite(float(objective)):
        raise RuntimeError(f"non-finite objective: {objective}")

    result: Dict[str, object] = {
        "dataset": name,
        "samples": int(len(dataset)),  # type: ignore[arg-type]
        "sample_index": int(sample_index),
        "input": input_desc,
        "elapsed_sec": round(time.time() - start, 3),
    }
    for key in METRIC_KEYS:
        if key in scalar_losses:
            result[key] = float(scalar_losses[key])
    return result


def _write_markdown(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    cols = (
        "dataset",
        "input",
        "loss_objective",
        "loss_camera",
        "loss_conf_depth",
        "loss_reg_depth",
        "loss_grad_depth",
        "loss_relative_pose",
        "elapsed_sec",
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(cols) + " |\n")
        handle.write("| " + " | ".join("---" for _ in cols) + " |\n")
        for row in rows:
            values = []
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.6f}"
                values.append(str(value).replace("|", "\\|"))
            handle.write("| " + " | ".join(values) + " |\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = train_mod.build_argparser()
    parser.description = "Run one forward+loss batch per dataset without backward"
    parser.add_argument("--loss_only_jsonl", type=str, default="")
    parser.add_argument("--loss_only_md", type=str, default="")
    parser.add_argument("--loss_only_sample_attempts", type=int, default=16)
    parser.add_argument(
        "--loss_only_train_mode",
        action="store_true",
        help="Use model.train() for forward. Default is eval() because no backward is performed.",
    )
    return parser


def main() -> None:
    args = _prepare_args(build_argparser().parse_args())
    train_mod.seed_everything(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    train_loader, _, train_scene_count, _ = train_mod.build_dataloaders(args)
    leaves = _unique_leaves(train_loader.dataset)
    print(f"[loss-only] train_scenes={train_scene_count} datasets={[(name, len(ds)) for name, ds in leaves]}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model = train_mod.build_model(args, device)
    if args.loss_only_train_mode:
        model.train()
    else:
        model.eval()
    criterion = _build_criterion(args, device)

    output_dir = Path(args.output_dir)
    jsonl_path = Path(args.loss_only_jsonl) if args.loss_only_jsonl else output_dir / "loss_only_one_step.jsonl"
    md_path = Path(args.loss_only_md) if args.loss_only_md else output_dir / "loss_only_one_step.md"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for name, dataset in leaves:
            print(f"[loss-only] {name}: start")
            try:
                result = _run_one_dataset(name, dataset, model, criterion, args, device, amp_dtype, amp_enabled)
                print("[loss-only-result] " + json.dumps(result, sort_keys=True))
            except Exception as exc:  # noqa: BLE001 - keep collecting other datasets
                result = {"dataset": name, "samples": int(len(dataset)), "error": repr(exc)}
                print("[loss-only-error] " + json.dumps(result, sort_keys=True))
            rows.append(result)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _write_markdown(md_path, rows)
    print(f"[loss-only] wrote jsonl: {jsonl_path}")
    print(f"[loss-only] wrote markdown: {md_path}")


if __name__ == "__main__":
    main()
