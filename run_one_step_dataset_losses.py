#!/usr/bin/env python
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import train as train_mod
from lingbot_map.data.dl3dv import DL3DVTrajectoryDataset
from lingbot_map.data.dynamic_replica import DynamicReplicaTrajectoryDataset
from lingbot_map.data.mapfree import MapfreeTrajectoryDataset
from lingbot_map.data.scannetpp import ScanNetppTrajectoryDataset
from lingbot_map.data.tartanair import TartanAirTrajectoryDataset


METRIC_KEYS = [
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
]


def first_scene(root: str) -> Path:
    root_path = Path(root)
    for entry in sorted(root_path.iterdir()):
        if entry.is_dir() and train_mod._valid_trajectory_name(entry.name):
            return entry
    raise RuntimeError(f"no scene found under {root}")


def make_args():
    args = train_mod.build_argparser().parse_args([])
    args.model_path = "/cpfs/user/guowenqi/lingbot-map/lingbot-map.pt"
    args.output_dir = "/cpfs/user/guowenqi/lingbot-map/runs/one_step_dataset_losses_20260527"
    args.max_steps = 1
    args.batch_size = 1
    args.num_workers = 0
    args.limit_train_batches = 1
    args.val_fraction = 0.0
    args.tensorboard = False
    args.save_every = 0
    args.val_every = 0
    args.log_every = 1
    args.print_input_every = 1
    args.image_size = int(os.environ.get("ONE_STEP_IMAGE_SIZE", "280"))
    args.depth_scale = 0.0
    args.use_mask = True
    args.clip_len = 48
    args.samples_per_scene = 1
    args.sequence_mode = "manip_4d_mixed"
    args.sample_strategy = "random_interval"
    args.random_stride_min = 10
    args.random_stride_max = 60
    args.random_interval_start = "first"
    args.max_sample_frames = 48
    args.min_sample_frames = 32
    args.num_scale_frames = 8
    args.num_frame_per_block = 1
    args.kv_cache_sliding_window = 64
    args.depth_frames_chunk_size = 2
    args.no_depth_activation_checkpoint = False
    args.freeze_dino_patch_embed = True
    args.freeze_aggregator = False
    args.freeze_camera = False
    args.freeze_depth = False
    args.freeze_point = False
    args.amp = True
    args.amp_dtype = "bf16"
    args.cpu = False
    args.allow_tf32 = True
    args.cudnn_benchmark = True
    return args


def make_dataset(name: str, args, num_views: int):
    common_manip = dict(
        clip_len=args.clip_len,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        sequence_mode=args.sequence_mode,
        view_ids=None,
        camera_names=None,
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
        m_stride_min=8,
        m_stride_max=24,
        s_views_min=4,
        s_views_max=6,
        m_num_views=4,
        m_num_times=0,
        m_views_min=3,
        m_views_max=6,
        mode_weights_initial=train_mod.parse_mode_weights("S=0.70,W=0.20,M=0.10"),
        mode_weights_final=train_mod.parse_mode_weights("S=0.30,W=0.40,M=0.30"),
        mode_warmup_start=2000,
        mode_warmup_end=8000,
        color_jitter_strength=float(os.environ.get("ONE_STEP_COLOR_JITTER_STRENGTH", "0.2")),
        color_jitter_prob=float(os.environ.get("ONE_STEP_COLOR_JITTER_PROB", "0.5")),
    )
    if name == "Manip_long3":
        return train_mod.ManipTrajectoryDataset([first_scene("/oss-guowenqi/Manip_long3/data")], **common_manip)
    if name == "Manip_long4":
        return train_mod.ManipTrajectoryDataset([first_scene("/oss-guowenqi/Manip_long4/data")], **common_manip)

    common = dict(
        num_views=num_views,
        image_size=args.image_size,
        patch_size=args.patch_size,
        preprocess_mode=args.preprocess_mode,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        samples_per_scene=args.samples_per_scene,
        color_jitter_strength=float(os.environ.get("ONE_STEP_COLOR_JITTER_STRENGTH", "0.2")),
        color_jitter_prob=float(os.environ.get("ONE_STEP_COLOR_JITTER_PROB", "0.5")),
        verbose=True,
    )
    if name == "DL3DV":
        return DL3DVTrajectoryDataset(
            root="/cpfs/shared/landmark/renkerui/data/dl3dv",
            split="train",
            min_interval=1,
            max_interval=32,
            video_prob=0.8,
            fix_interval_prob=0.6,
            block_shuffle=16,
            **common,
        )
    if name == "ScanNet++":
        return ScanNetppTrajectoryDataset(
            root="/shared/smartbot/renkerui/data/scannetppv2",
            split="train",
            min_interval=1,
            max_interval=30,
            video_prob=0.6,
            fix_interval_prob=0.6,
            block_shuffle=16,
            scan_max_workers=16,
            **common,
        )
    if name == "TartanAir":
        return TartanAirTrajectoryDataset(
            root="/cpfs/shared/landmark/renkerui/data/tartanair",
            split="train",
            min_interval=1,
            max_interval=32,
            video_prob=0.8,
            fix_interval_prob=0.6,
            block_shuffle=16,
            scan_max_workers=16,
            **common,
        )
    if name == "DynamicReplica":
        return DynamicReplicaTrajectoryDataset(
            root="/shared/smartbot/renkerui/data/dynamic_replica",
            split="train",
            min_interval=1,
            max_interval=64,
            video_prob=1.0,
            fix_interval_prob=1.0,
            block_shuffle=16,
            scan_max_workers=16,
            **common,
        )
    if name == "MapFree":
        return MapfreeTrajectoryDataset(
            root="/cpfs/shared/landmark/renkerui/data/mapfree",
            split="train",
            min_interval=1,
            max_interval=64,
            video_prob=0.8,
            fix_interval_prob=0.6,
            block_shuffle=16,
            scan_max_workers=16,
            **common,
        )
    raise ValueError(name)


def run_one(name: str, num_views: int) -> Dict[str, object]:
    args = make_args()
    train_mod.seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    dataset = make_dataset(name, args, num_views=num_views)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        collate_fn=train_mod.collate_rgbd_sequences,
    )
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model = train_mod.build_model(args, device)
    optimizer, scheduler = train_mod.build_optimizer_and_scheduler(model, args, steps_per_epoch=1)
    criterion = train_mod.VGGTStyleLoss(
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


    if os.environ.get("ONE_STEP_EVAL_MODE", "0") == "1":
        model.eval()
    else:
        model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = train_mod.canonicalize_to_first_frame(batch)
    batch = train_mod.normalize_scene_batch(
        batch,
        num_anchor_frames=min(args.num_scale_frames, int(batch["images"].shape[1])),
    )
    batch = train_mod.to_device(batch, device)
    input_desc = train_mod.format_batch_input(batch)

    start = time.time()
    model.clean_kv_cache()
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        predictions = model(
            batch["images"],
            num_frame_for_scale=min(args.num_scale_frames, int(batch["images"].shape[1])),
            num_frame_per_block=args.num_frame_per_block,
            depth_frames_chunk_size=args.depth_frames_chunk_size,
            causal_inference=True,
        )
        losses = criterion(predictions, batch)
        loss = losses["objective"]
    if not torch.isfinite(loss.detach()):
        raise RuntimeError(f"non-finite loss for {name}")

    if os.environ.get("ONE_STEP_SKIP_BACKWARD", "0") != "1":
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            args.grad_clip_norm,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    model.clean_kv_cache()

    scalar_losses = train_mod.loss_to_float_dict(losses)
    result = {
        "dataset": name,
        "num_views": num_views,
        "input": input_desc,
        "elapsed_sec": round(time.time() - start, 3),
    }
    result.update({key: scalar_losses.get(key) for key in METRIC_KEYS})

    del predictions, losses, loss, criterion, optimizer, scheduler, model, batch, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return result


def main() -> None:
    datasets = [
        "Manip_long3",
        "Manip_long4",
        "DL3DV",
        "ScanNet++",
        "TartanAir",
        "DynamicReplica",
        "MapFree",
    ]
    num_views = int(os.environ.get("ONE_STEP_NUM_VIEWS", "8"))
    out_path = Path(os.environ.get(
        "ONE_STEP_LOSS_JSONL",
        "/cpfs/user/guowenqi/lingbot-map/runs/one_step_dataset_losses_20260527/results.jsonl",
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for name in datasets:
            print(f"[dataset] {name} start", flush=True)
            try:
                result = run_one(name, num_views=num_views)
                print("[result] " + json.dumps(result, sort_keys=True), flush=True)
            except Exception as exc:
                result = {"dataset": name, "error": repr(exc)}
                print("[error] " + json.dumps(result, sort_keys=True), flush=True)
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
