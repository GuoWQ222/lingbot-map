#!/usr/bin/env python3
"""Multi-node DDP training for LingBot-MAP Manip fine-tuning.

This file reuses the dataset/model/loss utilities from ``train.py`` and only
replaces the orchestration layer: distributed initialization, sharded data
loading, DDP wrapping, rank-0 logging/checkpointing, and rank-0 validation.

Example, 6 nodes x 8 GPUs:

    # Run this on every node. Change NODE_RANK to 0..5 on each node.
    export MASTER_ADDR=<node0-ip-or-hostname>
    export MASTER_PORT=29500
    export NCCL_IB_DISABLE=0
    export NCCL_DEBUG=INFO

    torchrun \
      --nnodes=6 \
      --nproc_per_node=8 \
      --node_rank=${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      train_multinode.py \
      --data_roots /oss-guowenqi/Manip_long3/data /oss-guowenqi/Manip_long4/data \
      --scene_manifest runs/manip_long_train/manip_trajectory_manifest.txt \
      --model_path lingbot-map.pt \
      --output_dir runs/manip_long_train_ddp \
      --batch_size 1

Notes:
- This is DistributedDataParallel (DDP): each GPU keeps a full model copy and
  gradients are synchronized across all ranks. It increases throughput and
  global batch size, but it does not reduce per-GPU activation memory.
- ``--batch_size`` is per GPU. The global batch is
  ``batch_size * accum_steps * world_size``.
- ``--max_steps`` is the number of synchronized optimizer steps.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm.auto import tqdm

import train as single


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def is_dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_dist_ready() else 1


def rank() -> int:
    return dist.get_rank() if is_dist_ready() else 0


def is_main_process() -> bool:
    return rank() == 0


def print_rank0(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def broadcast_object(value, src: int = 0):
    if not is_dist_ready():
        return value
    obj_list = [value if rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def barrier() -> None:
    if is_dist_ready():
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model



def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def epoch_loader_seed(base_seed: int, epoch: int, global_rank: int, epoch_start_step: int) -> int:
    # SplitMix64-style mixing keeps nearby epochs/ranks/steps from producing
    # correlated worker seeds while staying deterministic across resumes.
    value = int(base_seed) & 0xFFFFFFFFFFFFFFFF
    value ^= (int(epoch) + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= (int(global_rank) + 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= int(epoch_start_step) & 0xFFFFFFFFFFFFFFFF
    return value


def set_sampler_global_step(sampler: Optional[Sampler[int]], step: int) -> None:
    if sampler is not None and hasattr(sampler, "set_global_step"):
        sampler.set_global_step(step)


def capture_rank_rng_state(device: torch.device) -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    else:
        state["cuda"] = None
    return state


def restore_rank_rng_state(state: Optional[Dict[str, object]], device: torch.device) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def gather_rank_object(value):
    if not is_dist_ready():
        return [value]
    gathered = [None for _ in range(world_size())]
    dist.all_gather_object(gathered, value)
    return gathered


def save_distributed_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    resume_epoch: int,
    resume_batch_idx: int,
    epoch_start_step: int,
) -> None:
    rank_state = {
        "rank": rank(),
        "rng": capture_rank_rng_state(device),
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    all_rank_states = gather_rank_object(rank_state)
    if not is_main_process():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": vars(args),
        "trainable_fingerprint": single._trainable_param_fingerprint(model),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": all_rank_states[0]["rng"] if all_rank_states else None,
        "ddp_rank_states": all_rank_states,
        "resume_state": {
            "epoch": int(resume_epoch),
            "batch_idx": int(resume_batch_idx),
            "epoch_start_step": int(epoch_start_step),
        },
    }
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

class RankShardSampler(Sampler[int]):
    """Shard a deterministic base sampler across DDP ranks.

    This is used for CurriculumMixtureSampler: every rank constructs the same
    global stream of ConcatDataset indices, then consumes positions
    ``rank, rank + world_size, ...``. That preserves the Manip-vs-external
    curriculum while ensuring ranks train on different samples.
    """

    def __init__(self, base_sampler: Sampler[int], num_replicas: int, rank_id: int) -> None:
        self.base_sampler = base_sampler
        self.num_replicas = max(1, int(num_replicas))
        self.rank_id = int(rank_id)

    def __iter__(self):
        total = (len(self.base_sampler) // self.num_replicas) * self.num_replicas
        for pos, index in enumerate(self.base_sampler):
            if pos >= total:
                break
            if pos % self.num_replicas == self.rank_id:
                yield index

    def __len__(self) -> int:
        return len(self.base_sampler) // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.base_sampler, "set_epoch", None)
        if callable(setter):
            setter(epoch)

    def set_global_step(self, step: int) -> None:
        setter = getattr(self.base_sampler, "set_global_step", None)
        if callable(setter):
            setter(step)

    def get_dataset_weights(self, step: Optional[int] = None):
        getter = getattr(self.base_sampler, "get_dataset_weights", None)
        if callable(getter):
            return getter(step)
        return {}


@contextlib.contextmanager
def silence_stdout(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle):
            yield


def init_distributed(args: argparse.Namespace) -> Tuple[int, int, int]:
    env_world_size = env_int("WORLD_SIZE", 1)
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = env_int("LOCAL_RANK", env_int("SLURM_LOCALID", 0))
    global_rank = env_int("RANK", 0)

    if env_world_size > 1:
        if args.cpu:
            backend = "gloo"
        else:
            backend = args.dist_backend
            if backend == "nccl" and not torch.cuda.is_available():
                raise RuntimeError("NCCL backend requested but CUDA is not available")
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=args.dist_timeout_min),
        )
        global_rank = dist.get_rank()
        env_world_size = dist.get_world_size()

    return global_rank, local_rank, env_world_size


def build_distributed_dataloaders(
    args: argparse.Namespace,
    global_rank: int,
    num_replicas: int,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[Sampler[int]], torch.Generator, int, int]:
    # If a manifest path is configured, rank 0 materializes it once and all
    # ranks then load the same scene universe. This avoids 64 concurrent OSS
    # directory scans at startup.
    if global_rank == 0 and args.scene_manifest:
        single.discover_trajectory_dirs(
            args.data_roots,
            max_scenes=args.max_scenes,
            manifest=args.scene_manifest,
            write_manifest=args.write_manifest,
            oss_uri_roots=single.parse_str_list(args.oss_uri_roots),
            ossutil_bin=args.ossutil_bin,
            ossutil_config=args.ossutil_config,
        )
    barrier()

    with silence_stdout(args.quiet_nonzero_ranks and global_rank != 0):
        base_train_loader, base_val_loader, train_scene_count, val_scene_count = single.build_dataloaders(args)

    train_dataset = base_train_loader.dataset
    base_sampler = base_train_loader.sampler
    train_sampler: Optional[Sampler[int]] = None
    shuffle = base_sampler is None

    if num_replicas > 1:
        if base_sampler is not None and hasattr(base_sampler, "get_dataset_weights"):
            per_rank_epoch_length = (
                max(1, int(args.limit_train_batches))
                if args.limit_train_batches > 0
                else max(1, math.ceil(len(base_sampler) / num_replicas))
            )
            if hasattr(base_sampler, "epoch_length"):
                base_sampler.epoch_length = per_rank_epoch_length * num_replicas
            train_sampler = RankShardSampler(base_sampler, num_replicas=num_replicas, rank_id=global_rank)
            shuffle = False
        else:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=num_replicas,
                rank=global_rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            shuffle = False
    elif base_sampler is not None:
        train_sampler = base_sampler
        shuffle = False

    train_generator = torch.Generator()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
        generator=train_generator,
        collate_fn=single.collate_rgbd_sequences,
    )

    val_loader = base_val_loader if global_rank == 0 else None
    return train_loader, val_loader, train_sampler, train_generator, train_scene_count, val_scene_count


def build_rank_model(args: argparse.Namespace, device: torch.device, global_rank: int, num_replicas: int) -> nn.Module:
    model_args = argparse.Namespace(**vars(args))
    if num_replicas > 1 and args.rank0_load_model and global_rank != 0:
        model_args.model_path = ""
    with silence_stdout(args.quiet_nonzero_ranks and global_rank != 0):
        model = single.build_model(model_args, device)

    if num_replicas <= 1:
        return model

    ddp_kwargs = dict(
        device_ids=[device.index] if device.type == "cuda" else None,
        output_device=device.index if device.type == "cuda" else None,
        find_unused_parameters=args.ddp_find_unused_parameters,
        broadcast_buffers=False,
        gradient_as_bucket_view=args.ddp_gradient_as_bucket_view,
        static_graph=args.ddp_static_graph,
    )
    return DDP(model, **ddp_kwargs)


def load_training_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    global_rank: int,
) -> Tuple[int, int, int, int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    if any(key.startswith("module.") for key in state.keys()):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    model_for_state = unwrap_model(model)
    missing, unexpected = model_for_state.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Resume aborted: checkpoint does not exactly match the current model. "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    saved_fp = ckpt.get("trainable_fingerprint")
    current_fp = single._trainable_param_fingerprint(model_for_state)
    if saved_fp is not None and saved_fp != current_fp:
        saved_set = {name for name, _ in saved_fp}
        current_set = {name for name, _ in current_fp}
        added = sorted(current_set - saved_set)[:8]
        removed = sorted(saved_set - current_set)[:8]
        raise RuntimeError(
            "Resume aborted: trainable-parameter set has changed since the checkpoint was saved. "
            f"added={added}, removed={removed}"
        )

    optimizer.load_state_dict(ckpt["optimizer"])
    if ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    rank_states = ckpt.get("ddp_rank_states")
    rank_state = None
    if rank_states:
        for item in rank_states:
            if item and int(item.get("rank", -1)) == int(global_rank):
                rank_state = item
                break
    if rank_state is None and global_rank == 0 and ckpt.get("rng") is not None:
        rank_state = {"rng": ckpt.get("rng"), "scaler": ckpt.get("scaler")}

    if scaler is not None:
        scaler_state = rank_state.get("scaler") if rank_state else ckpt.get("scaler")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
    restore_rank_rng_state(rank_state.get("rng") if rank_state else None, device)

    resume_state = ckpt.get("resume_state") or {}
    epoch = int(resume_state.get("epoch", ckpt.get("epoch", 0)))
    batch_idx = int(resume_state.get("batch_idx", 0))
    epoch_start_step = int(resume_state.get("epoch_start_step", ckpt.get("global_step", 0)))
    return epoch, int(ckpt.get("global_step", 0)), batch_idx, epoch_start_step


def reduce_loss_dict(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not is_dist_ready() or not metrics:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float32, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(world_size())
    return {key: float(value.detach().cpu()) for key, value in zip(keys, values)}


def all_ranks_boolean(value: bool, device: torch.device) -> bool:
    if not is_dist_ready():
        return value
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def summarize_startup(
    args: argparse.Namespace,
    output_dir: Path,
    global_rank: int,
    local_rank: int,
    num_replicas: int,
    train_batches: int,
    effective_train_batches: int,
    train_scene_count: int,
    val_scene_count: int,
) -> None:
    if global_rank != 0:
        return
    summary = f"""
========================================
 LingBot-MAP Manip DDP Training
========================================
[distributed]
  world_size    : {num_replicas}
  nnodes/gpus   : launched by torchrun
  rank0 local   : rank={global_rank}, local_rank={local_rank}
  backend       : {args.dist_backend}
  ddp buckets   : gradient_as_bucket_view={args.ddp_gradient_as_bucket_view}

[run]
  output_dir    : {output_dir}
  tensorboard   : {int(args.tensorboard)} ({args.tensorboard_dir or output_dir / "tensorboard"})
  checkpoints   : rank0 only

[data]
  train_scenes  : {train_scene_count}
  val_scenes    : {val_scene_count}
  train_batches : per_rank_total={train_batches}, per_epoch_running={effective_train_batches}

[sampling]
  mode          : {args.sequence_mode}
  strategy      : {args.sample_strategy}
  seq_len       : {args.max_sample_frames}
  image_size    : {args.image_size}
  depth_chunk   : {args.depth_frames_chunk_size}

[optimization]
  local_batch   : {args.batch_size}
  accum_steps   : {args.accum_steps}
  global_batch  : {args.batch_size * args.accum_steps * num_replicas}
  max_steps     : {args.max_steps}
  lr            : {args.lr} -> {args.min_lr}
========================================
"""
    print(summary, flush=True)


def train_distributed(args: argparse.Namespace) -> None:
    single.install_cuda_oom_hook()
    global_rank, local_rank, num_replicas = init_distributed(args)

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    single.seed_everything(args.seed + global_rank)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    output_dir = Path(args.output_dir)
    if global_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "args.json", "w", encoding="utf-8") as handle:
            json.dump(vars(args), handle, indent=2, sort_keys=True)
    barrier()

    train_loader, val_loader, train_sampler, train_generator, train_scene_count, val_scene_count = build_distributed_dataloaders(
        args,
        global_rank,
        num_replicas,
    )
    effective_train_batches = len(train_loader)
    if args.limit_train_batches > 0:
        effective_train_batches = min(effective_train_batches, args.limit_train_batches)

    summarize_startup(
        args,
        output_dir,
        global_rank,
        local_rank,
        num_replicas,
        len(train_loader),
        effective_train_batches,
        train_scene_count,
        val_scene_count,
    )

    amp_enabled = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16

    model = build_rank_model(args, device, global_rank, num_replicas)
    model_for_state = unwrap_model(model)
    trainable, total = single.count_trainable_params(model_for_state)
    print_rank0(f"[model] trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.2f}M")

    steps_per_epoch = max(1, effective_train_batches)
    with silence_stdout(args.quiet_nonzero_ranks and global_rank != 0):
        optimizer, scheduler = single.build_optimizer_and_scheduler(model, args, steps_per_epoch)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    start_epoch = 0
    global_step = 0
    resume_batch_idx = 0
    resume_epoch_start_step = 0
    if args.resume:
        start_epoch, global_step, resume_batch_idx, resume_epoch_start_step = load_training_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            global_rank,
        )
        print_rank0(
            f"[resume] {args.resume} epoch={start_epoch}, step={global_step}, "
            f"batch={resume_batch_idx}, epoch_start_step={resume_epoch_start_step}"
        )
    else:
        resume_epoch_start_step = global_step
    single._propagate_global_step(train_loader.dataset, global_step)
    set_sampler_global_step(train_sampler, global_step)
    if val_loader is not None:
        single._propagate_global_step(val_loader.dataset, global_step)
    barrier()

    criterion = single.VGGTStyleLoss(
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

    metric_keys = [
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

    tb_writer = single.create_tensorboard_writer(args, output_dir) if global_rank == 0 else None
    optimizer.zero_grad(set_to_none=True)

    total_train_steps = args.max_steps if args.max_steps > 0 else max(
        1,
        math.ceil(steps_per_epoch * args.epochs / max(1, args.accum_steps)),
    )
    step_progress = tqdm(
        total=total_train_steps,
        initial=min(global_step, total_train_steps),
        desc="ddp-train",
        unit="step",
        dynamic_ncols=True,
        disable=global_rank != 0,
    )

    start_time = time.time()
    last_input_print_step = -1
    epoch = start_epoch
    max_steps_drives_training = args.max_steps > 0

    while True:
        if args.max_steps > 0 and global_step >= args.max_steps:
            break
        if not max_steps_drives_training and epoch >= args.epochs:
            break

        model.train()
        model_for_state.train()
        epoch_start_step = global_step
        skip_batches = 0
        if epoch == start_epoch and resume_batch_idx > 0:
            skip_batches = min(int(resume_batch_idx), effective_train_batches)
            epoch_start_step = int(resume_epoch_start_step)

        single._propagate_global_step(train_loader.dataset, epoch_start_step)
        set_sampler_global_step(train_sampler, epoch_start_step)
        train_generator.manual_seed(epoch_loader_seed(args.seed, epoch, global_rank, epoch_start_step))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running: Dict[str, float] = {}
        running_count = 0
        replay_step = epoch_start_step

        for batch_idx, batch in enumerate(train_loader):
            if skip_batches > 0 and batch_idx < skip_batches:
                if (batch_idx + 1) % args.accum_steps == 0:
                    replay_step += 1
                    single._propagate_global_step(train_loader.dataset, replay_step)
                    set_sampler_global_step(train_sampler, replay_step)
                continue
            if skip_batches > 0:
                single._propagate_global_step(train_loader.dataset, global_step)
                set_sampler_global_step(train_sampler, global_step)
                skip_batches = 0
            if batch_idx >= effective_train_batches:
                break
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            if args.canonicalize_first_frame:
                batch = single.canonicalize_to_first_frame(batch)
            if args.normalize_scene:
                batch = single.normalize_scene_batch(
                    batch,
                    num_anchor_frames=min(args.num_scale_frames, int(batch["images"].shape[1])),
                )
            batch = single.to_device(batch, device)
            input_desc = single.format_batch_input(batch)
            single.CURRENT_OOM_CONTEXT.clear()
            single.CURRENT_OOM_CONTEXT.update(
                {
                    "rank": global_rank,
                    "local_rank": local_rank,
                    "world": num_replicas,
                    "phase": "prepare",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch": f"{batch_idx + 1}/{effective_train_batches}",
                    "input": input_desc,
                    "depth_chunk": args.depth_frames_chunk_size,
                }
            )
            if (
                global_rank == 0
                and args.print_input_every > 0
                and global_step != last_input_print_step
                and (global_step == 0 or global_step % args.print_input_every == 0)
            ):
                tqdm.write(
                    f"[input] step={global_step} epoch={epoch + 1} "
                    f"batch={batch_idx + 1}/{effective_train_batches} {input_desc}"
                )
                last_input_print_step = global_step

            should_step = (batch_idx + 1) % args.accum_steps == 0
            sync_context = contextlib.nullcontext()
            if isinstance(model, DDP) and not should_step:
                sync_context = model.no_sync()

            model_for_state.clean_kv_cache()
            single.CURRENT_OOM_CONTEXT["phase"] = "forward+loss"
            with sync_context:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    predictions = model(
                        batch["images"],
                        num_frame_for_scale=min(args.num_scale_frames, int(batch["images"].shape[1])),
                        num_frame_per_block=args.num_frame_per_block,
                        depth_frames_chunk_size=args.depth_frames_chunk_size,
                        causal_inference=True,
                    )
                    losses = criterion(predictions, batch)
                    loss = losses["objective"] / max(1, args.accum_steps)

                is_finite = torch.isfinite(loss.detach()).item()
                is_finite = all_ranks_boolean(bool(is_finite), device)
                if not is_finite:
                    if global_rank == 0:
                        tqdm.write("[warn] non-finite loss on at least one rank, skipping synchronized batch")
                    model_for_state.clean_kv_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue

                single.CURRENT_OOM_CONTEXT["phase"] = "backward"
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            model_for_state.clean_kv_cache()

            scalar_losses = single.loss_to_float_dict(losses)
            reduced_losses = reduce_loss_dict(scalar_losses, device)
            for key, value in reduced_losses.items():
                running[key] = running.get(key, 0.0) + value
            running_count += 1

            if should_step:
                single.CURRENT_OOM_CONTEXT["phase"] = "optimizer_step"
                if args.grad_clip_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        args.grad_clip_norm,
                    )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                single._propagate_global_step(train_loader.dataset, global_step)
                if getattr(train_loader, "sampler", None) is not None and hasattr(
                    train_loader.sampler, "set_global_step"
                ):
                    train_loader.sampler.set_global_step(global_step)
                if val_loader is not None:
                    single._propagate_global_step(val_loader.dataset, global_step)

                if (
                    device.type == "cuda"
                    and args.empty_cache_every > 0
                    and global_step % args.empty_cache_every == 0
                ):
                    torch.cuda.empty_cache()

                if global_rank == 0:
                    step_progress.update(1)
                    current_loss = reduced_losses.get("loss_objective", 0.0)
                    lr = optimizer.param_groups[0]["lr"]
                    seq_len = int(batch["images"].shape[1]) if torch.is_tensor(batch.get("images")) else 0
                    step_progress.set_postfix(
                        {
                            "epoch": epoch + 1,
                            "batch": f"{batch_idx + 1}/{effective_train_batches}",
                            "seq": seq_len,
                            "loss": f"{current_loss:.4f}",
                            "lr": f"{lr:.2e}",
                        },
                        refresh=False,
                    )

                    tb_metrics = dict(reduced_losses)
                    tb_metrics.update(single.tensorboard_input_metrics(batch))
                    tb_metrics.update(single.per_mode_loss_metrics(batch, reduced_losses))
                    tb_metrics["lr"] = lr
                    tb_metrics["epoch"] = float(epoch + 1)
                    tb_metrics["world_size"] = float(num_replicas)
                    tb_metrics["global_batch"] = float(args.batch_size * args.accum_steps * num_replicas)
                    manip_ds = single._find_dataset_with_attr(train_loader.dataset, "_compute_mode_weights")
                    if manip_ds is not None:
                        schedule_weights = manip_ds._compute_mode_weights(global_step)
                        for mode_key, mode_weight in schedule_weights.items():
                            tb_metrics[f"mode_weight/{mode_key}"] = float(mode_weight)
                    sampler_obj = getattr(train_loader, "sampler", None)
                    if sampler_obj is not None and hasattr(sampler_obj, "get_dataset_weights"):
                        mix_weights = sampler_obj.get_dataset_weights(global_step)
                        for src, weight in mix_weights.items():
                            tb_metrics[f"mix_weight/{src}"] = float(weight)
                    single.write_tensorboard_scalars(tb_writer, "train", tb_metrics, global_step)
                    if (
                        tb_writer is not None
                        and args.tensorboard_flush_every > 0
                        and global_step % args.tensorboard_flush_every == 0
                    ):
                        tb_writer.flush()

                    if args.log_every > 0 and global_step % args.log_every == 0 and running_count > 0:
                        averaged = {key: value / running_count for key, value in running.items()}
                        elapsed = time.time() - start_time
                        message = (
                            f"step={global_step} epoch={epoch + 1} "
                            f"lr={lr:.3e} elapsed={elapsed / 60:.1f}m "
                            f"input={input_desc} "
                            f"{single.format_metrics(averaged, metric_keys)}"
                        )
                        tqdm.write(f"[train] {message}")
                        running.clear()
                        running_count = 0

                if args.save_every > 0 and global_step % args.save_every == 0:
                    next_epoch = epoch
                    next_batch_idx = batch_idx + 1
                    next_epoch_start_step = epoch_start_step
                    if next_batch_idx >= effective_train_batches:
                        next_epoch = epoch + 1
                        next_batch_idx = 0
                        next_epoch_start_step = global_step
                    save_distributed_checkpoint(
                        output_dir / f"checkpoint_step_{global_step:08d}.pt",
                        model_for_state,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        args,
                        scaler,
                        device,
                        next_epoch,
                        next_batch_idx,
                        next_epoch_start_step,
                    )
                    barrier()

                should_validate = (
                    args.val_every > 0
                    and global_step % args.val_every == 0
                    and val_scene_count > 0
                )
                if should_validate:
                    if global_rank == 0:
                        if val_loader is not None:
                            val_metrics = single.run_validation(
                                model_for_state,
                                criterion,
                                val_loader,
                                args,
                                device,
                                amp_enabled,
                                amp_dtype,
                            )
                            if val_metrics:
                                tqdm.write(f"[val] step={global_step} {single.format_metrics(val_metrics, metric_keys)}")
                                single.write_tensorboard_scalars(tb_writer, "val", val_metrics, global_step)
                                if tb_writer is not None:
                                    tb_writer.flush()
                        model.train()
                        model_for_state.train()
                    barrier()

        epoch += 1
        if args.save_epoch_checkpoints:
            save_distributed_checkpoint(
                output_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                model_for_state,
                optimizer,
                scheduler,
                epoch,
                global_step,
                args,
                scaler,
                device,
                epoch,
                0,
                global_step,
            )
            barrier()

    if global_rank == 0:
        step_progress.close()
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
    save_distributed_checkpoint(
        output_dir / "checkpoint_last.pt",
        model_for_state,
        optimizer,
        scheduler,
        epoch,
        global_step,
        args,
        scaler,
        device,
        epoch,
        0,
        global_step,
    )
    if global_rank == 0:
        print(f"[done] DDP training finished at step={global_step}; checkpoints saved to {output_dir}")
    barrier()

    if is_dist_ready():
        dist.destroy_process_group()


def build_argparser() -> argparse.ArgumentParser:
    parser = single.build_argparser()
    parser.description = "Multi-node DDP fine-tuning for LingBot-MAP on Manip RGB-D trajectories"

    parser.add_argument("--local_rank", type=int, default=None, help="Optional local rank; torchrun normally sets LOCAL_RANK.")
    parser.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--dist_timeout_min", type=int, default=120)
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable only if DDP reports unused parameters.",
    )
    parser.add_argument("--ddp_gradient_as_bucket_view", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp_static_graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rank0_load_model",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load --model_path only on rank 0, then let DDP broadcast initial weights.",
    )
    parser.add_argument(
        "--quiet_nonzero_ranks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress routine stdout from nonzero ranks; stderr/tracebacks are still visible.",
    )
    parser.add_argument(
        "--save_epoch_checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default because distributed epochs are short and can create too many files.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train_distributed(args)


if __name__ == "__main__":
    main()
