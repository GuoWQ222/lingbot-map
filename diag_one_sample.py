"""Single-sample diagnostic: what does the model see, what does it predict, and
what does the GT actually look like?

We import the training-script utilities so the data path is byte-identical to
train.sh, then dump:
  * depth value stats per frame
  * camera intrinsics
  * GT extrinsics (w2c) and the c2w used for the loss
  * anchor-scale `s` from normalize_scene_batch
  * first-frame GT c2w (is it identity? if not the world is in scene-absolute frame)
  * GT pose encoding (absT_quaR_FoV)
  * Predicted pose encoding from the loaded checkpoint
  * Per-component absolute differences
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/cpfs/user/guowenqi/lingbot-map")
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    ManipTrajectoryDataset,
    build_argparser,
    build_model,
    normalize_scene_batch,
    to_device,
    w2c_to_c2w_extrinsics,
)
from lingbot_map.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402


def fmt(t: torch.Tensor) -> str:
    return np.array2string(t.detach().cpu().float().numpy(), precision=4, suppress_small=True)


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset using the manifest at SCENE_MANIFEST.
    manifest_path = Path(args.scene_manifest)
    scene_dirs = []
    with open(manifest_path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                scene_dirs.append(Path(ln))
    print(f"[manifest] {len(scene_dirs)} scenes")

    ds = ManipTrajectoryDataset(
        scene_dirs=scene_dirs[:64],
        clip_len=args.clip_len,
        image_size=args.image_size,
        patch_size=14,
        preprocess_mode="crop",
        sequence_mode=args.sequence_mode,
        sample_strategy=args.sample_strategy,
        frame_stride=args.frame_stride,
        random_stride_min=args.random_stride_min,
        random_stride_max=args.random_stride_max,
        random_interval_start=args.random_interval_start,
        max_sample_frames=args.max_sample_frames,
        min_sample_frames=args.min_sample_frames,
        depth_scale=args.depth_scale,
        use_mask=args.use_mask,
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
        mode_weights_initial={"S": 1.0, "W": 0.0, "M": 0.0},  # force S mode for determinism
        mode_weights_final={"S": 1.0, "W": 0.0, "M": 0.0},
        mode_warmup_start=args.mode_warmup_start,
        mode_warmup_end=args.mode_warmup_end,
    )

    sample = ds[0]
    # Stack to a batch of size 1
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v])
             for k, v in sample.items()}

    images = batch["images"]
    depths = batch["depths"]
    point_masks = batch["point_masks"]
    intrinsics = batch["intrinsics"]
    extrinsics_w2c = batch["extrinsics"]
    print(f"[shapes] images={tuple(images.shape)}  depths={tuple(depths.shape)}  "
          f"intr={tuple(intrinsics.shape)}  extr={tuple(extrinsics_w2c.shape)}  "
          f"point_masks={tuple(point_masks.shape)}  mode={batch['sample_mode']}")

    # ------------------------------------------------------------------
    # Per-frame raw depth stats (BEFORE anchor-scale normalization)
    # ------------------------------------------------------------------
    print("\n=== Per-frame DEPTH stats (raw, after depth_scale=%.0f) ===" % args.depth_scale)
    F = depths.shape[1]
    for f in range(F):
        d = depths[0, f]
        m = point_masks[0, f]
        valid = d[m]
        if valid.numel() > 0:
            print(f"  frame {f}: valid={valid.numel():>7d}/{m.numel()}  "
                  f"min={valid.min():.3f}  median={valid.median():.3f}  "
                  f"mean={valid.mean():.3f}  max={valid.max():.3f}")
        else:
            print(f"  frame {f}: NO valid pixels")

    # ------------------------------------------------------------------
    # GT camera frames
    # ------------------------------------------------------------------
    print("\n=== Intrinsics (frame 0) ===")
    print(fmt(intrinsics[0, 0]))
    print(f"  image HxW = {images.shape[-2]} x {images.shape[-1]}")

    print("\n=== GT extrinsics (w2c, frame 0..2) ===")
    for f in range(min(3, F)):
        print(f"frame {f}:\n{fmt(extrinsics_w2c[0, f])}")

    gt_c2w_pre = w2c_to_c2w_extrinsics(extrinsics_w2c)
    print("\n=== GT c2w (pre-normalize, frame 0..2) ===")
    for f in range(min(3, F)):
        print(f"frame {f} c2w:\n{fmt(gt_c2w_pre[0, f])}")

    # ------------------------------------------------------------------
    # Anchor-scale normalization
    # ------------------------------------------------------------------
    n_anchor = min(args.num_scale_frames, F)
    norm_batch = normalize_scene_batch({k: v for k, v in batch.items() if torch.is_tensor(v)},
                                       num_anchor_frames=n_anchor)
    s = float(norm_batch["anchor_scale"][0])
    print(f"\n=== Anchor scale (n_anchor={n_anchor}): s = {s:.4f}  "
          f"(==> normalized depth = depth / s)")

    print("\n=== GT c2w (POST-normalize, frame 0..2) ===")
    gt_c2w_norm = w2c_to_c2w_extrinsics(norm_batch["extrinsics"])
    for f in range(min(3, F)):
        print(f"frame {f} c2w:\n{fmt(gt_c2w_norm[0, f])}")

    # GT pose encoding
    gt_pose_enc = extri_intri_to_pose_encoding(
        gt_c2w_norm,
        norm_batch["intrinsics"],
        images.shape[-2:],
        pose_encoding_type="absT_quaR_FoV",
    )
    print("\n=== GT pose encoding (absT_quaR_FoV), frame 0..2 ===")
    for f in range(min(3, F)):
        v = gt_pose_enc[0, f]
        print(f"  frame {f}: T={fmt(v[:3])}  quat={fmt(v[3:7])}  fov={fmt(v[7:])}")

    # Normalized depth stats
    nd = norm_batch["depths"]
    pm = norm_batch["point_masks"]
    valid = nd[0][pm[0]]
    if valid.numel() > 0:
        print(f"\n=== Normalized depth stats (depth/s): "
              f"min={valid.min():.4f}  median={valid.median():.4f}  "
              f"mean={valid.mean():.4f}  max={valid.max():.4f}  "
              f"#valid={valid.numel()}")

    # ------------------------------------------------------------------
    # Model forward
    # ------------------------------------------------------------------
    print("\n[model] building & loading checkpoint...")
    model = build_model(args, device)
    model.eval()

    norm_batch_dev = to_device(norm_batch, device)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        pred = model(
            norm_batch_dev["images"],
            num_frame_for_scale=min(args.num_scale_frames, F),
            num_frame_per_block=args.num_frame_per_block,
            depth_frames_chunk_size=args.depth_frames_chunk_size,
            causal_inference=True,
        )

    pose_list = pred["pose_enc_list"]
    pred_pose_enc = pose_list[-1].float()  # last refinement stage
    print(f"\n=== Predicted pose encoding (last stage), frame 0..2 ===")
    for f in range(min(3, F)):
        v = pred_pose_enc[0, f]
        print(f"  frame {f}: T={fmt(v[:3])}  quat={fmt(v[3:7])}  fov={fmt(v[7:])}")

    print("\n=== |pred - gt| absolute difference, frame 0..2 ===")
    diff = (pred_pose_enc - gt_pose_enc.to(pred_pose_enc.device)).abs()
    for f in range(min(3, F)):
        v = diff[0, f]
        print(f"  frame {f}: dT={fmt(v[:3])}  dquat={fmt(v[3:7])}  dfov={fmt(v[7:])}")

    # Predicted depth stats
    if "depth" in pred:
        pd = pred["depth"]
        if isinstance(pd, (list, tuple)):
            pd = pd[-1]
        pd = pd.float()
        print(f"\n=== Predicted depth shape={tuple(pd.shape)}  "
              f"min={pd.min():.4f}  median={pd.median():.4f}  "
              f"mean={pd.mean():.4f}  max={pd.max():.4f}")
        # vs GT normalized depth
        gtd = norm_batch_dev["depths"]
        valid = norm_batch_dev["point_masks"]
        # squeeze any singleton channel
        if pd.dim() == gtd.dim() + 1 and pd.shape[-1] == 1:
            pd_cmp = pd.squeeze(-1)
        else:
            pd_cmp = pd
        if pd_cmp.shape == gtd.shape:
            err = (pd_cmp - gtd).abs()
            err_v = err[valid]
            print(f"  |pred_depth - gt_depth| over valid: mean={err_v.mean():.4f} "
                  f"median={err_v.median():.4f}  p90={torch.quantile(err_v, 0.9):.4f}")


if __name__ == "__main__":
    main()
