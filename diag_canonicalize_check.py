"""Same as diag_one_sample.py but recenters the world frame so the first
frame's c2w is identity, then recomputes the same losses to verify that this
is the root-cause of the high pose loss."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/cpfs/user/guowenqi/lingbot-map")
sys.path.insert(0, str(ROOT))

from train import (  # noqa: E402
    ManipTrajectoryDataset,
    VGGTStyleLoss,
    build_argparser,
    build_model,
    normalize_scene_batch,
    se3_3x4_to_4x4,
    to_device,
    w2c_to_c2w_extrinsics,
    loss_to_float_dict,
)


def canonicalize_to_first_frame(batch):
    """Make frame 0's c2w the identity by left-multiplying every w2c with
    c2w_0 (i.e. E_new = E_i @ inv(E_0_w2c) @ ... -> equivalent to expressing
    every camera in frame-0's coordinate system).

    Stored extrinsics are 3x4 OpenCV w2c. After this op: the first frame's
    extrinsics is exactly [I|0]. The world_points are also rotated/shifted
    into frame-0 coordinates so the depth-->world mapping stays consistent.
    """
    extr_w2c = batch["extrinsics"]  # [B,F,3,4]
    B, F = extr_w2c.shape[:2]

    extr_w2c_44 = se3_3x4_to_4x4(extr_w2c)            # [B,F,4,4]
    c2w = torch.linalg.inv(extr_w2c_44)               # [B,F,4,4]

    c2w_0 = c2w[:, :1]                                # [B,1,4,4]
    c2w_0_inv = torch.linalg.inv(c2w_0)               # this is w2c of frame 0

    # New c2w is c2w_0_inv @ c2w_i, so frame 0 becomes identity.
    new_c2w = torch.matmul(c2w_0_inv, c2w)
    new_w2c = torch.linalg.inv(new_c2w)

    new_extr_w2c = new_w2c[..., :3, :]                # back to 3x4

    # World points get rotated by c2w_0_inv too (they were in original world frame).
    wp = batch["world_points"]                        # [B,F,H,W,3]
    R = c2w_0_inv[..., :3, :3]                        # [B,1,3,3]
    t = c2w_0_inv[..., :3, 3]                         # [B,1,3]
    R_b = R.reshape(B, 1, 1, 1, 3, 3)
    t_b = t.reshape(B, 1, 1, 1, 3)
    wp_new = torch.einsum("bfhwij,bfhwj->bfhwi", R_b.expand(B, F, wp.shape[2], wp.shape[3], 3, 3), wp) + t_b

    new_batch = dict(batch)
    new_batch["extrinsics"] = new_extr_w2c
    new_batch["world_points"] = wp_new
    return new_batch


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest_path = Path(args.scene_manifest)
    scene_dirs = []
    with open(manifest_path) as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                scene_dirs.append(Path(ln))

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
        mode_weights_initial={"S": 1.0, "W": 0.0, "M": 0.0},
        mode_weights_final={"S": 1.0, "W": 0.0, "M": 0.0},
        mode_warmup_start=args.mode_warmup_start,
        mode_warmup_end=args.mode_warmup_end,
    )

    sample = ds[0]
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in sample.items()}

    F = batch["images"].shape[1]
    print(f"[shapes] images={tuple(batch['images'].shape)}  mode={batch['sample_mode']}")

    # Build the criterion with the same defaults as train.py uses.
    criterion = VGGTStyleLoss(
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

    print("[model] building & loading checkpoint...")
    model = build_model(args, device)
    model.eval()

    n_anchor = min(args.num_scale_frames, F)

    for label, do_canonicalize in [("ORIGINAL (no canonicalization)", False),
                                    ("CANONICALIZED (frame0 = identity)", True)]:
        b = {k: v for k, v in batch.items() if torch.is_tensor(v)}
        if do_canonicalize:
            b = canonicalize_to_first_frame(b)
        nb = normalize_scene_batch(b, num_anchor_frames=n_anchor)
        nb_dev = to_device(nb, device)
        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
            pred = model(
                nb_dev["images"],
                num_frame_for_scale=n_anchor,
                num_frame_per_block=args.num_frame_per_block,
                depth_frames_chunk_size=args.depth_frames_chunk_size,
                causal_inference=True,
            )
            losses = criterion(pred, nb_dev)
        scalars = loss_to_float_dict(losses)

        first_c2w = w2c_to_c2w_extrinsics(nb["extrinsics"])[0, 0]
        print(f"\n=== {label} ===")
        print(f"  frame-0 c2w (post-normalize):\n{first_c2w.detach().cpu().numpy().round(4)}")
        for k in ["loss_objective", "loss_camera", "loss_T", "loss_R", "loss_FL",
                  "loss_relative_pose", "loss_relative_rot", "loss_relative_trans",
                  "loss_conf_depth", "loss_reg_depth", "loss_grad_depth"]:
            if k in scalars:
                print(f"  {k:>22s} = {scalars[k]:.4f}")


if __name__ == "__main__":
    main()
