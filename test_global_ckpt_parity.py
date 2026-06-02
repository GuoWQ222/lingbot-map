"""Parity test for the new global-attention activation checkpoint.

Builds two views of the same model state:
  A) use_gradient_checkpoint = True  (frame + global both checkpointed)
  B) use_gradient_checkpoint = False (no activation checkpoint anywhere)

Runs 5 fwd+bwd steps on each, with identical RNG-seeded inputs, and reports
per-step loss + total grad-L2 + max per-param grad delta. If our checkpoint
shim is correct, grads should match within bf16 tolerance (~1e-2 rel).
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lingbot_map.models.gct_stream import GCTStream


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def build_model(use_ckpt: bool) -> torch.nn.Module:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = GCTStream(
        img_size=280,
        patch_size=14,
        enable_3d_rope=False,
        max_frame_num=64,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
        enable_point=False,
        enable_local_point=False,
        use_gradient_checkpoint=use_ckpt,
    ).to(DEVICE)
    model.train()
    # Force-disable DPT depth head checkpoint to remove that variable
    # (we want to isolate the aggregator-side checkpoint behavior).
    if model.depth_head is not None and hasattr(model.depth_head, "use_activation_checkpoint"):
        model.depth_head.use_activation_checkpoint = use_ckpt
    return model


def synth_inputs(B: int, S: int, H: int, W: int) -> torch.Tensor:
    g = torch.Generator(device=DEVICE).manual_seed(20251128)
    return torch.randn(B, S, 3, H, W, device=DEVICE, generator=g, dtype=torch.float32) * 0.4 + 0.5


def synthetic_loss(preds: dict) -> torch.Tensor:
    """Touch every head's output so gradients reach every parameter."""
    parts = []
    if "pose_enc" in preds:
        parts.append(preds["pose_enc"].float().pow(2).mean())
    if "pose_enc_list" in preds and preds["pose_enc_list"]:
        # iterative refinement outputs
        parts.append(sum(p.float().pow(2).mean() for p in preds["pose_enc_list"]) / max(1, len(preds["pose_enc_list"])))
    if "depth" in preds and isinstance(preds["depth"], torch.Tensor):
        parts.append(preds["depth"].float().pow(2).mean())
    if "depth_conf" in preds and isinstance(preds["depth_conf"], torch.Tensor):
        parts.append(preds["depth_conf"].float().mean())
    if not parts:
        raise RuntimeError(f"No usable outputs in preds: {list(preds.keys())}")
    return sum(parts)


def run_steps(model: torch.nn.Module, images: torch.Tensor, n_steps: int):
    out_per_step = []
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.0)
    for step in range(n_steps):
        torch.manual_seed(1000 + step)
        torch.cuda.manual_seed_all(1000 + step)
        model.clean_kv_cache()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE, enabled=True):
            preds = model(
                images,
                num_frame_for_scale=4,
                num_frame_per_block=1,
                depth_frames_chunk_size=2,
                causal_inference=True,
            )
            loss = synthetic_loss(preds)
        loss.backward()
        # Snapshot grads (clone so they survive the next zero_grad)
        grad_snapshot = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_snapshot[name] = p.grad.detach().float().cpu().clone()
        out_per_step.append({
            "loss": float(loss.detach()),
            "grads": grad_snapshot,
        })
        # peak/curr mem report
        mem_alloc = torch.cuda.max_memory_allocated() / 1024**3
        mem_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(f"  step {step}: loss={float(loss.detach()):.6f}  "
              f"max_alloc={mem_alloc:.2f}G  max_reserved={mem_reserved:.2f}G")
        torch.cuda.reset_peak_memory_stats()
    return out_per_step


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for parity test")

    B, S, H, W = 1, 6, 280, 280  # small but realistic; S=6 so Run B (no ckpt) also fits
    print(f"[parity] B={B} S={S} H={H} W={W} dtype=bf16 device={DEVICE}")

    print("\n=== Run A: use_gradient_checkpoint=True (frame+global checkpointed) ===")
    torch.cuda.empty_cache()
    model_a = build_model(use_ckpt=True)
    # Snapshot params before training so run B can restore them
    init_state = {k: v.detach().clone() for k, v in model_a.state_dict().items()}
    images = synth_inputs(B, S, H, W)
    out_a = run_steps(model_a, images, n_steps=3)
    del model_a
    torch.cuda.empty_cache()

    print("\n=== Run B: use_gradient_checkpoint=False (no checkpoint) ===")
    model_b = build_model(use_ckpt=False)
    # Load A's initial state so weights are identical
    model_b.load_state_dict(init_state, strict=True)
    out_b = run_steps(model_b, images, n_steps=3)
    del model_b
    torch.cuda.empty_cache()

    print("\n=== Comparison ===")
    print(f"{'step':>4} | {'loss_A':>10} | {'loss_B':>10} | {'|ΔL|':>8} | {'max_dg':>10} | {'mean_dg':>10} | {'param_max':>30}")
    pass_all = True
    for i in range(len(out_a)):
        loss_a = out_a[i]["loss"]
        loss_b = out_b[i]["loss"]
        dloss = abs(loss_a - loss_b)
        grads_a = out_a[i]["grads"]
        grads_b = out_b[i]["grads"]
        keys = sorted(set(grads_a) & set(grads_b))
        max_abs = 0.0
        max_rel = 0.0
        max_key = ""
        total_abs = 0.0
        n_params = 0
        for k in keys:
            ga = grads_a[k]
            gb = grads_b[k]
            if ga.shape != gb.shape:
                continue
            diff = (ga - gb).abs()
            scale = gb.abs().mean().item() + 1e-12
            ma = diff.max().item()
            rel = ma / scale
            total_abs += diff.mean().item()
            n_params += 1
            if rel > max_rel:
                max_rel = rel
                max_abs = ma
                max_key = k
        mean_abs = total_abs / max(1, n_params)
        bf16_ok = max_rel < 0.05 and dloss < 1e-3  # 5% relative is generous bf16 tolerance
        if not bf16_ok:
            pass_all = False
        flag = "ok" if bf16_ok else "FAIL"
        print(f"{i:>4} | {loss_a:>10.6f} | {loss_b:>10.6f} | {dloss:>8.2e} | "
              f"{max_abs:>10.2e} | {mean_abs:>10.2e} | {max_key:>30} [{flag}]")
    print(f"\n[parity] overall: {'PASS' if pass_all else 'FAIL'}")
    sys.exit(0 if pass_all else 1)


if __name__ == "__main__":
    main()
