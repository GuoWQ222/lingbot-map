"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Streaming inference (frame-by-frame with KV cache)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/

    # Streaming inference with keyframe KV caching
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/ --mode streaming --keyframe_interval 6

    # Windowed inference (for very long sequences, >500 frames)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10 --mode windowed --window_size 64

    # From video with custom FPS sampling
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10
"""

import argparse
import glob
import os
import sys
import tempfile
import time

# Must be set before `import torch` / any CUDA init. Reduces the reserved-vs-allocated
# memory gap by letting the caching allocator grow segments on demand instead of
# pre-reserving fixed-size blocks.
#
# Caveat: `expandable_segments:True` is **incompatible** with torch.compile's
# `cudagraph_trees` (PyTorch ≤2.8) — checkpoint pool state restore assumes the
# classic fixed-segment topology, and trips
# `RuntimeError: Expected curr_block->next == nullptr` during compiled warmup
# / replay. So when `--compile` is requested we skip the env override and let
# the default allocator run.
if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Foreground mask loading
# =============================================================================

def load_foreground_masks(image_paths, mask_dir, image_size, patch_size, invert=False):
    """Load binary foreground masks aligned with the preprocessed images.

    Mirrors the resize/center-crop in `load_and_preprocess_images(mode="crop")` so the
    returned mask is spatially aligned with the model's depth_conf output.

    Returns:
        np.ndarray of shape (S, H, W) with values in {0.0, 1.0}.
    """
    masks = []
    target_size = image_size
    for path in image_paths:
        basename = os.path.basename(path)
        mask_path = os.path.join(mask_dir, basename)
        if not os.path.exists(mask_path):
            stem = os.path.splitext(basename)[0]
            for ext in (".png", ".jpg", ".jpeg", ".bmp"):
                alt = os.path.join(mask_dir, stem + ext)
                if os.path.exists(alt):
                    mask_path = alt
                    break
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

        h, w = mask.shape[:2]
        new_w = target_size
        new_h = round(h * (new_w / w) / patch_size) * patch_size
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if new_h > target_size:
            start_y = (new_h - target_size) // 2
            mask = mask[start_y : start_y + target_size, :]

        keep = mask == 0 if invert else mask > 0
        masks.append(keep.astype(np.float32))

    return np.stack(masks, axis=0)


# =============================================================================
# Pretty printing
# =============================================================================

_BANNER_WIDTH = 64
_KEY_WIDTH = 20


def _hr(char="=", width=_BANNER_WIDTH):
    print(char * width)


def _banner(title, width=_BANNER_WIDTH):
    _hr("=", width)
    print(title.center(width))
    _hr("=", width)


def _section(title):
    print()
    print(f"[{title}]")


def _kv(key, value, key_width=_KEY_WIDTH):
    print(f"  {key:<{key_width}}: {value}")


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png,.JPG",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8,
                rotate_clockwise_90=False):
    """Load images from folder or video and preprocess into a tensor.

    Returns:
        (images, paths, resolved_image_folder): preprocessed tensor, file paths,
        and the folder containing the source images (for sky mask caching etc.).
    """
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    _section("input")
    _kv("source", image_folder if image_folder else video_path)
    _kv("num_files", f"{len(paths)} (stride={stride}, first_k={first_k})")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    _kv("preprocessed", f"{len(paths)} frames, {w}x{h} (canonical crop)")
    return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    _section("model")
    _kv("class", "GCTStream")
    _kv("input_image_size", args.image_size)
    _kv("model_image_size", args.model_image_size)
    _kv("patch_size", args.patch_size)
    _kv("num_scale_frames", args.num_scale_frames)
    _kv("kv_window", args.kv_cache_sliding_window)
    _kv("max_frame_num", args.max_frame_num)
    _kv("camera_iters", args.camera_num_iterations)
    _kv("attn_backend", "SDPA" if args.use_sdpa else "FlashInfer")

    model = GCTStream(
        img_size=args.model_image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    if args.model_path:
        _kv("checkpoint", args.model_path)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        _kv("missing_keys", len(missing))
        for k in missing:
            print(f"      - {k}")
        _kv("unexpected_keys", len(unexpected))
        for k in unexpected:
            print(f"      - {k}")
        _kv("status", "loaded")

    return model.to(device).eval()


# =============================================================================
# torch.compile (opt-in via --compile)
# =============================================================================

def compile_model(model):
    """Compile hot, fixed-shape modules with mode="reduce-overhead".

    Mirrors the targets in gct_profile.py:compile_model. Unlike the profile script,
    `model.point_head` is **kept** — the demo needs world_points for visualization.
    """
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")


def _warm_streaming(model, images, scale_frames, warm_stream_n, dtype,
                    passes=1, keyframe_interval=1):
    """Drive `clean_kv_cache → Phase 1 → N streaming forwards` `passes` times.

    Warmup inputs are sliced from the already-preprocessed ``images`` tensor, so
    their **spatial shape (H×W) and number of scale frames adapt to the user's
    input** — this is what makes the captured CUDA graphs match what
    ``inference_streaming`` will replay (reduce-overhead mode keys on shape).

    The streaming loop alternates keyframe / non-keyframe forwards according to
    ``keyframe_interval``, mirroring ``inference_streaming``'s call pattern so
    the ``skip_append`` (defer+append+attend+rollback) path is also captured
    during warmup.  Without this, the first non-keyframe in the real run hits
    cold orchestration code and can confuse cudagraph_trees' allocator
    checkpoint state.
    """
    num_avail = int(images.shape[0])
    scale_frames = max(1, min(int(scale_frames), num_avail))
    # Keep at least one streaming frame for the per-frame compile path; if the
    # user supplied <= scale_frames images, shrink scale to free a stream slot.
    if scale_frames >= num_avail:
        scale_frames = max(1, num_avail - 1)
    warm_stream_n = max(1, min(int(warm_stream_n), num_avail - scale_frames))
    kf_int = max(int(keyframe_interval), 1)

    # images: [S, 3, H, W] on device already; slice + add batch dim, no copy of
    # spatial dims so warmup shape == real inference shape (H, W).
    warm_scale = images[:scale_frames].unsqueeze(0).to(dtype)
    warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(dtype)

    for _ in range(passes):
        model.clean_kv_cache()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(
                warm_scale,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        for i in range(warm_stream_n):
            is_keyframe = (kf_int <= 1) or (i % kf_int == 0)
            if not is_keyframe:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(
                    warm_stream[:, i:i + 1],
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            if not is_keyframe:
                model._set_skip_append(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Wipe warmup KV so real inference_streaming starts clean (it also calls
    # clean_kv_cache internally, but this is defensive + makes intent obvious).
    model.clean_kv_cache()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rotate_clockwise_90", action="store_true",
                        help="Rotate source images 90° clockwise before preprocessing "
                             "(crop/resize then operates on the rotated aspect ratio)")

    # Model
    parser.add_argument("--model_path", type=str, default='/cpfs/shared/aigc/guowenqi/lingbot-map/lingbot-map.pt')
    parser.add_argument("--image_size", type=int, default=518,
                        help="Input image size (preprocessing). Images are resized/cropped to this size.")
    parser.add_argument("--model_image_size", type=int, default=None,
                        help="Image size used to construct the model (sizes pos_embed). "
                             "Defaults to --image_size. Set to the checkpoint's training resolution "
                             "when it differs from --image_size; ViT will runtime-interpolate pos_embed.")
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
                        help="streaming: frame-by-frame with KV cache; windowed: overlapping windows for long sequences")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Every N-th frame after scale frames is kept as a keyframe. 1 = every frame. "
            "Streaming: if unset, auto-selected (1 when num_frames <= 320, else ceil(num_frames / 320)) "
            "to bound KV cache. Windowed: defaults to 1; --window_size counts keyframes, so values >1 "
            "expand each window's actual-frame coverage to "
            "scale_frames + (window_size - scale_frames) * keyframe_interval.",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--camera_num_iterations", type=int, default=4,
                        help="Camera head iterative-refinement steps. Default 4; set 1 for faster inference "
                            "(skips 3 refinement passes at a small accuracy cost).")
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile hot modules (reduce-overhead) with a CUDA-graph warmup. "
                            "Streaming mode only; ~5 FPS faster at 518x378. Adds ~30-60 s warmup time.")
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload per-frame predictions to CPU during inference to cut GPU peak memory "
            "(on by default).  Use --no-offload_to_cpu to keep outputs on GPU.",
    )
    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16,
                        help="Overlap between windows in *actual frames*")
    parser.add_argument("--overlap_keyframes", type=int, default=None,
                        help="Overlap expressed in *keyframes* (takes precedence over "
                             "--overlap_size). Converted internally to "
                             "max(num_scale_frames, overlap_keyframes * keyframe_interval) "
                             "actual frames.  Recommended when --keyframe_interval > 1.")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="Directory of foreground masks (one per image, same basename). "
                             "Pixels where mask>0 are kept; background pixels are zeroed out of depth_conf "
                             "so the confidence threshold drops them from the point cloud.")
    parser.add_argument("--mask_invert", action="store_true",
                        help="Invert mask semantics: keep mask==0 pixels (treat zeros as foreground).")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export stride-sampled, resized/cropped images to this folder")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"

    if args.model_image_size is None:
        args.model_image_size = args.image_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print()
    _banner("LingBot-MAP Demo (3D Reconstruction)")
    _section("run")
    _kv("mode", args.mode)
    _kv("device", str(device))
    _kv("port", args.port)

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
    )

    # Export preprocessed images if requested
    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        _kv("export_to", args.export_preprocessed)
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )

    model = load_model(args, device)
    _kv("load_time", f"{time.time() - t0:.1f}s")

    # Pick inference dtype; autocast still runs for the ops that need fp32 (e.g. LayerNorm).
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Cast the aggregator (DINOv2-style trunk) to the inference dtype to remove the
    # redundant fp32 master weight copy + autocast bf16 weight cache (~2-3 GB saved,
    # no measurable quality change). gct_base._predict_* upcasts inputs to fp32 and
    # runs each head under `autocast(enabled=False)`, so camera/depth/point heads
    # keep fp32 weights automatically.
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        _kv("aggregator_dtype", str(dtype))
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = images.shape[0]

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
        else:
            args.keyframe_interval = 1

    if args.mode != "streaming" and args.keyframe_interval != 1:
        args.keyframe_interval = 1

    _section("inference")
    _kv("dtype", str(dtype))
    _kv("input_shape", tuple(images.shape))
    _kv("num_frames", num_frames)
    _kv("keyframe_interval", args.keyframe_interval)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _kv(
            "gpu_mem_pre",
            f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB",
        )

    # ── Inference ────────────────────────────────────────────────────────────
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device
            )

    _kv("duration", f"{time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        _kv(
            "gpu_peak",
            f"alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.max_memory_reserved()/1e9:.2f} GB",
        )

    # ── Post-process ─────────────────────────────────────────────────────────
    if args.offload_to_cpu:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]  # already CPU
    else:
        images_for_post = images

    predictions, images_cpu = postprocess(predictions, images_for_post)

    # ── Apply foreground mask to depth_conf ──────────────────────────────────
    if args.mask_dir is not None:
        _section("foreground_mask")
        _kv("mask_dir", args.mask_dir)
        _kv("invert", bool(args.mask_invert))
        fg_masks = load_foreground_masks(
            paths, args.mask_dir,
            image_size=args.image_size, patch_size=args.patch_size,
            invert=args.mask_invert,
        )
        depth_conf = predictions.get("depth_conf")
        if depth_conf is None:
            print("  warning: predictions has no 'depth_conf'; mask not applied")
        else:
            S, H, W = depth_conf.shape[-3:]
            if fg_masks.shape != (S, H, W):
                fg_resized = np.empty((S, H, W), dtype=np.float32)
                for i in range(S):
                    fg_resized[i] = cv2.resize(
                        fg_masks[i], (W, H), interpolation=cv2.INTER_NEAREST
                    )
                fg_masks = fg_resized
            kept = float(fg_masks.mean())
            _kv("kept_fraction", f"{kept:.3f}")
            mask_t = torch.from_numpy(fg_masks).to(depth_conf.dtype)
            predictions["depth_conf"] = depth_conf * mask_t

    # ── Visualize ────────────────────────────────────────────────────────────
    _section("viewer")
    _kv("url", f"http://localhost:{args.port}")
    _kv("conf_threshold", args.conf_threshold)
    _kv("downsample_factor", args.downsample_factor)
    _kv("point_size", args.point_size)
    _kv("mask_sky", bool(args.mask_sky))
    _hr()
    print()

    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        )
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
