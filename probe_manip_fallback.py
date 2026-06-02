"""Fast probe of ManipTrajectoryDataset's W/T linspace-fallback rate.

Reads only the line count of each scene's ``{camera}_pose.txt`` to get the
trajectory length per camera, then runs Monte-Carlo simulations of the W /
T mode walkers (byte-faithful copies of train.py's logic). Image I/O and
pose-file parsing are skipped entirely, so 2000 iterations finish in seconds.

Usage:
    python probe_manip_fallback.py                          # defaults match train.sh
    python probe_manip_fallback.py --n 5000
    python probe_manip_fallback.py --min 8 --max 64         # compare 8-64 vs 16-64
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import numpy as np


WRIST_DIR_RE = re.compile(r"^realsense_.+$")
STATIC_DIR_RE = re.compile(r"^surround_.+$")


def count_lines(path: Path) -> int:
    try:
        # Reading in chunks beats opening the whole file as a list — pose files
        # can be a few hundred KB. Counting via os.read keeps it ~10× faster.
        n = 0
        with open(path, "rb") as f:
            while True:
                buf = f.read(1 << 16)
                if not buf:
                    break
                n += buf.count(b"\n")
        return n
    except OSError:
        return 0


def read_manifest(path: Path) -> list[Path]:
    with open(path, "r") as f:
        return [Path(line.strip()) for line in f if line.strip()]


def scan_scene(scene_dir: Path) -> dict:
    """Return {'wrist': [(name, L), ...], 'static': [(name, L), ...]}.

    Each scene_dir contains per-camera subdirectories, each holding a
    ``<camera>_pose.txt`` file whose line count equals the trajectory length
    for that camera. Cameras with no pose file (or 0 lines) are skipped.
    """
    wrist: list[tuple[str, int]] = []
    static: list[tuple[str, int]] = []
    try:
        entries = list(scene_dir.iterdir())
    except OSError:
        return {"wrist": wrist, "static": static}
    for entry in entries:
        if not entry.is_dir():
            continue
        name = entry.name
        pose_path = entry / f"{name}_pose.txt"
        if not pose_path.is_file():
            continue
        L = count_lines(pose_path)
        if L <= 0:
            continue
        if WRIST_DIR_RE.match(name):
            wrist.append((name, L))
        elif STATIC_DIR_RE.match(name):
            static.append((name, L))
    return {"wrist": wrist, "static": static}


def simulate_W(L: int, max_frames: int, min_frames: int,
               stride_min: int, stride_max: int) -> tuple[int, int, bool, int, bool]:
    """Replicate ``_sample_mode_W`` walker (forward, then backward, then linspace).

    Returns (fwd, bwd, bwd_used, final, fallback).
    """
    start = random.randint(0, L - 1) if L > 1 else 0

    def walk(direction: int) -> int:
        sel = 1
        idx = start
        while sel < max_frames:
            step = random.randint(stride_min, stride_max)
            nxt = idx + direction * step
            if nxt < 0 or nxt >= L:
                break
            sel += 1
            idx = nxt
        return sel

    fwd_len = walk(+1)
    bwd_used = False
    bwd_len = 0
    natural = fwd_len
    if fwd_len < min_frames:
        bwd_len = walk(-1)
        if bwd_len > fwd_len:
            natural = bwd_len
            bwd_used = True
    will_fallback = (natural < min_frames and L >= min_frames)
    final_len = min_frames if will_fallback else natural
    return fwd_len, bwd_len, bwd_used, final_len, will_fallback


def simulate_T(L: int, max_frames: int, min_frames: int,
               stride_min: int, stride_max: int) -> tuple[int, int, bool, int, bool]:
    """Replicate ``_sample_mode_T`` (forward, then backward, then linspace).

    Returns (fwd, bwd, bwd_used, final, fallback).
    """
    start = random.randint(0, L - 1) if L > 1 else 0

    def walk(direction: int) -> int:
        sel = 1
        idx = start
        while sel < max_frames:
            step = random.randint(stride_min, stride_max)
            nxt = idx + direction * step
            if nxt < 0 or nxt >= L:
                break
            sel += 1
            idx = nxt
        return sel

    fwd_len = walk(+1)
    bwd_used = False
    bwd_len = 0
    natural = fwd_len
    if fwd_len < min_frames:
        bwd_len = walk(-1)
        if bwd_len > fwd_len:
            natural = bwd_len
            bwd_used = True
    will_fallback = (natural < min_frames and L >= min_frames)
    final_len = min_frames if will_fallback else natural
    return fwd_len, bwd_len, bwd_used, final_len, will_fallback


def histogram(values, bins, label: str) -> None:
    hist, _ = np.histogram(values, bins=bins)
    total = max(1, len(values))
    peak = max(hist) if len(hist) > 0 else 1
    print(f"\n  {label} histogram (n={len(values)}):")
    for i, c in enumerate(hist):
        lo, hi = bins[i], bins[i + 1]
        bar = "#" * int(50 * c / peak) if peak > 0 else ""
        pct = 100.0 * c / total
        print(f"    [{lo:>3},{hi:>3})  n={c:>5}  ({pct:>5.1f}%)  {bar}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="runs/seqlen_probe/T64/manip_trajectory_manifest.txt",
    )
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of (scene, view) walker simulations.")
    parser.add_argument("--max_scenes", type=int, default=0,
                        help="Limit how many distinct scenes are scanned (0 = all).")
    parser.add_argument("--max", type=int, default=64, help="MAX_SAMPLE_FRAMES")
    parser.add_argument("--min", type=int, default=16, help="MIN_SAMPLE_FRAMES")
    parser.add_argument("--w_stride_min", type=int, default=10)
    parser.add_argument("--w_stride_max", type=int, default=60)
    parser.add_argument("--t_stride_min", type=int, default=15)
    parser.add_argument("--t_stride_max", type=int, default=60)
    parser.add_argument("--long5_marker", default="Manip_long5")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    scene_dirs = read_manifest(manifest_path)
    print(f"[probe] manifest: {manifest_path}")
    print(f"[probe] {len(scene_dirs)} scenes in manifest", flush=True)

    if args.max_scenes > 0 and args.max_scenes < len(scene_dirs):
        rng_scene = random.Random(args.seed)
        scene_dirs = rng_scene.sample(scene_dirs, args.max_scenes)
        print(f"[probe] sub-sampling to {len(scene_dirs)} scenes for the scan",
              flush=True)

    # Scan trajectory lengths for each scene (slow part: ~50-200ms/scene on OSS).
    print("[probe] scanning pose-file lengths...", flush=True)
    long34_pool: list[list[tuple[str, int]]] = []  # list of wrist-view lists
    long5_pool: list[list[tuple[str, int]]] = []   # list of static-view lists (T mode)
    skipped = 0
    for i, scene_dir in enumerate(scene_dirs):
        info = scan_scene(scene_dir)
        is_long5 = args.long5_marker in str(scene_dir)
        if is_long5:
            views = info["static"] or info["wrist"]
            if not views:
                skipped += 1
                continue
            long5_pool.append(views)
        else:
            views = info["wrist"] or info["static"]
            if not views:
                skipped += 1
                continue
            long34_pool.append(views)
        if (i + 1) % 200 == 0:
            print(f"  scanned {i + 1}/{len(scene_dirs)}", flush=True)

    print(f"[probe] scan done. long3/4 scenes: {len(long34_pool)}, "
          f"long5 scenes: {len(long5_pool)}, skipped: {skipped}", flush=True)

    if not long34_pool and not long5_pool:
        print("[probe] no valid scenes found; aborting"); return

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Monte-Carlo simulation: each iteration picks a scene proportional to pool
    # size (matches uniform random.choice over scene_dirs in production).
    total_pool = len(long34_pool) + len(long5_pool)
    W = dict(count=0, bwd_used=0, fallback=0, fwd_lens=[], bwd_lens=[],
             final_lens=[], L_dist=[])
    T = dict(count=0, bwd_used=0, fallback=0, fwd_lens=[], bwd_lens=[],
             final_lens=[], L_dist=[])

    for _ in range(args.n):
        # Pick scene type weighted by actual pool sizes
        if random.random() < len(long5_pool) / total_pool:
            views = random.choice(long5_pool)
            _name, L = random.choice(views)
            fwd, bwd, bwd_used, final_len, did_fb = simulate_T(
                L, args.max, args.min, args.t_stride_min, args.t_stride_max
            )
            T["count"] += 1
            T["L_dist"].append(L)
            T["fwd_lens"].append(fwd)
            T["bwd_lens"].append(bwd if bwd > 0 else fwd)
            T["final_lens"].append(final_len)
            if bwd_used:
                T["bwd_used"] += 1
            if did_fb:
                T["fallback"] += 1
        else:
            views = random.choice(long34_pool)
            _name, L = random.choice(views)
            fwd, bwd, bwd_used, final_len, did_fb = simulate_W(
                L, args.max, args.min, args.w_stride_min, args.w_stride_max
            )
            W["count"] += 1
            W["L_dist"].append(L)
            W["fwd_lens"].append(fwd)
            W["bwd_lens"].append(bwd if bwd > 0 else fwd)
            W["final_lens"].append(final_len)
            if bwd_used:
                W["bwd_used"] += 1
            if did_fb:
                W["fallback"] += 1

    print(f"\n=== Probe results: n={args.n} simulations, MIN={args.min}, MAX={args.max} ===")
    print(f"  pool composition: long3/4 ({len(long34_pool)}) vs long5 ({len(long5_pool)}) "
          f"-> sampled W={W['count']}, T={T['count']}")

    bins = [0, args.min, args.min + 1, 24, 32, 40, 48, 56, args.max, args.max + 1]

    def report(label: str, S: dict, stride_min: int, stride_max: int) -> None:
        n = S["count"]
        if n == 0:
            print(f"\n[{label}] no samples"); return
        fb = S["fallback"]
        bu = S["bwd_used"]
        print(f"\n[{label}]  stride={stride_min}-{stride_max}  n={n}")
        print(f"  fallback rate:    {fb}/{n} = {100.0 * fb / n:.2f}%")
        print(f"  backward-walk used: {bu}/{n} = {100.0 * bu / n:.2f}%")
        L_arr = np.array(S["L_dist"])
        print(f"  trajectory L: min={L_arr.min()}, p25={int(np.percentile(L_arr, 25))}, "
              f"median={int(np.median(L_arr))}, p75={int(np.percentile(L_arr, 75))}, "
              f"max={L_arr.max()}")
        if S.get("fwd_lens"):
            Fw = np.array(S["fwd_lens"])
            print(f"  fwd walk len: min={Fw.min()}, median={int(np.median(Fw))}, "
                  f"max={Fw.max()}, mean={Fw.mean():.1f}")
        if S.get("bwd_lens"):
            Bw = np.array(S["bwd_lens"])
            print(f"  bwd walk len: min={Bw.min()}, median={int(np.median(Bw))}, "
                  f"max={Bw.max()}, mean={Bw.mean():.1f}")
        F = np.array(S["final_lens"])
        print(f"  final clip len: min={F.min()}, median={int(np.median(F))}, "
              f"max={F.max()}, mean={F.mean():.1f}")
        histogram(S["final_lens"], bins, "final_len")

    report("W (Manip_long3/4 wrist)", W, args.w_stride_min, args.w_stride_max)
    report("T (Manip_long5 surround)", T, args.t_stride_min, args.t_stride_max)


if __name__ == "__main__":
    main()
