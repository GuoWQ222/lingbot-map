from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any


def finite_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def summarize_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {"clips_evaluated": len(rows)}
    depth_prefixes = sorted(
        {
            key.rsplit("_", 1)[0]
            for row in rows
            for key in row
            if key.startswith("depth_") and finite_float(row.get(key)) is not None
        }
    )
    depth: dict[str, Any] = {}
    for prefix in depth_prefixes:
        metric = prefix.removeprefix("depth_")
        vals = [v for row in rows if (v := finite_float(row.get(prefix))) is not None]
        if vals:
            depth[metric] = mean(vals)
    if depth:
        depth["n_scene_rows"] = len(rows)
        out["depth_scene_mean"] = depth

    camera_keys = [
        "cam_ate_rmse",
        "cam_rpe_trans_rmse",
        "cam_rpe_rot_rmse_deg",
        "cam_fov_h_deg_mae",
        "cam_fov_w_deg_mae",
    ]
    camera: dict[str, Any] = {}
    for key in camera_keys:
        vals = [v for row in rows if (v := finite_float(row.get(key))) is not None]
        if vals:
            camera[f"{key}_mean"] = mean(vals)
            camera[f"{key}_median"] = float(median(vals))
            camera[f"{key}_n"] = len(vals)
    if camera:
        out["camera"] = camera

    pc_keys = ["pc_ACC", "pc_Completeness", "pc_CD", "pc_n_pred_points", "pc_n_gt_points"]
    pointcloud: dict[str, Any] = {}
    for key in pc_keys:
        vals = [v for row in rows if (v := finite_float(row.get(key))) is not None]
        if vals:
            pointcloud[key.removeprefix("pc_")] = mean(vals)
            pointcloud[f"{key.removeprefix('pc_')}_n"] = len(vals)
    if pointcloud:
        out["pointcloud"] = pointcloud
    return out


def weighted_depth_from_shards(metrics: list[dict[str, Any]]) -> dict[str, float]:
    weighted: dict[str, tuple[float, float]] = {}
    valid_pixels_total = 0.0
    n_frames_total = 0.0
    for data in metrics:
        depth = data.get("modes", {}).get("manip_track", {}).get("overall", {}).get("depth", {})
        if not isinstance(depth, dict):
            continue
        n_frames = finite_float(depth.get("n_frames")) or 0.0
        n_frames_total += n_frames
        valid_pixels_total += finite_float(depth.get("valid_pixels_total")) or 0.0
        for key, value in depth.items():
            if key in {"n_frames", "valid_pixels_total"}:
                continue
            val = finite_float(value)
            if val is None or n_frames <= 0:
                continue
            acc, weight = weighted.get(key, (0.0, 0.0))
            weighted[key] = (acc + val * n_frames, weight + n_frames)
    out = {key: acc / weight for key, (acc, weight) in weighted.items() if weight > 0}
    if n_frames_total:
        out["n_frames"] = n_frames_total
    if valid_pixels_total:
        out["valid_pixels_total"] = valid_pixels_total
    return out


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: aggregate_streamvggt_shards.py OUTPUT_ROOT")
    output_root = Path(sys.argv[1]).resolve()
    shard_dirs = sorted(output_root.glob("shard_*_of_*"))
    metric_paths = [p / "metrics.json" for p in shard_dirs if (p / "metrics.json").is_file()]
    if not metric_paths:
        raise SystemExit(f"no shard metrics found under {output_root}")

    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    rows: list[dict[str, str]] = []
    all_fields: set[str] = set()
    for shard_dir in shard_dirs:
        csv_path = shard_dir / "per_scene.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["shard"] = shard_dir.name
                rows.append(row)
                all_fields.update(row.keys())

    if rows:
        combined_csv = output_root / "combined_per_scene.csv"
        fieldnames = ["shard"] + sorted(k for k in all_fields if k != "shard")
        with combined_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    by_group: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_group.setdefault(row.get("metric_group") or "unknown", []).append(row)

    template = metrics[0]
    combined = {
        "source": "aggregate_streamvggt_shards.py",
        "output_root": str(output_root),
        "n_shards_found": len(metric_paths),
        "metric_paths": [str(path) for path in metric_paths],
        "model": template.get("model"),
        "train_args_json": template.get("train_args_json"),
        "split": template.get("split"),
        "eval_strategy": template.get("eval_strategy"),
        "eval_num_frames": template.get("eval_num_frames"),
        "image_size": template.get("image_size"),
        "depth_align": template.get("modes", {}).get("manip_track", {}).get("depth_align"),
        "camera_align": template.get("modes", {}).get("manip_track", {}).get("camera_align"),
        "pointcloud_align": template.get("modes", {}).get("manip_track", {}).get("pointcloud_align"),
        "skipped_total": sum(
            int(data.get("modes", {}).get("manip_track", {}).get("skipped", 0))
            for data in metrics
        ),
        "overall": summarize_rows(rows),
        "groups": {group: summarize_rows(group_rows) for group, group_rows in sorted(by_group.items())},
        "weighted_shard_depth": weighted_depth_from_shards(metrics),
    }

    out_path = output_root / "combined_metrics.json"
    out_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[aggregate_streamvggt_shards] wrote {out_path}")
    if rows:
        print(f"[aggregate_streamvggt_shards] wrote {output_root / 'combined_per_scene.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
