from pathlib import Path

import numpy as np
import torch
from PIL import Image

import train as T
from eval_model_geometry import load_frame_with_model_geometry


class FakeDataset:
    depth_scale = 1.0
    min_depth = 0.0
    max_depth = 0.0
    use_mask = True

    def _load_camera_for_entry(self, scene_dir: Path, entry: T.FrameEntry):
        intrinsics = np.array(
            [[10.0, 0.0, 1.0], [0.0, 20.0, 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        extrinsics = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        return intrinsics, torch.from_numpy(extrinsics)

    def _apply_color_jitter(self, rgb, jitter_params):
        return rgb


def test_load_frame_with_model_geometry_syncs_depth_mask_intrinsics_and_world_points(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (4, 2), (255, 0, 0)).save(rgb_path)
    np.save(depth_path, np.full((2, 4), 2.0, dtype=np.float32))
    Image.fromarray(np.ones((2, 4), dtype=np.uint8) * 255).save(mask_path)

    entry = T.FrameEntry(
        frame_id=7,
        view_id=3,
        rgb_path=rgb_path,
        depth_path=depth_path,
        mask_path=mask_path,
    )

    def model_geometry(width: int, height: int):
        assert (width, height) == (4, 2)
        return {
            "new_width": 8,
            "new_height": 4,
            "crop_left": 0,
            "crop_top": 0,
            "crop_width": 8,
            "crop_height": 4,
            "pad_left": 0,
            "pad_right": 0,
            "pad_top": 0,
            "pad_bottom": 0,
        }

    image, depth, point_mask, intrinsics, extrinsics, world_points = load_frame_with_model_geometry(
        FakeDataset(),
        tmp_path,
        entry,
        model_geometry,
        rgb_resample=Image.Resampling.NEAREST,
    )

    assert tuple(image.shape) == (3, 4, 8)
    assert tuple(depth.shape) == (4, 8)
    assert tuple(point_mask.shape) == (4, 8)
    assert bool(point_mask.all())
    assert torch.allclose(depth, torch.full((4, 8), 2.0))
    assert torch.allclose(intrinsics[0], torch.tensor([20.0, 0.0, 2.0]))
    assert torch.allclose(intrinsics[1], torch.tensor([0.0, 40.0, 1.0]))
    assert torch.allclose(extrinsics, torch.eye(4)[:3])
    assert tuple(world_points.shape) == (4, 8, 3)
    assert torch.isfinite(world_points).all()
