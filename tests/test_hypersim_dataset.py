import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_fake_train_helpers():
    fake_train = types.ModuleType("train")
    fake_train.compute_preprocess_geometry = (
        lambda width, height, image_size, patch_size, preprocess_mode: None
    )
    fake_train.apply_preprocess_to_image = (
        lambda image, geometry, resample, fill: image
    )
    fake_train.preprocess_intrinsics = (
        lambda K, width, height, geometry: torch.from_numpy(np.asarray(K, dtype=np.float32))
    )
    fake_train.depth_to_world_points = (
        lambda depth, intrinsics, extrinsics: torch.zeros(
            (*depth.shape, 3), dtype=torch.float32
        )
    )
    sys.modules["train"] = fake_train


def _write_pose(path: Path):
    path.write_text(
        "\n".join(
            [
                "1 0 0 0",
                "0 1 0 0",
                "0 0 1 0",
                "10 0 4",
                "0 10 3",
                "0 0 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_scene(root: Path, scene: str, num_frames: int, *, missing_pose: bool = False):
    scene_root = root / scene
    for subdir in ("rgb", "depth", "pose"):
        (scene_root / subdir).mkdir(parents=True, exist_ok=True)
    for idx in range(num_frames):
        stem = f"{idx:04d}"
        Image.new("RGB", (8, 6), color=(idx, 2 * idx, 3 * idx)).save(
            scene_root / "rgb" / f"{stem}.jpg"
        )
        Image.fromarray(np.full((6, 8), 1000, dtype=np.uint16)).save(
            scene_root / "depth" / f"{stem}.png"
        )
        if not missing_pose or idx != num_frames - 1:
            _write_pose(scene_root / "pose" / f"{stem}.txt")


class HypersimDatasetTest(unittest.TestCase):
    def setUp(self):
        _install_fake_train_helpers()

    def test_pose_file_rows_are_read_as_world_to_camera(self):
        from lingbot_map.data.hypersim import _read_pose_file

        with tempfile.TemporaryDirectory() as tmp:
            pose_path = Path(tmp) / "pose.txt"
            pose_path.write_text(
                "\n".join(
                    [
                        "1 0 0 1.5",
                        "0 1 0 -2.0",
                        "0 0 1 3.0",
                        "10 0 4",
                        "0 10 3",
                        "0 0 1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            w2c, K = _read_pose_file(str(pose_path))

            np.testing.assert_allclose(
                w2c,
                np.array(
                    [
                        [1.0, 0.0, 0.0, 1.5],
                        [0.0, 1.0, 0.0, -2.0],
                        [0.0, 0.0, 1.0, 3.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            )
            np.testing.assert_allclose(
                K,
                np.array(
                    [[10.0, 0.0, 4.0], [0.0, 10.0, 3.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            )

    def test_discovers_valid_ai_scenes_and_reads_depth_in_meters(self):
        from lingbot_map.data.hypersim import HypersimTrajectoryDataset

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_scene(root, "ai_valid_cam_00", 5)
            _write_scene(root, "not_ai_scene", 5)
            _write_scene(root, "ai_incomplete_cam_00", 5, missing_pose=True)

            dataset = HypersimTrajectoryDataset(
                str(root),
                num_views=3,
                min_views=3,
                image_size=8,
                patch_size=1,
                samples_per_scene=1,
                io_max_workers=1,
            )

            self.assertEqual(dataset.sequences, ["ai_valid_cam_00"])
            self.assertEqual(dataset.max_interval, 4)
            self.assertEqual(dataset.video_prob, 0.5)
            self.assertEqual(dataset.fix_interval_prob, 0.5)
            self.assertEqual(dataset.block_shuffle, 16)

            sample = dataset[0]
            self.assertEqual(sample["sample_mode"], "hypersim")
            self.assertEqual(sample["images"].shape, (3, 3, 6, 8))
            self.assertEqual(sample["depths"].shape, (3, 6, 8))
            self.assertTrue(torch.allclose(sample["depths"], torch.ones_like(sample["depths"])))
            self.assertTrue(torch.all(sample["point_masks"]))


if __name__ == "__main__":
    unittest.main()
