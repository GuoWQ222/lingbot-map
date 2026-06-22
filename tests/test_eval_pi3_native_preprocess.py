import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_pi3
from train import FrameEntry


class EvalPi3NativePreprocessTest(unittest.TestCase):
    def test_pi3_native_geometry_matches_load_images_new_width_rule(self):
        square = eval_pi3.compute_pi3_preprocess_geometry(512, 512, 518)
        self.assertEqual(square["new_width"], 518)
        self.assertEqual(square["new_height"], 518)
        self.assertEqual(square["crop_width"], 518)
        self.assertEqual(square["crop_height"], 518)

        landscape = eval_pi3.compute_pi3_preprocess_geometry(640, 480, 518)
        self.assertEqual(landscape["new_width"], 518)
        self.assertEqual(landscape["new_height"], 392)
        self.assertEqual(landscape["crop_width"], 518)
        self.assertEqual(landscape["crop_height"], 392)

    def test_native_dataset_outputs_gt_on_pi3_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "Manip_long6" / "data" / "fake_scene"
            root.mkdir(parents=True)
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.npy"
            Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(rgb_path)
            np.save(depth_path, np.ones((480, 640), dtype=np.float32))
            entry = FrameEntry(
                frame_id=0,
                view_id=0,
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=None,
                camera_name="realsense_left",
            )
            dataset = eval_pi3.Pi3NativeEvalDataset(
                [root],
                eval_mode="wrist_track",
                eval_num_frames=1,
                eval_wrist_camera_name="realsense_left",
                eval_surround_camera_name="surround_cam_moving",
                eval_seed=42,
                clip_len=1,
                image_size=518,
                patch_size=14,
                preprocess_mode="resize",
                sequence_mode="all_views",
                view_ids=[],
                camera_names=[],
                sample_strategy="fixed_stride",
                frame_stride=1,
                random_stride_min=1,
                random_stride_max=1,
                random_interval_start="first",
                max_sample_frames=1,
                min_sample_frames=1,
                depth_scale=1.0,
                min_depth=1e-6,
                max_depth=0.0,
                use_mask=False,
                invert_cam_extrinsics=False,
                samples_per_scene=1,
            )

            intrinsics = np.eye(3, dtype=np.float32)
            extrinsics = torch.eye(4, dtype=torch.float32)[:3]
            with mock.patch.object(dataset, "_load_camera_for_entry", return_value=(intrinsics, extrinsics)):
                image, depth, mask, loaded_intrinsics, _, world_points = dataset._load_one(root, entry)

        self.assertEqual(tuple(image.shape), (3, 392, 518))
        self.assertEqual(tuple(depth.shape), (392, 518))
        self.assertEqual(tuple(mask.shape), (392, 518))
        self.assertEqual(tuple(world_points.shape), (392, 518, 3))
        self.assertAlmostEqual(float(loaded_intrinsics[0, 0]), 518.0 / 640.0)
        self.assertAlmostEqual(float(loaded_intrinsics[1, 1]), 392.0 / 480.0)

    def test_local_pi3_loader_fallback_matches_native_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_path = Path(tmpdir) / "rgb.png"
            Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8)).save(rgb_path)

            images = eval_pi3.load_images_like_pi3([str(rgb_path)], new_width=518)

        self.assertEqual(tuple(images.shape), (1, 3, 392, 518))
        self.assertGreaterEqual(float(images.min()), 0.0)
        self.assertLessEqual(float(images.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
