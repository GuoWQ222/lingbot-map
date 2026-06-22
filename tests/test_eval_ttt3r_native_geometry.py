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

import eval_ttt3r
from train import FrameEntry


class EvalTTT3RNativeGeometryTest(unittest.TestCase):
    def test_square_512_ttt3r_geometry_is_center_crop_384_by_512(self):
        geometry = eval_ttt3r.compute_ttt3r_preprocess_geometry(512, 512, 512)

        self.assertEqual(geometry["new_width"], 512)
        self.assertEqual(geometry["new_height"], 512)
        self.assertEqual(geometry["crop_left"], 0)
        self.assertEqual(geometry["crop_top"], 64)
        self.assertEqual(geometry["crop_width"], 512)
        self.assertEqual(geometry["crop_height"], 384)

    def test_predictions_are_not_stretched_to_a_different_metric_grid(self):
        predictions = {
            "depth": torch.zeros(1, 1, 384, 512, 1),
            "depth_conf": torch.ones(1, 1, 384, 512, 1),
            "world_points": torch.zeros(1, 1, 384, 512, 3),
        }

        with self.assertRaisesRegex(ValueError, "TTT3R prediction shape"):
            eval_ttt3r.crop_predictions_to_hw(predictions, (512, 512))

    def test_native_loader_outputs_gt_on_ttt3r_grid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "Manip_long6" / "data" / "fake_scene"
            root.mkdir(parents=True)
            rgb_path = root / "rgb.png"
            depth_path = root / "depth.npy"
            Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8)).save(rgb_path)
            np.save(depth_path, np.ones((512, 512), dtype=np.float32))
            entry = FrameEntry(
                frame_id=0,
                view_id=0,
                rgb_path=rgb_path,
                depth_path=depth_path,
                mask_path=None,
                camera_name="realsense_left",
            )
            dataset = eval_ttt3r.TTT3RNativeEvalDataset(
                [root],
                eval_mode="wrist_track",
                eval_num_frames=1,
                eval_wrist_camera_name="realsense_left",
                eval_surround_camera_name="surround_cam_moving",
                eval_seed=42,
                clip_len=1,
                image_size=512,
                patch_size=16,
                preprocess_mode="crop",
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

        self.assertEqual(tuple(image.shape), (3, 384, 512))
        self.assertEqual(tuple(depth.shape), (384, 512))
        self.assertEqual(tuple(mask.shape), (384, 512))
        self.assertEqual(tuple(world_points.shape), (384, 512, 3))
        self.assertAlmostEqual(float(loaded_intrinsics[1, 2]), -64.0)


if __name__ == "__main__":
    unittest.main()
