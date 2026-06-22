import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_vggt
from train import compute_preprocess_geometry


class EvalVGGTNativePreprocessTest(unittest.TestCase):
    def test_args_force_vggt_native_preprocess_independent_of_training_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args_path = Path(tmpdir) / "args.json"
            args_path.write_text(
                json.dumps(
                    {
                        "image_size": 280,
                        "patch_size": 16,
                        "preprocess_mode": "pad",
                    }
                )
            )

            eval_args = argparse.Namespace(
                train_args_json=str(args_path),
                output_dir=str(Path(tmpdir) / "eval"),
                vggt_repo="/cpfs/user/guowenqi/vggt",
                model_name="facebook/VGGT-1B",
                model_weights="/cpfs/user/guowenqi/vggt/model.pt",
                strict_load=False,
                split="val",
                max_scenes_eval=1,
                eval_shard_count=1,
                eval_shard_index=0,
                num_workers=0,
                device="cpu",
                per_scene_csv=True,
                save_predictions=False,
                print_every=1,
                eval_strategy="left_moving_tracks",
                eval_num_frames=64,
                eval_wrist_camera_name="realsense_left",
                eval_surround_camera_name="surround_cam_moving",
                eval_seed=42,
                image_size=518,
                depth_align="pi3_scale_shift",
                secondary_depth_align="",
                geometry_normalization="none",
                camera_align="sim3",
                pointcloud_metrics=True,
                pointcloud_max_points=100000,
                pointcloud_align="pi3_icp",
                pointcloud_icp_threshold=0.1,
                pointcloud_icp_max_iterations=30,
                pointcloud_icp_backend="open3d",
                pointcloud_kdtree_workers=1,
                pointcloud_workers=1,
                amp=True,
                amp_dtype="bf16",
            )

            args = eval_vggt.args_from_run_json(eval_args)

        self.assertEqual(args.input_preprocess, "vggt_native")
        self.assertEqual(args.image_size, 518)
        self.assertEqual(args.patch_size, 14)
        self.assertEqual(args.preprocess_mode, "crop")
        self.assertEqual(args.vggt_preprocess_mode, "crop")

    def test_vggt_native_geometry_matches_upstream_crop_size_rule(self):
        square = compute_preprocess_geometry(512, 512, 518, 14, "crop")
        self.assertEqual(square["new_width"], 518)
        self.assertEqual(square["new_height"], 518)
        self.assertEqual(square["crop_width"], 518)
        self.assertEqual(square["crop_height"], 518)

        landscape = compute_preprocess_geometry(640, 480, 518, 14, "crop")
        self.assertEqual(landscape["new_width"], 518)
        self.assertEqual(landscape["new_height"], 392)
        self.assertEqual(landscape["crop_width"], 518)
        self.assertEqual(landscape["crop_height"], 392)


if __name__ == "__main__":
    unittest.main()
