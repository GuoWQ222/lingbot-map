import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_streamvggt


class EvalStreamVGGTNativePreprocessTest(unittest.TestCase):
    def test_args_force_streamvggt_native_preprocess_independent_of_training_args(self):
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
                streamvggt_repo="/cpfs/user/guowenqi/StreamVGGT",
                model_weights="/cpfs/user/guowenqi/StreamVGGT/checkpoints.pth",
                strict_load=False,
                forward_mode="stream",
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
                streamvggt_preprocess_mode="crop",
                depth_align="pi3_scale_shift",
                secondary_depth_align="",
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

            args = eval_streamvggt.args_from_run_json(eval_args)

        self.assertEqual(args.input_preprocess, "vggt_native")
        self.assertEqual(args.image_size, 518)
        self.assertEqual(args.patch_size, 14)
        self.assertEqual(args.preprocess_mode, "crop")
        self.assertEqual(args.streamvggt_preprocess_mode, "crop")
        self.assertEqual(args.vggt_preprocess_mode, "crop")


if __name__ == "__main__":
    unittest.main()
