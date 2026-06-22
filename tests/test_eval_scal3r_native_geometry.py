import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAL3R_ROOT = REPO_ROOT.parent / "Scal3R"
for path in (REPO_ROOT, SCAL3R_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import eval_scal3r
import train as T


class EmptyLoader:
    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())


class EvalScal3RNativeGeometryTest(unittest.TestCase):
    def test_scal3r_geometry_matches_scal3r_preprocess_and_saved_intrinsics(self):
        from scal3r.utils.base_utils import DotDict
        from scal3r.utils.cam_utils import write_camera
        from scal3r.utils.image_utils import load_and_preprocess_images

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_path = tmp_path / "frame.png"
            Image.new("RGB", (853, 481), (31, 63, 127)).save(image_path)

            dataset_cfg = DotDict(
                render_ratio=1.0,
                rot90=None,
                proc_max_size=518,
                proc_align_size=14,
                center_crop=True,
                focal_ratio=1.0,
                cam_param_type="abs_quat_fov",
                use_world_coord=True,
            )
            sequence, scal3r_h, scal3r_w = load_and_preprocess_images(
                [str(image_path)],
                dataset_cfg,
                preprocess_workers=1,
            )
            scal3r_k = sequence[0].ixt.numpy()

            geometry = eval_scal3r.compute_scal3r_preprocess_geometry(
                width=853,
                height=481,
                proc_max_size=518,
                proc_align_size=14,
                center_crop=True,
                focal_ratio=1.0,
            )
            dummy_k = eval_scal3r.build_scal3r_dummy_intrinsics(853, 481, focal_ratio=1.0)
            lingbot_k = T.preprocess_intrinsics(dummy_k, 853, 481, geometry).numpy()

            self.assertEqual((geometry["crop_height"], geometry["crop_width"]), (scal3r_h, scal3r_w))
            np.testing.assert_allclose(lingbot_k, scal3r_k, rtol=0.0, atol=1e-5)

            camera_dir = tmp_path / "camera"
            cameras = {
                "000000": DotDict(
                    H=scal3r_h,
                    W=scal3r_w,
                    K=scal3r_k,
                    R=np.eye(3, dtype=np.float32),
                    T=np.zeros(3, dtype=np.float32),
                )
            }
            write_camera(cameras, str(camera_dir))
            saved_k = eval_scal3r.load_scal3r_intrinsics(camera_dir / "intri.yml", 1)[0]
            np.testing.assert_allclose(saved_k, lingbot_k, rtol=0.0, atol=1e-5)

    def test_evaluate_one_mode_uses_scal3r_native_loader(self):
        args = argparse.Namespace(
            split="val",
            max_scenes_eval=5,
            eval_shard_count=4,
            eval_shard_index=2,
            eval_num_frames=64,
            image_size=518,
            depth_align="pi3_scale_shift",
            secondary_depth_align="",
            pointcloud_metrics=False,
            pointcloud_workers=0,
            pointcloud_kdtree_workers=1,
            pointcloud_align="pi3_icp",
            pointcloud_icp_backend="open3d",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(eval_scal3r, "build_scal3r_native_eval_loader", return_value=(EmptyLoader(), [])) as native_loader:
                with mock.patch.object(eval_scal3r.E, "build_eval_loader") as shared_loader:
                    with self.assertRaisesRegex(RuntimeError, "No batches evaluated"):
                        eval_scal3r.evaluate_one_mode(
                            args,
                            torch.device("cpu"),
                            "left_moving_tracks",
                            Path(tmpdir),
                        )

        native_loader.assert_called_once_with(args, eval_mode="left_moving_tracks")
        shared_loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
