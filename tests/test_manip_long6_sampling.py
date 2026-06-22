import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import (
    FrameEntry,
    ManipTrajectoryDataset,
    OPENCV_TO_GENMANIP_CAMERA_ROTATION,
    parse_mode_weights,
    read_manip_long6_camera_arrays,
)


def _entry(camera_name: str, view_id: int, frame_id: int) -> FrameEntry:
    base = Path("/tmp") / camera_name
    return FrameEntry(
        frame_id=frame_id,
        view_id=view_id,
        rgb_path=base / "rgb" / f"{frame_id:06d}.png",
        depth_path=base / "depth" / f"{frame_id:06d}.png",
        mask_path=None,
        camera_name=camera_name,
        pose_path=base / "pose.txt",
    )


def _long6_entries(num_frames: int = 80):
    cameras = [
        "realsense_left",
        "realsense_right",
        "surround_cam_fixed",
        "surround_cam_moving",
    ]
    return [
        _entry(camera_name, view_id, frame_id)
        for view_id, camera_name in enumerate(cameras)
        for frame_id in range(num_frames)
    ]


class _FakeCameraStageCache:
    def __init__(self, root: Path):
        self.stage_root = root / "stage"
        self.calls = []

    def should_stage(self, source_path):
        return True

    def resolve_dir(self, **kwargs):
        raise AssertionError("Manip fallback should stage one camera, not the whole scene")

    def stage_dir(self, *, dataset, relative_key, oss_uri, dest_root, count_entry=True):
        self.calls.append((dataset, relative_key, oss_uri, Path(dest_root)))
        camera_dir = Path(dest_root) / relative_key
        (camera_dir / "rgb").mkdir(parents=True, exist_ok=True)
        (camera_dir / "depth_npy").mkdir(exist_ok=True)
        np.save(camera_dir / "intrinsics.npy", np.eye(3, dtype=np.float32))
        np.save(camera_dir / "camera2env_extrinsics.npy", np.repeat(np.eye(4, dtype=np.float32)[None], 32, axis=0))
        for frame_id in range(32):
            stem = f"{frame_id:06d}"
            (camera_dir / "rgb" / f"{stem}.png").write_bytes(b"x")
            np.save(camera_dir / "depth_npy" / f"{stem}.npy", np.zeros((2, 2), dtype=np.float32))
        return camera_dir

    def enable_fallback_from_exception(self, exc, *, source_path=None):
        return False


class ManipLong6SamplingTest(unittest.TestCase):
    def _dataset(self, mode_weights: str) -> ManipTrajectoryDataset:
        return ManipTrajectoryDataset(
            [Path("/oss-guowenqi/Manip_long6/data/fake_scene")],
            clip_len=64,
            image_size=16,
            patch_size=1,
            sequence_mode="manip_4d_mixed",
            max_sample_frames=4,
            min_sample_frames=2,
            w_stride_min=8,
            w_stride_max=8,
            moving_stride_min=6,
            moving_stride_max=6,
            fixed_stride_min=16,
            fixed_stride_max=16,
            long6_root_marker="Manip_long6",
            long6_mode_weights=parse_mode_weights(mode_weights),
        )


    def test_fallback_stages_only_selected_long6_camera(self):
        with tempfile.TemporaryDirectory(prefix="Manip_long6_stage_test_") as tmpdir:
            cache = _FakeCameraStageCache(Path(tmpdir))
            dataset = ManipTrajectoryDataset(
                [Path("/oss-guowenqi/Manip_long6/data/fake_scene")],
                clip_len=64,
                image_size=16,
                patch_size=1,
                sequence_mode="manip_4d_mixed",
                max_sample_frames=4,
                min_sample_frames=2,
                long6_root_marker="Manip_long6",
                long6_mode_weights=parse_mode_weights("W=0,T=1,F=0"),
                oss_stage_cache=cache,
                data_roots=["/oss-guowenqi/Manip_long6/data"],
                oss_uri_roots=["oss://bucket/guowenqi/Manip_long6/data"],
            )

            def fake_load(scene_dir, entry, jitter_params=None):
                self.assertIn("surround_cam_moving", str(entry.rgb_path))
                return (
                    torch.zeros(3, 16, 16),
                    torch.zeros(16, 16),
                    torch.ones(16, 16, dtype=torch.bool),
                    torch.eye(3),
                    torch.zeros(3, 4),
                    torch.zeros(16, 16, 3),
                )

            with mock.patch.object(dataset, "_load_one", side_effect=fake_load):
                sample = dataset[0]

            self.assertEqual(sample["sample_mode"], "T")
            self.assertEqual(len(cache.calls), 1)
            dataset_name, relative_key, oss_uri, _ = cache.calls[0]
            self.assertEqual(dataset_name, "manip")
            self.assertEqual(relative_key, "fake_scene/surround_cam_moving")
            self.assertEqual(oss_uri, "oss://bucket/guowenqi/Manip_long6/data/fake_scene/surround_cam_moving/")

    def test_long6_wrist_mode_uses_w_stride(self):
        dataset = self._dataset("W=1,T=0,F=0")

        with mock.patch("train.random.randint", side_effect=lambda lo, hi: lo):
            selected, mode = dataset._sample_manip_4d_mixed(
                _long6_entries(),
                Path("/oss-guowenqi/Manip_long6/data/fake_scene"),
            )

        self.assertEqual(mode, "W")
        self.assertTrue(all(item.camera_name.startswith("realsense") for item in selected))
        self.assertEqual([item.frame_id for item in selected], [0, 8, 16, 24])

    def test_long6_moving_mode_uses_moving_stride(self):
        dataset = self._dataset("W=0,T=1,F=0")

        with mock.patch("train.random.randint", side_effect=lambda lo, hi: lo):
            selected, mode = dataset._sample_manip_4d_mixed(
                _long6_entries(),
                Path("/oss-guowenqi/Manip_long6/data/fake_scene"),
            )

        self.assertEqual(mode, "T")
        self.assertTrue(all(item.camera_name == "surround_cam_moving" for item in selected))
        self.assertEqual([item.frame_id for item in selected], [0, 6, 12, 18])

    def test_long6_fixed_mode_uses_fixed_stride(self):
        dataset = self._dataset("W=0,T=0,F=1")

        with mock.patch("train.random.randint", side_effect=lambda lo, hi: lo):
            selected, mode = dataset._sample_manip_4d_mixed(
                _long6_entries(),
                Path("/oss-guowenqi/Manip_long6/data/fake_scene"),
            )

        self.assertEqual(mode, "F")
        self.assertTrue(all(item.camera_name == "surround_cam_fixed" for item in selected))
        self.assertEqual([item.frame_id for item in selected], [0, 16, 32, 48])

    def test_long6_omitted_modes_are_not_used_as_fallback(self):
        dataset = self._dataset("W=1,T=1,F=1")

        with mock.patch.object(
            dataset,
            "_sample_single_camera_walk",
            side_effect=RuntimeError("single-camera modes unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "single-camera modes unavailable"):
                dataset._sample_long6_mixed({}, {}, [], [])


    def test_long6_pose_arrays_convert_genmanip_c2e_to_opencv_w2c(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            camera_dir = Path(tmpdir)
            intrinsics = np.eye(3, dtype=np.float32)
            genmanip_c2e = np.eye(4, dtype=np.float32)
            genmanip_c2e[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            np.save(camera_dir / "intrinsics.npy", intrinsics)
            np.save(camera_dir / "camera2env_extrinsics.npy", genmanip_c2e[None])

            loaded_intrinsics, extrinsics_by_frame = read_manip_long6_camera_arrays(
                camera_dir / "camera2env_extrinsics.npy"
            )

        opencv_to_genmanip = np.eye(4, dtype=np.float32)
        opencv_to_genmanip[:3, :3] = OPENCV_TO_GENMANIP_CAMERA_ROTATION
        expected_w2c = np.linalg.inv(genmanip_c2e @ opencv_to_genmanip).astype(np.float32)
        np.testing.assert_allclose(loaded_intrinsics, intrinsics)
        np.testing.assert_allclose(extrinsics_by_frame[0], expected_w2c, rtol=1e-6, atol=1e-6)

    def test_rejects_non_long6_scene_paths(self):
        with self.assertRaisesRegex(ValueError, "Manip_long6"):
            ManipTrajectoryDataset(
                [Path("/oss-guowenqi/Manip_long5/data/fake_scene")],
                clip_len=64,
                image_size=16,
                patch_size=1,
                sequence_mode="manip_4d_mixed",
            )

    def test_ignores_legacy_images_depth_real_layout(self):
        with tempfile.TemporaryDirectory(prefix="Manip_long6_") as tmpdir:
            scene_dir = Path(tmpdir) / "scene"
            camera_dir = scene_dir / "realsense_left"
            (camera_dir / "images").mkdir(parents=True)
            (camera_dir / "depth_real").mkdir()
            (camera_dir / "images" / "000000.png").write_bytes(b"not-a-real-image")
            (camera_dir / "depth_real" / "000000.png").write_bytes(b"not-a-real-depth")
            (camera_dir / "realsense_left_pose.txt").write_text("# 1 0 0\n# 0 1 0\n# 0 0 1\n0 0 0 0 1 0 0 0\n")

            dataset = ManipTrajectoryDataset(
                [scene_dir],
                clip_len=64,
                image_size=16,
                patch_size=1,
                sequence_mode="manip_4d_mixed",
            )

            self.assertEqual(dataset._manip_entries_for_scene(scene_dir), [])


if __name__ == "__main__":
    unittest.main()
