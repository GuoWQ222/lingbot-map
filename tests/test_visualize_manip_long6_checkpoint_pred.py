import sys
import unittest
from pathlib import Path

import torch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.pose_enc import extri_intri_to_pose_encoding
from visualize_manip_long6_checkpoint_surround_moving_pred import (
    camera_focus_from_pose,
    format_camera_pose,
    parse_camera_pose,
    set_camera_handle_scale,
    split_camera_visibility_flags,
    uniform_sample_entries,
    unproject_depth_with_c2w_pose,
    visible_camera_flags,
)


class UniformCheckpointVisualizationSamplingTest(unittest.TestCase):
    def test_uniform_sample_entries_includes_first_and_last_without_duplicates(self):
        entries = list(range(100))

        sampled = uniform_sample_entries(entries, 64)

        self.assertEqual(len(sampled), 64)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 99)
        self.assertEqual(len(set(sampled)), 64)
        self.assertEqual(sampled, sorted(sampled))

    def test_uniform_sample_entries_keeps_all_entries_when_shorter_than_target(self):
        entries = list(range(12))

        sampled = uniform_sample_entries(entries, 64)

        self.assertEqual(sampled, entries)


class CheckpointVisualizationUnprojectionTest(unittest.TestCase):
    def test_unproject_depth_with_c2w_pose_treats_pose_translation_as_camera_center(self):
        depth = torch.ones(1, 1, 2, 2, 1)
        c2w = torch.tensor(
            [[[[1.0, 0.0, 0.0, 1.0],
               [0.0, 1.0, 0.0, 2.0],
               [0.0, 0.0, 1.0, 3.0]]]]
        )
        intrinsics = torch.tensor(
            [[[[1.0, 0.0, 1.0],
               [0.0, 1.0, 1.0],
               [0.0, 0.0, 1.0]]]]
        )
        pose_enc = extri_intri_to_pose_encoding(
            c2w,
            intrinsics,
            image_size_hw=(2, 2),
            pose_encoding_type="absT_quaR_FoV",
        )

        points = unproject_depth_with_c2w_pose(depth, pose_enc)

        expected = torch.tensor(
            [[[[[0.0, 1.0, 4.0],
                [1.0, 1.0, 4.0]],
               [[0.0, 2.0, 4.0],
                [1.0, 2.0, 4.0]]]]]
        )
        torch.testing.assert_close(points, expected)


class CheckpointVisualizationCameraVisibilityTest(unittest.TestCase):
    def test_camera_visibility_follows_visible_point_mode(self):
        self.assertEqual(
            visible_camera_flags("Current frame", 2, 5, show_cameras=True),
            [False, False, True, False, False],
        )
        self.assertEqual(
            visible_camera_flags("0 to current frame", 2, 5, show_cameras=True),
            [True, True, True, False, False],
        )
        self.assertEqual(
            visible_camera_flags("All frames", 2, 5, show_cameras=False),
            [False, False, False, False, False],
        )

    def test_split_camera_visibility_tracks_pred_and_gt_toggles(self):
        pred_flags, gt_flags = split_camera_visibility_flags(
            "0 to current frame",
            current_frame=2,
            frame_count=5,
            show_cameras=True,
            show_pred=True,
            show_gt=False,
        )

        self.assertEqual(pred_flags, [True, True, True, False, False])
        self.assertEqual(gt_flags, [False, False, False, False, False])

        pred_flags, gt_flags = split_camera_visibility_flags(
            "Current frame",
            current_frame=3,
            frame_count=5,
            show_cameras=True,
            show_pred=False,
            show_gt=True,
        )

        self.assertEqual(pred_flags, [False, False, False, False, False])
        self.assertEqual(gt_flags, [False, False, False, True, False])

    def test_camera_focus_from_pose_matches_dataloader_view_jump(self):
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        position, look_at, up = camera_focus_from_pose(pose)

        np.testing.assert_allclose(position, np.array([1.0, 2.0, 2.85], dtype=np.float32))
        np.testing.assert_allclose(look_at, np.array([1.0, 2.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(up, np.array([0.0, -1.0, 0.0], dtype=np.float32))


class CheckpointVisualizationViewPoseControlsTest(unittest.TestCase):
    def test_format_camera_pose_outputs_plain_seven_numbers(self):
        text = format_camera_pose([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])

        self.assertEqual(text, "1.000000 2.000000 3.000000 1.000000 0.000000 0.000000 0.000000")
        self.assertEqual(len(text.split()), 7)
        position, wxyz = parse_camera_pose(text)
        self.assertEqual(position, (1.0, 2.0, 3.0))
        self.assertEqual(wxyz, (1.0, 0.0, 0.0, 0.0))

    def test_parse_camera_pose_accepts_plain_seven_numbers(self):
        position, wxyz = parse_camera_pose("1 2 3 0.5 0.5 -0.5 -0.5")

        self.assertEqual(position, (1.0, 2.0, 3.0))
        self.assertEqual(wxyz, (0.5, 0.5, -0.5, -0.5))


if __name__ == "__main__":
    unittest.main()
