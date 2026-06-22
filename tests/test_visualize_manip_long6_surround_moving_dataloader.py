import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualize_manip_long6_surround_moving_dataloader import (
    format_camera_pose,
    parse_camera_pose,
    set_camera_handle_scale,
    uniform_sample_entries,
    visible_camera_flags,
)


class ManipLong6SurroundMovingUniformSamplingTest(unittest.TestCase):
    def test_uniform_sample_entries_matches_checkpoint_viewer_policy(self):
        entries = list(range(100))

        sampled = uniform_sample_entries(entries, 64)

        self.assertEqual(len(sampled), 64)
        self.assertEqual(sampled[0], 0)
        self.assertEqual(sampled[-1], 99)
        self.assertEqual(len(set(sampled)), 64)
        self.assertEqual(sampled, sorted(sampled))

    def test_uniform_sample_entries_keeps_all_entries_when_shorter_than_target(self):
        self.assertEqual(uniform_sample_entries(list(range(12)), 64), list(range(12)))


class ManipLong6SurroundMovingCameraVisibilityTest(unittest.TestCase):
    def test_camera_visibility_follows_visible_point_mode(self):
        self.assertEqual(
            visible_camera_flags("Current frame", 2, 5, show_cameras=True),
            [False, False, True, False, False],
        )
        self.assertEqual(
            visible_camera_flags("0 to current frame", 2, 5, show_cameras=True),
            [True, True, True, False, False],
        )


class ManipLong6SurroundMovingViewPoseControlsTest(unittest.TestCase):
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
