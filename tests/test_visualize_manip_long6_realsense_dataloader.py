import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualize_manip_long6_realsense_dataloader import (
    set_camera_handle_scale,
    visible_camera_flags,
    visible_frame_indices,
)


class ManipLong6RealsenseViserControlsTest(unittest.TestCase):
    def test_current_frame_mode_clamps_to_one_frame(self):
        self.assertEqual(visible_frame_indices("Current frame", -2, 5), [0])
        self.assertEqual(visible_frame_indices("Current frame", 3, 5), [3])
        self.assertEqual(visible_frame_indices("Current frame", 9, 5), [4])

    def test_up_to_current_mode_includes_zero_through_current(self):
        self.assertEqual(visible_frame_indices("0 to current frame", -2, 5), [0])
        self.assertEqual(visible_frame_indices("0 to current frame", 3, 5), [0, 1, 2, 3])
        self.assertEqual(visible_frame_indices("0 to current frame", 9, 5), [0, 1, 2, 3, 4])

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


if __name__ == "__main__":
    unittest.main()
