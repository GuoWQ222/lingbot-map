import unittest

import numpy as np

from visualize_scene import opencv_camera_frame, set_client_to_opencv_camera_view


class _FakeCamera:
    def __init__(self):
        self.position = None
        self.up_direction = None
        self.look_at = None


class _FakeClient:
    def __init__(self):
        self.camera = _FakeCamera()


class VisualizeSceneCameraViewTest(unittest.TestCase):
    def test_opencv_camera_frame_uses_z_forward_and_negative_y_up(self):
        c2w = np.eye(4, dtype=np.float64)

        center, forward, up = opencv_camera_frame(c2w, centroid=np.zeros(3))

        np.testing.assert_allclose(center, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(forward, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(up, [0.0, -1.0, 0.0])

    def test_set_client_view_backs_off_without_setting_quaternion(self):
        client = _FakeClient()

        set_client_to_opencv_camera_view(
            client,
            camera_center=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            forward=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            up=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            look_distance=2.0,
            backoff=0.25,
        )

        self.assertEqual(client.camera.position, (1.0, 2.0, 2.75))
        self.assertEqual(client.camera.up_direction, (0.0, -1.0, 0.0))
        self.assertEqual(client.camera.look_at, (1.0, 2.0, 5.0))
        self.assertFalse(hasattr(client.camera, "wxyz"))


if __name__ == "__main__":
    unittest.main()
