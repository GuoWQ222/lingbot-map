import numpy as np
import torch

from visualize_pi3_native_gt_batch import extract_frame_clouds


def test_extract_frame_clouds_unprojects_depth_with_batch_intrinsics_and_extrinsics():
    batch = {
        "images": torch.tensor(
            [[[[[0.0, 0.5], [1.0, 0.25]], [[0.0, 0.5], [1.0, 0.25]], [[0.0, 0.5], [1.0, 0.25]]]]],
            dtype=torch.float32,
        ),
        "depths": torch.ones(1, 1, 2, 2, dtype=torch.float32),
        "point_masks": torch.ones(1, 1, 2, 2, dtype=torch.bool),
        "intrinsics": torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3),
        "extrinsics": torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]]]),
        "frame_indices": torch.tensor([[7]], dtype=torch.long),
    }

    frame_points, frame_colors, c2w, intrinsics, images, frame_ids = extract_frame_clouds(
        batch, max_points_per_frame=0, point_stride=1, seed=0
    )

    np.testing.assert_allclose(
        frame_points[0],
        np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        frame_colors[0],
        np.array([[0, 0, 0], [127, 127, 127], [255, 255, 255], [63, 63, 63]], dtype=np.uint8),
    )
    np.testing.assert_allclose(c2w[0], np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(intrinsics[0], np.eye(3, dtype=np.float32))
    np.testing.assert_allclose(images[0].shape, (2, 2, 3))
    assert frame_ids == [7]
