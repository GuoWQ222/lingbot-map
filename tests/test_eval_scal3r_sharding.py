import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import eval_scal3r


class EmptyLoader:
    def __len__(self):
        return 0

    def __iter__(self):
        return iter(())


class EvalScal3RShardingTest(unittest.TestCase):
    def test_evaluate_one_mode_delegates_sharding_to_scal3r_native_loader(self):
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
        calls = []

        def fake_build_eval_loader(call_args, eval_mode):
            calls.append(
                {
                    "split": call_args.split,
                    "max_scenes_eval": call_args.max_scenes_eval,
                    "eval_shard_count": call_args.eval_shard_count,
                    "eval_shard_index": call_args.eval_shard_index,
                    "eval_mode": eval_mode,
                }
            )
            return EmptyLoader(), [Path(f"scene_{idx}") for idx in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(eval_scal3r, "build_scal3r_native_eval_loader", side_effect=fake_build_eval_loader):
                with self.assertRaisesRegex(RuntimeError, "No batches evaluated"):
                    eval_scal3r.evaluate_one_mode(
                        args,
                        eval_scal3r.torch.device("cpu"),
                        "left_moving_tracks",
                        Path(tmpdir),
                    )

        self.assertEqual(
            calls,
            [
                {
                    "split": "val",
                    "max_scenes_eval": 5,
                    "eval_shard_count": 4,
                    "eval_shard_index": 2,
                    "eval_mode": "left_moving_tracks",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
