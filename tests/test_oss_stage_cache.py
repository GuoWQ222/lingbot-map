import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.data.oss_stage_cache import OssStageCache
from train import discover_trajectory_dirs_with_rclone


def _write_fake_rclone(path: Path, fake_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import os, shutil, sys",
                f"fake_root = {str(fake_root)!r}",
                "args = sys.argv[1:]",
                "expected_config = os.environ.get('EXPECTED_RCLONE_CONFIG')",
                "if expected_config:",
                "    assert '--config' in args, args",
                "    idx = args.index('--config')",
                "    assert args[idx + 1] == expected_config, args",
                "    del args[idx:idx + 2]",
                "pos = [arg for arg in args if not arg.startswith('-')]",
                "if pos and pos[0] == 'lsd':",
                "    print('          -1 2026-06-13 00:00:00        -1 scene_a')",
                "    print('          -1 2026-06-13 00:00:00        -1 scene_b')",
                "    print('          -1 2026-06-13 00:00:00        -1 .hidden')",
                "    print('          -1 2026-06-13 00:00:00        -1 scene_claim_tmp')",
                "    sys.exit(0)",
                "if pos and pos[0] == 'copy':",
                "    pos = pos[1:]",
                "src, dst = pos[0], pos[1]",
                "assert src.startswith('aliyunoss:'), src",
                "rel = src[len('aliyunoss:'):].rstrip('/')",
                "src_path = os.path.join(fake_root, rel)",
                "if os.path.isdir(src_path):",
                "    shutil.copytree(src_path, dst, dirs_exist_ok=True)",
                "else:",
                "    os.makedirs(os.path.dirname(dst), exist_ok=True)",
                "    shutil.copy2(src_path, dst)",
            ]
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class OssStageCacheTest(unittest.TestCase):
    def test_discovers_trajectory_dirs_with_rclone_lsd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            rclone = tmp / "rclone"
            _write_fake_rclone(rclone, tmp / "unused_fake_oss")

            config_path = "/cpfs/user/guowenqi/rclone/rclone.conf"
            with patch.dict(os.environ, {"EXPECTED_RCLONE_CONFIG": config_path}):
                trajectories = discover_trajectory_dirs_with_rclone(
                    ["/mounted/Manip_long6/data"],
                    ["oss://pjlab-bjpai-sim/guowenqi/Manip_long6/data"],
                    rclone_bin=str(rclone),
                    rclone_config=config_path,
                    rclone_remote="aliyunoss",
                )

            self.assertEqual(
                trajectories,
                [
                    Path("/mounted/Manip_long6/data/scene_a"),
                    Path("/mounted/Manip_long6/data/scene_b"),
                ],
            )

    def test_reuses_completed_stage_and_eviction_is_lru(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            fake_root = tmp / "fake_oss"
            for name in ("scene_a", "scene_b", "scene_c"):
                source_dir = fake_root / "bucket" / "prefix" / "Manip_long6" / "data" / name
                source_dir.mkdir(parents=True)
                (source_dir / "payload.txt").write_text(name)

            rclone = tmp / "rclone"
            _write_fake_rclone(rclone, fake_root)
            config_path = "/cpfs/user/guowenqi/rclone/rclone.conf"
            cache = OssStageCache(
                stage_root=tmp / "stage",
                mount_root=tmp / "missing_mount",
                rclone_bin=rclone,
                rclone_config=config_path,
                rclone_remote="aliyunoss",
                max_entries=2,
                enabled=True,
            )
            cache.enable_fallback("unit test")

            with patch.dict(os.environ, {"EXPECTED_RCLONE_CONFIG": config_path}):
                a = cache.stage_dir(
                    dataset="manip",
                    relative_key="scene_a",
                    oss_uri="oss://bucket/prefix/Manip_long6/data/scene_a/",
                )
                b = cache.stage_dir(
                    dataset="co3d",
                    relative_key="cup/scene_b",
                    oss_uri="oss://bucket/prefix/Manip_long6/data/scene_b/",
                )
            self.assertEqual((a / "payload.txt").read_text(), "scene_a")
            self.assertEqual((b / "payload.txt").read_text(), "scene_b")

            with patch.dict(os.environ, {"EXPECTED_RCLONE_CONFIG": config_path}):
                cache.stage_dir(
                    dataset="manip",
                    relative_key="scene_a",
                    oss_uri="oss://bucket/prefix/Manip_long6/data/scene_a/",
                )
                c = cache.stage_dir(
                    dataset="manip",
                    relative_key="scene_c",
                    oss_uri="oss://bucket/prefix/Manip_long6/data/scene_c/",
                )

            self.assertTrue((a / "payload.txt").exists())
            self.assertFalse((b / "payload.txt").exists())
            self.assertEqual((c / "payload.txt").read_text(), "scene_c")


if __name__ == "__main__":
    unittest.main()
