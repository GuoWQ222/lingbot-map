import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class EvalLauncherDefaultsTest(unittest.TestCase):
    def test_manip_long6_launchers_default_to_moving_surround_camera(self):
        launcher_names = [
            "eval.sh",
            "eval_fast_parallel.sh",
            "eval_loger.sh",
            "eval_pi3.sh",
            "eval_scal3r.sh",
            "eval_streamvggt.sh",
            "eval_ttt3r.sh",
            "eval_vggt.sh",
        ]
        assignment_re = re.compile(
            r'^EVAL_SURROUND_CAMERA_NAME="\$\{EVAL_SURROUND_CAMERA_NAME(?::?-)([^}]+)\}"$'
        )

        for launcher_name in launcher_names:
            with self.subTest(launcher=launcher_name):
                launcher = REPO_ROOT / launcher_name
                defaults = [
                    match.group(1)
                    for line in launcher.read_text(encoding="utf-8").splitlines()
                    if (match := assignment_re.match(line.strip()))
                ]
                self.assertEqual(defaults, ["surround_cam_moving"])


if __name__ == "__main__":
    unittest.main()
