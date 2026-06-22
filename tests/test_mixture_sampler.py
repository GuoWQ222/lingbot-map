import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.data.mixture_sampler import CurriculumMixtureSampler


class CurriculumMixtureSamplerTest(unittest.TestCase):
    def test_external_weights_are_relative_within_external_share(self):
        sampler = CurriculumMixtureSampler(
            manip_size=10,
            external_sizes=[10, 10, 10],
            external_names=["dl3dv", "scannetpp", "co3d"],
            external_weights=[4.0, 4.0, 1.5],
            epoch_length=8,
            p_manip_start=0.60,
            p_manip_end=0.80,
            warmup_start=10,
            warmup_end=20,
            seed=0,
        )

        weights0 = sampler.get_dataset_weights(0)
        self.assertAlmostEqual(weights0["manip"], 0.60)
        self.assertAlmostEqual(weights0["dl3dv"], 0.40 * 4.0 / 9.5)
        self.assertAlmostEqual(weights0["scannetpp"], 0.40 * 4.0 / 9.5)
        self.assertAlmostEqual(weights0["co3d"], 0.40 * 1.5 / 9.5)

        weights20 = sampler.get_dataset_weights(20)
        self.assertAlmostEqual(weights20["manip"], 0.80)
        self.assertAlmostEqual(weights20["dl3dv"], 0.20 * 4.0 / 9.5)
        self.assertAlmostEqual(weights20["scannetpp"], 0.20 * 4.0 / 9.5)
        self.assertAlmostEqual(weights20["co3d"], 0.20 * 1.5 / 9.5)

    def test_external_weight_mapping_defaults_missing_names_to_one(self):
        sampler = CurriculumMixtureSampler(
            manip_size=10,
            external_sizes=[10, 10],
            external_names=["dl3dv", "unknown"],
            external_weights={"dl3dv": 3.0},
            epoch_length=8,
            p_manip_start=0.75,
            p_manip_end=0.75,
            seed=0,
        )

        weights = sampler.get_dataset_weights(0)
        self.assertAlmostEqual(weights["dl3dv"], 0.25 * 3.0 / 4.0)
        self.assertAlmostEqual(weights["unknown"], 0.25 * 1.0 / 4.0)


if __name__ == "__main__":
    unittest.main()
