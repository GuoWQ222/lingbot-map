from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "scripts" / "train"


class ARKitScenesHighResWiringTest(unittest.TestCase):
    def test_arkitscenes_highres_is_wired_as_external_dataset(self):
        train_py = (TRAIN_DIR / "train.py").read_text()
        train_sh = (TRAIN_DIR / "train.sh").read_text()
        adapter = ROOT / "lingbot_map" / "data" / "arkitscenes_highres.py"

        self.assertTrue(adapter.exists())
        self.assertIn(
            "from lingbot_map.data.arkitscenes_highres import ARKitScenesHighResTrajectoryDataset",
            train_py,
        )
        self.assertIn('external_train.append(("arkitscenes_highres", len(arkit_train)))', train_py)
        self.assertIn('"arkitscenes_highres"', train_py)

        self.assertIn("ARKITSCENES_HIGHRES_ROOT", train_sh)
        self.assertIn("--arkitscenes_highres_root", train_sh)
        self.assertIn("--arkitscenes_highres_max_interval", train_sh)
        self.assertIn("--arkitscenes_highres_block_shuffle", train_sh)

    def test_arkitscenes_highres_adapter_keeps_base3d_sampling_contract(self):
        adapter_text = (ROOT / "lingbot_map" / "data" / "arkitscenes_highres.py").read_text()

        self.assertIn("base3d-clean/datasets/arkitscenes_highres.py", adapter_text)
        self.assertIn("_get_seq_from_start_id", adapter_text)
        self.assertIn("max_interval: int = 32", adapter_text)
        self.assertIn("block_shuffle: Optional[int] = 16", adapter_text)
        self.assertIn('"sample_mode": "arkitscenes_highres"', adapter_text)


if __name__ == "__main__":
    unittest.main()
