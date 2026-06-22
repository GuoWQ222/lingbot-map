from pathlib import Path

import torch
from PIL import Image

import eval_loger


def test_pointcloud_metrics_default_matches_eval_py() -> None:
    args = eval_loger.build_argparser().parse_args(["--train_args_json", "args.json"])

    assert args.pointcloud_metrics is True


def test_pointcloud_metrics_can_be_disabled() -> None:
    args = eval_loger.build_argparser().parse_args(
        ["--train_args_json", "args.json", "--no_pointcloud_metrics"]
    )

    assert args.pointcloud_metrics is False


def test_loger_native_input_preprocess_is_default() -> None:
    args = eval_loger.build_argparser().parse_args(["--train_args_json", "args.json"])

    assert args.loger_input_preprocess == "native"


def test_loger_native_resolution_defaults_to_loger_demo_geometry() -> None:
    args = eval_loger.build_argparser().parse_args(["--train_args_json", "args.json"])

    assert args.loger_native_width == 504
    assert args.loger_native_height == 280


def test_loger_native_geometry_uses_independent_width_and_height() -> None:
    geometry = eval_loger.compute_loger_preprocess_geometry(
        640,
        480,
        target_width=504,
        target_height=280,
    )

    assert geometry["new_width"] == 504
    assert geometry["new_height"] == 280
    assert geometry["crop_width"] == 504
    assert geometry["crop_height"] == 280


def test_loger_native_loader_matches_loger_rgb_lanczos_tensor(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 2), (255, 0, 0)).save(image_path)

    images = eval_loger._load_loger_native_images_from_paths(
        [str(image_path)],
        target_hw=(14, 28),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert tuple(images.shape) == (1, 1, 3, 14, 28)
    assert torch.allclose(images[0, 0, 0], torch.ones((14, 28)))
    assert torch.allclose(images[0, 0, 1:], torch.zeros((2, 14, 28)))


def test_loger_rgb_paths_flatten_default_collate_shape() -> None:
    collated = [("a.png",), ("b.png",)]

    assert eval_loger._flatten_collated_rgb_paths(collated) == [["a.png", "b.png"]]


def test_loger_input_preprocess_controls_reused_pi3_input_mode(tmp_path: Path) -> None:
    args_json = tmp_path / "args.json"
    args_json.write_text("{}")

    native_args = eval_loger.build_argparser().parse_args(["--train_args_json", str(args_json)])
    native_ns = eval_loger.coerce_loger_args_from_json(native_args)
    assert native_ns.pi3_input_mode == "native"
    assert native_ns.pi3_native_width == 504
    assert native_ns.loger_native_height == 280

    lingbot_args = eval_loger.build_argparser().parse_args([
        "--train_args_json", str(args_json),
        "--loger_input_preprocess", "lingbot",
    ])
    lingbot_ns = eval_loger.coerce_loger_args_from_json(lingbot_args)
    assert lingbot_ns.pi3_input_mode == "lingbot"
