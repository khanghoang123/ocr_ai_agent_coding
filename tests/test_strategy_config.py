from __future__ import annotations

from ocr_pipeline.experiment_config import load_experiment_config


def test_phase1_strategy_configs_load():
    noise = load_experiment_config("configs/refine_noise_filter.yaml")
    rotated = load_experiment_config("configs/rotated_crop.yaml")
    combined = load_experiment_config("configs/refine_combined.yaml")

    assert noise.enable_noise_box_filter is True
    assert rotated.enable_rotated_crop is True
    assert combined.enable_noise_box_filter is True
    assert combined.enable_rotated_crop is True
    assert combined.crop_padding_ratio == 0.08


def test_old_configs_keep_safe_strategy_defaults():
    baseline = load_experiment_config("configs/baseline.yaml")

    assert baseline.enable_noise_box_filter is False
    assert baseline.enable_rotated_crop is True
    assert baseline.debug_save_intermediate_crops is False
    assert baseline.tiny_box_height_ratio == 0.45
    assert baseline.crop_strategy == "basic"
    assert baseline.enable_line_deskew is False


def test_unsupported_options_still_limited_to_declared_flags():
    deskew = load_experiment_config("configs/deskew_on.yaml")

    assert deskew.unsupported_options == ["enable_deskew"]


def test_phase2b_crop_geometry_config_loads():
    config = load_experiment_config("configs/phase2B_crop_geometry.yaml")

    assert config.crop_strategy == "validated"
    assert config.enable_line_deskew is True
    assert config.min_crop_height_ratio == 0.60
    assert config.vertical_padding_ratio == 0.18
    assert config.horizontal_padding_ratio == 0.25
    assert config.unsupported_options == []
