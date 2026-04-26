from __future__ import annotations

from ocr_pipeline.experiment_config import load_experiment_config


def test_config_loads_baseline_defaults():
    config = load_experiment_config("configs/baseline.yaml")
    assert config.name == "baseline"
    assert config.rectify_line is True
    assert config.enable_local_contrast is True
    assert config.unsupported_options == []


def test_config_marks_unsupported_options():
    config = load_experiment_config("configs/deskew_on.yaml")
    assert "enable_deskew" in config.unsupported_options
