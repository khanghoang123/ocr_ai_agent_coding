from __future__ import annotations

from ocr_pipeline.config import settings


def test_model_catalog_exposes_three_inference_models():
    catalog = settings.get_model_catalog()
    keys = [item["key"] for item in catalog]

    assert settings.get_default_model_key() == "experiment_B_50k"
    assert keys == ["baseline_10k", "baseline_50k", "experiment_B_50k"]


def test_model_configs_resolve_existing_weight_paths():
    for key in ("baseline_10k", "baseline_50k", "experiment_B_50k"):
        cfg = settings.get_model_config(key)
        assert cfg["weights_exists"] is True
        assert cfg["weights_path"].endswith("best_model.pth")
