from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    description: str = "Baseline OCR pipeline configuration"
    model_key: str | None = None
    crop_padding: int | None = None
    crop_padding_ratio: float | None = None
    rectify_line: bool = True
    enable_deskew: bool = False
    enable_document_perspective_correction: bool = False
    enable_local_contrast: bool = True
    detect_on_upscaled_image: bool = True
    upscale_factor: float = 1.0
    enable_vietnamese_postprocess: bool = False
    flag_digit_noise: bool = False
    debug: bool = True
    enable_noise_box_filter: bool = False
    tiny_box_height_ratio: float = 0.45
    tiny_box_width_ratio: float = 0.08
    tiny_box_min_ink_occupancy: float = 0.01
    enable_rotated_crop: bool = True
    debug_save_intermediate_crops: bool = False
    crop_strategy: str = "basic"
    enable_line_deskew: bool = False
    min_crop_height_ratio: float = 0.75
    vertical_padding_ratio: float = 0.35
    horizontal_padding_ratio: float = 0.60
    max_deskew_angle: float = 8.0
    unsupported_options: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw", None)
        return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Experiment config must be a mapping: {config_path}")

    allowed = set(ExperimentConfig.__dataclass_fields__) - {"raw", "unsupported_options"}
    kwargs = {key: value for key, value in payload.items() if key in allowed}
    config = ExperimentConfig(**kwargs)
    config.raw = dict(payload)

    unsupported: list[str] = []
    if config.enable_deskew:
        unsupported.append("enable_deskew")
    if config.enable_document_perspective_correction:
        unsupported.append("enable_document_perspective_correction")
    config.unsupported_options = unsupported
    return config


def dump_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, allow_unicode=True, sort_keys=False)
