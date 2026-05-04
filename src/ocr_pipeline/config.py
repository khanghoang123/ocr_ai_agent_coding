"""
Vietnamese Handwritten OCR — Centralised Configuration

Uses Pydantic BaseSettings for environment-variable override.
All values can be overridden via .env file or environment
variables (prefixed with OCR_).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = two levels up from this file (src/ocr_pipeline/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _cuda_available() -> bool:
    """Check if CUDA is available without importing torch at module load time."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class Settings(BaseSettings):
    """Centralised application configuration.

    Override any value via environment variable (prefix: OCR_) or .env file.
    Example:  OCR_DEFAULT_MODEL=baseline python -m app.main
    """

    model_config = SettingsConfigDict(
        env_prefix="OCR_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Project paths ────────────────────────────────────────
    project_root: Path = PROJECT_ROOT
    models_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"
    dataset_dir: Path = PROJECT_ROOT / "Dataset" / "data"

    # ── Model selection ──────────────────────────────────────
    default_model: str = "experiment_B_50k"
    model_registry_path: Path = PROJECT_ROOT / "models" / "model_registry.yaml"

    # ── PaddleOCR detector settings ──────────────────────────
    # Detection settings (values copied from notebook 05_crawl_and_detect_CLEANED.ipynb)
    det_use_gpu: bool = False
    det_use_angle_cls: bool = False
    det_lang: str = "vi"
    det_db_thresh: float = 0.15
    det_db_box_thresh: float = 0.10
    det_db_unclip_ratio: float = 1.8
    det_limit_side_len: int = 1920
    det_limit_type: str = "max"
    det_db_score_mode: str = "slow" # 'slow' is better for precise polygons on curved lines
    det_db_box_type: str = "poly"  # 'poly' allows multi-point polygons for curved lines

    # Phase 3 default detector backend. Kraken BLLA was the winner of the
    # Tier-1 leaderboard on tests/test/ (best full_width_band_rate, best
    # recall on cursive handwriting). Override via DETECTOR_BACKEND env var
    # — accepted values are paddle / surya / craft / kraken. The legacy
    # silent OpenCV grid fallback is OFF regardless of backend.
    detector_backend: str = "kraken"

    # ── VietOCR recognizer settings ──────────────────────────
    rec_architecture: str = "vgg_seq2seq"
    rec_image_height: int = 32
    rec_image_max_width: int = 690
    rec_image_min_width: int = 32
    rec_device: str = "cuda" if _cuda_available() else "cpu"

    # No-repeat-ngram constraint for the seq2seq decoder. Blocks any next
    # token that would cause the last n tokens to match an ngram already
    # emitted earlier in the same line. Target: the repeating-digit
    # attractor ("0101010..." / "NDNDND...") observed on baseline_50k
    # against slanted handwriting crops.
    # 0 = disabled (legacy vietocr greedy/beam). 3 = default; blocks
    # 3-gram repeats. Natural Vietnamese trigram repeats within a single
    # line are rare (< 1% of GT), so the benefit is expected to dominate.
    rec_no_repeat_ngram_size: int = 3

    # KenLM 5-gram rescoring. When ``rec_kenlm_path`` points to an
    # existing ``.bin`` or ``.arpa`` file AND ``rec_kenlm_beam_width``
    # is > 1, the recognizer runs a shallow beam search, produces K
    # candidates per line, and picks the best by::
    #
    #     score = gamma * acoustic_logprob
    #           + alpha * (lm_logprob / max(1, word_count))
    #           + beta  * word_count
    #
    # When the file is missing or kenlm is not installed, the
    # rescorer silently degrades to a no-op and the recognizer
    # returns the top-1 acoustic hypothesis unchanged. Do NOT set
    # ``rec_kenlm_beam_width=1`` when the LM is configured — a beam
    # of 1 produces only the top-1 candidate and there is nothing to
    # rescore. Typical values: alpha=0.5, beta=0.1, beam_width=5.
    rec_kenlm_path: Optional[str] = None
    rec_kenlm_alpha: float = 0.5
    rec_kenlm_beta: float = 0.1
    rec_kenlm_gamma: float = 1.0
    rec_kenlm_beam_width: int = 1  # 1 = disabled

    # ── Pipeline settings ────────────────────────────────────
    crop_padding: int = 4           # Pixels to add around each detected bbox
    min_line_height: int = 8        # Minimum line height (px) — filters noise
    min_line_width: int = 20        # Minimum line width (px)
    row_tolerance_ratio: float = 0.5  # Row grouping tolerance (fraction of median height)

    # ── FastAPI settings ─────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Vietnamese Handwritten OCR API"
    api_version: str = "1.0.0"
    max_upload_size: int = 50 * 1024 * 1024  # 50MB per file

    # ── Supported formats ────────────────────────────────────
    supported_image_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    supported_doc_exts: tuple[str, ...] = (".pdf",)

    @field_validator("models_dir", "data_dir", mode="before")
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    def get_model_registry(self) -> dict:
        """Load and return the full model registry as a dict."""
        with open(self.model_registry_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_default_model_key(self) -> str:
        """Return the effective default model key."""
        registry = self.get_model_registry()
        registry_default = registry.get("default")

        if self.default_model and self.default_model in registry.get("models", {}):
            return self.default_model
        if registry_default in registry.get("models", {}):
            return registry_default
        raise ValueError("No valid default model configured in model registry.")

    def get_model_config(self, model_key: Optional[str] = None) -> dict:
        """Return config dict for the specified model key (or default)."""
        registry = self.get_model_registry()
        key = model_key or self.get_default_model_key()
        if key not in registry["models"]:
            available = list(registry["models"].keys())
            raise ValueError(f"Model '{key}' not found. Available: {available}")
        cfg = dict(registry["models"][key])
        cfg["key"] = key
        # Resolve weights path relative to project root
        resolved_weights = self.project_root / cfg["weights_path"]
        cfg["weights_path"] = str(resolved_weights)
        cfg["weights_exists"] = resolved_weights.exists()
        return cfg

    def get_model_catalog(self, inference_only: bool = True) -> list[dict]:
        """Return model metadata for the API/frontend model picker."""
        registry = self.get_model_registry()
        models = []
        for key in registry.get("models", {}):
            cfg = self.get_model_config(key)
            if inference_only and cfg.get("status") != "inference":
                continue
            if not cfg["weights_exists"]:
                continue
            models.append({
                "key": key,
                "name": cfg.get("name", key),
                "display_name": cfg.get("display_name", cfg.get("name", key)),
                "description": cfg.get("description", ""),
                "architecture": cfg.get("architecture", ""),
                "training_iters": cfg.get("training_iters"),
                "cer": cfg.get("cer"),
                "wer": cfg.get("wer"),
                "exact_match": cfg.get("exact_match"),
                "status": cfg.get("status", "unknown"),
                "recommended": bool(cfg.get("recommended", False)),
                "is_default": key == self.get_default_model_key(),
            })
        return models


# Singleton — import this everywhere
settings = Settings()
