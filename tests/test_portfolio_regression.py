from __future__ import annotations

from pathlib import Path

import pytest

from tests.portfolio_regression import MANIFEST_PATH, OUTPUT_ROOT, load_manifest, run_portfolio_regression


def _runtime_ready() -> tuple[bool, str]:
    try:
        import paddleocr  # noqa: F401
        import vietocr  # noqa: F401
    except ImportError as exc:
        return False, f"runtime dependency missing: {exc}"

    try:
        from ocr_pipeline.config import settings

        config = settings.get_model_config(settings.get_default_model_key())
    except Exception as exc:
        return False, f"model config unavailable: {exc}"

    if not Path(config["weights_path"]).exists():
        return False, f"weights not found: {config['weights_path']}"

    return True, "ready"


def test_portfolio_manifest_is_valid():
    fixtures = load_manifest()

    assert MANIFEST_PATH.exists()
    assert len(fixtures) >= 5
    assert any(fixture.curved_lines for fixture in fixtures)
    for fixture in fixtures:
        assert fixture.image_path.exists(), fixture.image_path
        assert fixture.line_min > 0
        assert fixture.line_max >= fixture.line_min
        assert fixture.fallback_rate_max > 0
        assert fixture.blank_margin_gain_min >= 0


def test_portfolio_regression_real_images():
    ready, reason = _runtime_ready()
    if not ready:
        pytest.skip(reason)

    summary = run_portfolio_regression(output_root=OUTPUT_ROOT)

    assert len(summary["fixtures"]) >= 5
    assert summary["curved_branch_fixtures"]
