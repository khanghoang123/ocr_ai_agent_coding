from __future__ import annotations

from ocr_pipeline.validation.harness import run_all_validations


def test_ocr_validation_harness_passes():
    summary = run_all_validations()
    assert summary["passed"] is True
    assert len(summary["results"]) == 4
    failed = [r for r in summary["results"] if not r["passed"]]
    assert failed == []
