from __future__ import annotations

from ocr_pipeline.recognizer.vietnamese_postprocess import safe_vietnamese_postprocess


def test_safe_postprocess_normalizes_whitespace_and_keeps_valid_numbers():
    result = safe_vietnamese_postprocess("  Năm  1945 \t lớp 11.11 ngày 24/04/2026  ")
    assert result.cleaned_text == "Năm 1945 lớp 11.11 ngày 24/04/2026"
    assert result.flags == []


def test_safe_postprocess_flags_digit_noise_without_rewriting():
    result = safe_vietnamese_postprocess("Tôi yêu V1ệt N4m")
    assert result.cleaned_text == "Tôi yêu V1ệt N4m"
    assert "digit_noise" in result.flags
    assert result.suspicious_tokens == ["V1ệt", "N4m"]
