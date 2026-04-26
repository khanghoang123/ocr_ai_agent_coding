from __future__ import annotations

from ocr_pipeline.validation.metrics import (
    auto_research_score,
    character_error_rate,
    digit_noise_rate,
    word_error_rate,
)


def test_cer_wer_exact_and_errors():
    assert character_error_rate("xin chào", "xin chào") == 0.0
    assert word_error_rate("xin chào", "xin chào") == 0.0
    assert character_error_rate("xin cho", "xin chào") > 0.0
    assert word_error_rate("xin cho", "xin chào") == 0.5


def test_digit_noise_rate_ignores_valid_numbers():
    text = "Năm 1945, lớp 11.11 thi ngày 24/04/2026 ở phòng A2"
    assert digit_noise_rate(text) == 0.0
    assert digit_noise_rate("Tôi yêu V1ệt N4m") > 0.0


def test_auto_research_score_uses_weighted_formula():
    score = auto_research_score(
        {
            "cer": 0.10,
            "wer": 0.20,
            "digit_noise_rate": 0.30,
            "normalized_line_count_error": 0.40,
        }
    )
    assert score == 0.18
