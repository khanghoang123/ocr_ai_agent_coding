"""Tests for the new VietOCR-hallucination diagnostic metrics (Phase 2C)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocr_pipeline.validation.metrics import (  # noqa: E402
    abnormal_symbol_rate,
    diagnostic_diff,
    diagnostic_metrics,
    garbage_text_ratio,
    is_garbage_line,
    is_hallucinated_line,
    repeated_number_sequences,
    uppercase_garbage_tokens,
)


def test_clean_vietnamese_is_not_hallucinated():
    text = "Hôm nay trời rất đẹp và tôi đi học."
    assert is_hallucinated_line(text) is False
    assert is_garbage_line(text) is False
    assert uppercase_garbage_tokens(text) == []
    assert repeated_number_sequences(text) == []
    assert abnormal_symbol_rate(text) == 0.0


def test_uppercase_abbreviation_garbage_is_flagged():
    text = "ND TP UBND đã ban hành quyết định"
    assert uppercase_garbage_tokens(text) == ["ND", "TP", "UBND"]
    assert is_hallucinated_line(text) is True


def test_repeated_binary_sequences_are_flagged():
    text = "abc 010101 def 1111 ghi"
    assert "010101" in repeated_number_sequences(text)
    assert "1111" in repeated_number_sequences(text)
    assert is_hallucinated_line(text) is True


def test_long_digit_run_is_flagged():
    # ≥5 contiguous digits that don't look like a year/date.
    text = "Mã số 12345678 không hợp lệ"
    assert any(seq.startswith("12345") for seq in repeated_number_sequences(text))
    assert is_hallucinated_line(text) is True


def test_abnormal_symbols_increase_rate():
    text = "Tôi đi học #@%*^ ngày mai"
    rate = abnormal_symbol_rate(text)
    assert rate > 0.05


def test_garbage_text_ratio_counts_blank_and_short_lines():
    lines = [
        "",
        "ab",
        "Một câu tiếng Việt bình thường.",
        "Another normal looking line of text.",
    ]
    # Two of four lines are garbage (empty + 2-char).
    assert garbage_text_ratio(lines) == 0.5


def test_diagnostic_metrics_aggregates():
    lines = [
        "Hôm nay trời đẹp",
        "ND TP UBND",            # uppercase garbage
        "abc 010101 def",        # binary repetition
    ]
    m = diagnostic_metrics(lines)
    assert m["line_count"] == 3
    assert m["hallucinated_line_count"] == 2
    assert 0.0 < m["hallucinated_line_rate"] < 1.0
    assert m["uppercase_garbage_token_count"] >= 3
    assert m["repeated_number_sequence_count"] >= 1


def test_diagnostic_diff_verdicts():
    before = diagnostic_metrics(
        [
            "ND TP UBND",
            "abc 010101 def",
            "Một câu tiếng Việt",
        ]
    )
    after_better = diagnostic_metrics(
        [
            "Một câu tiếng Việt",
            "Một câu khác",
            "Một câu thứ ba",
        ]
    )
    after_worse = diagnostic_metrics(
        [
            "ND TP UBND XXX YYY",
            "abc 010101 def 1111",
            "ZZZ 0000 #@%*",
        ]
    )
    assert diagnostic_diff(before, after_better)["verdict"] == "improved"
    assert diagnostic_diff(before, after_worse)["verdict"] == "worsened"
    assert diagnostic_diff(before, before)["verdict"] == "no_change"
