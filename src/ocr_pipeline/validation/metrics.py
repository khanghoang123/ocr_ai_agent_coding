from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


def _edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    a = list(left)
    b = list(right)
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        curr = [i]
        for j, item_b in enumerate(b, start=1):
            cost = 0 if item_a == item_b else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def normalize_for_metric(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    return re.sub(r"\s+", " ", normalized).strip()


def character_error_rate(prediction: str, reference: str) -> float:
    ref = normalize_for_metric(reference)
    pred = normalize_for_metric(prediction)
    if not ref:
        return 0.0 if not pred else 1.0
    return _edit_distance(pred, ref) / len(ref)


def word_error_rate(prediction: str, reference: str) -> float:
    ref_words = normalize_for_metric(reference).split()
    pred_words = normalize_for_metric(prediction).split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return _edit_distance(pred_words, ref_words) / len(ref_words)


VALID_NUMBER_RE = re.compile(
    r"""
    (?<![\w])
    (?:
        \d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?
        |\d{4}
        |\d{1,3}(?:[.,]\d{1,3})+
        |\d+[A-Za-zÀ-ỹ]?
        |[A-Za-zÀ-ỹ]?\d+
    )
    (?![\w])
    """,
    re.VERBOSE,
)
SUSPICIOUS_DIGIT_RE = re.compile(r"(?=\S*[A-Za-zÀ-ỹ])(?=\S*\d)\S+")


def suspicious_digit_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for match in SUSPICIOUS_DIGIT_RE.finditer(normalize_for_metric(text)):
        token = match.group(0).strip(".,;:!?()[]{}\"'")
        if not token or VALID_NUMBER_RE.fullmatch(token):
            continue
        letters = sum(ch.isalpha() for ch in token)
        digits = sum(ch.isdigit() for ch in token)
        if letters >= 2 and digits >= 1:
            tokens.append(token)
    return tokens


def digit_noise_rate(texts: str | Iterable[str]) -> float:
    if isinstance(texts, str):
        text = texts
    else:
        text = "\n".join(texts)
    normalized = normalize_for_metric(text)
    if not normalized:
        return 0.0
    digit_count = sum(ch.isdigit() for ch in normalized)
    if digit_count == 0:
        return 0.0
    suspicious_digits = sum(sum(ch.isdigit() for ch in token) for token in suspicious_digit_tokens(normalized))
    return suspicious_digits / digit_count


def line_count_error(predicted_count: int, expected_count: int | None) -> int | None:
    if expected_count is None:
        return None
    return abs(int(predicted_count) - int(expected_count))


def normalized_line_count_error(predicted_count: int, expected_count: int | None) -> float | None:
    error = line_count_error(predicted_count, expected_count)
    if error is None:
        return None
    return error / max(int(expected_count), 1)


def aggregate_nullable(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def auto_research_score(metrics: dict) -> float | None:
    cer = metrics.get("cer")
    wer = metrics.get("wer")
    digit_noise = metrics.get("digit_noise_rate")
    line_error = metrics.get("normalized_line_count_error")
    components = [
        (0.50, cer),
        (0.30, wer),
        (0.10, digit_noise),
        (0.10, line_error),
    ]
    available = [(weight, float(value)) for weight, value in components if value is not None]
    if not available:
        return None
    weight_sum = sum(weight for weight, _ in available)
    return sum(weight * value for weight, value in available) / weight_sum


@dataclass
class ImageMetric:
    image_path: str
    cer: float | None
    wer: float | None
    digit_noise_rate: float
    line_count_error: int | None
    normalized_line_count_error: float | None
    detection_count: int
    crop_flag_count: int
