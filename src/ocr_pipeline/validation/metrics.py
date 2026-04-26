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

# Characters that should never appear in normal Vietnamese handwritten text.
# Vietnamese uses Latin letters with diacritics, digits, and standard punctuation.
# Anything outside this set is treated as an abnormal symbol (likely VietOCR
# decoding noise on a malformed crop). We deliberately exclude rare ASCII
# symbols like # @ % ^ & * $ that are almost never present in handwritten
# paragraphs and are a common hallucination signal.
ABNORMAL_SYMBOL_RE = re.compile(r"[^A-Za-zÀ-ỹà-ỹ0-9\s\.,;:!\?\(\)\[\]\{\}'\"\-—–_/«»“”‘’]")

# Tokens that look like ALL-CAPS Latin abbreviations (>= 2 letters), which are
# the classic VietOCR hallucination pattern reported by the project (ND, TP,
# UBND, etc.) when fed a poor crop. Excludes pure-digit tokens.
UPPERCASE_TOKEN_RE = re.compile(r"\b[A-ZĐÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴ]{2,}\b")

# Repeated 0/1 sequences (≥4 chars), a textbook VietOCR hallucination on
# blurry / over-cropped lines (e.g. "010101", "1111", "0000110").
REPEATED_BINARY_RE = re.compile(r"(?:[01]){4,}")
# More general: any digit run of length ≥5 that is not a year/date-like token
LONG_DIGIT_RUN_RE = re.compile(r"\d{5,}")


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


# ─────────────────────────────────────────────────────────────────────────────
# Hallucination diagnostic metrics.
#
# These are designed for the "no ground truth" evaluation regime described in
# the task spec: we cannot compute CER/WER, but we *can* detect output text
# that has the structural fingerprints of VietOCR hallucination — long digit
# runs, all-caps Latin abbreviations, abnormal Unicode symbols, mixed
# digit-letter garbage. They are pure functions of the predicted text.
# ─────────────────────────────────────────────────────────────────────────────


def abnormal_symbol_count(text: str) -> int:
    """Number of characters outside the normal Vietnamese character set."""
    return len(ABNORMAL_SYMBOL_RE.findall(normalize_for_metric(text)))


def abnormal_symbol_rate(text: str) -> float:
    """Fraction of characters that are not part of normal Vietnamese text."""
    normalized = normalize_for_metric(text)
    if not normalized:
        return 0.0
    bad = abnormal_symbol_count(normalized)
    visible = sum(1 for ch in normalized if not ch.isspace())
    if visible == 0:
        return 0.0
    return bad / visible


def uppercase_garbage_tokens(text: str) -> list[str]:
    """All-caps Latin tokens of length ≥2 (e.g. ND, TP, UBND).

    Vietnamese handwriting is overwhelmingly lowercase. When VietOCR is fed a
    bad crop, it tends to fall back to short uppercase abbreviation tokens
    that exist heavily in its training distribution — this is the classic
    'ND/TP/UBND' hallucination reported in this project.
    """
    return UPPERCASE_TOKEN_RE.findall(normalize_for_metric(text))


def uppercase_garbage_token_count(text: str) -> int:
    return len(uppercase_garbage_tokens(text))


def uppercase_garbage_token_rate(text: str) -> float:
    """Fraction of whitespace-separated tokens that look like all-caps garbage."""
    normalized = normalize_for_metric(text)
    if not normalized:
        return 0.0
    tokens = normalized.split()
    if not tokens:
        return 0.0
    return uppercase_garbage_token_count(normalized) / len(tokens)


def repeated_number_sequences(text: str) -> list[str]:
    """Repeated 0/1 sequences and long pure-digit runs.

    VietOCR's classic hallucination on noisy/curved crops is to emit long runs
    of '0' and '1' (its decoder collapses to high-frequency tokens). We also
    flag any pure-digit run of ≥5 characters as suspicious — Vietnamese
    handwritten prose almost never contains 5+ consecutive bare digits, so on
    this corpus a long bare-digit run is a reliable hallucination signal.

    Date- and number-like strings (``12/03/2024``, ``1.234,56``, ``2024``) are
    *not* matched by ``LONG_DIGIT_RUN_RE`` to begin with — the regex only
    matches *bare* digit runs of length ≥5, so things containing ``/``, ``-``,
    ``.``, ``,`` or that are ≤4 digits long never reach this function.
    """
    normalized = normalize_for_metric(text)
    matches: list[str] = []
    matches.extend(REPEATED_BINARY_RE.findall(normalized))
    for run in LONG_DIGIT_RUN_RE.findall(normalized):
        if run in matches:
            continue
        matches.append(run)
    return matches


def repeated_number_sequence_count(text: str) -> int:
    return len(repeated_number_sequences(text))


def suspicious_digit_token_count(text: str) -> int:
    return len(suspicious_digit_tokens(text))


def is_garbage_line(text: str) -> bool:
    """Heuristic for whether a single line of OCR output is garbage.

    Mirrors `_is_garbage_text` in `validation.debug_analyzer` so we have one
    canonical implementation usable from both sides of the pipeline.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    letters = sum(ch.isalpha() for ch in stripped)
    digits = sum(ch.isdigit() for ch in stripped)
    alnum = letters + digits
    punctuation = sum((not ch.isalnum() and not ch.isspace()) for ch in stripped)
    if alnum == 0:
        return True
    digit_ratio = digits / max(alnum, 1)
    punctuation_ratio = punctuation / max(len(stripped), 1)
    uppercase_tokens = [token for token in stripped.split() if len(token) >= 2 and token.isupper()]
    uppercase_ratio = len(uppercase_tokens) / max(len(stripped.split()), 1)
    return (
        digit_ratio > 0.55
        or punctuation_ratio > 0.35
        or (uppercase_ratio > 0.65 and letters > 8)
        or len(stripped) <= 2
    )


def garbage_text_ratio(lines: Iterable[str]) -> float:
    """Fraction of lines flagged as garbage."""
    line_list = [line for line in lines]
    if not line_list:
        return 0.0
    return sum(1 for line in line_list if is_garbage_line(line)) / len(line_list)


def is_hallucinated_line(text: str) -> bool:
    """Detect VietOCR hallucination on a single line.

    A line is flagged as hallucinated if it shows ANY of:
    - all-caps Latin abbreviation tokens (ND/TP/UBND-style)
    - long digit runs / repeated 0-1 sequences
    - abnormal-symbol rate > 5%
    - mixed digit-letter "suspicious" tokens
    """
    if not text:
        return False
    if uppercase_garbage_token_count(text) >= 1:
        return True
    if repeated_number_sequence_count(text) >= 1:
        return True
    if abnormal_symbol_rate(text) > 0.05:
        return True
    if suspicious_digit_token_count(text) >= 1:
        return True
    return False


def diagnostic_metrics(lines: Iterable[str]) -> dict[str, float | int]:
    """Aggregate diagnostic metrics over a list of OCR lines for a single image.

    Returns a flat dict suitable for serialization to JSONL.
    """
    line_list = [line or "" for line in lines]
    full_text = "\n".join(line_list)
    line_count = len(line_list)
    hallucinated = [line for line in line_list if is_hallucinated_line(line)]
    return {
        "line_count": line_count,
        "char_count": len(full_text),
        "digit_noise_rate": digit_noise_rate(full_text),
        "garbage_text_ratio": garbage_text_ratio(line_list),
        "high_digit_noise_line_rate": (
            sum(1 for line in line_list if digit_noise_rate(line) >= 0.20) / max(line_count, 1)
        ),
        "abnormal_symbol_rate": abnormal_symbol_rate(full_text),
        "abnormal_symbol_count": abnormal_symbol_count(full_text),
        "suspicious_digit_token_count": suspicious_digit_token_count(full_text),
        "uppercase_garbage_token_count": uppercase_garbage_token_count(full_text),
        "uppercase_garbage_token_rate": uppercase_garbage_token_rate(full_text),
        "repeated_number_sequence_count": repeated_number_sequence_count(full_text),
        "hallucinated_line_count": len(hallucinated),
        "hallucinated_line_rate": len(hallucinated) / max(line_count, 1),
    }


def diagnostic_diff(before: dict, after: dict) -> dict[str, float | int | str]:
    """Compute before→after deltas for the diagnostic metrics dict.

    Negative deltas on the `*_rate` / `*_count` keys are improvements.
    Returns a verdict in {"improved", "worsened", "no_change"} based on
    whether the *hallucinated_line_rate* moved by more than 1 percentage point.
    """
    keys = set(before.keys()) | set(after.keys())
    diff: dict[str, float | int | str] = {}
    for key in keys:
        b = before.get(key, 0)
        a = after.get(key, 0)
        try:
            diff[f"delta_{key}"] = float(a) - float(b)
        except (TypeError, ValueError):
            continue
    halluc_delta = float(diff.get("delta_hallucinated_line_rate", 0.0))
    garbage_delta = float(diff.get("delta_garbage_text_ratio", 0.0))
    digit_delta = float(diff.get("delta_digit_noise_rate", 0.0))
    score = halluc_delta + 0.5 * garbage_delta + 0.5 * digit_delta
    if score < -0.01:
        diff["verdict"] = "improved"
    elif score > 0.01:
        diff["verdict"] = "worsened"
    else:
        diff["verdict"] = "no_change"
    diff["diagnostic_score_delta"] = score
    return diff


# ── Detector-level metrics (Phase 3) ─────────────────────────────────────────
#
# These metrics measure detector quality directly, *without* needing
# ground-truth boxes. They are the project's leaderboard signal for
# comparing detector backends (Surya / CRAFT / Kraken / Paddle).
#
# All metrics expect a list of polygons. A polygon is any iterable that
# can be coerced to an (N, 2) numpy array (lists of (x, y) tuples,
# 4×2 quads, etc.).


def _polygon_bbox(polygon) -> tuple[float, float, float, float]:
    """Return the axis-aligned bbox (x1, y1, x2, y2) of a polygon."""
    import numpy as _np

    pts = _np.asarray(polygon, dtype=float).reshape(-1, 2)
    return (
        float(_np.min(pts[:, 0])),
        float(_np.min(pts[:, 1])),
        float(_np.max(pts[:, 0])),
        float(_np.max(pts[:, 1])),
    )


def full_width_band_rate(polygons, image_width: int, ratio: float = 0.95) -> float:
    """Fraction of polygons whose width covers >= ``ratio`` of the image width.

    Detector-quality fingerprint: the silent OpenCV grid fallback used to
    emit fixed-width horizontal bands across the page. Healthy detectors
    score ~0.0; the fallback scores ~1.0.
    """
    if not polygons or image_width <= 0:
        return 0.0
    threshold = float(image_width) * float(ratio)
    full_count = 0
    for poly in polygons:
        x1, _, x2, _ = _polygon_bbox(poly)
        if (x2 - x1) >= threshold:
            full_count += 1
    return full_count / float(len(polygons))


def mean_box_aspect_ratio(polygons) -> float:
    """Mean of width/height across all polygons.

    Healthy text lines are wide (aspect ~ 8–25). Single-character
    detections are ~1; full-page bands explode upwards.
    """
    if not polygons:
        return 0.0
    ratios: list[float] = []
    for poly in polygons:
        x1, y1, x2, y2 = _polygon_bbox(poly)
        height = max(y2 - y1, 1.0)
        width = max(x2 - x1, 0.0)
        ratios.append(width / height)
    return float(sum(ratios) / len(ratios))


def mean_distinct_x1_per_page(polygons, bin_size: int = 10) -> float:
    """Number of distinct *left edges*, rounded to ``bin_size``-pixel bins.

    Real handwritten pages have varied paragraph indents and word starts,
    so distinct-x1 counts are typically 6+. Grid-fallback bands all start
    at x=0, so this metric collapses to 1.

    Returned as a float to match the rest of the metrics module — the
    caller can int() it where needed.
    """
    if not polygons:
        return 0.0
    bin_size = max(int(bin_size), 1)
    seen: set[int] = set()
    for poly in polygons:
        x1, _, _, _ = _polygon_bbox(poly)
        seen.add(int(round(x1 / bin_size)) * bin_size)
    return float(len(seen))


def detector_metrics(polygons, image_width: int, image_height: int) -> dict:
    """Bundle the three detector-quality metrics + raw count + median height.

    This is the single function the experiment runner calls per page.
    """
    import numpy as _np

    heights: list[float] = []
    for poly in polygons:
        _, y1, _, y2 = _polygon_bbox(poly)
        heights.append(max(y2 - y1, 1.0))
    median_height = float(_np.median(heights)) if heights else 0.0

    return {
        "detection_count": len(polygons),
        "full_width_band_rate": full_width_band_rate(polygons, image_width),
        "mean_box_aspect_ratio": mean_box_aspect_ratio(polygons),
        "mean_distinct_x1_per_page": mean_distinct_x1_per_page(polygons),
        "median_box_height": median_height,
        "image_width": float(image_width),
        "image_height": float(image_height),
    }
