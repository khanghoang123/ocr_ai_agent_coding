from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from ocr_pipeline.validation.metrics import digit_noise_rate, suspicious_digit_tokens


@dataclass
class SafePostprocessResult:
    raw_text: str
    cleaned_text: str
    flags: list[str] = field(default_factory=list)
    digit_noise_rate: float = 0.0
    suspicious_tokens: list[str] = field(default_factory=list)


def safe_vietnamese_postprocess(text: str, flag_digit_noise: bool = True) -> SafePostprocessResult:
    raw = text or ""
    cleaned = unicodedata.normalize("NFC", raw)
    cleaned = re.sub(r"[ \t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines()).strip()

    suspicious = suspicious_digit_tokens(cleaned) if flag_digit_noise else []
    flags: list[str] = []
    if suspicious:
        flags.append("digit_noise")
    rate = digit_noise_rate(cleaned) if flag_digit_noise else 0.0
    return SafePostprocessResult(
        raw_text=raw,
        cleaned_text=cleaned,
        flags=flags,
        digit_noise_rate=rate,
        suspicious_tokens=suspicious,
    )


def safe_postprocess_lines(
    lines: list[str],
    flag_digit_noise: bool = True,
) -> list[SafePostprocessResult]:
    return [safe_vietnamese_postprocess(line, flag_digit_noise=flag_digit_noise) for line in lines]
