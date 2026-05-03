#!/usr/bin/env python3
"""
Train a Vietnamese 5-gram KenLM from a plain-text corpus.

Usage:
    python scripts/train_kenlm.py \\
        --input data/processed/train_annotation.txt \\
        --output models/kenlm/vi_5gram.bin \\
        --order 5 \\
        [--trust-symbols]

The training corpus may be either:

  * One sentence per line (preferred).
  * A VietOCR-style annotation file where each line is
    ``<image_path>\\t<ground_truth_text>``. The script detects this
    layout automatically and extracts the text column.

The output is a quantised KenLM binary (``.bin``) suitable for
``kenlm.Model(path)`` at inference time.

Requirements:
  * ``lmplz`` and ``build_binary`` in ``$PATH`` (build from
    https://github.com/kpu/kenlm). This script does NOT depend on any
    Python package beyond the standard library.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def extract_text_column(line: str) -> str:
    """Return the text portion of a corpus line.

    Supports both ``<text>`` and ``<path>\\t<text>`` layouts. Any
    line that contains a single tab is treated as the VietOCR
    annotation layout.
    """
    line = line.rstrip("\r\n")
    if "\t" in line:
        parts = line.split("\t")
        # "path\ttext" or "path\ttext\textra" — take the last column as text
        return parts[-1]
    return line


_WS_RE = re.compile(r"\s+")


def normalise(line: str) -> str:
    """Apply KenLM-safe normalisation: NFC + whitespace collapse.

    We deliberately do NOT lowercase or strip accents — the LM must
    preserve Vietnamese diacritics so that it can reward the correct
    diacritic variant of an ambiguous OCR token (e.g. ``động`` over
    ``dong``).
    """
    cleaned = unicodedata.normalize("NFC", line)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def prepare_corpus(input_path: Path, tmp_dir: Path) -> Path:
    """Write one sentence per line to a normalised scratch file."""
    scratch = tmp_dir / "corpus.txt"
    kept = 0
    total = 0
    with input_path.open("r", encoding="utf-8") as src, scratch.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw in src:
            total += 1
            text = normalise(extract_text_column(raw))
            if text:
                dst.write(text + "\n")
                kept += 1
    logger.info(
        "Prepared corpus: %d/%d lines retained at %s",
        kept, total, scratch,
    )
    if kept == 0:
        raise SystemExit(
            f"Input {input_path} produced no usable lines after "
            "normalisation — check the file format."
        )
    return scratch


def run_lmplz(
    corpus: Path,
    arpa_out: Path,
    order: int,
    trust_symbols: bool,
    extra_args: list[str],
) -> None:
    lmplz = shutil.which("lmplz")
    if lmplz is None:
        raise SystemExit(
            "lmplz not found in $PATH. Build KenLM from "
            "https://github.com/kpu/kenlm and make sure `lmplz` and "
            "`build_binary` are on PATH."
        )
    cmd = [lmplz, "-o", str(order)]
    if trust_symbols:
        cmd.append("--skip_symbols")
    cmd.extend(extra_args)
    logger.info("Running: %s < %s > %s", " ".join(cmd), corpus, arpa_out)
    with corpus.open("rb") as stdin, arpa_out.open("wb") as stdout:
        proc = subprocess.run(cmd, stdin=stdin, stdout=stdout, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"lmplz failed with exit code {proc.returncode}")


def run_build_binary(arpa_in: Path, bin_out: Path) -> None:
    bb = shutil.which("build_binary")
    if bb is None:
        raise SystemExit(
            "build_binary not found in $PATH. Build KenLM and add it to PATH."
        )
    # ``trie -q 8 -b 8`` is a good default for our small corpora.
    cmd = [bb, "trie", "-q", "8", "-b", "8", str(arpa_in), str(bin_out)]
    logger.info("Running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"build_binary failed with exit code {proc.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to training corpus (one sentence per line, or "
             "tab-separated VietOCR annotation file).",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output KenLM binary path (.bin). An intermediate .arpa "
             "is written alongside.",
    )
    parser.add_argument(
        "--order", type=int, default=5, help="N-gram order (default 5).",
    )
    parser.add_argument(
        "--trust-symbols", action="store_true",
        help="Pass --skip_symbols to lmplz (allow <unk> / <s> / </s> "
             "in the input without failing).",
    )
    parser.add_argument(
        "--lmplz-arg", action="append", default=[],
        help="Extra args forwarded to lmplz (can be repeated).",
    )
    parser.add_argument(
        "--keep-arpa", action="store_true",
        help="Keep the intermediate .arpa alongside the .bin.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Corpus not found: %s", args.input)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    arpa_out = args.output.with_suffix(".arpa")

    with tempfile.TemporaryDirectory(prefix="kenlm_train_") as tmp:
        tmp_path = Path(tmp)
        corpus = prepare_corpus(args.input, tmp_path)
        run_lmplz(
            corpus=corpus,
            arpa_out=arpa_out,
            order=args.order,
            trust_symbols=args.trust_symbols,
            extra_args=list(args.lmplz_arg),
        )
        run_build_binary(arpa_out, args.output)

    if not args.keep_arpa and arpa_out.exists():
        arpa_out.unlink()

    size_mb = args.output.stat().st_size / 1e6
    logger.info(
        "KenLM binary written to %s (%.2f MB). Set "
        "OCR_REC_KENLM_PATH=%s to enable rescoring.",
        args.output, size_mb, args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
