from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

from ocr_pipeline.config import settings


class PostProcessError(RuntimeError):
    pass


@dataclass
class SymSpellCorrector:
    max_edit_distance: int = 1

    def __post_init__(self) -> None:
        self._sym_spell = None
        self._dict_built = False

    def _ensure_loaded(self) -> None:
        if self._dict_built:
            return
        try:
            from symspellpy import SymSpell, Verbosity
        except ImportError as e:
            raise PostProcessError(
                "symspellpy is required for SymSpell post-processing. "
                "Install with: pip install symspellpy"
            ) from e

        train_path = settings.data_dir / "processed" / "train_annotation.txt"
        if not train_path.exists():
            raise PostProcessError(
                f"Training annotations not found: {train_path}. "
                "Cannot build SymSpell dictionary."
            )

        sym_spell = SymSpell(max_dictionary_edit_distance=self.max_edit_distance, prefix_length=7)
        with open(train_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    text = parts[1]
                    for word in text.split():
                        word_clean = re.sub(r"^\\W+|\\W+$", "", word)
                        if word_clean:
                            sym_spell.create_dictionary_entry(word_clean, 1)

        self._sym_spell = sym_spell
        self._dict_built = True

    def correct_lines(self, lines: list[str]) -> list[str]:
        from symspellpy import Verbosity

        self._ensure_loaded()
        corrected = []
        for line in lines:
            words = line.split()
            corrected_words = []
            for word in words:
                mo = re.match(r"^(\\W*)(.*?)(\\W*)$", word)
                if mo:
                    prefix, core_word, suffix = mo.groups()
                else:
                    prefix, core_word, suffix = "", word, ""
                if core_word == "":
                    corrected_words.append(word)
                    continue
                suggestions = self._sym_spell.lookup(
                    core_word, Verbosity.CLOSEST, max_edit_distance=self.max_edit_distance
                )
                if suggestions:
                    best_match = suggestions[0].term
                    if core_word.istitle():
                        best_match = best_match.title()
                    elif core_word.isupper():
                        best_match = best_match.upper()
                    corrected_words.append(prefix + best_match + suffix)
                else:
                    corrected_words.append(word)
            corrected.append(" ".join(corrected_words))
        return corrected


@dataclass
class BartphoCorrector:
    model_name: str = "bmd1905/vietnamese-correction-v2"
    max_length: int = 256

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as e:
            raise PostProcessError(
                "transformers is required for BARTpho post-processing. "
                "Install with: pip install transformers sentencepiece"
            ) from e

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        except (OSError, ValueError) as e:
            raise PostProcessError(
                f"Failed to load BARTpho model '{self.model_name}': {e}"
            ) from e

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def correct_lines(self, lines: list[str]) -> list[str]:
        self._ensure_loaded()
        corrected: list[str] = []
        for line in lines:
            inputs = self._tokenizer(
                line,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model.generate(
                **inputs,
                max_length=self.max_length,
            )
            corrected.append(
                self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            )
        return corrected


class PostProcessor:
    def __init__(
        self,
        symspell: Optional[SymSpellCorrector] = None,
        bartpho: Optional[BartphoCorrector] = None,
    ):
        self.symspell = symspell or SymSpellCorrector()
        self.bartpho = bartpho or BartphoCorrector()

    def process_lines(self, lines: list[str]) -> dict[str, list[str]]:
        symspell_lines = self.symspell.correct_lines(lines)
        bartpho_lines = self.bartpho.correct_lines(lines)
        return {
            "symspell": symspell_lines,
            "bartpho": bartpho_lines,
        }


_post_processor: Optional[PostProcessor] = None


def get_post_processor() -> PostProcessor:
    global _post_processor
    if _post_processor is None:
        _post_processor = PostProcessor()
    return _post_processor
