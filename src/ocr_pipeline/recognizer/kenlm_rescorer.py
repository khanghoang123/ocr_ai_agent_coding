"""
KenLM Vietnamese 5-gram rescorer for the VietOCR recognizer path.

The recognizer's fine-tuned ``baseline_50k`` checkpoint is known to emit
semantically wrong but locally plausible outputs on hard handwriting
crops (e.g. ``Ninn 41.100 VPV 101 SD TP Trên…``). A character-level
no-repeat-ngram constraint (PR #6) suppresses the repeating-digit
attractor but does not help these local mistakes.

This module runs a shallow beam over the recognizer's own decoder and
picks the candidate that maximises::

    score(candidate) = acoustic_logprob + alpha * lm_logprob
                       + beta * word_count(candidate)

following the classic Bahdanau / Mozilla DeepSpeech scoring rule.

The rescorer is loaded **lazily** and is a no-op whenever the KenLM
``.bin`` (or ``.arpa``) file referenced by ``model_path`` does not
exist. This is deliberate:

* KenLM itself is an optional dependency (not in ``requirements.txt``).
* The trained Vietnamese LM is user-provided (see
  ``scripts/train_kenlm.py``) and is **not** committed to the repo.
* In CI, tests mock the scorer. In production, a missing file simply
  falls back to the recognizer's raw top-1 output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One (text, acoustic_logprob) hypothesis from the recognizer."""

    text: str
    acoustic_logprob: float = 0.0


@dataclass
class RescoreResult:
    """Rescoring outcome for a single line."""

    text: str
    acoustic_logprob: float
    lm_logprob: float
    word_count: int
    score: float
    # Where this candidate came from. "rescored" means the rescorer
    # selected a non-top-1 candidate; "top1" means the top-1 acoustic
    # hypothesis was already the best; "disabled" means the LM file
    # was not loadable and we passed the input through untouched.
    source: str = "top1"


@dataclass
class KenLMRescorer:
    """Rescore recognizer N-best candidates with a Vietnamese 5-gram KenLM.

    ``alpha`` weights the LM log-probability, ``beta`` is a word-count
    bonus (prevents the LM from preferring shorter candidates just
    because each token contributes a negative log-prob), and
    ``gamma`` is the acoustic weight (usually 1.0 — kept configurable
    for ablation).

    The scorer **case-insensitively normalises** the candidate before
    scoring against the LM: Vietnamese OCR noise often manifests as
    stray uppercase (``CTP`` vs ``ctp``), and our training corpus
    typically uses natural case. The original (case-preserving)
    candidate text is what gets returned — only the LM lookup is
    lowercased.

    Args:
        model_path: Path to a ``.bin`` or ``.arpa`` KenLM model. If it
            does not exist, the rescorer operates in no-op mode and
            always returns the top-1 acoustic candidate.
        alpha: LM log-prob weight. Larger values trust the LM more.
        beta: Word-count bonus. Larger values prefer longer outputs.
        gamma: Acoustic log-prob weight. Usually 1.0.
        case_insensitive_lm: If True, lowercase the candidate text
            before computing the LM log-prob. Defaults to True.
    """

    model_path: str | Path | None = None
    alpha: float = 0.5
    beta: float = 0.1
    gamma: float = 1.0
    case_insensitive_lm: bool = True

    _model: object | None = field(default=None, init=False, repr=False)
    _load_attempted: bool = field(default=False, init=False, repr=False)
    _disabled_reason: str | None = field(default=None, init=False, repr=False)

    # ── Loading ──────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True

        if self.model_path is None:
            self._disabled_reason = "no model_path configured"
            return
        path = Path(self.model_path)
        if not path.exists():
            self._disabled_reason = f"model file not found: {path}"
            logger.info(
                "KenLMRescorer disabled — %s. The recognizer will return "
                "the top-1 acoustic candidate unchanged.",
                self._disabled_reason,
            )
            return

        try:
            import kenlm  # noqa: WPS433 — intentional lazy import
        except ImportError as exc:
            self._disabled_reason = (
                f"kenlm package not installed ({exc}). Install with "
                "`pip install kenlm` to enable LM rescoring."
            )
            logger.info("KenLMRescorer disabled — %s", self._disabled_reason)
            return

        try:
            self._model = kenlm.Model(str(path))
            logger.info(
                "KenLMRescorer loaded model from %s "
                "(alpha=%.3f, beta=%.3f, gamma=%.3f)",
                path, self.alpha, self.beta, self.gamma,
            )
        except (OSError, RuntimeError) as exc:
            self._disabled_reason = (
                f"kenlm.Model failed to load {path}: {exc}"
            )
            logger.warning("KenLMRescorer disabled — %s", self._disabled_reason)
            self._model = None

    @property
    def is_enabled(self) -> bool:
        """True if an LM is loaded and rescoring is active."""
        self._ensure_loaded()
        return self._model is not None

    @property
    def disabled_reason(self) -> str | None:
        """Human-readable reason the rescorer is operating as a no-op,
        or None if the LM is active."""
        self._ensure_loaded()
        return self._disabled_reason

    # ── Scoring ──────────────────────────────────────────────────

    def lm_logprob(self, text: str) -> float:
        """Return the KenLM log10-probability of ``text``.

        Returns 0.0 if the model is not loaded (no-op mode).
        """
        self._ensure_loaded()
        if self._model is None:
            return 0.0
        query = text.lower() if self.case_insensitive_lm else text
        # kenlm.Model.score returns log10-probability with sentence
        # boundary tokens added by default.
        return float(self._model.score(query, bos=True, eos=True))

    def rescore_candidates(
        self,
        candidates: Iterable[Candidate],
    ) -> RescoreResult:
        """Pick the best candidate for a single line.

        The scoring rule is::

            score = gamma * acoustic_logprob
                  + alpha * lm_logprob
                  + beta  * word_count

        Ties are broken by the input order (stable). If the rescorer
        is disabled (no LM, no kenlm, bad file), we return the first
        candidate untouched and mark ``source="disabled"``.
        """
        cand_list = list(candidates)
        if not cand_list:
            return RescoreResult(
                text="",
                acoustic_logprob=0.0,
                lm_logprob=0.0,
                word_count=0,
                score=0.0,
                source="empty",
            )

        self._ensure_loaded()
        if self._model is None:
            top1 = cand_list[0]
            return RescoreResult(
                text=top1.text,
                acoustic_logprob=top1.acoustic_logprob,
                lm_logprob=0.0,
                word_count=len(top1.text.split()),
                score=float(self.gamma * top1.acoustic_logprob),
                source="disabled",
            )

        best_idx = 0
        best_score = float("-inf")
        best_details: dict[str, float] = {}

        for idx, cand in enumerate(cand_list):
            lm = self.lm_logprob(cand.text)
            words = len(cand.text.split())
            # Length-normalise the LM logprob so a long-but-fluent
            # candidate is not punished simply for having more
            # words. ``word_count`` bonus is applied separately to
            # let the caller tune how much the rescorer biases
            # toward longer outputs.
            lm_per_word = lm / max(1, words)
            score = (
                self.gamma * cand.acoustic_logprob
                + self.alpha * lm_per_word
                + self.beta * words
            )
            if score > best_score:
                best_idx = idx
                best_score = score
                best_details = {
                    "lm_logprob": lm,
                    "word_count": float(words),
                }

        chosen = cand_list[best_idx]
        source = "top1" if best_idx == 0 else "rescored"
        return RescoreResult(
            text=chosen.text,
            acoustic_logprob=chosen.acoustic_logprob,
            lm_logprob=best_details.get("lm_logprob", 0.0),
            word_count=int(best_details.get("word_count", 0)),
            score=best_score,
            source=source,
        )

    # ── Convenience constructors ─────────────────────────────────

    @classmethod
    def from_settings(cls) -> "KenLMRescorer":
        """Build a rescorer from the global Settings defaults.

        Reads ``rec_kenlm_path``, ``rec_kenlm_alpha``,
        ``rec_kenlm_beta``, ``rec_kenlm_gamma`` from Settings.
        """
        from ocr_pipeline.config import settings

        return cls(
            model_path=(
                Path(settings.rec_kenlm_path)
                if settings.rec_kenlm_path
                else None
            ),
            alpha=float(settings.rec_kenlm_alpha),
            beta=float(settings.rec_kenlm_beta),
            gamma=float(settings.rec_kenlm_gamma),
        )


class _NullRescorer:
    """Stand-in that always returns the top-1 candidate unchanged.

    Used internally by the recognizer when the caller did not attach a
    KenLMRescorer. Keeps the code path identical regardless of whether
    rescoring is enabled.
    """

    is_enabled = False
    disabled_reason = "no rescorer attached"

    def rescore_candidates(
        self, candidates: Iterable[Candidate]
    ) -> RescoreResult:
        cand_list = list(candidates)
        if not cand_list:
            return RescoreResult("", 0.0, 0.0, 0, 0.0, source="empty")
        top1 = cand_list[0]
        return RescoreResult(
            text=top1.text,
            acoustic_logprob=top1.acoustic_logprob,
            lm_logprob=0.0,
            word_count=len(top1.text.split()),
            score=top1.acoustic_logprob,
            source="disabled",
        )
