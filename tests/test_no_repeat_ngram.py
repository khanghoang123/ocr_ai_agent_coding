"""Tests for the no-repeat-ngram decoder constraint.

The constraint is implemented as a helper (`_banned_next_tokens`) and a
custom greedy decode method on `VietOCRRecognizer`. We test the helper
thoroughly on known token-id patterns, and verify that the recognizer
constructor / config plumbing carries the knob end-to-end.

We deliberately do not exercise the actual VietOCR model here — that
path requires the ~90MB fine-tuned checkpoint and torch, neither of
which is available on the CI quality image. The helper is pure Python
and fully covers the algorithmic correctness.
"""

from __future__ import annotations

from ocr_pipeline.recognizer.vietocr_recognizer import (
    VietOCRRecognizer,
    _banned_next_tokens,
)


class TestBannedNextTokens:
    def test_empty_prefix_returns_empty_set(self) -> None:
        assert _banned_next_tokens([], 3) == set()

    def test_prefix_shorter_than_ngram_returns_empty_set(self) -> None:
        # With ngram_size=3 we need at least 3 prior tokens to have seen
        # any ngram, so any shorter prefix is unconstrained.
        assert _banned_next_tokens([1], 3) == set()
        assert _banned_next_tokens([1, 2], 3) == set()

    def test_ngram_size_one_is_disabled(self) -> None:
        # An ngram_size of 1 would ban every previously-seen token,
        # which is unusable for a language model. We define size<=1 as
        # effectively off.
        assert _banned_next_tokens([0, 1, 2, 3], 1) == set()
        assert _banned_next_tokens([0, 1, 2, 3], 0) == set()

    def test_repeat_digit_attractor_is_blocked(self) -> None:
        # Classic seq2seq pathology: "010101..." — after [0, 1, 0, 1] is
        # emitted, the next token MUST NOT be 0 (which would complete
        # the 3-gram (1, 0, 1)->0 → prefix (1, 0) already saw 1 and 0
        # following it, so BOTH are banned).
        banned = _banned_next_tokens([0, 1, 0, 1], 3)
        # Walk the 3-grams of [0, 1, 0, 1]:
        #   (0, 1) -> 0 at position 0-2
        #   (1, 0) -> 1 at position 1-3
        # Current prefix (last n-1 = 2 tokens) is (0, 1). The prefix
        # (0, 1) was historically followed by 0. So 0 is banned.
        assert 0 in banned
        assert banned == {0}

    def test_longer_repeat_run_bans_continuation(self) -> None:
        # [0, 1, 0, 1, 0, 1] — whichever trigram pattern we pick, the
        # constraint must block the next step from extending the run.
        # Current prefix (0, 1) historically precedes 0 (at indexes 0-2
        # and 2-4). So 0 is banned; the model is forced to emit
        # something non-zero (ending the repetition).
        assert _banned_next_tokens([0, 1, 0, 1, 0, 1], 3) == {0}

    def test_three_token_attractor(self) -> None:
        # "ND-ND-ND..." pattern — using arbitrary token ids 78/68.
        # [78, 68, 78, 68, 78] => prefix (68, 78) precedes 68 at
        # positions (1, 2, 3). So 68 is banned.
        assert _banned_next_tokens([78, 68, 78, 68, 78], 3) == {68}

    def test_natural_text_not_over_constrained(self) -> None:
        # Simulate a normal Vietnamese line of distinct tokens: natural
        # trigrams are unique, so nothing is banned.
        tokens = [1, 5, 8, 10, 15, 20, 25, 30]
        assert _banned_next_tokens(tokens, 3) == set()

    def test_natural_repeat_of_a_word_does_block_trigram(self) -> None:
        # Occasionally the same 3-char word appears twice in a line
        # (e.g. "các các"). Our constraint DOES block the 3rd
        # reoccurrence of the exact trigram; this is the known trade-off.
        # "cac cac cac" with spaces → roughly [c,a,c,_,c,a,c,_,c,a]
        #   pos 0-2: (c,a,c)   — prefix (c,a) followed by c
        #   pos 4-6: (c,a,c)   — prefix (c,a) followed by c
        # prefix (c,a) precedes c; at next step c is banned.
        # This is intentional: a third consecutive "cac" is vastly more
        # likely to be an attractor than real text on a single line.
        tokens = [99, 98, 99, 10, 99, 98, 99, 10, 99, 98]
        assert _banned_next_tokens(tokens, 3) == {99}

    def test_ngram_size_four_requires_longer_prefix(self) -> None:
        # [0, 1, 0, 1, 0, 1] with ngram_size=4:
        #   4-grams: (0,1,0,1), (1,0,1,0), (0,1,0,1). Prefix (last 3) is
        #   (1, 0, 1). The 4-gram (1, 0, 1, ?) was seen once with ? = 0.
        #   So 0 is banned.
        assert _banned_next_tokens([0, 1, 0, 1, 0, 1], 4) == {0}


class TestRecognizerConstructorWiring:
    def test_default_constructor_disables_constraint(self) -> None:
        # Default parameter is 0 = disabled → legacy vietocr path.
        rec = VietOCRRecognizer(
            weights_path="/nonexistent/weights.pth",
            architecture="vgg_seq2seq",
        )
        assert rec.no_repeat_ngram_size == 0

    def test_explicit_constructor_sets_constraint(self) -> None:
        rec = VietOCRRecognizer(
            weights_path="/nonexistent/weights.pth",
            architecture="vgg_seq2seq",
            no_repeat_ngram_size=3,
        )
        assert rec.no_repeat_ngram_size == 3

    def test_none_is_normalised_to_zero(self) -> None:
        # Some callers may pass None to mean "default"; our init should
        # coerce to int, and `None or 0` is 0 = disabled.
        rec = VietOCRRecognizer(
            weights_path="/nonexistent/weights.pth",
            architecture="vgg_seq2seq",
            no_repeat_ngram_size=None,  # type: ignore[arg-type]
        )
        assert rec.no_repeat_ngram_size == 0

    def test_settings_default_is_three(self) -> None:
        # The global default in Settings is 3 (we want the constraint
        # ON by default in production).
        from ocr_pipeline.config import Settings

        settings = Settings()
        assert settings.rec_no_repeat_ngram_size == 3


class TestExperimentConfigWiring:
    def test_default_is_none_meaning_use_settings(self) -> None:
        # Experiment runners default to `None` so the global Settings
        # value flows through. An explicit integer overrides it.
        from ocr_pipeline.experiment_config import ExperimentConfig

        cfg = ExperimentConfig()
        assert cfg.rec_no_repeat_ngram_size is None

    def test_explicit_override_is_preserved(self) -> None:
        from ocr_pipeline.experiment_config import ExperimentConfig

        cfg = ExperimentConfig(rec_no_repeat_ngram_size=4)
        assert cfg.rec_no_repeat_ngram_size == 4

    def test_zero_disables_for_this_experiment(self) -> None:
        # A calibration experiment may want to measure the baseline
        # decoder with the constraint off.
        from ocr_pipeline.experiment_config import ExperimentConfig

        cfg = ExperimentConfig(rec_no_repeat_ngram_size=0)
        assert cfg.rec_no_repeat_ngram_size == 0
