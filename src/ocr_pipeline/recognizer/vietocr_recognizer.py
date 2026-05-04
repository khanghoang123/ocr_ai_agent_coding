"""
VietOCR text recognition module.

Wraps the VietOCR Predictor with:
- Lazy model loading (load once, infer many)
- Batch inference support
- Graceful handling of edge-case crops (very long, very short lines)
- Performance timing

Extracted and refactored from notebooks/08_train_experiment_B.ipynb
and notebooks/04_evaluate_baseline.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from ocr_pipeline.recognizer.kenlm_rescorer import Candidate

logger = logging.getLogger(__name__)


def _banned_next_tokens(prev_tokens: list[int], ngram_size: int) -> set[int]:
    """Return the set of tokens that must NOT be emitted next because
    doing so would reproduce an (ngram_size)-gram already seen earlier
    in ``prev_tokens``.

    For example with ngram_size=3 and prev_tokens = [1, 5, 0, 1, 0, 1],
    the last two tokens are (0, 1). The prefix (0, 1) already appears at
    positions (2, 3) and (4, 5); the token that followed it was 0 both
    times, so emitting 0 again would reproduce the 3-gram (0, 1, 0).
    We therefore ban {0} as the next token.
    """
    if ngram_size <= 1 or len(prev_tokens) < ngram_size:
        return set()

    seen: dict[tuple[int, ...], set[int]] = {}
    # Walk every (ngram_size - 1)-prefix in prev_tokens and record the
    # token that historically followed it.
    for i in range(len(prev_tokens) - ngram_size + 1):
        key = tuple(prev_tokens[i : i + ngram_size - 1])
        next_tok = prev_tokens[i + ngram_size - 1]
        seen.setdefault(key, set()).add(next_tok)

    current_prefix = tuple(prev_tokens[-(ngram_size - 1) :])
    return seen.get(current_prefix, set())


@dataclass
class RecognitionResult:
    text: str
    probability: float
    decode_mode: str
    pre_norm_ratio: float
    post_norm_ratio: float
    baseline_confidence: float
    baseline_offset: float
    normalized_preview: np.ndarray | None = None


@dataclass
class RecognitionPreprocessResult:
    image: Image.Image
    preview_rgb: np.ndarray
    pre_norm_ratio: float
    post_norm_ratio: float
    baseline_confidence: float
    baseline_offset: float


class RecognitionPreprocessor:
    def __init__(
        self,
        target_height: int = 32,
        target_ratio: float = 4.0,
        min_ratio: float = 3.5,
        max_ratio: float = 5.0,
        baseline_target_ratio: float = 0.70,
        min_anchor_width: int = 64,
        enable_local_contrast: bool = True,
    ):
        self.target_height = target_height
        self.target_ratio = target_ratio
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.baseline_target_ratio = baseline_target_ratio
        self.min_anchor_width = min_anchor_width
        self.enable_local_contrast = enable_local_contrast

    def preprocess(
        self,
        image: Image.Image,
        max_width: int,
    ) -> RecognitionPreprocessResult:
        crop_rgb = np.array(image.convert("RGB"))
        if crop_rgb.size == 0:
            blank = np.full((self.target_height, self.target_height, 3), 255, dtype=np.uint8)
            return RecognitionPreprocessResult(
                image=Image.fromarray(blank),
                preview_rgb=blank,
                pre_norm_ratio=1.0,
                post_norm_ratio=1.0,
                baseline_confidence=0.0,
                baseline_offset=0.0,
            )

        pre_norm_ratio = float(crop_rgb.shape[1]) / float(max(crop_rgb.shape[0], 1))
        contrast_rgb = self._normalize_contrast(crop_rgb) if self.enable_local_contrast else crop_rgb
        resized = self._resize_height(contrast_rgb)
        baseline_confidence, baseline_position = self._estimate_baseline(resized)
        padded = self._pad_to_ratio(resized, max_width=max_width)
        baseline_offset = 0.0

        if (
            baseline_confidence > 0.7
            and resized.shape[1] >= self.min_anchor_width
            and baseline_position is not None
        ):
            target_baseline = self.baseline_target_ratio * float(padded.shape[0] - 1)
            baseline_offset = float(target_baseline - baseline_position)
            padded = self._shift_vertical(padded, baseline_offset)

        post_norm_ratio = float(padded.shape[1]) / float(max(padded.shape[0], 1))
        normalized = Image.fromarray(padded)
        return RecognitionPreprocessResult(
            image=normalized,
            preview_rgb=padded,
            pre_norm_ratio=pre_norm_ratio,
            post_norm_ratio=post_norm_ratio,
            baseline_confidence=float(baseline_confidence),
            baseline_offset=float(baseline_offset),
        )

    def _normalize_contrast(self, crop_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(gray)
        return cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    def _resize_height(self, crop_rgb: np.ndarray) -> np.ndarray:
        height, width = crop_rgb.shape[:2]
        if height <= 0:
            return crop_rgb
        scale = self.target_height / float(height)
        target_width = max(1, int(round(width * scale)))
        return cv2.resize(
            crop_rgb,
            (target_width, self.target_height),
            interpolation=cv2.INTER_CUBIC if scale >= 1.0 else cv2.INTER_AREA,
        )

    def _pad_to_ratio(self, crop_rgb: np.ndarray, max_width: int) -> np.ndarray:
        height, width = crop_rgb.shape[:2]
        ratio = width / float(max(height, 1))
        target_width = width
        if ratio < self.min_ratio:
            target_width = int(round(self.min_ratio * height))
        elif ratio < self.target_ratio:
            target_width = int(round(self.target_ratio * height))

        target_width = min(max(target_width, width), max_width)
        if target_width <= width:
            return crop_rgb

        pad_total = target_width - width
        left = pad_total // 2
        right = pad_total - left
        return cv2.copyMakeBorder(
            crop_rgb,
            0,
            0,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

    def _estimate_baseline(self, crop_rgb: np.ndarray) -> tuple[float, float | None]:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        )

        bottoms = []
        for col_idx in range(binary.shape[1]):
            ys = np.where(binary[:, col_idx] > 0)[0]
            if ys.size < 2:
                continue
            bottoms.append(float(ys[-1]))

        if len(bottoms) < max(12, int(binary.shape[1] * 0.3)):
            return 0.0, None

        bottoms_arr = np.array(bottoms, dtype=np.float32)
        kernel_size = max(5, min(21, (len(bottoms_arr) // 8) * 2 + 1))
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smooth = np.convolve(bottoms_arr, kernel, mode="same")
        residual = float(np.mean(np.abs(bottoms_arr - smooth)))
        coverage = len(bottoms_arr) / float(max(binary.shape[1], 1))
        confidence = np.clip(coverage * np.exp(-residual / 3.5), 0.0, 1.0)
        return float(confidence), float(np.median(smooth))

    @staticmethod
    def _shift_vertical(crop_rgb: np.ndarray, offset: float) -> np.ndarray:
        if abs(offset) < 0.5:
            return crop_rgb
        transform = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, offset]], dtype=np.float32)
        return cv2.warpAffine(
            crop_rgb,
            transform,
            (crop_rgb.shape[1], crop_rgb.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )


class VietOCRRecognizer:
    """Singleton-friendly VietOCR recognizer.

    Usage:
        rec = VietOCRRecognizer.from_settings()        # or
        rec = VietOCRRecognizer(weights_path="...", device="cuda")
        text = rec.recognize(pil_image)
        texts = rec.recognize_batch([img1, img2, ...])
    """

    def __init__(
        self,
        weights_path: str | Path,
        architecture: str = "vgg_seq2seq",
        image_height: int = 32,
        image_max_width: int = 690,
        image_min_width: int = 32,
        device: str = "cpu",
        enable_local_contrast: bool = True,
        no_repeat_ngram_size: int = 0,
        kenlm_rescorer: object | None = None,
        kenlm_beam_width: int = 1,
    ):
        self.weights_path = str(weights_path)
        self.architecture = architecture
        self.image_height = image_height
        self.image_max_width = image_max_width
        self.image_min_width = image_min_width
        self.device = device
        # Seq2seq decoder no-repeat-ngram constraint. 0 disables the
        # constraint (legacy vietocr path); >0 blocks any token that
        # would complete an ngram of this size already emitted earlier
        # in the same line. Used to suppress the repeating-digit
        # attractor observed with fine-tuned checkpoints.
        self.no_repeat_ngram_size = int(no_repeat_ngram_size or 0)
        # KenLM 5-gram rescoring. When ``kenlm_beam_width > 1`` AND a
        # ``KenLMRescorer`` is attached, the recognizer maintains
        # ``beam_width`` partial hypotheses per decoder step, then
        # picks the best by (acoustic + alpha*LM + beta*word_count).
        # If the LM file is missing or kenlm is not installed, the
        # rescorer silently degrades to a no-op and the top-1
        # acoustic hypothesis is returned.
        self.kenlm_beam_width = max(1, int(kenlm_beam_width or 1))
        self.kenlm_rescorer = kenlm_rescorer
        self._preprocessor = RecognitionPreprocessor(
            target_height=image_height,
            enable_local_contrast=enable_local_contrast,
        )
        self._predictor = None   # Lazy init
        self._decode_mode = "greedy"

    # ── Loading ───────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return

        if not Path(self.weights_path).exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}\n"
                "Download the weights and place them at the path above,\n"
                "or update models/model_registry.yaml."
            )

        # Apply Pillow ≥10 compatibility patch BEFORE vietocr imports
        self._patch_pillow()

        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        logger.info(
            "Loading VietOCR model '%s' from %s (device=%s) ...",
            self.architecture, self.weights_path, self.device,
        )

        cfg = Cfg.load_config_from_name(self.architecture)
        cfg["weights"] = self.weights_path
        cfg["device"] = self.device
        cfg["cnn"]["pretrained"] = False
        cfg["dataset"]["image_height"]    = self.image_height
        cfg["dataset"]["image_max_width"] = self.image_max_width
        cfg["dataset"]["image_min_width"] = self.image_min_width
        predictor_cfg = cfg.setdefault("predictor", {})
        beam_enabled = isinstance(predictor_cfg, dict)
        # Our custom greedy/beam paths handle the no-repeat-ngram
        # constraint and KenLM rescoring. Force vietocr's own
        # beamsearch off whenever we intend to run a custom decode
        # path, so the predictor's internal decode logic doesn't
        # override ours.
        custom_decode = (
            self.no_repeat_ngram_size > 0 or self.kenlm_beam_width > 1
        )
        if beam_enabled:
            predictor_cfg["beamsearch"] = False if custom_decode else True
        try:
            self._predictor = Predictor(cfg)
            self._decode_mode = self._compute_decode_mode(beam_enabled)
        except Exception:
            if not beam_enabled:
                raise
            predictor_cfg["beamsearch"] = False
            self._predictor = Predictor(cfg)
            self._decode_mode = self._compute_decode_mode(beam_enabled=False)
        logger.info("VietOCR model loaded successfully.")

    def _compute_decode_mode(self, beam_enabled: bool) -> str:
        parts: list[str] = []
        if self.kenlm_beam_width > 1:
            parts.append(f"beam{self.kenlm_beam_width}")
        elif self.no_repeat_ngram_size > 0:
            parts.append("greedy")
        else:
            parts.append("beam" if beam_enabled else "greedy")
        if self.no_repeat_ngram_size > 0:
            parts.append(f"no_repeat_ngram_{self.no_repeat_ngram_size}")
        if self.kenlm_beam_width > 1 and getattr(
            self.kenlm_rescorer, "is_enabled", False
        ):
            parts.append("kenlm")
        return "+".join(parts)

    @staticmethod
    def _patch_pillow() -> None:
        """Patch PIL._util for Pillow ≥10 compatibility.

        Pillow ≥10.0 removed `is_path` and `is_directory` from PIL._util.
        VietOCR/torchvision may call these at import time.
        This patch (from notebook 08) must run BEFORE any vietocr import.
        """
        try:
            from PIL._util import is_directory as _test  # noqa: F401
        except ImportError:
            import os
            import PIL._util as _pu
            if not hasattr(_pu, "is_path"):
                _pu.is_path = lambda f: isinstance(f, (str, os.PathLike))
            if not hasattr(_pu, "is_directory"):
                _pu.is_directory = lambda f: os.path.isdir(f)

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize(self, image: Image.Image) -> str:
        """Recognize text in a single PIL RGB Image crop.

        Edge cases:
          - Very long lines: VietOCR truncates at image_max_width; we feed as-is
            (the model was trained with max_width=690, longer crops are rescaled)
          - Empty/blank crops: returns empty string
          - Grayscale inputs: converted to RGB automatically
        """
        self._ensure_loaded()
        try:
            prepared = self._preprocessor.preprocess(image, max_width=self.image_max_width)
            text, _ = self._predict_with_optional_probability(prepared.image)
            return text
        except Exception as e:
            logger.warning("Recognition failed for image (%s): %s", image.size, e)
            return ""

    def recognize_batch(
        self,
        images: list[Image.Image],
        return_prob: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
        """Recognize text for a batch of PIL Images.

        Args:
            images:      List of RGB PIL crops.
            return_prob: If True, return (text, confidence) tuples.

        Returns:
            List of strings, or list of (string, float) if return_prob=True.

        Note:
            VietOCR's Predictor does not expose native batch inference cleanly.
            We iterate with predict() here; the model still benefits from GPU
            through torch's internal batching when images are similar sizes.
        """
        detailed = self.recognize_batch_detailed(images, include_preview=False)
        if return_prob:
            return [(item.text, item.probability) for item in detailed]
        return [item.text for item in detailed]

    def recognize_batch_detailed(
        self,
        images: list[Image.Image],
        include_preview: bool = False,
    ) -> list[RecognitionResult]:
        self._ensure_loaded()
        results: list[RecognitionResult] = []
        for img in images:
            prepared = self._preprocessor.preprocess(img, max_width=self.image_max_width)
            preview = prepared.preview_rgb if include_preview else None
            try:
                text, prob = self._predict_with_optional_probability(prepared.image)
                results.append(
                    RecognitionResult(
                        text=text,
                        probability=prob,
                        decode_mode=self._decode_mode,
                        pre_norm_ratio=prepared.pre_norm_ratio,
                        post_norm_ratio=prepared.post_norm_ratio,
                        baseline_confidence=prepared.baseline_confidence,
                        baseline_offset=prepared.baseline_offset,
                        normalized_preview=preview,
                    )
                )
            except Exception as e:
                logger.warning("Batch recognition failed for one crop: %s", e)
                results.append(
                    RecognitionResult(
                        text="",
                        probability=0.0,
                        decode_mode=self._decode_mode,
                        pre_norm_ratio=prepared.pre_norm_ratio,
                        post_norm_ratio=prepared.post_norm_ratio,
                        baseline_confidence=prepared.baseline_confidence,
                        baseline_offset=prepared.baseline_offset,
                        normalized_preview=preview,
                    )
                )
        return results

    def _predict_no_repeat_ngram(
        self,
        image: Image.Image,
        max_seq_length: int = 128,
        sos_token: int = 1,
        eos_token: int = 2,
    ) -> tuple[str, float, float]:
        """Greedy seq2seq decode with a no-repeat-ngram constraint.

        At each decoder step we take the top-K logits over the vocab,
        then mask out any token that would cause the last
        ``no_repeat_ngram_size`` tokens to match an ngram already
        emitted earlier in the same line. This suppresses the
        repeating-digit attractor ("0101010...", "232323...", "NDND...")
        that fine-tuned ``baseline_50k`` produces on slanted
        handwriting crops.

        The math and data flow mirror ``vietocr.tool.translate.translate``
        so that token ids are compatible with ``self._predictor.vocab``.

        Returns ``(text, confidence, avg_logprob)`` where:
        * ``confidence`` is ``mean(p_i)`` over non-special tokens — the
          legacy per-step probability used by the API.
        * ``avg_logprob`` is ``mean(log(p_i))`` over the same tokens —
          commensurate with the per-token average log-prob returned by
          ``_predict_beam_no_repeat_ngram``. Combining greedy and beam
          candidates in the rescorer requires both scores live in the
          same space; ``log(mean(p))`` would be biased upward for
          greedy by Jensen's inequality and would systematically beat
          beam alternatives at the rescoring step.
        """
        import math

        import torch
        from torch.nn.functional import softmax
        from vietocr.tool.translate import process_input

        model = self._predictor.model
        vocab = self._predictor.vocab
        device = self._predictor.device

        img = process_input(
            image,
            self.image_height,
            self.image_min_width,
            self.image_max_width,
        ).to(device)

        n = int(self.no_repeat_ngram_size)
        model.eval()
        with torch.no_grad():
            src = model.cnn(img)
            memory = model.transformer.forward_encoder(src)

            batch_size = img.size(0)
            # token_seq[b] = list of token ids decoded so far for row b
            token_seq: list[list[int]] = [[sos_token] for _ in range(batch_size)]
            prob_seq: list[list[float]] = [[1.0] for _ in range(batch_size)]
            finished = [False] * batch_size

            for _ in range(max_seq_length):
                # Feed the full prefix each step (memory is cached).
                # vietocr's forward_decoder expects the target tensor as
                # [T, B] — see vietocr/tool/translate.py::translate
                # (`torch.LongTensor(translated_sentence)` is built as a
                # [T, B] tensor because `translated_sentence` is a list
                # where each element contains the t-th token for every
                # batch row). We mirror that shape exactly.
                max_t = max(len(s) for s in token_seq)
                padded = [s + [eos_token] * (max_t - len(s)) for s in token_seq]
                # padded is [B, T] → transpose to [T, B].
                tgt_inp = torch.LongTensor(padded).to(device).transpose(0, 1)
                output, memory = model.transformer.forward_decoder(tgt_inp, memory)
                # Both Seq2Seq and Transformer decoders return
                # output shape [B, T, V] for this input. Take the last
                # time-step per batch row.
                if output.dim() == 3 and output.size(0) == batch_size:
                    last_logits = output[:, -1, :]
                else:
                    last_logits = output[-1, :, :]
                last_probs = softmax(last_logits, dim=-1).to("cpu")

                # Mask ngram-repeats per batch row.
                if n > 0:
                    for b in range(batch_size):
                        if finished[b]:
                            continue
                        banned = _banned_next_tokens(token_seq[b], n)
                        if banned:
                            last_probs[b, list(banned)] = 0.0

                # Greedy pick.
                values, indices = torch.topk(last_probs, 1, dim=-1)
                indices = indices[:, 0].tolist()
                values = values[:, 0].tolist()

                for b in range(batch_size):
                    if finished[b]:
                        continue
                    token_seq[b].append(int(indices[b]))
                    prob_seq[b].append(float(values[b]))
                    if int(indices[b]) == eos_token:
                        finished[b] = True

                if all(finished):
                    break

            # batch_size is 1 in our wrapper path.
            tokens = token_seq[0]
            probs = prob_seq[0]
            text = vocab.decode(tokens)
            valid = [p for p, t in zip(probs, tokens) if t > 3]
            confidence = float(sum(valid) / len(valid)) if valid else 0.0
            if valid:
                avg_logprob = float(
                    sum(math.log(max(p, 1e-30)) for p in valid) / len(valid)
                )
            else:
                avg_logprob = 0.0
            return str(text), confidence, avg_logprob

    def _predict_beam_no_repeat_ngram(
        self,
        image: Image.Image,
        beam_width: int,
        max_seq_length: int = 128,
        sos_token: int = 1,
        eos_token: int = 2,
    ) -> list[tuple[str, float]]:
        """Beam-search seq2seq decode with a no-repeat-ngram constraint.

        Returns a list of ``(text, acoustic_mean_prob)`` candidates
        sorted best-first by cumulative acoustic log-probability. The
        length of the list is ``<= beam_width`` (beams that finish at
        ``eos`` are kept; beams that do not finish before
        ``max_seq_length`` are truncated at the last step).

        The implementation batches the K beams into one
        ``forward_decoder`` call per step to keep the cost close to K×
        the greedy path. Memory caching follows the same pattern as
        ``_predict_no_repeat_ngram``.

        Acoustic score is computed as the **sum of log-probabilities**
        along the decoded sequence (natural log, base-e). The
        per-candidate ``acoustic_mean_prob`` returned as the second
        tuple element matches the confidence reported by the greedy
        path (mean of per-step probs excluding special tokens) so
        the metric pipeline is unchanged.
        """
        import math

        import torch
        from torch.nn.functional import softmax
        from vietocr.tool.translate import process_input

        model = self._predictor.model
        vocab = self._predictor.vocab
        device = self._predictor.device

        img = process_input(
            image,
            self.image_height,
            self.image_min_width,
            self.image_max_width,
        ).to(device)

        n = int(self.no_repeat_ngram_size)
        K = max(1, int(beam_width))
        model.eval()
        with torch.no_grad():
            src = model.cnn(img)
            memory_1 = model.transformer.forward_encoder(src)
            # ``memory_1`` carries encoder state. For vietocr's seq2seq
            # architecture it is a ``(hidden, encoder_outputs)`` tuple
            # where ``hidden`` is the RNN state (which *evolves* over
            # decoder steps) and ``encoder_outputs`` is constant. For
            # the transformer architecture it is a single tensor.
            #
            # Each beam maintains its own ``memory`` so that the
            # per-beam hidden trajectory stays consistent when beams
            # diverge. On each step we stack the K beams' memories
            # into one tensor, call ``forward_decoder`` once, then
            # scatter the updated memory back per beam.

            # Each beam has: token list, list of step probs, finished flag,
            # cumulative acoustic logprob, and its own encoder memory
            # snapshot (the RNN hidden state at the end of its prefix).
            beams = [
                {
                    "tokens": [sos_token],
                    "probs": [1.0],
                    "finished": False,
                    "logprob": 0.0,
                    "memory": memory_1,
                }
            ]
            completed: list[dict] = []

            # GNMT-style length penalty (cf. Wu et al. 2016,
            # eq. 14). Softer than pure avg-logprob: at alpha=0.6 a
            # 30-token beam is penalised ~2.8x, not 30x, so long-but-
            # correct outputs survive against short-but-truncated ones.
            _LP_ALPHA = 0.6

            def _length_norm_score(b: dict) -> float:
                steps = max(1, len(b["tokens"]) - 1)
                penalty = ((5.0 + steps) / 6.0) ** _LP_ALPHA
                return b["logprob"] / penalty

            def _avg_logprob(b: dict) -> float:
                # For the rescorer interface we still pass per-token
                # average so candidates of different lengths are
                # comparable when combined with the LM score.
                steps = max(1, len(b["tokens"]) - 1)
                return b["logprob"] / steps

            for _ in range(max_seq_length):
                if not beams:
                    break

                # vietocr's seq2seq ``forward_decoder`` slices
                # ``tgt[-1]``, so we only need the last token per beam
                # (shape ``[1, N_alive]``). The transformer path
                # tolerates a single-step input too.
                last_tokens = [b["tokens"][-1] for b in beams]
                tgt_inp = torch.LongTensor([last_tokens]).to(device)
                step_memory = self._stack_memories(
                    [b["memory"] for b in beams]
                )
                output, new_memory = model.transformer.forward_decoder(
                    tgt_inp, step_memory
                )
                # Output shape varies between seq2seq ([B,1,V]) and
                # transformer ([T,B,V] / [B,T,V]). Grab the last
                # timestep along whichever axis has size == N.
                if output.dim() == 3 and output.size(0) == len(beams):
                    last_logits = output[:, -1, :]
                else:
                    last_logits = output[-1, :, :]
                last_probs = softmax(last_logits, dim=-1).to("cpu")

                # Split the updated memory back into per-beam slices.
                per_beam_memory = [
                    self._slice_stacked_memory(new_memory, i)
                    for i in range(len(beams))
                ]

                # Mask ngram-repeats per beam.
                if n > 0:
                    for b_idx, beam in enumerate(beams):
                        banned = _banned_next_tokens(beam["tokens"], n)
                        if banned:
                            last_probs[b_idx, list(banned)] = 0.0

                # Per-beam top-K expansion → candidate set.
                topk_vals, topk_idx = torch.topk(last_probs, K, dim=-1)

                candidates: list[dict] = []
                for b_idx, beam in enumerate(beams):
                    for k in range(K):
                        tok = int(topk_idx[b_idx, k].item())
                        p = float(topk_vals[b_idx, k].item())
                        lp = math.log(max(p, 1e-30))
                        candidates.append({
                            "tokens": beam["tokens"] + [tok],
                            "probs": beam["probs"] + [p],
                            "finished": tok == eos_token,
                            "logprob": beam["logprob"] + lp,
                            "memory": per_beam_memory[b_idx],
                        })

                # Split into completed-this-step (EOS) and alive.
                alive: list[dict] = []
                for c in candidates:
                    if c["finished"]:
                        # Drop the memory on completed beams — we
                        # won't step them again and it wastes RAM.
                        c.pop("memory", None)
                        completed.append(c)
                    else:
                        alive.append(c)

                # Sort ALIVE beams by raw cumulative logprob (standard
                # beam search). Length normalisation is applied only at
                # *final* ranking so we don't prematurely cull longer
                # partials that happen to have a few uncertain tokens.
                alive.sort(key=lambda b: b["logprob"], reverse=True)
                beams = alive[:K]

                # Early stop only when every alive beam is guaranteed
                # to be worse than the best completed beam, even
                # without penalising length. Since each extra token
                # can only decrease ``logprob``, we stop when the
                # single-best alive beam already scores below the
                # best completed beam under length-normalisation.
                if completed and beams:
                    best_done = max(_length_norm_score(b) for b in completed)
                    # Use length-norm on alive too so the comparison
                    # is length-fair. If even the best alive (after
                    # adding a hypothetical full-length tail) can't
                    # beat best_done, stop.
                    best_alive = _length_norm_score(beams[0])
                    if best_done > best_alive:
                        break

            # Finalise: any alive beams at max_seq_length count as
            # (truncated) candidates too.
            for b in beams:
                b.pop("memory", None)
                completed.append(b)

            # Deduplicate by text (keep best logprob per unique text).
            by_text: dict[str, dict] = {}
            for b in completed:
                text = str(vocab.decode(b["tokens"]))
                if text not in by_text or b["logprob"] > by_text[text]["logprob"]:
                    b = dict(b, _text=text)
                    by_text[text] = b

            # Rank final candidates by GNMT length-normalised logprob
            # so the top-K returned is a meaningful shortlist.
            final = sorted(
                by_text.values(), key=_length_norm_score, reverse=True
            )[:K]

            results: list[tuple[str, float, float]] = []
            for beam in final:
                text = beam["_text"]
                valid = [p for p, t in zip(beam["probs"], beam["tokens"]) if t > 3]
                conf = float(sum(valid) / len(valid)) if valid else 0.0
                # Return LENGTH-NORMALISED acoustic logprob so
                # candidates of different lengths are comparable when
                # the rescorer combines them with the LM score.
                results.append((text, conf, _avg_logprob(beam)))

            return results

    @staticmethod
    def _batch_dim(t) -> int:
        """Return the batch dimension of a vietocr encoder tensor.

        Seq2seq hidden state is ``[B, D]`` (batch dim 0), while
        ``encoder_outputs`` and transformer memory are ``[T, B, D]``
        (batch dim 1). We use the rule: dim 1 if the tensor is 3D,
        else dim 0. This matches vietocr's internal shape conventions
        (see vietocr.model.seqmodel.seq2seq.Seq2Seq.forward_decoder
        and vietocr.model.seqmodel.transformer.LanguageTransformer).
        """
        return 1 if t.dim() == 3 else 0

    @classmethod
    def _stack_memories(cls, memories):
        """Stack a list of per-beam memories into a single batched memory.

        Each element in ``memories`` has batch dim = 1 (or is already
        batched, in which case we just concatenate along the batch
        axis). This lets us call ``forward_decoder`` once for all K
        beams and then scatter the output back.
        """
        import torch

        if not memories:
            return memories
        first = memories[0]
        if torch.is_tensor(first):
            axis = cls._batch_dim(first)
            return torch.cat(memories, dim=axis)
        if isinstance(first, (tuple, list)):
            parts = []
            for i in range(len(first)):
                parts.append(cls._stack_memories([m[i] for m in memories]))
            return type(first)(parts)
        return first

    @classmethod
    def _slice_stacked_memory(cls, memory, index: int):
        """Return the single-beam slice at ``index`` of a stacked memory.

        The returned slice keeps the batch dim (size 1), i.e. shapes
        ``[1, D]`` or ``[T, 1, D]``, so that it can be round-tripped
        through ``_stack_memories`` again on the next step.
        """
        import torch

        if torch.is_tensor(memory):
            axis = cls._batch_dim(memory)
            # ``narrow`` keeps the dim with size 1 (vs indexing which
            # drops it) so the shape stays compatible with the
            # decoder's expected input.
            return memory.narrow(axis, index, 1)
        if isinstance(memory, (tuple, list)):
            return type(memory)(
                cls._slice_stacked_memory(m, index) for m in memory
            )
        return memory

    def _predict_candidates(
        self,
        image: Image.Image,
    ) -> list["Candidate"]:
        """Return the recognizer's top-K candidates for one crop.

        When ``kenlm_beam_width > 1``, runs a beam-search decode (with
        the no-repeat-ngram mask if enabled) and returns the full
        candidate list. Otherwise, runs the existing greedy path and
        returns a single candidate.

        The result is ``list[Candidate]`` where ``Candidate.text`` is
        the decoded string and ``Candidate.acoustic_logprob`` is the
        sum of per-step natural-log probabilities (0.0 for the
        greedy path which does not accumulate).
        """
        from ocr_pipeline.recognizer.kenlm_rescorer import Candidate

        if self.kenlm_beam_width > 1:
            triples = self._predict_beam_no_repeat_ngram(
                image, beam_width=self.kenlm_beam_width
            )
            # Narrow beams (K=5) sometimes drop the argmax path at a
            # shared-prefix fork (e.g. "việc " vs "trình ") because the
            # model's next-token distribution diffuses after a space.
            # Always include the greedy prediction as a candidate so
            # rescoring can never do worse than the legacy path.
            #
            # The greedy path's acoustic score is the **per-token mean
            # log-prob** (``mean(log(p_i))``), matching what
            # ``_predict_beam_no_repeat_ngram`` returns for beam
            # candidates. Using ``log(mean(p_i))`` here would be biased
            # upward by Jensen's inequality and would systematically
            # beat any beam alternative regardless of its LM score.
            candidates: list[Candidate] = []
            seen_texts: set[str] = set()
            # ``_predict_no_repeat_ngram`` is the same greedy decoder
            # whether or not the ngram mask is active (``n=0`` skips
            # masking). Call it unconditionally so the per-token
            # ``mean(log(p))`` we get back is computed identically for
            # the rescorer, regardless of the ngram setting.
            greedy_text, _greedy_conf, greedy_acoustic = (
                self._predict_no_repeat_ngram(image)
            )
            greedy_text = str(greedy_text)
            if greedy_text and greedy_text not in seen_texts:
                candidates.append(
                    Candidate(
                        text=greedy_text,
                        acoustic_logprob=greedy_acoustic,
                    )
                )
                seen_texts.add(greedy_text)
            for (t, _conf, lp) in triples:
                if t in seen_texts:
                    continue
                candidates.append(Candidate(text=t, acoustic_logprob=float(lp)))
                seen_texts.add(t)
            return candidates
        if self.no_repeat_ngram_size > 0:
            text, _conf, _avg_lp = self._predict_no_repeat_ngram(image)
            return [Candidate(text=text, acoustic_logprob=0.0)]
        # Legacy vietocr path — text only.
        prediction = self._predictor.predict(image, return_prob=True)
        if isinstance(prediction, tuple):
            text = prediction[0] or ""
        else:
            text = prediction or ""
        return [Candidate(text=str(text), acoustic_logprob=0.0)]

    def _predict_with_optional_probability(
        self,
        image: Image.Image,
    ) -> tuple[str, float]:
        # Beam + KenLM rescoring path. Uses ``_predict_candidates``
        # which always includes the greedy prediction so rescoring
        # cannot regress vs the legacy path.
        if self.kenlm_beam_width > 1:
            from ocr_pipeline.recognizer.kenlm_rescorer import _NullRescorer

            candidates = self._predict_candidates(image)
            if not candidates:
                return "", 0.0
            rescorer = self.kenlm_rescorer or _NullRescorer()
            result = rescorer.rescore_candidates(candidates)
            return result.text, 0.0

        if self.no_repeat_ngram_size > 0:
            text, conf, _avg_lp = self._predict_no_repeat_ngram(image)
            return text, conf
        prediction = self._predictor.predict(image, return_prob=True)
        if isinstance(prediction, tuple):
            text = prediction[0] if len(prediction) > 0 else ""
            probability = prediction[1] if len(prediction) > 1 else None
        else:
            text = prediction
            probability = None

        if text is None:
            text = ""

        if probability is None:
            logger.debug(
                "Predictor returned no probability in %s mode; keeping text and using 0.0 confidence.",
                self._decode_mode,
            )
            return str(text), 0.0

        try:
            return str(text), float(probability)
        except (TypeError, ValueError):
            logger.debug(
                "Predictor returned non-numeric probability %r in %s mode; using 0.0 confidence.",
                probability,
                self._decode_mode,
            )
            return str(text), 0.0

    def recognize_timed(
        self, images: list[Image.Image]
    ) -> tuple[list[str], float]:
        """Recognize a list of images and return (texts, elapsed_ms)."""
        t0 = time.perf_counter()
        texts = self.recognize_batch(images)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return texts, elapsed_ms

    @classmethod
    def from_settings(cls, model_key: Optional[str] = None) -> "VietOCRRecognizer":
        """Create a recognizer from the global settings + model registry."""
        from ocr_pipeline.config import settings
        from ocr_pipeline.recognizer.kenlm_rescorer import KenLMRescorer

        model_cfg = settings.get_model_config(model_key)
        beam_width = int(settings.rec_kenlm_beam_width or 1)
        rescorer = None
        if beam_width > 1:
            # We attach the rescorer unconditionally when beam>1; the
            # rescorer itself decides at load time whether to operate
            # in no-op mode (missing .bin, no kenlm, etc.).
            rescorer = KenLMRescorer.from_settings()
        return cls(
            weights_path=model_cfg["weights_path"],
            architecture=model_cfg["architecture"],
            image_height=model_cfg["image_height"],
            image_max_width=model_cfg["image_max_width"],
            image_min_width=model_cfg["image_min_width"],
            device=settings.rec_device,
            no_repeat_ngram_size=settings.rec_no_repeat_ngram_size,
            kenlm_rescorer=rescorer,
            kenlm_beam_width=beam_width,
        )
