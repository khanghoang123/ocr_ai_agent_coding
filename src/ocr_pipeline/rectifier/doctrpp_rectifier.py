"""DocTr++ wrapper.

DocTr++ ("Deep Unrestricted Document Image Rectification", TMM 2023, Feng et al.)
is a Transformer-based dewarping network that handles unrestricted document
deformations — including curved, perspective-distorted, hand-held photos.

This module is a thin inference wrapper that:
  - Loads a pre-converted DocTr++ checkpoint when available at the configured
    path (``models/rectifier/doctrpp.pth`` by default), OR a TorchScript /
    ONNX export of the same.
  - If the checkpoint or required dependencies are missing, the rectifier
    *gracefully reports `applied=False`* with a clear diagnostic so the
    HybridRectifier can fall back to OpenCV. This is intentional: we never
    want a missing weight file to crash the whole pipeline.

Why we don't vendor the full architecture in-tree:
  - The official DocTr++ repo is GPL-style licensed (CeCILL); vendoring would
    contaminate the project. We only call into it if the user has it
    installed locally OR provides a TorchScript export.
  - For "no heavy retraining" use cases, a TorchScript / ONNX export is the
    practical deployment path: a single self-contained file, no source
    dependency.

Expected weight formats
-----------------------
1. **TorchScript (.pt / .torchscript)** — produced by `torch.jit.trace`/
   `torch.jit.script` from the official DocTr++ checkpoint. Auto-detected.
2. **ONNX (.onnx)** — runs via `onnxruntime` if available.
3. **Raw .pth state_dict** — only loadable if the official `doctr_plus`
   architecture module is importable from the local environment (e.g. user
   has `pip install -e` on a clone of fh2019ustc/DocTr-Plus).

If none of these are available, this rectifier returns ``applied=False`` and
lets the caller (HybridRectifier) fall back to OpenCV.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ocr_pipeline.rectifier.base import (
    Rectifier,
    RectifierResult,
    from_numpy_rgb,
    to_numpy_rgb,
)

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS_CANDIDATES = (
    "models/rectifier/doctrpp.torchscript",
    "models/rectifier/doctrpp.pt",
    "models/rectifier/doctrpp.onnx",
    "models/rectifier/doctrpp.pth",
)


class DocTrPlusRectifier(Rectifier):
    """Pretrained DocTr++ inference wrapper with graceful degradation."""

    name = "doctrpp"

    def __init__(
        self,
        weights_path: str | None = None,
        device: str = "cpu",
        input_size: int = 288,
        **_: object,
    ):
        self.weights_path = self._resolve_weights_path(weights_path)
        self.device = device
        self.input_size = int(input_size)
        self._backend_kind: str | None = None
        self._model: Any = None
        self._load_error: str | None = None
        self._tried_load = False

    # ── Public API ────────────────────────────────────────────────────────

    def rectify(self, image: Image.Image) -> RectifierResult:
        if not self._tried_load:
            self._try_load()

        if self._model is None:
            return RectifierResult(
                image=image,
                backend_used=self.name,
                applied=False,
                confidence=0.0,
                diagnostics={
                    "reason": "doctrpp_unavailable",
                    "load_error": self._load_error,
                    "expected_weights_at": self.weights_path,
                },
            )

        try:
            warped = self._infer(image)
        except Exception as exc:  # pragma: no cover - runtime defensive
            logger.warning("DocTr++ inference failed: %s", exc)
            return RectifierResult(
                image=image,
                backend_used=self.name,
                applied=False,
                confidence=0.0,
                diagnostics={"reason": "doctrpp_inference_error", "error": str(exc)},
            )

        return RectifierResult(
            image=warped,
            backend_used=self.name,
            applied=True,
            confidence=0.9,  # DocTr++ doesn't return its own confidence
            diagnostics={
                "backend_kind": self._backend_kind,
                "input_size": list(image.size),
                "output_size": list(warped.size),
            },
        )

    # ── Loading ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_weights_path(explicit: str | None) -> str | None:
        if explicit and Path(explicit).exists():
            return str(Path(explicit).resolve())
        for candidate in DEFAULT_WEIGHTS_CANDIDATES:
            p = Path.cwd() / candidate
            if p.exists():
                return str(p.resolve())
        return None

    def _try_load(self) -> None:
        self._tried_load = True
        if self.weights_path is None:
            self._load_error = (
                f"DocTr++ weights not found. Looked for {DEFAULT_WEIGHTS_CANDIDATES}. "
                "Will fall back to OpenCV rectifier."
            )
            return

        ext = Path(self.weights_path).suffix.lower()
        try:
            if ext in {".torchscript", ".pt"}:
                self._load_torchscript()
            elif ext == ".onnx":
                self._load_onnx()
            elif ext == ".pth":
                self._load_state_dict()
            else:
                self._load_error = f"Unrecognized weights extension: {ext}"
        except Exception as exc:
            self._load_error = f"DocTr++ load failed: {exc}"
            logger.warning("%s", self._load_error)
            self._model = None

    def _load_torchscript(self) -> None:
        import torch

        self._model = torch.jit.load(self.weights_path, map_location=self.device)
        self._model.eval()
        self._backend_kind = "torchscript"

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for ONNX backend") from exc
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        self._model = ort.InferenceSession(self.weights_path, providers=providers)
        self._backend_kind = "onnx"

    def _load_state_dict(self) -> None:
        try:
            from doctr_plus.docres import DocTrPlus  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Loading raw .pth state_dicts requires the original DocTr-Plus "
                "package (`pip install git+https://github.com/fh2019ustc/DocTr-Plus`). "
                "Use a TorchScript or ONNX export instead for self-contained deployment."
            ) from exc
        import torch

        model = DocTrPlus()
        state = torch.load(self.weights_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.eval()
        self._model = model.to(self.device)
        self._backend_kind = "pth"

    # ── Inference ─────────────────────────────────────────────────────────

    def _infer(self, image: Image.Image) -> Image.Image:
        rgb = to_numpy_rgb(image)
        h, w = rgb.shape[:2]
        # DocTr++ expects square input around 288×288. Pad to square then resize.
        size = max(h, w)
        pad_h = size - h
        pad_w = size - w
        padded = np.pad(
            rgb,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        resized = self._resize_pil(padded, self.input_size)
        net_input = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]

        if self._backend_kind == "onnx":
            outputs = self._model.run(None, {self._model.get_inputs()[0].name: net_input})
            warped_small = outputs[0][0]
        else:
            import torch

            with torch.no_grad():
                inp = torch.from_numpy(net_input).to(self.device)
                out = self._model(inp)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                warped_small = out.detach().cpu().numpy()[0]

        # Output: either an image (3,H,W) or a flow field (2,H,W). Handle both.
        if warped_small.shape[0] == 2:
            # Flow field — apply to original padded image.
            warped = self._apply_flow(padded, warped_small)
        else:
            warped = warped_small.transpose(1, 2, 0)
            warped = (warped.clip(0, 1) * 255.0).astype(np.uint8)
            warped = self._resize_pil(warped, max(h, w))

        # Crop the rectified output back to the original aspect ratio.
        warped = warped[: int(warped.shape[0] * (h / size)), : int(warped.shape[1] * (w / size))]
        return from_numpy_rgb(warped)

    @staticmethod
    def _resize_pil(arr: np.ndarray, side: int) -> np.ndarray:
        import cv2

        return cv2.resize(arr, (side, side), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _apply_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
        import cv2

        h, w = image.shape[:2]
        flow_resized = np.stack(
            [
                cv2.resize(flow[0], (w, h), interpolation=cv2.INTER_LINEAR),
                cv2.resize(flow[1], (w, h), interpolation=cv2.INTER_LINEAR),
            ]
        )
        # Flow is in normalized [-1, 1] coords. Convert to absolute pixel maps.
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = ((flow_resized[0] + 1.0) * 0.5 * (w - 1)).astype(np.float32)
        map_y = ((flow_resized[1] + 1.0) * 0.5 * (h - 1)).astype(np.float32)
        # Fall back to identity if the flow is degenerate.
        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            return image
        warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return warped
