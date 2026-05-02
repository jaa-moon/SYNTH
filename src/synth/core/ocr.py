"""OCR pipeline — image preprocessing and text extraction.

This module provides :class:`DocumentScanner`, which chains an OpenCV
preprocessing pipeline with EasyOCR to extract clean text from document
images.  Hardware selection is automatic: CUDA → MPS → CPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from synth.core.device import detect_device
from synth.core.exceptions import ImageLoadError, NoTextFoundError

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────
ImageArray = np.ndarray  # uint8 HxW or HxWxC


# ── Preprocessing config ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreprocessConfig:
    """Tunable knobs for the OpenCV preprocessing pipeline.

    Attributes:
        adaptive_block_size: Neighbourhood size for adaptive thresholding
            (must be odd, ≥ 3).
        adaptive_c: Constant subtracted from the weighted mean.
        denoise_strength: Filter strength *h* for ``fastNlMeansDenoising``.
            Higher = more aggressive.
    """

    adaptive_block_size: int = 11
    adaptive_c: int = 2
    denoise_strength: int = 10


# ── Main class ────────────────────────────────────────────────────────────────

@dataclass
class DocumentScanner:
    """End-to-end document OCR: preprocess → extract → return text.

    Parameters:
        languages: BCP-47 language codes for EasyOCR (default: ``["en"]``).
        config: Preprocessing tunables.  Uses sensible defaults when omitted.

    Example::

        scanner = DocumentScanner()
        print(scanner.device)          # "mps" on Apple Silicon
        text = scanner.extract_text("receipt.jpg")
    """

    languages: list[str] = field(default_factory=lambda: ["en"])
    config: PreprocessConfig = field(default_factory=PreprocessConfig)

    # ── Resolved at post-init ─────────────────────────────────────────────
    device: str = field(init=False)
    _reader: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.device = detect_device()
        self._reader = self._init_reader()
        logger.info(
            "DocumentScanner ready · device=%s · languages=%s",
            self.device,
            self.languages,
        )

    # ── Reader bootstrap ──────────────────────────────────────────────────

    def _init_reader(self) -> object:
        """Lazily import and initialise the EasyOCR ``Reader``.

        EasyOCR natively supports CUDA but **not** MPS.  When MPS is the
        detected device the reader falls back to CPU for OCR while the
        device hint is preserved for downstream consumers (e.g. the
        authenticator's transformer models, which *do* support MPS).
        """
        import easyocr  # heavy import — deferred on purpose

        use_gpu = self.device == "cuda"

        if self.device == "mps":
            logger.info(
                "EasyOCR does not support MPS — OCR will run on CPU; "
                "MPS device hint preserved for transformer models"
            )

        reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            verbose=False,
        )
        return reader

    # ── Preprocessing pipeline ────────────────────────────────────────────

    def preprocess(self, image_path: str | Path) -> ImageArray:
        """Load an image and run the full OpenCV cleanup pipeline.

        Pipeline stages:
            1. **Load** — read from disk via ``cv2.imread``.
            2. **Grayscale** — convert BGR → single-channel intensity.
            3. **Adaptive threshold** — Gaussian-weighted local threshold
               that removes shadows and compensates for uneven lighting.
            4. **Denoise** — non-local means denoising to suppress scanner
               noise and JPEG artifacts.

        Args:
            image_path: Path to the source image file.

        Returns:
            Cleaned ``uint8`` grayscale numpy array ready for OCR.

        Raises:
            FileNotFoundError: If *image_path* does not exist.
            ImageLoadError: If OpenCV cannot decode the file.
        """
        path = Path(image_path).resolve()

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # 1 · Load
        img = cv2.imread(str(path))
        if img is None:
            raise ImageLoadError(path, reason="OpenCV could not decode the file")

        # 2 · Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3 · Adaptive threshold (shadow removal)
        thresh = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=self.config.adaptive_block_size,
            C=self.config.adaptive_c,
        )

        # 4 · Denoise
        denoised: ImageArray = cv2.fastNlMeansDenoising(
            thresh,
            h=self.config.denoise_strength,
        )

        logger.debug(
            "Preprocessed %s → shape=%s dtype=%s",
            path.name,
            denoised.shape,
            denoised.dtype,
        )
        return denoised

    # ── Text extraction ───────────────────────────────────────────────────

    def extract_text(self, image_path: str | Path) -> str:
        """Run the full pipeline: preprocess → OCR → joined text.

        Args:
            image_path: Path to the source image.

        Returns:
            Extracted text with one OCR line per ``\\n``-delimited line.

        Raises:
            NoTextFoundError: If the image contains no readable characters.
            FileNotFoundError: If *image_path* does not exist.
            ImageLoadError: If the image cannot be decoded.
        """
        processed = self.preprocess(image_path)

        # EasyOCR returns List[Tuple[bbox, text, confidence]]
        results = self._reader.readtext(processed)  # type: ignore[union-attr]

        lines: list[str] = [
            text.strip()
            for _bbox, text, _conf in results
            if text.strip()
        ]

        if not lines:
            raise NoTextFoundError(image_path)

        extracted = "\n".join(lines)
        logger.info(
            "Extracted %d line(s) (%d chars) from %s",
            len(lines),
            len(extracted),
            Path(image_path).name,
        )
        return extracted

    # ── Raw results (advanced) ────────────────────────────────────────────

    def extract_raw(
        self, image_path: str | Path
    ) -> list[tuple[list[list[int]], str, float]]:
        """Return raw EasyOCR results with bounding boxes and confidence.

        Each element is ``(bbox, text, confidence)`` where *bbox* is a list
        of four ``[x, y]`` corner points.

        Raises:
            NoTextFoundError: If the image contains no readable characters.
        """
        processed = self.preprocess(image_path)
        results = self._reader.readtext(processed)  # type: ignore[union-attr]

        if not results:
            raise NoTextFoundError(image_path)

        return results  # type: ignore[return-value]
