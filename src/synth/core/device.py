"""Shared hardware detection utility.

Centralised device resolution used by both the OCR pipeline
(EasyOCR) and the authenticator pipeline (HuggingFace / PyTorch).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Return the best available compute device string.

    Priority order:
        1. **cuda** — NVIDIA GPU via CUDA
        2. **mps**  — Apple Silicon GPU via Metal Performance Shaders
        3. **cpu**  — Fallback

    Returns:
        One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if torch.cuda.is_available():
        device = "cuda"
        detail = torch.cuda.get_device_name(0)
        logger.info("Hardware probe: CUDA available → %s", detail)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.info("Hardware probe: Apple MPS available")
    else:
        device = "cpu"
        logger.info("Hardware probe: No GPU detected → CPU fallback")
    return device


def get_torch_device() -> torch.device:
    """Return a ``torch.device`` for the best available hardware.

    Convenience wrapper around :func:`detect_device` for modules
    that need an actual ``torch.device`` object.
    """
    return torch.device(detect_device())
