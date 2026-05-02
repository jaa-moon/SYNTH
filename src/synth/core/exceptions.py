"""Synth custom exceptions."""

from __future__ import annotations

from pathlib import Path


class SynthError(Exception):
    """Base exception for all Synth errors."""


class NoTextFoundError(SynthError):
    """Raised when no readable text can be extracted from a file."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        super().__init__(
            f"No readable text found in: {self.file_path.name}"
        )


class ImageLoadError(SynthError):
    """Raised when an image cannot be loaded or decoded."""

    def __init__(self, image_path: str | Path, reason: str = "unknown") -> None:
        self.image_path = Path(image_path)
        super().__init__(
            f"Failed to load image '{self.image_path.name}': {reason}"
        )
