"""Core business logic for Synth."""

from synth.core.auth import (
    APIEndpointConfig,
    AuthResult,
    BaseAuthenticator,
    DetectorFactory,
    LocalHFAuthenticator,
    UniversalAPIAuthenticator,
)
from synth.core.device import detect_device, get_torch_device
from synth.core.exceptions import ImageLoadError, NoTextFoundError, SynthError
from synth.core.ocr import DocumentScanner, PreprocessConfig

__all__ = [
    # Auth
    "APIEndpointConfig",
    "AuthResult",
    "BaseAuthenticator",
    "DetectorFactory",
    "LocalHFAuthenticator",
    "UniversalAPIAuthenticator",
    # Device
    "detect_device",
    "get_torch_device",
    # Exceptions
    "ImageLoadError",
    "NoTextFoundError",
    "SynthError",
    # OCR
    "DocumentScanner",
    "PreprocessConfig",
]
