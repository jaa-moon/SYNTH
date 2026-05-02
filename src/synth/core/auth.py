"""AI content authentication — Strategy Pattern implementation.

This module provides a pluggable detection system where **any** model
(local HuggingFace checkpoint, OpenAI, Anthropic, or custom HTTP API)
can be slotted in behind a common :class:`BaseAuthenticator` interface.

Use :class:`DetectorFactory` to instantiate the correct strategy by name.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

from synth.core.device import detect_device, get_torch_device
from synth.core.exceptions import SynthError

logger = logging.getLogger(__name__)

# Load .env once at import time
load_dotenv()


# ── Verdict constants ─────────────────────────────────────────────────────────

VERDICT_HUMAN = "human"
VERDICT_AI = "ai"
VERDICT_MIXED = "mixed"


# ── Result type ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthResult:
    """Structured result from an AI detection run.

    Attributes:
        score: Probability that the text is AI-generated
            (``0.0`` = definitely human, ``1.0`` = definitely AI).
        verdict: One of ``"human"``, ``"ai"``, or ``"mixed"``.
        reasoning: Human-readable explanation of the verdict.
        model: Identifier of the model that produced this result.
    """

    score: float
    verdict: str
    reasoning: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "score": self.score,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "model": self.model,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy interface
# ══════════════════════════════════════════════════════════════════════════════


class BaseAuthenticator(ABC):
    """Abstract strategy interface for AI content detection.

    All authenticators **must** implement :meth:`detect`, returning a
    standardised :class:`AuthResult` regardless of the underlying model.
    """

    @abstractmethod
    def detect(self, text: str) -> AuthResult:
        """Analyse *text* and return an AI-authorship verdict.

        Args:
            text: The content to analyse.

        Returns:
            :class:`AuthResult` with score, verdict, reasoning, and model.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this strategy."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 1 — Local HuggingFace model
# ══════════════════════════════════════════════════════════════════════════════


class LocalHFAuthenticator(BaseAuthenticator):
    """Detect AI-generated text using a local HuggingFace transformer.

    Default model: ``roberta-base-openai-detector`` (OpenAI's GPT-2 output
    detector fine-tuned on RoBERTa).

    **Device selection** is fully automatic::

        cuda  →  NVIDIA GPU (fastest)
        mps   →  Apple Silicon Metal  (M-series Macs)
        cpu   →  Fallback

    Example::

        auth = LocalHFAuthenticator()
        result = auth.detect("Some text to check...")
        print(result.verdict)  # "ai" | "human" | "mixed"
    """

    DEFAULT_MODEL = "roberta-base-openai-detector"

    # ── Verdict thresholds ────────────────────────────────────────────────
    AI_THRESHOLD: float = 0.75
    HUMAN_THRESHOLD: float = 0.25

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._device_str = detect_device()
        self._device = get_torch_device()
        self._pipeline = self._load_pipeline()
        logger.info(
            "LocalHFAuthenticator ready · model=%s · device=%s",
            self._model_name,
            self._device,
        )

    # ── Device & model bootstrap ──────────────────────────────────────────

    def _load_pipeline(self) -> Any:
        """Load the HuggingFace ``text-classification`` pipeline.

        The pipeline is placed on the best available device automatically.
        ``top_k=None`` ensures all label scores are returned.
        """
        from transformers import pipeline as hf_pipeline  # heavy — deferred

        return hf_pipeline(
            "text-classification",
            model=self._model_name,
            device=self._device,
            top_k=None,  # return scores for every label
        )

    # ── Core detection ────────────────────────────────────────────────────

    def detect(self, text: str) -> AuthResult:
        """Run the local model on *text* and return a verdict.

        The input is truncated to 512 tokens to respect the model's
        context window.

        Raises:
            SynthError: If *text* is empty or whitespace-only.
        """
        if not text.strip():
            raise SynthError("Cannot authenticate empty text")

        # Most classification models cap at 512 tokens
        results = self._pipeline(text[:512])

        # results is List[List[Dict]] when top_k=None
        label_scores: dict[str, float] = {
            r["label"]: r["score"] for r in results[0]
        }

        ai_score = self._extract_ai_score(label_scores)
        verdict = self._score_to_verdict(ai_score)

        return AuthResult(
            score=round(ai_score, 4),
            verdict=verdict,
            reasoning=self._build_reasoning(ai_score, verdict, label_scores),
            model=self._model_name,
        )

    # ── Label parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_ai_score(label_scores: dict[str, float]) -> float:
        """Extract the AI probability from heterogeneous label formats.

        Supports common naming conventions:
            - ``LABEL_1`` / ``LABEL_0`` (roberta-base-openai-detector)
            - ``Fake`` / ``Real``
            - ``ai`` / ``human``
        """
        # Try known AI-positive labels first
        for ai_label in ("LABEL_1", "Fake", "fake", "ai", "AI", "machine"):
            if ai_label in label_scores:
                return label_scores[ai_label]

        # Binary fallback: assume second label is AI
        labels = list(label_scores.keys())
        if len(labels) == 2:
            return label_scores[labels[1]]

        return max(label_scores.values())

    @staticmethod
    def _score_to_verdict(score: float) -> str:
        """Map a continuous score to a discrete verdict."""
        if score >= LocalHFAuthenticator.AI_THRESHOLD:
            return VERDICT_AI
        if score <= LocalHFAuthenticator.HUMAN_THRESHOLD:
            return VERDICT_HUMAN
        return VERDICT_MIXED

    @staticmethod
    def _build_reasoning(
        score: float,
        verdict: str,
        raw: dict[str, float],
    ) -> str:
        pct = score * 100
        raw_str = ", ".join(f"{k}={v:.3f}" for k, v in raw.items())
        return (
            f"AI probability: {pct:.1f}%. "
            f"Verdict: {verdict}. "
            f"Raw model scores: [{raw_str}]"
        )

    @property
    def name(self) -> str:
        return f"local:{self._model_name}"


# ══════════════════════════════════════════════════════════════════════════════
#  Strategy 2 — Universal remote API
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class APIEndpointConfig:
    """Configuration for a remote detection API.

    Can be hydrated from:

    * **Environment variables / ``.env``** — via :meth:`from_env`
    * **JSON config file** — via :meth:`from_json`

    ``payload_template`` uses ``{text}`` as a placeholder that gets
    substituted with the actual input text at request time.
    """

    base_url: str
    api_key: str
    model: str = ""

    # Auth
    auth_header: str = "Authorization"
    auth_prefix: str = "Bearer"

    # Request shape — {text} is replaced with input
    payload_template: dict[str, Any] = field(default_factory=dict)

    # Response parsing — dot-notation paths into the JSON response
    #   e.g. "choices.0.message.content" for OpenAI
    score_path: str = "score"
    label_path: str = "label"
    reasoning_path: str = "reasoning"

    # Network
    timeout_seconds: float = 30.0

    # ── Constructors ──────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        *,
        cli_api_key: str | None = None,
        cli_base_url: str | None = None,
        cli_model: str | None = None,
    ) -> APIEndpointConfig:
        """Build config from environment / ``.env`` file.

        CLI arguments take priority over environment variables.

        Args:
            cli_api_key:  API key passed via ``--api-key``.  Overrides env.
            cli_base_url: Endpoint URL passed via ``--api-url``.  Overrides env.
            cli_model:    Model identifier passed via ``--agent``.  Overrides env.

        Required (from CLI **or** env):
            ``SYNTH_API_BASE_URL`` — Full endpoint URL.
            ``SYNTH_API_KEY``      — Authentication key.

        Optional:
            ``SYNTH_API_MODEL``    — Model identifier.
            ``SYNTH_PAYLOAD_MAP``  — Path to a JSON payload mapping file.
        """
        load_dotenv()

        # CLI arguments override environment variables
        base_url = cli_base_url or os.getenv("SYNTH_API_BASE_URL", "")
        api_key = cli_api_key or os.getenv("SYNTH_API_KEY", "")
        model = cli_model or os.getenv("SYNTH_API_MODEL", "")

        if not base_url:
            raise SynthError(
                "API base URL is not set. "
                "Pass --api-url on the command line, or set "
                "SYNTH_API_BASE_URL in your .env file, or run "
                "'synth configure' to set it up interactively."
            )
        if not api_key:
            raise SynthError(
                "API key is not set. "
                "Pass --api-key on the command line, or set "
                "SYNTH_API_KEY in your .env file, or run "
                "'synth configure' to set it up interactively."
            )

        # Optional payload mapping file
        payload_map_path = os.getenv("SYNTH_PAYLOAD_MAP", "")
        payload_template: dict[str, Any] = {}
        overrides: dict[str, Any] = {}

        if payload_map_path:
            mapping = cls._load_json(Path(payload_map_path))
            payload_template = mapping.pop("payload_template", {})
            # Pull any override keys that match our fields
            for key in (
                "score_path",
                "label_path",
                "reasoning_path",
                "auth_header",
                "auth_prefix",
                "timeout_seconds",
            ):
                if key in mapping:
                    overrides[key] = mapping[key]

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            payload_template=payload_template,
            **overrides,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> APIEndpointConfig:
        """Load a complete config from a JSON file.

        The JSON must contain at least ``base_url`` and ``api_key``.
        All other fields are optional and will use defaults.
        """
        data = cls._load_json(Path(path))
        return cls(**data)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise SynthError(f"Config file not found: {path}")
        return json.loads(path.read_text())  # type: ignore[no-any-return]


class UniversalAPIAuthenticator(BaseAuthenticator):
    """Generic API-based AI content detector.

    Connects to **any** HTTP endpoint — OpenAI, Anthropic, or a custom
    detection service — through a configurable payload template and
    dot-notation response path mapping.

    Lifecycle::

        # From .env
        auth = UniversalAPIAuthenticator()

        # From explicit config
        cfg = APIEndpointConfig(base_url="https://...", api_key="sk-...")
        auth = UniversalAPIAuthenticator(config=cfg)

        # From JSON file
        cfg = APIEndpointConfig.from_json("config/openai.json")
        auth = UniversalAPIAuthenticator(config=cfg)

        result = auth.detect("Text to check...")

    Supports context-manager protocol for clean shutdown::

        with UniversalAPIAuthenticator() as auth:
            result = auth.detect(text)
    """

    def __init__(
        self,
        config: APIEndpointConfig | None = None,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        api_model: str | None = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = APIEndpointConfig.from_env(
                cli_api_key=api_key,
                cli_base_url=api_url,
                cli_model=api_model,
            )
        self._client = httpx.Client(
            base_url=self._config.base_url,
            headers=self._build_headers(),
            timeout=self._config.timeout_seconds,
        )
        logger.info(
            "UniversalAPIAuthenticator ready · endpoint=%s · model=%s",
            self._config.base_url,
            self._config.model or "(default)",
        )

    # ── Request construction ──────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        cfg = self._config
        return {
            cfg.auth_header: f"{cfg.auth_prefix} {cfg.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, text: str) -> dict[str, Any]:
        """Render the payload, substituting ``{text}`` placeholders.

        If no ``payload_template`` is configured, falls back to an
        OpenAI-compatible chat-completion request that asks the model
        to act as an AI content detector.
        """
        template = self._config.payload_template

        if not template:
            # Sensible default — OpenAI / Anthropic chat format
            return {
                "model": self._config.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI content detector. Analyse the "
                            "following text and respond with valid JSON "
                            "containing exactly three keys: "
                            '"score" (float 0.0=human to 1.0=AI), '
                            '"verdict" ("human", "ai", or "mixed"), '
                            'and "reasoning" (one sentence explanation).'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                "response_format": {"type": "json_object"},
            }

        return self._deep_substitute(template, text)

    def _deep_substitute(self, obj: Any, text: str) -> Any:
        """Recursively replace ``{text}`` placeholders in nested structures."""
        if isinstance(obj, str):
            return obj.replace("{text}", text)
        if isinstance(obj, dict):
            return {k: self._deep_substitute(v, text) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_substitute(item, text) for item in obj]
        return obj

    # ── Response parsing ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_path(data: Any, path: str) -> Any:
        """Walk a dot-notation path through nested dicts/lists.

        Example::

            _resolve_path(data, "choices.0.message.content")
            # → data["choices"][0]["message"]["content"]
        """
        current = data
        for key in path.split("."):
            try:
                if isinstance(current, list):
                    current = current[int(key)]
                elif isinstance(current, dict):
                    current = current[key]
                else:
                    raise SynthError(
                        f"Cannot traverse '{path}': "
                        f"unexpected type {type(current).__name__} at '{key}'"
                    )
            except (KeyError, IndexError, ValueError) as exc:
                raise SynthError(
                    f"Response path '{path}' failed at '{key}': {exc}"
                ) from exc
        return current

    def _parse_response(self, data: dict[str, Any]) -> AuthResult:
        """Extract score / verdict / reasoning from the API response.

        Handles two response shapes:
            1. **Structured API** — score, label, reasoning at direct paths.
            2. **LLM chat API** — a JSON string embedded in the message
               content (e.g. OpenAI ``choices.0.message.content``).
        """
        cfg = self._config

        raw_content = self._resolve_path(data, cfg.score_path)

        # Case 1: LLM response — content is a JSON string
        if isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
                return AuthResult(
                    score=round(float(parsed.get("score", 0.5)), 4),
                    verdict=str(parsed.get("verdict", VERDICT_MIXED)),
                    reasoning=str(
                        parsed.get("reasoning", "No reasoning provided")
                    ),
                    model=cfg.model or cfg.base_url,
                )
            except (json.JSONDecodeError, AttributeError):
                # Not JSON — treat the whole string as reasoning
                return AuthResult(
                    score=0.5,
                    verdict=VERDICT_MIXED,
                    reasoning=raw_content[:500],
                    model=cfg.model or cfg.base_url,
                )

        # Case 2: Structured response — score is numeric
        score = float(raw_content)
        verdict = str(self._resolve_path(data, cfg.label_path))
        reasoning = str(self._resolve_path(data, cfg.reasoning_path))

        return AuthResult(
            score=round(score, 4),
            verdict=verdict,
            reasoning=reasoning,
            model=cfg.model or cfg.base_url,
        )

    # ── Core detection ────────────────────────────────────────────────────

    def detect(self, text: str) -> AuthResult:
        """Send *text* to the remote API and return a verdict.

        Raises:
            SynthError: On empty input, HTTP errors, or unparseable responses.
        """
        if not text.strip():
            raise SynthError("Cannot authenticate empty text")

        payload = self._build_payload(text)

        try:
            response = self._client.post("", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise SynthError(
                f"API request failed ({exc.response.status_code}): "
                f"{exc.response.text[:500]}"
            ) from exc
        except httpx.RequestError as exc:
            raise SynthError(f"API connection error: {exc}") from exc

        return self._parse_response(response.json())

    # ── Lifecycle ─────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"api:{self._config.model or self._config.base_url}"

    def close(self) -> None:
        """Shut down the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> UniversalAPIAuthenticator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════


class DetectorFactory:
    """Create authenticator instances by strategy name.

    Built-in strategies:
        - ``"local"`` → :class:`LocalHFAuthenticator`
        - ``"api"``   → :class:`UniversalAPIAuthenticator`

    Custom strategies can be registered at runtime::

        DetectorFactory.register("my_custom", MyCustomAuth)
        detector = DetectorFactory.create("my_custom", **opts)
    """

    _REGISTRY: dict[str, type[BaseAuthenticator]] = {
        "local": LocalHFAuthenticator,
        "api": UniversalAPIAuthenticator,
    }

    @classmethod
    def create(cls, strategy: str, **kwargs: Any) -> BaseAuthenticator:
        """Instantiate an authenticator by strategy name.

        Args:
            strategy: One of the registered strategy names.
            **kwargs: Forwarded to the authenticator constructor.

        Raises:
            SynthError: If *strategy* is not recognised.
        """
        key = strategy.lower().strip()

        if key not in cls._REGISTRY:
            available = ", ".join(sorted(cls._REGISTRY))
            raise SynthError(
                f"Unknown strategy '{strategy}'. Available: {available}"
            )

        logger.info("DetectorFactory: creating '%s' authenticator", key)
        return cls._REGISTRY[key](**kwargs)

    @classmethod
    def register(
        cls, name: str, authenticator_cls: type[BaseAuthenticator]
    ) -> None:
        """Register a custom authenticator at runtime.

        This allows third-party plugins to extend Synth without
        modifying the source::

            from synth.core.auth import DetectorFactory, BaseAuthenticator

            class MyDetector(BaseAuthenticator):
                ...

            DetectorFactory.register("my_detector", MyDetector)
        """
        cls._REGISTRY[name.lower().strip()] = authenticator_cls
        logger.info("Registered custom strategy: '%s'", name)

    @classmethod
    def available(cls) -> list[str]:
        """Return a sorted list of all registered strategy names."""
        return sorted(cls._REGISTRY)
