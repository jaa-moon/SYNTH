"""Synth CLI — AI Content Authenticator & OCR Engine.

Entry point for the ``synth`` command. All subcommands are defined here
and delegate to :mod:`synth.core` for business logic.
"""

from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.status import Status

from synth import __version__
from synth.cli.display import (
    build_results_table,
    build_summary_panel,
    build_text_panel,
    console,
    print_banner,
)
from synth.core.auth import AuthResult, DetectorFactory
from synth.core.exceptions import NoTextFoundError, SynthError
from synth.core.extractors import ALL_SUPPORTED_EXTENSIONS, get_extractor

logger = logging.getLogger(__name__)

# ── Supported file extensions (all formats) ───────────────────────────────────

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}

# ── CLI enum for --engine ─────────────────────────────────────────────────────


class EngineChoice(str, Enum):
    """Detection engine strategy."""

    local = "local"
    api = "api"


# ── App setup ─────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="synth",
    help="[bold cyan]Synth[/bold cyan] — AI Content Authenticator & OCR Engine.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ── Version callback ─────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    """Print the version and exit."""
    if value:
        console.print(
            f"[bold cyan]synth[/bold cyan] [dim]v{__version__}[/dim]"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(  # noqa: UP007
        None,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable debug logging.",
    ),
) -> None:
    """Synth — AI Content Authenticator & OCR Engine."""
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  synth verify
# ══════════════════════════════════════════════════════════════════════════════


def _collect_files(target: Path) -> list[Path]:
    """Return a sorted list of supported files from *target*.

    If *target* is a single file, returns a one-element list.
    If *target* is a directory, recursively globs for supported files.
    """
    if target.is_file():
        if target.suffix.lower() not in ALL_SUPPORTED_EXTENSIONS:
            console.print(
                f"[bold red]✗[/bold red] Unsupported file type: "
                f"[yellow]{target.suffix}[/yellow]. "
                f"Supported: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
            )
            raise typer.Exit(code=1)
        return [target]

    if target.is_dir():
        files = sorted(
            f
            for f in target.rglob("*")
            if f.is_file() and f.suffix.lower() in ALL_SUPPORTED_EXTENSIONS
        )
        if not files:
            console.print(
                f"[bold red]✗[/bold red] No supported files found in "
                f"[yellow]{target}[/yellow]"
            )
            raise typer.Exit(code=1)
        return files

    console.print(
        f"[bold red]✗[/bold red] Path not found: [yellow]{target}[/yellow]"
    )
    raise typer.Exit(code=1)


def _extract_text_from_file(
    file_path: Path,
    lang_list: list[str],
) -> str:
    """Extract text from any supported file using the appropriate extractor."""
    extractor = get_extractor(file_path, ocr_languages=lang_list)
    return extractor.extract(file_path)


def _process_single_file(
    file_path: Path,
    detector: object,
    show_text: bool,
    lang_list: list[str],
) -> tuple[str, str, AuthResult] | None:
    """Extract text + detect on a single file. Returns (filename, text, result) or None."""
    filename = file_path.name

    try:
        text = _extract_text_from_file(file_path, lang_list)
    except NoTextFoundError:
        console.print(
            f"  [dim yellow]⚠ {filename}:[/dim yellow] No readable text found — skipped"
        )
        return None
    except Exception as exc:
        console.print(
            f"  [dim red]✗ {filename}:[/dim red] Extraction failed — {exc}"
        )
        return None

    try:
        result = detector.detect(text)  # type: ignore[union-attr]
    except SynthError as exc:
        console.print(
            f"  [dim red]✗ {filename}:[/dim red] Detection failed — {exc}"
        )
        return None

    return filename, text, result


@app.command()
def verify(
    path: Path = typer.Argument(
        ...,
        help="Path to a file or directory. Supports images, PDF, DOCX, TXT, and more.",
        exists=True,
        readable=True,
        resolve_path=True,
    ),
    engine: EngineChoice = typer.Option(
        EngineChoice.local,
        "--engine",
        "-e",
        help="Detection strategy: [cyan]local[/cyan] (HuggingFace) or [cyan]api[/cyan] (remote LLM).",
    ),
    agent: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--agent",
        "-a",
        help="Model name. E.g. [cyan]roberta-base-openai-detector[/cyan] for local, or [cyan]gpt-4o[/cyan] for API.",
    ),
    api_key: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--api-key",
        "-k",
        help="API key for remote LLM (overrides .env). Only used with [cyan]--engine api[/cyan].",
    ),
    api_url: Optional[str] = typer.Option(  # noqa: UP007
        None,
        "--api-url",
        "-u",
        help="API endpoint URL (overrides .env). Only used with [cyan]--engine api[/cyan].",
    ),
    show_text: bool = typer.Option(
        True,
        "--show-text/--no-text",
        help="Show extracted text panel with highlighted AI patterns.",
    ),
    languages: str = typer.Option(
        "en",
        "--lang",
        "-l",
        help="Comma-separated OCR language codes (e.g. [cyan]en,fr[/cyan]). Used for images and scanned PDFs.",
    ),
) -> None:
    """Verify whether content is human-written or AI-generated.

    Accepts a single file or a directory for batch processing.
    Supports images (PNG, JPG, …), PDF, DOCX, TXT, MD, and Pages files.

    \b
    Examples:
        synth verify photo.png
        synth verify report.pdf --engine api --api-key sk-... --agent gpt-4o
        synth verify ./documents/ --engine api --agent gpt-4o
        synth verify essay.docx --engine api --api-key sk-... --api-url https://api.openai.com/v1/chat/completions
        synth verify scan.jpg --lang en,es --no-text
    """
    # ── Banner ────────────────────────────────────────────────────────────
    print_banner()

    # ── Collect files ─────────────────────────────────────────────────────
    files = _collect_files(path)
    is_batch = len(files) > 1

    if is_batch:
        console.print(
            f"[bold]Found [cyan]{len(files)}[/cyan] file(s) in "
            f"[cyan]{path}[/cyan][/bold]\n"
        )

    # ── Initialise detection model (with spinner) ─────────────────────────
    lang_list = [l.strip() for l in languages.split(",")]

    detector_kwargs: dict[str, object] = {}
    if engine == EngineChoice.local:
        if agent:
            detector_kwargs["model_name"] = agent
    else:
        # API engine — pass CLI overrides
        if api_key:
            detector_kwargs["api_key"] = api_key
        if api_url:
            detector_kwargs["api_url"] = api_url
        if agent:
            detector_kwargs["api_model"] = agent

    with Status(
        "[bold cyan]Initialising detection model…[/bold cyan]",
        spinner="dots",
        console=console,
    ) as status:
        status.update("[bold cyan]Loading detection model…[/bold cyan]")
        detector = DetectorFactory.create(engine.value, **detector_kwargs)

    console.print("[bold green]✓[/bold green] Detection model loaded successfully\n")

    # ── Process files ─────────────────────────────────────────────────────
    results: list[tuple[str, AuthResult]] = []
    texts: dict[str, str] = {}

    if is_batch:
        # ── Batch mode with progress bar ──────────────────────────────
        with Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=40, style="cyan", complete_style="bold cyan"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task("Scanning files…", total=len(files))

            for file_path in files:
                progress.update(
                    task,
                    description=f"Processing [cyan]{file_path.name}[/cyan]",
                )

                outcome = _process_single_file(
                    file_path, detector, show_text, lang_list
                )

                if outcome is not None:
                    filename, text, result = outcome
                    results.append((filename, result))
                    texts[filename] = text

                progress.advance(task)

        console.print()

    else:
        # ── Single file mode with spinner ─────────────────────────────
        file_path = files[0]

        with Status(
            f"[bold cyan]Analysing [white]{file_path.name}[/white]…[/bold cyan]",
            spinner="dots",
            console=console,
        ):
            outcome = _process_single_file(
                file_path, detector, show_text, lang_list
            )

        if outcome is None:
            raise typer.Exit(code=1)

        filename, text, result = outcome
        results.append((filename, result))
        texts[filename] = text

    # ── Display results ───────────────────────────────────────────────────
    if not results:
        console.print(
            "[bold red]✗[/bold red] No files could be processed successfully."
        )
        raise typer.Exit(code=1)

    # Results table
    table = build_results_table(results)
    console.print(table)
    console.print()

    # Text panels (single file or when --show-text is on)
    if show_text:
        for filename, result in results:
            if filename in texts:
                panel = build_text_panel(
                    texts[filename], result, filename=filename
                )
                console.print(panel)
                console.print()

    # Batch summary
    if is_batch:
        summary = build_summary_panel(results)
        console.print(summary)

    # ── Exit code based on results ────────────────────────────────────────
    ai_detected = any(r.verdict == "ai" for _, r in results)
    if ai_detected:
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  synth configure
# ══════════════════════════════════════════════════════════════════════════════

# Provider presets for the interactive wizard
_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o",
        "key_prefix": "sk-",
        "payload_map": "",
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-sonnet-4-20250514",
        "key_prefix": "sk-ant-",
        "payload_map": "",
    },
    "google": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "default_model": "gemini-2.5-flash",
        "key_prefix": "AI",
        "payload_map": "",
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434/v1/chat/completions",
        "default_model": "llama3",
        "key_prefix": "",
        "payload_map": "",
    },
}


@app.command()
def configure() -> None:
    """Interactive setup wizard for API configuration.

    Walks you through selecting an LLM provider, entering your API key,
    and choosing a model. Saves the configuration to a ``.env`` file.

    \b
    Supported providers:
        • OpenAI  (GPT-4o, GPT-4, etc.)
        • Anthropic  (Claude)
        • Google Gemini
        • Ollama  (local models)
        • Custom  (any OpenAI-compatible endpoint)
    """
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt

    print_banner()

    console.print(
        Panel(
            "[bold]Welcome to the Synth API Configuration Wizard![/bold]\n\n"
            "This will help you set up a remote LLM for AI content detection.\n"
            "You can always re-run [cyan]synth configure[/cyan] to update your settings.",
            title="[bold cyan]⚙ Configuration[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # ── Step 1: Choose provider ───────────────────────────────────────────
    console.print("\n[bold]Step 1:[/bold] Choose your LLM provider\n")

    providers = list(_PROVIDER_PRESETS.keys()) + ["custom"]
    for idx, key in enumerate(providers, 1):
        if key == "custom":
            label = "Custom (any OpenAI-compatible API)"
        else:
            label = _PROVIDER_PRESETS[key]["name"]
        console.print(f"  [cyan]{idx}[/cyan]. {label}")

    choice = Prompt.ask(
        "\n[bold]Select provider[/bold]",
        choices=[str(i) for i in range(1, len(providers) + 1)],
        default="1",
    )
    provider_key = providers[int(choice) - 1]

    # ── Step 2: API URL ───────────────────────────────────────────────────
    console.print()

    if provider_key == "custom":
        base_url = Prompt.ask(
            "[bold]Step 2:[/bold] Enter your API endpoint URL",
        )
    else:
        preset = _PROVIDER_PRESETS[provider_key]
        default_url = preset["base_url"]
        base_url = Prompt.ask(
            f"[bold]Step 2:[/bold] API endpoint URL",
            default=default_url,
        )

    # ── Step 3: API Key ───────────────────────────────────────────────────
    console.print()

    if provider_key == "ollama":
        api_key = Prompt.ask(
            "[bold]Step 3:[/bold] API key (leave empty for local Ollama)",
            default="ollama",
        )
    else:
        api_key = Prompt.ask(
            "[bold]Step 3:[/bold] Enter your API key",
        )

    # ── Step 4: Model ─────────────────────────────────────────────────────
    console.print()

    if provider_key == "custom":
        model = Prompt.ask(
            "[bold]Step 4:[/bold] Model identifier",
            default="",
        )
    else:
        default_model = _PROVIDER_PRESETS.get(provider_key, {}).get("default_model", "")
        model = Prompt.ask(
            "[bold]Step 4:[/bold] Model to use",
            default=default_model,
        )

    # ── Step 5: Save location ─────────────────────────────────────────────
    console.print()

    env_path = Path.cwd() / ".env"
    home_env = Path.home() / ".synth.env"

    save_location = Prompt.ask(
        f"[bold]Step 5:[/bold] Save config to",
        choices=["local", "home"],
        default="local",
    )

    target_path = env_path if save_location == "local" else home_env

    # ── Build .env content ────────────────────────────────────────────────
    env_lines = [
        "# ──────────────────────────────────────────────",
        "#  Synth — API Configuration",
        f"#  Generated by 'synth configure' for {_PROVIDER_PRESETS.get(provider_key, {}).get('name', 'Custom')}",
        "# ──────────────────────────────────────────────",
        "",
        f"SYNTH_API_BASE_URL={base_url}",
        f"SYNTH_API_KEY={api_key}",
        f"SYNTH_API_MODEL={model}",
    ]

    # Add payload map for Anthropic (different auth style)
    if provider_key == "anthropic":
        payload_map = Path(__file__).parent.parent.parent.parent / "config" / "payload_anthropic.json"
        if payload_map.exists():
            env_lines.append(f"SYNTH_PAYLOAD_MAP={payload_map}")

    env_content = "\n".join(env_lines) + "\n"

    # ── Confirm and write ─────────────────────────────────────────────────
    console.print()
    console.print(
        Panel(
            f"[bold]Provider:[/bold]  {_PROVIDER_PRESETS.get(provider_key, {}).get('name', 'Custom')}\n"
            f"[bold]Endpoint:[/bold]  {base_url}\n"
            f"[bold]API Key:[/bold]   {'*' * (len(api_key) - 4) + api_key[-4:] if len(api_key) > 4 else '****'}\n"
            f"[bold]Model:[/bold]     {model or '(default)'}\n"
            f"[bold]Save to:[/bold]   {target_path}",
            title="[bold cyan]📋 Configuration Summary[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if not Confirm.ask("\n[bold]Save this configuration?[/bold]", default=True):
        console.print("[yellow]Configuration cancelled.[/yellow]")
        raise typer.Exit()

    target_path.write_text(env_content, encoding="utf-8")
    console.print(
        f"\n[bold green]✓[/bold green] Configuration saved to "
        f"[cyan]{target_path}[/cyan]\n"
    )
    console.print(
        "[dim]You can now use:[/dim]\n"
        "  [cyan]synth verify <file> --engine api[/cyan]\n"
        "\n"
        "[dim]Or override on the fly:[/dim]\n"
        "  [cyan]synth verify <file> --engine api --api-key sk-... --agent gpt-4o[/cyan]"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  synth info
# ══════════════════════════════════════════════════════════════════════════════


@app.command()
def info() -> None:
    """Show system information and available hardware."""
    from synth.core.device import detect_device

    print_banner()

    device = detect_device()

    info_lines = [
        f"[bold]Version:[/bold]       {__version__}",
        f"[bold]Python:[/bold]        {sys.version.split()[0]}",
        f"[bold]Compute:[/bold]       [cyan]{device}[/cyan]",
        f"[bold]Strategies:[/bold]    {', '.join(DetectorFactory.available())}",
        f"[bold]File types:[/bold]    {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}",
    ]

    try:
        import torch

        info_lines.append(f"[bold]PyTorch:[/bold]       {torch.__version__}")
        info_lines.append(
            f"[bold]CUDA:[/bold]          {'✓ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ not available'}"
        )
        info_lines.append(
            f"[bold]MPS:[/bold]           {'[green]✓ available[/green]' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else '[red]✗ not available[/red]'}"
        )
    except ImportError:
        pass

    # Check API configuration
    api_configured = bool(os.getenv("SYNTH_API_KEY"))
    api_status = "[green]✓ configured[/green]" if api_configured else "[yellow]✗ not configured (run 'synth configure')[/yellow]"
    info_lines.append(f"[bold]API:[/bold]           {api_status}")

    from rich.panel import Panel

    console.print(
        Panel(
            "\n".join(info_lines),
            title="[bold cyan]System Info[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
