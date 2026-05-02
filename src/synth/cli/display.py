"""Rich terminal UI components for Synth.

Reusable display helpers: banner, result tables, text panels,
and progress bars. Keeps the main CLI module focused on command logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from synth.core.auth import AuthResult

console = Console()

# ── ASCII Banner ──────────────────────────────────────────────────────────────

BANNER = r"""
[bold cyan]  ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗[/bold cyan]
[bold cyan]  ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║[/bold cyan]
[bold cyan]  ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║[/bold cyan]
[bold cyan]  ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║[/bold cyan]
[bold cyan]  ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║[/bold cyan]
[bold cyan]  ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝[/bold cyan]
[dim]  AI Content Authenticator & OCR Engine[/dim]
"""


def print_banner() -> None:
    """Print the stylised SYNTH ASCII banner."""
    console.print(BANNER)


# ── Verdict colours ───────────────────────────────────────────────────────────

_VERDICT_STYLES: dict[str, str] = {
    "human": "bold green",
    "ai": "bold red",
    "mixed": "bold yellow",
}


def _verdict_styled(verdict: str) -> str:
    """Return a Rich-markup-wrapped verdict string."""
    style = _VERDICT_STYLES.get(verdict, "bold white")
    tag = verdict.upper()
    return f"[{style}]{tag}[/{style}]"


def _score_styled(score: float) -> str:
    """Return colour-coded score: green ≤0.3, yellow 0.3–0.7, red ≥0.7."""
    pct = f"{score * 100:.1f}%"
    if score <= 0.3:
        return f"[green]{pct}[/green]"
    if score <= 0.7:
        return f"[yellow]{pct}[/yellow]"
    return f"[red]{pct}[/red]"


# ── Results table ─────────────────────────────────────────────────────────────


def build_results_table(
    results: list[tuple[str, AuthResult]],
) -> Table:
    """Build a Rich table summarising verification results.

    Args:
        results: List of ``(filename, AuthResult)`` tuples.

    Returns:
        A styled :class:`rich.table.Table`.
    """
    table = Table(
        title="[bold]Verification Results[/bold]",
        title_style="cyan",
        border_style="dim cyan",
        header_style="bold white",
        row_styles=["", "dim"],
        expand=True,
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("File", style="white", ratio=3)
    table.add_column("Verdict", justify="center", width=10)
    table.add_column("AI Score", justify="center", width=10)
    table.add_column("Model", style="dim", ratio=2)

    for idx, (filename, result) in enumerate(results, 1):
        table.add_row(
            str(idx),
            filename,
            _verdict_styled(result.verdict),
            _score_styled(result.score),
            result.model,
        )

    return table


# ── Text panel ────────────────────────────────────────────────────────────────

_HIGHLIGHT_PATTERNS = [
    "however",
    "furthermore",
    "moreover",
    "in conclusion",
    "it is important to note",
    "significantly",
    "consequently",
    "as a result",
    "in summary",
    "overall",
    "notably",
    "essentially",
]


def build_text_panel(
    text: str,
    result: AuthResult,
    filename: str = "",
    max_chars: int = 800,
) -> Panel:
    """Build a Rich panel showing extracted text with AI patterns highlighted.

    Common AI-associated connector words are highlighted in red to give
    the user a visual sense of synthetic patterning.

    Args:
        text: The extracted text content.
        result: The detection result.
        filename: Optional filename for the panel title.
        max_chars: Truncate text beyond this limit.
    """
    # Truncate long texts
    display_text = text[:max_chars]
    if len(text) > max_chars:
        display_text += f"\n[dim]… ({len(text) - max_chars} more characters)[/dim]"

    # Highlight synthetic patterns
    rich_text = Text(display_text)
    for pattern in _HIGHLIGHT_PATTERNS:
        rich_text.highlight_words(
            [pattern, pattern.capitalize(), pattern.upper()],
            style="bold red underline",
        )

    # Build subtitle
    verdict_display = _verdict_styled(result.verdict)
    score_display = _score_styled(result.score)
    subtitle = f"Verdict: {verdict_display}  │  AI Score: {score_display}"

    title = f"[bold cyan]📄 {filename}[/bold cyan]" if filename else "[bold cyan]Text Analysis[/bold cyan]"

    return Panel(
        Align.left(rich_text),
        title=title,
        subtitle=subtitle,
        border_style="cyan",
        padding=(1, 2),
        expand=True,
    )


# ── Summary panel ────────────────────────────────────────────────────────────


def build_summary_panel(
    results: list[tuple[str, AuthResult]],
) -> Panel:
    """Build a summary panel with aggregate statistics."""
    total = len(results)
    ai_count = sum(1 for _, r in results if r.verdict == "ai")
    human_count = sum(1 for _, r in results if r.verdict == "human")
    mixed_count = sum(1 for _, r in results if r.verdict == "mixed")
    avg_score = sum(r.score for _, r in results) / total if total else 0.0

    lines = [
        f"[bold]Files scanned:[/bold]  {total}",
        f"[bold green]Human:[/bold green]          {human_count}",
        f"[bold red]AI-generated:[/bold red]   {ai_count}",
        f"[bold yellow]Mixed:[/bold yellow]          {mixed_count}",
        f"[bold]Avg AI score:[/bold]   {_score_styled(avg_score)}",
    ]

    return Panel(
        "\n".join(lines),
        title="[bold cyan]📊 Batch Summary[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    )
