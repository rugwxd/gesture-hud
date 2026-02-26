"""CLI entry point for AR Spellcaster."""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import load_config, setup_logging
from src.core.engine import SpellEngine

console = Console()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AR Spellcaster — Cast Magic with Hand Gestures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Spells:
  Swipe up         → Fireball (flaming orb projectile)
  Point up + hold  → Lightning (electric bolt to fingertip)
  Fist hold        → Shield (hexagonal force barrier)
  Open palm tap    → Force Push (expanding shockwave)
  Pinch + release  → Teleport (screen glitch effect)
  Swipe left/right → Wind (debris sweep)

Controls:
  'q' or ESC → quit
        """,
    )
    parser.add_argument(
        "--source", "-s",
        default=None,
        help="Video file path (default: webcam)",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable spell sound effects",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Load configuration
    settings = load_config(args.config)
    if args.verbose:
        settings.logging.level = "DEBUG"
    setup_logging(settings.logging)

    if args.no_audio:
        settings.audio.enabled = False

    # Display banner
    console.print()
    console.print(Panel.fit(
        "[bold magenta]AR Spellcaster[/bold magenta] — Cast Magic with Your Hands",
        border_style="magenta",
    ))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row(
        "Camera",
        f"Device {settings.camera.device_id}" if not args.source else args.source,
    )
    table.add_row("Resolution", f"{settings.camera.width}x{settings.camera.height}")
    table.add_row("Max Particles", str(settings.particles.max_particles))
    table.add_row("Audio", "Enabled" if settings.audio.enabled else "Disabled")
    console.print(table)

    console.print()
    console.print("[dim]Press 'q' or ESC to quit. Use hand gestures to cast spells.[/dim]")
    console.print()

    # Run the spell engine
    engine = SpellEngine(settings)

    try:
        engine.run(source=args.source)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        engine.stop()
        console.print("[green]Spellcaster stopped.[/green]")


if __name__ == "__main__":
    main()
