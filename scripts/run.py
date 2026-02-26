"""CLI entry point for Gesture HUD."""

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
from src.core.engine import HUDEngine

console = Console()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gesture HUD — Iron Man-Style AR Command Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Point finger    → targeting reticle follows
  Fist (hold)     → lock on to nearest object
  Open palm       → trigger scan mode
  Swipe left/right → switch HUD modes
  Thumbs up (hold) → take screenshot
  'q' or ESC       → quit
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
        "--no-detection",
        action="store_true",
        help="Disable object detection (faster startup)",
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

    if args.no_detection:
        settings.detection.run_every_n_frames = 999999

    # Display banner
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Gesture HUD[/bold cyan] — AR Command Interface",
        border_style="cyan",
    ))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Camera", f"Device {settings.camera.device_id}" if not args.source else args.source)
    table.add_row("Resolution", f"{settings.camera.width}x{settings.camera.height}")
    table.add_row("Detection", "Enabled" if not args.no_detection else "Disabled")
    table.add_row("Effects", "Glow + Scanlines + Holographic")
    console.print(table)

    console.print()
    console.print("[dim]Press 'q' or ESC to quit. Swipe to switch modes.[/dim]")
    console.print()

    # Run the HUD engine
    engine = HUDEngine(settings)

    try:
        engine.run(source=args.source)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    finally:
        engine.stop()
        console.print("[green]Gesture HUD stopped.[/green]")


if __name__ == "__main__":
    main()
