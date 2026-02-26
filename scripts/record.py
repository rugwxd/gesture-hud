"""Record AR Spellcaster output to video file."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from rich.console import Console

from src.audio.player import AudioPlayer
from src.config import load_config, setup_logging
from src.core.renderer import SpellRenderer
from src.effects.glow import apply_glow
from src.effects.screen import ScreenEffects
from src.gestures.recognizer import GestureRecognizer
from src.gestures.tracker import GestureEvent, GestureTracker
from src.particles.engine import ParticleEngine
from src.spells.fireball import Fireball
from src.spells.force_push import ForcePush
from src.spells.lightning import Lightning
from src.spells.registry import SpellRegistry
from src.spells.shield import Shield
from src.spells.teleport import Teleport
from src.spells.wind import Wind
from src.vision.camera import Camera
from src.vision.hands import HandTracker

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record AR Spellcaster output")
    parser.add_argument("--source", "-s", default=None, help="Video file (default: webcam)")
    parser.add_argument("--output", "-o", default=None, help="Output path")
    parser.add_argument("--config", "-c", default=None, help="Config YAML path")
    args = parser.parse_args()

    settings = load_config(args.config)
    setup_logging(settings.logging)

    output = args.output
    if not output:
        Path("data/recordings").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"data/recordings/spell_{timestamp}.mp4"

    # Initialize components
    camera = Camera(settings.camera, source=args.source) if args.source else Camera(settings.camera)
    if not camera.open():
        console.print("[red]Failed to open camera[/red]")
        return

    hand_tracker = HandTracker(settings.hands)
    recognizer = GestureRecognizer()
    tracker = GestureTracker(settings.gestures)
    particles = ParticleEngine(max_particles=settings.particles.max_particles)
    screen_fx = ScreenEffects()
    audio = AudioPlayer(enabled=False)  # No audio during recording
    registry = SpellRegistry(
        particles=particles, screen_fx=screen_fx, audio=audio,
        max_mana=settings.spells.max_mana, mana_regen=settings.spells.mana_regen,
    )
    renderer = SpellRenderer(settings)

    # Register spells
    registry.register("hold_start_fist", Shield)
    registry.register("hold_start_open_palm", ForcePush)
    registry.register("hold_start_point", Lightning)
    registry.register("swipe_left", Wind)
    registry.register("swipe_right", Wind)
    registry.register("hold_start_pinch", Teleport)
    registry.register("swipe_up", Fireball)

    # Video writer
    w = settings.camera.width
    h = settings.camera.height
    fps = settings.camera.fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    console.print(f"[green]Recording to {output}[/green]")
    frame_count = 0
    last_time = time.time()

    try:
        while True:
            frame_data = camera.read()
            if frame_data is None:
                break

            frame = frame_data.image
            frame_count += 1
            now = time.time()
            dt = min(now - last_time, 0.1)
            last_time = now

            # Process
            hands = hand_tracker.process(frame)
            hand_x = hand_y = None
            gesture_name = ""
            if hands:
                result = recognizer.classify(hands[0])
                state = tracker.update(result)
                hand_x, hand_y = hands[0].center.x, hands[0].center.y
                gesture_name = result.gesture.name.lower()
            else:
                state = tracker.update(None)

            if state and state.event != GestureEvent.NONE:
                registry.handle_event(
                    state.event, hand_x or 0.5, hand_y or 0.5, gesture_name,
                )

            registry.update(dt, hand_x, hand_y)
            particles.update(dt)
            screen_fx.update(dt)

            frame = particles.render(frame)
            frame = registry.render(frame)
            if settings.particles.glow_enabled and particles.count > 0:
                frame = apply_glow(frame, intensity=settings.particles.glow_intensity)
            frame = screen_fx.apply(frame)
            renderer.draw_landmarks(frame, hands)
            renderer.draw_mana_bar(frame, registry.mana)

            writer.write(frame)

            cv2.imshow("Recording - AR Spellcaster", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

            if frame_count % 30 == 0:
                console.print(f"[dim]Frame {frame_count}...[/dim]", end="\r")

    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped.[/yellow]")
    finally:
        writer.release()
        camera.release()
        hand_tracker.release()
        cv2.destroyAllWindows()
        console.print(f"\n[green]Saved: {output} ({frame_count} frames)[/green]")


if __name__ == "__main__":
    main()
