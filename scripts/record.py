"""Record HUD output to video file."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from rich.console import Console

from src.config import load_config, setup_logging
from src.core.engine import HUDEngine

console = Console()


class RecordingEngine(HUDEngine):
    """Extended HUD engine that records output to video."""

    def __init__(self, settings, output_path: str) -> None:
        super().__init__(settings)
        self._output_path = output_path
        self._writer: cv2.VideoWriter | None = None

    def run(self, source: str | None = None) -> None:
        """Override run to add video recording."""
        if source:
            from src.vision.camera import Camera

            self.camera = Camera(self.settings.camera, source=source)

        if not self.camera.open():
            return

        # Setup video writer
        width = self.settings.camera.width
        height = self.settings.camera.height
        fps = self.settings.camera.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._output_path, fourcc, fps, (width, height))

        self._running = True
        console.print(f"[green]Recording to {self._output_path}[/green]")

        try:
            while self._running:
                frame_data = self.camera.read()
                if frame_data is None:
                    break

                frame = frame_data.image
                self._frame_count += 1

                # Process through pipeline (same as parent)
                hands = self.hand_tracker.process(frame)
                gesture_result = None
                if hands:
                    gesture_result = self.gesture_recognizer.classify(hands[0])
                    gesture_state = self.gesture_tracker.update(gesture_result)
                else:
                    gesture_state = self.gesture_tracker.update(None)

                detections = self.detector.detect(frame)
                detection_dicts = [d.to_dict() for d in detections]

                from src.hud.widgets import HUDState

                hud_state = HUDState(
                    frame_size=(frame.shape[1], frame.shape[0]),
                    hands=hands,
                    gesture_state=gesture_state,
                    detections=detection_dicts,
                    mode=self._current_mode,
                    fps=0,
                    frame_number=self._frame_count,
                )

                overlay = self.renderer.create_overlay(frame.shape[1], frame.shape[0])
                self.widget_registry.update_all(hud_state)
                overlay = self.widget_registry.render_all(overlay, hud_state)
                self._current_mode = self.mode_menu.current_mode

                result = self.renderer.composite(frame, overlay)
                result = self.effects.apply(result)
                self._draw_landmarks(result, hands)

                # Write frame
                self._writer.write(result)

                cv2.imshow("Recording - Gesture HUD", result)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break

                # Show progress
                if self._frame_count % 30 == 0:
                    console.print(f"[dim]Frame {self._frame_count}...[/dim]", end="\r")

        finally:
            if self._writer:
                self._writer.release()
            self._cleanup()
            console.print(f"\n[green]Saved: {self._output_path} ({self._frame_count} frames)[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Gesture HUD output")
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
        output = f"data/recordings/hud_{timestamp}.mp4"

    engine = RecordingEngine(settings, output)
    try:
        engine.run(source=args.source)
    except KeyboardInterrupt:
        console.print("\n[yellow]Recording stopped.[/yellow]")


if __name__ == "__main__":
    main()
