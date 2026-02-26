"""Main HUD engine orchestrating all components."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import Settings
from src.core.renderer import HUDRenderer
from src.effects.holographic import EffectsPipeline
from src.gestures.recognizer import GestureRecognizer, GestureType
from src.gestures.tracker import GestureEvent, GestureTracker
from src.hud.menu import ModeMenu
from src.hud.object_tags import ObjectTags
from src.hud.radar import RadarWidget
from src.hud.stats import StatsPanel
from src.hud.targeting import TargetingReticle
from src.hud.widgets import HUDMode, HUDState, WidgetRegistry
from src.vision.camera import Camera
from src.vision.detector import ObjectDetector
from src.vision.hands import HandTracker

logger = logging.getLogger(__name__)


class HUDEngine:
    """Main engine that orchestrates camera capture, hand tracking, gesture
    recognition, object detection, HUD rendering, and effects.

    This is the core loop that ties everything together.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Vision components
        self.camera = Camera(settings.camera)
        self.hand_tracker = HandTracker(settings.hands)
        self.detector = ObjectDetector(settings.detection)

        # Gesture components
        self.gesture_recognizer = GestureRecognizer()
        self.gesture_tracker = GestureTracker(settings.gestures)

        # Rendering
        self.renderer = HUDRenderer(settings.hud)
        self.effects = EffectsPipeline(settings.effects)

        # HUD widgets
        self.widget_registry = WidgetRegistry()
        self.mode_menu = ModeMenu()
        self._current_mode = HUDMode.COMBAT

        self._setup_widgets()

        # State
        self._running = False
        self._frame_count = 0

    def _setup_widgets(self) -> None:
        """Register all HUD widgets."""
        primary = tuple(self.settings.hud.color_primary)
        secondary = tuple(self.settings.hud.color_secondary)

        self.widget_registry.register(self.mode_menu)
        self.widget_registry.register(TargetingReticle(color=primary))
        self.widget_registry.register(StatsPanel(color=primary))
        self.widget_registry.register(RadarWidget(color=primary))
        self.widget_registry.register(ObjectTags(color=primary, secondary_color=secondary))

    def run(self, source: str | None = None) -> None:
        """Run the main HUD loop.

        Args:
            source: Optional video file path. If None, uses webcam.
        """
        if source:
            self.camera = Camera(self.settings.camera, source=source)

        if not self.camera.open():
            logger.error("Failed to open camera")
            return

        self._running = True
        logger.info("HUD engine started")

        try:
            while self._running:
                self._process_frame()

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # q or ESC
                    break
                elif key == ord("s"):
                    self._take_screenshot()
        finally:
            self._cleanup()

    def _process_frame(self) -> None:
        """Process a single frame through the full pipeline."""
        frame_data = self.camera.read()
        if frame_data is None:
            self._running = False
            return

        frame = frame_data.image
        self._frame_count += 1

        # 1. Hand tracking
        hands = self.hand_tracker.process(frame)

        # 2. Gesture recognition (use first hand)
        gesture_result = None
        gesture_state = None
        if hands:
            gesture_result = self.gesture_recognizer.classify(hands[0])
            gesture_state = self.gesture_tracker.update(gesture_result)
        else:
            gesture_state = self.gesture_tracker.update(None)

        # 3. Object detection
        detections = self.detector.detect(frame)
        detection_dicts = [d.to_dict() for d in detections]

        # 4. Check for screenshot gesture
        if gesture_state and gesture_state.event == GestureEvent.HOLD_START:
            if gesture_state.current_gesture == GestureType.THUMBS_UP:
                self._take_screenshot(frame)

        # 5. Build HUD state
        width, height = frame.shape[1], frame.shape[0]
        hud_state = HUDState(
            frame_size=(width, height),
            hands=hands,
            gesture_state=gesture_state,
            detections=detection_dicts,
            mode=self._current_mode,
            fps=0,
            frame_number=self._frame_count,
        )

        # 6. Update and render widgets
        overlay = self.renderer.create_overlay(width, height)
        self.widget_registry.update_all(hud_state)
        overlay = self.widget_registry.render_all(overlay, hud_state)

        # Update mode from menu widget
        self._current_mode = self.mode_menu.current_mode

        # 7. Composite overlay onto frame
        result = self.renderer.composite(frame, overlay)

        # 8. Apply effects
        result = self.effects.apply(result)

        # 9. Draw hand landmarks (subtle)
        self._draw_landmarks(result, hands)

        # 10. Display
        cv2.imshow("Gesture HUD", result)

    def _draw_landmarks(self, frame: np.ndarray, hands: list) -> None:
        """Draw subtle hand landmark connections."""
        color = tuple(c // 2 for c in self.settings.hud.color_primary)
        height, width = frame.shape[:2]

        for hand in hands:
            points = [(int(lm.x * width), int(lm.y * height)) for lm in hand.landmarks]

            # Draw connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17),  # Palm
            ]

            for start, end in connections:
                if start < len(points) and end < len(points):
                    cv2.line(frame, points[start], points[end], color, 1, cv2.LINE_AA)

            # Draw small dots at joints
            for pt in points:
                cv2.circle(frame, pt, 2, color, -1)

    def _take_screenshot(self, frame: np.ndarray | None = None) -> None:
        """Save the current frame as a screenshot."""
        output_dir = Path(self.settings.recording.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"screenshot_{timestamp}.png"

        if frame is not None:
            cv2.imwrite(str(filepath), frame)
            logger.info("Screenshot saved: %s", filepath)

    def _cleanup(self) -> None:
        """Release all resources."""
        self._running = False
        self.camera.release()
        self.hand_tracker.release()
        self.detector.release()
        cv2.destroyAllWindows()
        logger.info("HUD engine stopped")

    def stop(self) -> None:
        """Signal the engine to stop."""
        self._running = False
