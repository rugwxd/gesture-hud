"""Tests for object tagging overlay widget."""

from __future__ import annotations

import numpy as np

from src.gestures.recognizer import GestureType
from src.gestures.tracker import GestureState
from src.hud.object_tags import ObjectTags
from src.hud.widgets import HUDMode, HUDState


class TestObjectTags:
    def setup_method(self):
        self.tags = ObjectTags(color=(0, 255, 200), secondary_color=(255, 170, 0))

    def test_creation(self):
        assert self.tags.name == "object_tags"
        assert HUDMode.COMBAT in self.tags.active_modes

    def test_render_with_detections(self, sample_detections):
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(
            frame_size=(640, 480),
            detections=sample_detections,
        )

        result = self.tags.render(overlay, state)
        assert np.any(result > 0)

    def test_render_no_detections(self):
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(frame_size=(640, 480))

        result = self.tags.render(overlay, state)
        # No detections, no scan — overlay should be empty
        assert result is overlay

    def test_scan_animation_triggered(self):
        state = HUDState(
            frame_size=(640, 480),
            gesture_state=GestureState(
                current_gesture=GestureType.OPEN_PALM,
            ),
        )
        self.tags.update(state)
        assert self.tags._scan_active is True

    def test_scan_animation_renders(self):
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(
            frame_size=(640, 480),
            gesture_state=GestureState(
                current_gesture=GestureType.OPEN_PALM,
            ),
        )
        self.tags.update(state)
        self.tags._scan_y = 200  # Mid-frame

        result = self.tags.render(overlay, state)
        assert np.any(result > 0)
