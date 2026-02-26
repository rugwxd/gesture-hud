"""Tests for targeting reticle widget."""

from __future__ import annotations

import numpy as np

from src.gestures.recognizer import GestureType
from src.gestures.tracker import GestureState
from src.hud.targeting import TargetingReticle
from src.hud.widgets import HUDMode, HUDState
from src.vision.hands import Point


class TestTargetingReticle:
    def setup_method(self):
        self.reticle = TargetingReticle(color=(0, 255, 200))

    def test_creation(self):
        assert self.reticle.name == "targeting"
        assert HUDMode.COMBAT in self.reticle.active_modes
        assert HUDMode.NAVIGATION not in self.reticle.active_modes

    def test_update_with_hand(self):
        state = HUDState(
            frame_size=(640, 480),
            gesture_state=GestureState(
                current_gesture=GestureType.POINT,
                index_tip=Point(0.5, 0.5),
            ),
        )
        self.reticle.update(state)
        assert self.reticle._target_x == 320
        assert self.reticle._target_y == 240

    def test_render_returns_overlay(self):
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(frame_size=(640, 480))

        # Update first to set position
        state.gesture_state = GestureState(
            current_gesture=GestureType.POINT,
            index_tip=Point(0.5, 0.5),
        )
        self.reticle.update(state)

        result = self.reticle.render(overlay, state)
        assert result is overlay
        # Check that something was drawn (overlay should have non-zero pixels)
        assert np.any(result > 0)

    def test_no_render_at_origin(self):
        """Should not render when position is (0, 0)."""
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(frame_size=(640, 480))

        result = self.reticle.render(overlay, state)
        assert not np.any(result > 0)

    def test_lock_on_with_fist(self):
        state = HUDState(
            frame_size=(640, 480),
            gesture_state=GestureState(
                current_gesture=GestureType.FIST,
                is_holding=True,
                index_tip=Point(0.5, 0.5),
            ),
        )
        self.reticle.update(state)
        assert self.reticle._locked is True
