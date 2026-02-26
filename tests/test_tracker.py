"""Tests for gesture state machine."""

from __future__ import annotations

import time

from src.config import GesturesConfig
from src.gestures.recognizer import GestureResult, GestureType
from src.gestures.tracker import GestureEvent, GestureTracker
from src.vision.hands import HandData, Point


def make_gesture_result(gesture_type: GestureType, center_x: float = 0.5) -> GestureResult:
    """Helper to create a GestureResult with minimal hand data."""
    landmarks = [Point(0.5, 0.7)] * 21
    landmarks[0] = Point(0.5, 0.8)
    landmarks[5] = Point(0.44, 0.62)
    landmarks[8] = Point(center_x, 0.32)
    landmarks[9] = Point(0.50, 0.60)
    landmarks[13] = Point(0.56, 0.62)
    landmarks[17] = Point(0.62, 0.65)
    hand = HandData(landmarks=landmarks, handedness="Right", confidence=0.9, finger_extended={})
    return GestureResult(gesture=gesture_type, confidence=0.9, hand_data=hand)


class TestGestureTracker:
    def setup_method(self):
        self.config = GesturesConfig(debounce_frames=2, hold_duration=0.3, tap_max_duration=0.2)
        self.tracker = GestureTracker(self.config)

    def test_initial_state(self):
        state = self.tracker.update(None)
        assert state.current_gesture == GestureType.NONE
        assert state.event == GestureEvent.NONE

    def test_gesture_debounce(self):
        result = make_gesture_result(GestureType.FIST)

        # First update: not yet confirmed (debounce_frames=2)
        state = self.tracker.update(result)
        # Second update: now confirmed
        state = self.tracker.update(result)
        assert state.current_gesture == GestureType.FIST

    def test_hold_detection(self):
        result = make_gesture_result(GestureType.FIST)

        # Get through debounce
        self.tracker.update(result)
        self.tracker.update(result)

        # Simulate time passing
        self.tracker._gesture_start_time = time.time() - 0.5

        state = self.tracker.update(result)
        assert state.is_holding is True
        assert state.event == GestureEvent.HOLD_START

    def test_no_hand_resets(self):
        result = make_gesture_result(GestureType.FIST)

        # Get a gesture going
        self.tracker.update(result)
        self.tracker.update(result)

        # Remove hand
        state = self.tracker.update(None)
        assert state.current_gesture == GestureType.NONE

    def test_reset(self):
        result = make_gesture_result(GestureType.FIST)
        self.tracker.update(result)
        self.tracker.update(result)

        self.tracker.reset()
        state = self.tracker.update(None)
        assert state.current_gesture == GestureType.NONE

    def test_hand_center_tracked(self):
        result = make_gesture_result(GestureType.POINT, center_x=0.6)
        state = self.tracker.update(result)
        assert state.hand_center is not None
