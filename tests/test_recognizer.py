"""Tests for gesture recognition."""

from __future__ import annotations

from src.gestures.recognizer import GestureRecognizer, GestureType
from src.vision.hands import HandData, Point


class TestGestureRecognizer:
    def setup_method(self):
        self.recognizer = GestureRecognizer()

    def test_open_palm(self, hand_data_open_palm):
        result = self.recognizer.classify(hand_data_open_palm)
        assert result.gesture == GestureType.OPEN_PALM
        assert result.confidence > 0.8

    def test_fist(self, hand_data_fist):
        result = self.recognizer.classify(hand_data_fist)
        assert result.gesture == GestureType.FIST
        assert result.confidence > 0.8

    def test_point(self, hand_data_point):
        result = self.recognizer.classify(hand_data_point)
        assert result.gesture == GestureType.POINT
        assert result.confidence > 0.7

    def test_pinch_gesture(self):
        """Test pinch detection when thumb and index tips are close."""
        landmarks = [Point(0.5, 0.7)] * 21
        landmarks[0] = Point(0.5, 0.8)  # Wrist

        # Thumb tip and index tip very close
        landmarks[3] = Point(0.42, 0.55)
        landmarks[4] = Point(0.45, 0.50)  # Thumb tip
        landmarks[6] = Point(0.44, 0.52)
        landmarks[8] = Point(0.46, 0.51)  # Index tip (close to thumb)

        # Other fingers curled
        landmarks[10] = Point(0.50, 0.53)
        landmarks[12] = Point(0.50, 0.63)
        landmarks[14] = Point(0.56, 0.55)
        landmarks[16] = Point(0.56, 0.65)
        landmarks[18] = Point(0.62, 0.57)
        landmarks[20] = Point(0.62, 0.67)

        from src.vision.hands import compute_finger_states

        states = compute_finger_states(landmarks)
        hand = HandData(
            landmarks=landmarks, handedness="Right", confidence=0.9, finger_extended=states
        )
        result = self.recognizer.classify(hand)
        assert result.gesture == GestureType.PINCH

    def test_result_contains_hand_data(self, hand_data_open_palm):
        result = self.recognizer.classify(hand_data_open_palm)
        assert result.hand_data is hand_data_open_palm
        assert result.handedness == "Right"

    def test_none_gesture_for_ambiguous(self):
        """Test that ambiguous poses return NONE."""
        landmarks = [Point(0.5, 0.7)] * 21
        landmarks[0] = Point(0.5, 0.8)

        # Three fingers extended (no matching gesture)
        landmarks[6] = Point(0.44, 0.52)
        landmarks[8] = Point(0.44, 0.32)  # Index up
        landmarks[10] = Point(0.50, 0.50)
        landmarks[12] = Point(0.50, 0.30)  # Middle up
        landmarks[14] = Point(0.56, 0.52)
        landmarks[16] = Point(0.56, 0.32)  # Ring up
        landmarks[18] = Point(0.62, 0.57)
        landmarks[20] = Point(0.62, 0.67)  # Pinky curled

        # Thumb curled
        landmarks[3] = Point(0.43, 0.73)
        landmarks[4] = Point(0.44, 0.74)

        from src.vision.hands import compute_finger_states

        states = compute_finger_states(landmarks)
        hand = HandData(
            landmarks=landmarks, handedness="Right", confidence=0.9, finger_extended=states
        )
        result = self.recognizer.classify(hand)
        assert result.gesture == GestureType.NONE
