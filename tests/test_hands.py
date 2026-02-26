"""Tests for hand tracking module."""

from __future__ import annotations

from src.vision.hands import (
    Finger,
    HandData,
    Point,
    compute_finger_states,
)


class TestPoint:
    def test_creation(self):
        p = Point(0.5, 0.3)
        assert p.x == 0.5
        assert p.y == 0.3

    def test_pixel_conversion(self):
        p = Point(0.5, 0.25)
        px, py = p.pixel(640, 480)
        assert px == 320
        assert py == 120

    def test_distance(self):
        p1 = Point(0.0, 0.0)
        p2 = Point(0.3, 0.4)
        assert abs(p1.distance_to(p2) - 0.5) < 0.001

    def test_distance_to_self(self):
        p = Point(0.5, 0.5)
        assert p.distance_to(p) == 0.0


class TestHandData:
    def test_properties(self, sample_landmarks_open_palm):
        hand = HandData(
            landmarks=sample_landmarks_open_palm,
            handedness="Right",
            confidence=0.95,
        )
        assert hand.handedness == "Right"
        assert hand.confidence == 0.95
        assert hand.wrist == sample_landmarks_open_palm[0]
        assert hand.index_tip == sample_landmarks_open_palm[8]
        assert hand.thumb_tip == sample_landmarks_open_palm[4]

    def test_center(self, sample_landmarks_open_palm):
        hand = HandData(
            landmarks=sample_landmarks_open_palm,
            handedness="Right",
            confidence=0.95,
        )
        center = hand.center
        assert 0.0 <= center.x <= 1.0
        assert 0.0 <= center.y <= 1.0

    def test_num_fingers_extended(self, hand_data_open_palm, hand_data_fist):
        assert hand_data_open_palm.num_fingers_extended == 5
        assert hand_data_fist.num_fingers_extended == 0


class TestFingerStates:
    def test_open_palm(self, sample_landmarks_open_palm):
        states = compute_finger_states(sample_landmarks_open_palm)
        assert states[Finger.INDEX] is True
        assert states[Finger.MIDDLE] is True
        assert states[Finger.RING] is True
        assert states[Finger.PINKY] is True
        assert states[Finger.THUMB] is True

    def test_fist(self, sample_landmarks_fist):
        states = compute_finger_states(sample_landmarks_fist)
        assert states[Finger.INDEX] is False
        assert states[Finger.MIDDLE] is False
        assert states[Finger.RING] is False
        assert states[Finger.PINKY] is False

    def test_point(self, sample_landmarks_point):
        states = compute_finger_states(sample_landmarks_point)
        assert states[Finger.INDEX] is True
        assert states[Finger.MIDDLE] is False
        assert states[Finger.RING] is False
        assert states[Finger.PINKY] is False
