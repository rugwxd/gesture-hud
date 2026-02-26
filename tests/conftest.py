"""Shared test fixtures for the Gesture HUD test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import (
    CameraConfig,
    DetectionConfig,
    EffectsConfig,
    GesturesConfig,
    HandsConfig,
    HUDConfig,
    LoggingConfig,
    RecordingConfig,
    Settings,
)
from src.vision.hands import HandData, Point


@pytest.fixture
def test_settings():
    """Create test settings with sensible defaults."""
    return Settings(
        camera=CameraConfig(device_id=0, width=640, height=480, fps=30),
        hands=HandsConfig(max_hands=2, min_detection_confidence=0.7),
        detection=DetectionConfig(model="yolov8n.pt", confidence_threshold=0.5),
        gestures=GesturesConfig(swipe_threshold=0.15, hold_duration=0.5, debounce_frames=5),
        hud=HUDConfig(color_primary=[0, 255, 200]),
        effects=EffectsConfig(glow_enabled=True, scanlines_enabled=True),
        recording=RecordingConfig(output_dir="data/recordings"),
        logging=LoggingConfig(level="WARNING"),
    )


@pytest.fixture
def sample_frame():
    """Create a sample BGR frame (640x480)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def bright_frame():
    """Create a frame with bright areas for testing effects."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add bright rectangle
    frame[100:200, 200:400] = (200, 220, 240)
    # Add bright circle area
    frame[300:350, 300:350] = (255, 255, 255)
    return frame


@pytest.fixture
def sample_landmarks_open_palm():
    """Create landmarks for an open palm gesture (all fingers extended)."""
    # Simplified: all tips above their PIP joints, thumb tip far from wrist
    landmarks = [Point(0.5, 0.7)] * 21  # Default positions

    # Wrist
    landmarks[0] = Point(0.5, 0.8)

    # Thumb: tip far from wrist
    landmarks[1] = Point(0.42, 0.72)
    landmarks[2] = Point(0.38, 0.65)
    landmarks[3] = Point(0.35, 0.58)
    landmarks[4] = Point(0.32, 0.50)  # Tip

    # Index: tip above PIP
    landmarks[5] = Point(0.44, 0.62)
    landmarks[6] = Point(0.44, 0.52)  # PIP
    landmarks[7] = Point(0.44, 0.42)
    landmarks[8] = Point(0.44, 0.32)  # Tip (above PIP)

    # Middle: tip above PIP
    landmarks[9] = Point(0.50, 0.60)
    landmarks[10] = Point(0.50, 0.50)  # PIP
    landmarks[11] = Point(0.50, 0.40)
    landmarks[12] = Point(0.50, 0.30)  # Tip

    # Ring: tip above PIP
    landmarks[13] = Point(0.56, 0.62)
    landmarks[14] = Point(0.56, 0.52)  # PIP
    landmarks[15] = Point(0.56, 0.42)
    landmarks[16] = Point(0.56, 0.32)  # Tip

    # Pinky: tip above PIP
    landmarks[17] = Point(0.62, 0.65)
    landmarks[18] = Point(0.62, 0.55)  # PIP
    landmarks[19] = Point(0.62, 0.45)
    landmarks[20] = Point(0.62, 0.35)  # Tip

    return landmarks


@pytest.fixture
def sample_landmarks_fist():
    """Create landmarks for a fist gesture (no fingers extended)."""
    landmarks = [Point(0.5, 0.7)] * 21

    # Wrist
    landmarks[0] = Point(0.5, 0.8)

    # Thumb: tip close to wrist (curled)
    landmarks[1] = Point(0.45, 0.75)
    landmarks[2] = Point(0.42, 0.72)
    landmarks[3] = Point(0.43, 0.73)  # IP
    landmarks[4] = Point(0.44, 0.74)  # Tip close to IP

    # Index: tip below PIP (curled)
    landmarks[5] = Point(0.44, 0.65)
    landmarks[6] = Point(0.44, 0.55)  # PIP
    landmarks[7] = Point(0.44, 0.60)
    landmarks[8] = Point(0.44, 0.65)  # Tip below PIP

    # Middle: curled
    landmarks[9] = Point(0.50, 0.63)
    landmarks[10] = Point(0.50, 0.53)  # PIP
    landmarks[11] = Point(0.50, 0.58)
    landmarks[12] = Point(0.50, 0.63)  # Tip below PIP

    # Ring: curled
    landmarks[13] = Point(0.56, 0.65)
    landmarks[14] = Point(0.56, 0.55)  # PIP
    landmarks[15] = Point(0.56, 0.60)
    landmarks[16] = Point(0.56, 0.65)  # Tip below PIP

    # Pinky: curled
    landmarks[17] = Point(0.62, 0.67)
    landmarks[18] = Point(0.62, 0.57)  # PIP
    landmarks[19] = Point(0.62, 0.62)
    landmarks[20] = Point(0.62, 0.67)  # Tip below PIP

    return landmarks


@pytest.fixture
def sample_landmarks_point():
    """Create landmarks for a pointing gesture (only index extended)."""
    landmarks = [Point(0.5, 0.7)] * 21

    # Wrist
    landmarks[0] = Point(0.5, 0.8)

    # Thumb: curled
    landmarks[1] = Point(0.45, 0.75)
    landmarks[2] = Point(0.42, 0.72)
    landmarks[3] = Point(0.43, 0.73)
    landmarks[4] = Point(0.44, 0.74)

    # Index: extended (tip above PIP)
    landmarks[5] = Point(0.44, 0.62)
    landmarks[6] = Point(0.44, 0.52)  # PIP
    landmarks[7] = Point(0.44, 0.42)
    landmarks[8] = Point(0.44, 0.32)  # Tip above PIP

    # Middle: curled
    landmarks[9] = Point(0.50, 0.63)
    landmarks[10] = Point(0.50, 0.53)
    landmarks[11] = Point(0.50, 0.58)
    landmarks[12] = Point(0.50, 0.63)

    # Ring: curled
    landmarks[13] = Point(0.56, 0.65)
    landmarks[14] = Point(0.56, 0.55)
    landmarks[15] = Point(0.56, 0.60)
    landmarks[16] = Point(0.56, 0.65)

    # Pinky: curled
    landmarks[17] = Point(0.62, 0.67)
    landmarks[18] = Point(0.62, 0.57)
    landmarks[19] = Point(0.62, 0.62)
    landmarks[20] = Point(0.62, 0.67)

    return landmarks


@pytest.fixture
def hand_data_open_palm(sample_landmarks_open_palm):
    """HandData for open palm gesture."""
    from src.vision.hands import compute_finger_states

    states = compute_finger_states(sample_landmarks_open_palm)
    return HandData(
        landmarks=sample_landmarks_open_palm,
        handedness="Right",
        confidence=0.95,
        finger_extended=states,
    )


@pytest.fixture
def hand_data_fist(sample_landmarks_fist):
    """HandData for fist gesture."""
    from src.vision.hands import compute_finger_states

    states = compute_finger_states(sample_landmarks_fist)
    return HandData(
        landmarks=sample_landmarks_fist,
        handedness="Right",
        confidence=0.95,
        finger_extended=states,
    )


@pytest.fixture
def hand_data_point(sample_landmarks_point):
    """HandData for pointing gesture."""
    from src.vision.hands import compute_finger_states

    states = compute_finger_states(sample_landmarks_point)
    return HandData(
        landmarks=sample_landmarks_point,
        handedness="Right",
        confidence=0.95,
        finger_extended=states,
    )


@pytest.fixture
def sample_detections():
    """Create sample detection results."""
    return [
        {
            "x1": 100,
            "y1": 100,
            "x2": 250,
            "y2": 300,
            "confidence": 0.92,
            "class_id": 0,
            "label": "person",
        },
        {
            "x1": 400,
            "y1": 150,
            "x2": 550,
            "y2": 350,
            "confidence": 0.78,
            "class_id": 67,
            "label": "cell phone",
        },
    ]
