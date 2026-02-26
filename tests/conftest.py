"""Shared test fixtures for the AR Spellcaster test suite."""

from __future__ import annotations

import numpy as np
import pytest

from src.audio.player import AudioPlayer
from src.config import (
    AudioConfig,
    CameraConfig,
    GesturesConfig,
    HandsConfig,
    LoggingConfig,
    ParticlesConfig,
    RecordingConfig,
    Settings,
    SpellsConfig,
)
from src.effects.screen import ScreenEffects
from src.particles.engine import ParticleEngine
from src.spells.registry import SpellRegistry
from src.vision.hands import HandData, Point


@pytest.fixture
def test_settings():
    """Create test settings with sensible defaults."""
    return Settings(
        camera=CameraConfig(device_id=0, width=640, height=480, fps=30),
        hands=HandsConfig(max_hands=2, min_detection_confidence=0.7),
        gestures=GesturesConfig(
            swipe_threshold=0.08, hold_duration=0.5, debounce_frames=5,
        ),
        particles=ParticlesConfig(max_particles=500, glow_enabled=False),
        spells=SpellsConfig(max_mana=100, mana_regen=8.0),
        audio=AudioConfig(enabled=False),
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
    frame[100:200, 200:400] = (200, 220, 240)
    frame[300:350, 300:350] = (255, 255, 255)
    return frame


@pytest.fixture
def particle_engine():
    """Create a test particle engine."""
    return ParticleEngine(max_particles=500)


@pytest.fixture
def screen_effects():
    """Create a test screen effects manager."""
    return ScreenEffects()


@pytest.fixture
def audio_player():
    """Create a disabled audio player for testing."""
    return AudioPlayer(enabled=False)


@pytest.fixture
def spell_registry(particle_engine, screen_effects, audio_player):
    """Create a test spell registry."""
    return SpellRegistry(
        particles=particle_engine,
        screen_fx=screen_effects,
        audio=audio_player,
        max_mana=100,
        mana_regen=8.0,
    )


@pytest.fixture
def sample_landmarks_open_palm():
    """Create landmarks for an open palm gesture (all fingers extended)."""
    landmarks = [Point(0.5, 0.7)] * 21

    landmarks[0] = Point(0.5, 0.8)  # Wrist
    landmarks[1] = Point(0.42, 0.72)
    landmarks[2] = Point(0.38, 0.65)
    landmarks[3] = Point(0.35, 0.58)
    landmarks[4] = Point(0.32, 0.50)  # Thumb tip
    landmarks[5] = Point(0.44, 0.62)
    landmarks[6] = Point(0.44, 0.52)  # Index PIP
    landmarks[7] = Point(0.44, 0.42)
    landmarks[8] = Point(0.44, 0.32)  # Index tip
    landmarks[9] = Point(0.50, 0.60)
    landmarks[10] = Point(0.50, 0.50)  # Middle PIP
    landmarks[11] = Point(0.50, 0.40)
    landmarks[12] = Point(0.50, 0.30)  # Middle tip
    landmarks[13] = Point(0.56, 0.62)
    landmarks[14] = Point(0.56, 0.52)  # Ring PIP
    landmarks[15] = Point(0.56, 0.42)
    landmarks[16] = Point(0.56, 0.32)  # Ring tip
    landmarks[17] = Point(0.62, 0.65)
    landmarks[18] = Point(0.62, 0.55)  # Pinky PIP
    landmarks[19] = Point(0.62, 0.45)
    landmarks[20] = Point(0.62, 0.35)  # Pinky tip

    return landmarks


@pytest.fixture
def sample_landmarks_fist():
    """Create landmarks for a fist gesture (no fingers extended)."""
    landmarks = [Point(0.5, 0.7)] * 21

    landmarks[0] = Point(0.5, 0.8)
    landmarks[1] = Point(0.45, 0.75)
    landmarks[2] = Point(0.42, 0.72)
    landmarks[3] = Point(0.43, 0.73)
    landmarks[4] = Point(0.44, 0.74)
    landmarks[5] = Point(0.44, 0.65)
    landmarks[6] = Point(0.44, 0.55)
    landmarks[7] = Point(0.44, 0.60)
    landmarks[8] = Point(0.44, 0.65)
    landmarks[9] = Point(0.50, 0.63)
    landmarks[10] = Point(0.50, 0.53)
    landmarks[11] = Point(0.50, 0.58)
    landmarks[12] = Point(0.50, 0.63)
    landmarks[13] = Point(0.56, 0.65)
    landmarks[14] = Point(0.56, 0.55)
    landmarks[15] = Point(0.56, 0.60)
    landmarks[16] = Point(0.56, 0.65)
    landmarks[17] = Point(0.62, 0.67)
    landmarks[18] = Point(0.62, 0.57)
    landmarks[19] = Point(0.62, 0.62)
    landmarks[20] = Point(0.62, 0.67)

    return landmarks


@pytest.fixture
def sample_landmarks_point():
    """Create landmarks for a pointing gesture (only index extended)."""
    landmarks = [Point(0.5, 0.7)] * 21

    landmarks[0] = Point(0.5, 0.8)
    landmarks[1] = Point(0.45, 0.75)
    landmarks[2] = Point(0.42, 0.72)
    landmarks[3] = Point(0.43, 0.73)
    landmarks[4] = Point(0.44, 0.74)
    landmarks[5] = Point(0.44, 0.62)
    landmarks[6] = Point(0.44, 0.52)
    landmarks[7] = Point(0.44, 0.42)
    landmarks[8] = Point(0.44, 0.32)
    landmarks[9] = Point(0.50, 0.63)
    landmarks[10] = Point(0.50, 0.53)
    landmarks[11] = Point(0.50, 0.58)
    landmarks[12] = Point(0.50, 0.63)
    landmarks[13] = Point(0.56, 0.65)
    landmarks[14] = Point(0.56, 0.55)
    landmarks[15] = Point(0.56, 0.60)
    landmarks[16] = Point(0.56, 0.65)
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
