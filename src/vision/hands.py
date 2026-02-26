"""MediaPipe hand tracking wrapper with landmark extraction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np

from src.config import HandsConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MODEL_PATH = PROJECT_ROOT / "data" / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


class Finger(IntEnum):
    """Finger identifiers."""

    THUMB = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    PINKY = 4


# Tip and PIP landmark indices per finger
FINGER_TIPS = {
    Finger.THUMB: THUMB_TIP,
    Finger.INDEX: INDEX_TIP,
    Finger.MIDDLE: MIDDLE_TIP,
    Finger.RING: RING_TIP,
    Finger.PINKY: PINKY_TIP,
}

FINGER_PIPS = {
    Finger.THUMB: THUMB_IP,
    Finger.INDEX: INDEX_PIP,
    Finger.MIDDLE: MIDDLE_PIP,
    Finger.RING: RING_PIP,
    Finger.PINKY: PINKY_PIP,
}

FINGER_MCPS = {
    Finger.THUMB: THUMB_MCP,
    Finger.INDEX: INDEX_MCP,
    Finger.MIDDLE: MIDDLE_MCP,
    Finger.RING: RING_MCP,
    Finger.PINKY: PINKY_MCP,
}


@dataclass
class Point:
    """A 2D point with normalized coordinates (0-1)."""

    x: float
    y: float

    def pixel(self, width: int, height: int) -> tuple[int, int]:
        """Convert to pixel coordinates."""
        return int(self.x * width), int(self.y * height)

    def distance_to(self, other: Point) -> float:
        """Euclidean distance to another point."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class HandData:
    """Processed hand tracking data for a single hand."""

    landmarks: list[Point]
    handedness: str  # "Left" or "Right"
    confidence: float
    finger_extended: dict[Finger, bool] = field(default_factory=dict)

    @property
    def wrist(self) -> Point:
        return self.landmarks[WRIST]

    @property
    def index_tip(self) -> Point:
        return self.landmarks[INDEX_TIP]

    @property
    def thumb_tip(self) -> Point:
        return self.landmarks[THUMB_TIP]

    @property
    def middle_tip(self) -> Point:
        return self.landmarks[MIDDLE_TIP]

    @property
    def center(self) -> Point:
        """Center of the palm (average of MCP joints)."""
        mcp_indices = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        avg_x = sum(self.landmarks[i].x for i in mcp_indices) / len(mcp_indices)
        avg_y = sum(self.landmarks[i].y for i in mcp_indices) / len(mcp_indices)
        return Point(avg_x, avg_y)

    @property
    def num_fingers_extended(self) -> int:
        return sum(1 for ext in self.finger_extended.values() if ext)


def compute_finger_states(landmarks: list[Point]) -> dict[Finger, bool]:
    """Determine which fingers are extended based on landmark positions.

    For thumb: compare tip-to-wrist distance vs IP-to-wrist distance.
    For other fingers: tip is above PIP (lower y = higher on screen).
    """
    states: dict[Finger, bool] = {}
    wrist = landmarks[WRIST]

    # Thumb: tip farther from wrist than IP joint
    thumb_tip_dist = landmarks[THUMB_TIP].distance_to(wrist)
    thumb_ip_dist = landmarks[THUMB_IP].distance_to(wrist)
    states[Finger.THUMB] = thumb_tip_dist > thumb_ip_dist * 1.1

    # Other fingers: tip above PIP (y decreases going up)
    for finger in [Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY]:
        tip_y = landmarks[FINGER_TIPS[finger]].y
        pip_y = landmarks[FINGER_PIPS[finger]].y
        states[finger] = tip_y < pip_y

    return states


def _download_model() -> bool:
    """Download the hand landmarker model if not present."""
    if MODEL_PATH.exists():
        return True

    logger.info("Downloading hand landmarker model...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        import urllib.request

        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        logger.info("Model downloaded to %s", MODEL_PATH)
        return True
    except Exception as exc:
        logger.error("Failed to download model: %s", exc)
        return False


class HandTracker:
    """MediaPipe hand tracking wrapper using the Tasks API.

    Processes frames and returns structured hand data with finger states.
    Uses HandLandmarker from mediapipe.tasks.python.vision.
    """

    def __init__(self, config: HandsConfig) -> None:
        self.config = config
        self._landmarker = None
        self._initialized = False
        self._frame_timestamp_ms = 0

    def initialize(self) -> bool:
        """Initialize MediaPipe HandLandmarker."""
        try:
            if not _download_model():
                return False

            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                HandLandmarker,
                HandLandmarkerOptions,
                RunningMode,
            )

            base_options = BaseOptions(
                model_asset_path=str(MODEL_PATH)
            )
            options = HandLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_hands=self.config.max_hands,
                min_hand_detection_confidence=self.config.min_detection_confidence,
                min_hand_presence_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._landmarker = HandLandmarker.create_from_options(options)
            self._initialized = True
            logger.info(
                "Hand tracker initialized (max_hands=%d)", self.config.max_hands
            )
            return True
        except ImportError:
            logger.error("mediapipe not installed. Run: pip install mediapipe")
            return False
        except Exception as exc:
            logger.error("Failed to initialize hand tracker: %s", exc)
            return False

    def process(self, frame: np.ndarray) -> list[HandData]:
        """Process a frame and return detected hands.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            List of HandData for each detected hand.
        """
        if not self._initialized:
            if not self.initialize():
                return []

        import mediapipe as mp

        # Convert BGR to RGB
        rgb = cv2_to_rgb(frame)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Increment timestamp (must be monotonically increasing for VIDEO mode)
        self._frame_timestamp_ms += 33  # ~30 FPS

        try:
            results = self._landmarker.detect_for_video(
                mp_image, self._frame_timestamp_ms
            )
        except Exception as exc:
            logger.debug("Hand detection error: %s", exc)
            return []

        if not results.hand_landmarks:
            return []

        hands: list[HandData] = []
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            # Extract landmarks as Points
            landmarks = [
                Point(x=lm.x, y=lm.y) for lm in hand_landmarks
            ]

            # Get handedness
            handedness = "Right"
            confidence = 0.0
            if results.handedness and idx < len(results.handedness):
                category = results.handedness[idx][0]
                handedness = category.category_name
                confidence = category.score

            # Compute finger states
            finger_states = compute_finger_states(landmarks)

            hands.append(
                HandData(
                    landmarks=landmarks,
                    handedness=handedness,
                    confidence=confidence,
                    finger_extended=finger_states,
                )
            )

        return hands

    def release(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._initialized = False


def cv2_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB."""
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
