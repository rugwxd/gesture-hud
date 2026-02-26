"""MediaPipe hand tracking wrapper with landmark extraction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from src.config import HandsConfig

logger = logging.getLogger(__name__)

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


class HandTracker:
    """MediaPipe hand tracking wrapper.

    Processes frames and returns structured hand data with finger states.
    """

    def __init__(self, config: HandsConfig) -> None:
        self.config = config
        self._hands = None
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize MediaPipe Hands. Lazy init to avoid import at module level."""
        try:
            import mediapipe as mp

            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.config.max_hands,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
            self._initialized = True
            logger.info("Hand tracker initialized (max_hands=%d)", self.config.max_hands)
            return True
        except ImportError:
            logger.error("mediapipe not installed. Run: pip install mediapipe")
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

        # MediaPipe expects RGB
        rgb = cv2_to_rgb(frame)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        hands: list[HandData] = []
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract landmarks as Points
            landmarks = [
                Point(x=lm.x, y=lm.y) for lm in hand_landmarks.landmark
            ]

            # Get handedness
            handedness = "Right"
            confidence = 0.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                classification = results.multi_handedness[idx].classification[0]
                handedness = classification.label
                confidence = classification.score

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
        if self._hands is not None:
            self._hands.close()
            self._hands = None
            self._initialized = False


def cv2_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert BGR frame to RGB."""
    import cv2

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
