"""Gesture recognition from hand landmarks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

from src.vision.hands import Finger, HandData

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Recognized gesture types."""

    NONE = auto()
    FIST = auto()
    OPEN_PALM = auto()
    POINT = auto()
    PINCH = auto()
    THUMBS_UP = auto()
    PEACE = auto()


@dataclass
class GestureResult:
    """Result of gesture classification for a single hand."""

    gesture: GestureType
    confidence: float
    hand_data: HandData

    @property
    def handedness(self) -> str:
        return self.hand_data.handedness


class GestureRecognizer:
    """Rule-based gesture classifier using finger states and landmark geometry.

    Classifies gestures based on which fingers are extended and the
    spatial relationships between key landmarks.
    """

    def classify(self, hand: HandData) -> GestureResult:
        """Classify the current hand pose into a gesture.

        Args:
            hand: Processed hand tracking data.

        Returns:
            GestureResult with the detected gesture type.
        """
        ext = hand.finger_extended
        num_extended = hand.num_fingers_extended

        # Fist: no fingers extended
        if num_extended == 0:
            return GestureResult(gesture=GestureType.FIST, confidence=0.9, hand_data=hand)

        # Open palm: all 5 fingers extended
        if num_extended == 5:
            return GestureResult(gesture=GestureType.OPEN_PALM, confidence=0.9, hand_data=hand)

        # Pinch: thumb and index tips close together
        if self._is_pinch(hand):
            return GestureResult(gesture=GestureType.PINCH, confidence=0.85, hand_data=hand)

        # Thumbs up: only thumb extended, hand oriented vertically
        if ext.get(Finger.THUMB, False) and num_extended == 1:
            if self._is_thumb_up_orientation(hand):
                return GestureResult(
                    gesture=GestureType.THUMBS_UP, confidence=0.85, hand_data=hand
                )

        # Point: only index finger extended
        if ext.get(Finger.INDEX, False) and num_extended == 1:
            return GestureResult(gesture=GestureType.POINT, confidence=0.9, hand_data=hand)

        # Point variant: thumb + index extended (natural pointing)
        if (
            ext.get(Finger.INDEX, False)
            and ext.get(Finger.THUMB, False)
            and num_extended == 2
        ):
            return GestureResult(gesture=GestureType.POINT, confidence=0.8, hand_data=hand)

        # Peace sign: index + middle extended
        if (
            ext.get(Finger.INDEX, False)
            and ext.get(Finger.MIDDLE, False)
            and num_extended == 2
        ):
            return GestureResult(gesture=GestureType.PEACE, confidence=0.85, hand_data=hand)

        return GestureResult(gesture=GestureType.NONE, confidence=0.5, hand_data=hand)

    def _is_pinch(self, hand: HandData) -> bool:
        """Check if thumb and index fingertips are close together."""
        distance = hand.thumb_tip.distance_to(hand.index_tip)
        # Pinch threshold: distance less than ~5% of frame
        return distance < 0.05

    def _is_thumb_up_orientation(self, hand: HandData) -> bool:
        """Check if the thumb is pointing upward (hand vertical)."""
        thumb_tip = hand.thumb_tip
        wrist = hand.wrist
        # Thumb should be significantly above the wrist
        return (wrist.y - thumb_tip.y) > 0.08
