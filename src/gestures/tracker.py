"""Gesture state machine for detecting taps, holds, and swipes."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto

from src.config import GesturesConfig
from src.gestures.recognizer import GestureResult, GestureType
from src.vision.hands import Point

logger = logging.getLogger(__name__)


class GestureEvent(Enum):
    """Events emitted by the gesture state machine."""

    NONE = auto()
    TAP = auto()
    HOLD_START = auto()
    HOLD_END = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()


@dataclass
class GestureState:
    """Current state of the gesture tracking system."""

    current_gesture: GestureType = GestureType.NONE
    event: GestureEvent = GestureEvent.NONE
    gesture_start_time: float = 0.0
    is_holding: bool = False
    hand_center: Point | None = None
    index_tip: Point | None = None


class GestureTracker:
    """Tracks gesture transitions over time and detects composite events.

    Handles debouncing, tap/hold detection, and swipe recognition
    from hand center velocity.
    """

    def __init__(self, config: GesturesConfig) -> None:
        self.config = config
        self._current_gesture = GestureType.NONE
        self._gesture_start_time = 0.0
        self._is_holding = False
        self._debounce_counter = 0
        self._pending_gesture = GestureType.NONE

        # Position history for swipe detection
        self._position_history: deque[tuple[Point, float]] = deque(maxlen=30)

    def update(self, gesture_result: GestureResult | None) -> GestureState:
        """Update the state machine with a new gesture classification.

        Args:
            gesture_result: Latest gesture classification result, or None if no hand detected.

        Returns:
            Current GestureState with any triggered events.
        """
        now = time.time()
        event = GestureEvent.NONE

        if gesture_result is None:
            # No hand detected — reset state
            if self._current_gesture != GestureType.NONE:
                if self._is_holding:
                    event = GestureEvent.HOLD_END
                    self._is_holding = False
                elif self._check_tap(now):
                    event = GestureEvent.TAP
                self._current_gesture = GestureType.NONE
            return GestureState(
                current_gesture=GestureType.NONE,
                event=event,
            )

        new_gesture = gesture_result.gesture
        hand_data = gesture_result.hand_data

        # Track hand center position for swipe detection
        center = hand_data.center
        self._position_history.append((center, now))

        # Check for swipe regardless of gesture type
        swipe_event = self._detect_swipe()
        if swipe_event != GestureEvent.NONE:
            event = swipe_event
            self._position_history.clear()

        # Debounce gesture changes
        if new_gesture != self._current_gesture:
            if new_gesture == self._pending_gesture:
                self._debounce_counter += 1
            else:
                self._pending_gesture = new_gesture
                self._debounce_counter = 1

            if self._debounce_counter >= self.config.debounce_frames:
                # Gesture change confirmed
                if self._is_holding:
                    event = GestureEvent.HOLD_END
                    self._is_holding = False
                elif self._check_tap(now):
                    event = GestureEvent.TAP

                self._current_gesture = new_gesture
                self._gesture_start_time = now
                self._debounce_counter = 0
        else:
            self._debounce_counter = 0
            self._pending_gesture = GestureType.NONE

            # Check for hold
            if not self._is_holding and self._current_gesture != GestureType.NONE:
                duration = now - self._gesture_start_time
                if duration >= self.config.hold_duration:
                    self._is_holding = True
                    if event == GestureEvent.NONE:
                        event = GestureEvent.HOLD_START

        return GestureState(
            current_gesture=self._current_gesture,
            event=event,
            gesture_start_time=self._gesture_start_time,
            is_holding=self._is_holding,
            hand_center=center,
            index_tip=hand_data.index_tip,
        )

    def _check_tap(self, now: float) -> bool:
        """Check if the previous gesture was a tap (short duration)."""
        if self._gesture_start_time == 0:
            return False
        duration = now - self._gesture_start_time
        return 0 < duration < self.config.tap_max_duration

    def _detect_swipe(self) -> GestureEvent:
        """Detect swipe from hand center velocity."""
        if len(self._position_history) < 5:
            return GestureEvent.NONE

        # Compare recent position to position ~0.3s ago
        recent_pos, recent_time = self._position_history[-1]
        oldest_pos, oldest_time = self._position_history[0]

        dt = recent_time - oldest_time
        if dt < 0.1 or dt > 0.5:
            return GestureEvent.NONE

        dx = recent_pos.x - oldest_pos.x
        dy = recent_pos.y - oldest_pos.y

        threshold = self.config.swipe_threshold

        # Horizontal swipe takes priority
        if abs(dx) > threshold and abs(dx) > abs(dy):
            return GestureEvent.SWIPE_RIGHT if dx > 0 else GestureEvent.SWIPE_LEFT

        # Vertical swipe
        if abs(dy) > threshold and abs(dy) > abs(dx):
            return GestureEvent.SWIPE_DOWN if dy > 0 else GestureEvent.SWIPE_UP

        return GestureEvent.NONE

    def reset(self) -> None:
        """Reset the tracker state."""
        self._current_gesture = GestureType.NONE
        self._gesture_start_time = 0.0
        self._is_holding = False
        self._debounce_counter = 0
        self._pending_gesture = GestureType.NONE
        self._position_history.clear()
