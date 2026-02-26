"""Targeting reticle HUD widget that follows hand position."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from src.hud.widgets import HUDMode, HUDState, HUDWidget


class TargetingReticle(HUDWidget):
    """Animated targeting reticle that tracks the index finger position.

    Features:
    - Crosshair with rotating outer ring
    - Distance readout from center
    - Lock-on animation when fist gesture is held
    """

    name = "targeting"
    active_modes = {HUDMode.COMBAT, HUDMode.SCAN}

    def __init__(self, color: tuple[int, int, int] = (0, 255, 200)) -> None:
        super().__init__()
        self.color = color
        self._target_x = 0
        self._target_y = 0
        self._smooth_x = 0.0
        self._smooth_y = 0.0
        self._locked = False
        self._rotation_angle = 0.0
        self._smoothing = 0.3

    def update(self, state: HUDState) -> None:
        width, height = state.frame_size

        if state.gesture_state and state.gesture_state.index_tip:
            tip = state.gesture_state.index_tip
            self._target_x = int(tip.x * width)
            self._target_y = int(tip.y * height)

        # Smooth movement
        self._smooth_x += (self._target_x - self._smooth_x) * self._smoothing
        self._smooth_y += (self._target_y - self._smooth_y) * self._smoothing

        # Check for lock-on (fist hold)
        from src.gestures.recognizer import GestureType

        if state.gesture_state:
            self._locked = (
                state.gesture_state.current_gesture == GestureType.FIST
                and state.gesture_state.is_holding
            )

        # Rotate outer ring
        self._rotation_angle += 2.0
        if self._rotation_angle >= 360:
            self._rotation_angle -= 360

    def render(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        cx = int(self._smooth_x)
        cy = int(self._smooth_y)

        if cx == 0 and cy == 0:
            return overlay

        color = (0, 100, 255) if self._locked else self.color
        size = 40 if self._locked else 30

        # Outer rotating ring segments
        for i in range(4):
            start = self._rotation_angle + i * 90
            end = start + 60
            cv2.ellipse(
                overlay, (cx, cy), (size + 10, size + 10),
                0, start, end, color, 1, cv2.LINE_AA,
            )

        # Inner crosshair
        gap = size // 3
        cv2.line(overlay, (cx - size, cy), (cx - gap, cy), color, 1, cv2.LINE_AA)
        cv2.line(overlay, (cx + gap, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
        cv2.line(overlay, (cx, cy - size), (cx, cy - gap), color, 1, cv2.LINE_AA)
        cv2.line(overlay, (cx, cy + gap), (cx, cy + size), color, 1, cv2.LINE_AA)

        # Center dot
        cv2.circle(overlay, (cx, cy), 3, color, -1, cv2.LINE_AA)

        # Corner ticks
        tick = size + 5
        tick_len = 8
        for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            tx = cx + dx * tick
            ty = cy + dy * tick
            cv2.line(overlay, (tx, ty), (tx + dx * tick_len, ty), color, 1, cv2.LINE_AA)
            cv2.line(overlay, (tx, ty), (tx, ty + dy * tick_len), color, 1, cv2.LINE_AA)

        # Coordinate readout
        width, height = state.frame_size
        norm_x = cx / max(width, 1)
        norm_y = cy / max(height, 1)
        coord_text = f"({norm_x:.2f}, {norm_y:.2f})"
        cv2.putText(
            overlay, coord_text, (cx + size + 15, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

        # Lock-on indicator
        if self._locked:
            cv2.putText(
                overlay, "LOCKED", (cx + size + 15, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1, cv2.LINE_AA,
            )
            # Pulsing outer ring
            pulse = int(5 * math.sin(time.time() * 8))
            cv2.circle(overlay, (cx, cy), size + 20 + pulse, (0, 100, 255), 1, cv2.LINE_AA)

        return overlay
