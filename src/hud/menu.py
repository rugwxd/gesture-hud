"""Gesture-controlled HUD mode switching menu."""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.gestures.tracker import GestureEvent
from src.hud.widgets import HUDMode, HUDState, HUDWidget

# Mode cycle order
MODE_ORDER = [HUDMode.COMBAT, HUDMode.SCAN, HUDMode.NAVIGATION]

MODE_LABELS = {
    HUDMode.COMBAT: "COMBAT",
    HUDMode.SCAN: "SCAN",
    HUDMode.NAVIGATION: "NAV",
}

MODE_COLORS = {
    HUDMode.COMBAT: (0, 100, 255),  # Orange-red
    HUDMode.SCAN: (0, 255, 200),  # Cyan
    HUDMode.NAVIGATION: (255, 200, 0),  # Blue-ish
}


class ModeMenu(HUDWidget):
    """Displays current HUD mode and handles mode switching via swipe gestures.

    Shows:
    - Current mode indicator at top center
    - Mode transition animation on switch
    - Mode name with icon styling
    """

    name = "mode_menu"
    active_modes = {HUDMode.COMBAT, HUDMode.SCAN, HUDMode.NAVIGATION}

    def __init__(self) -> None:
        super().__init__()
        self._current_mode = HUDMode.COMBAT
        self._transition_start = 0.0
        self._transitioning = False
        self._transition_from = HUDMode.COMBAT

    @property
    def current_mode(self) -> HUDMode:
        return self._current_mode

    def update(self, state: HUDState) -> None:
        if state.gesture_state is None:
            return

        event = state.gesture_state.event

        # Swipe left/right to cycle modes
        if event == GestureEvent.SWIPE_LEFT:
            self._switch_mode(-1)
        elif event == GestureEvent.SWIPE_RIGHT:
            self._switch_mode(1)

        # Update state mode
        state.mode = self._current_mode

        # Fade out transition
        if self._transitioning:
            if time.time() - self._transition_start > 1.0:
                self._transitioning = False

    def _switch_mode(self, direction: int) -> None:
        """Cycle to the next/previous mode."""
        current_idx = MODE_ORDER.index(self._current_mode)
        new_idx = (current_idx + direction) % len(MODE_ORDER)
        self._transition_from = self._current_mode
        self._current_mode = MODE_ORDER[new_idx]
        self._transitioning = True
        self._transition_start = time.time()

    def render(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        height, width = overlay.shape[:2]
        mode_label = MODE_LABELS[self._current_mode]
        mode_color = MODE_COLORS[self._current_mode]

        # Mode indicator at top center
        text_size = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = (width - text_size[0]) // 2
        ty = 35

        # Background bar
        bar_w = text_size[0] + 40
        bar_x = (width - bar_w) // 2
        cv2.rectangle(overlay, (bar_x, 8), (bar_x + bar_w, 45), (15, 15, 15), -1)
        cv2.rectangle(overlay, (bar_x, 8), (bar_x + bar_w, 45), mode_color, 1)

        # Diamond indicators for each mode
        for idx, mode in enumerate(MODE_ORDER):
            dx = bar_x + 12 + idx * 14
            dy = 27
            size = 4
            is_active = mode == self._current_mode
            pts = np.array(
                [[dx, dy - size], [dx + size, dy], [dx, dy + size], [dx - size, dy]],
                dtype=np.int32,
            )
            if is_active:
                cv2.fillPoly(overlay, [pts], mode_color)
            else:
                cv2.polylines(overlay, [pts], True, (80, 80, 80), 1)

        # Mode text
        cv2.putText(
            overlay, mode_label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA,
        )

        # Transition flash effect
        if self._transitioning:
            elapsed = time.time() - self._transition_start
            alpha = max(0.0, 1.0 - elapsed / 0.5)
            if alpha > 0:
                flash_color = tuple(int(c * alpha * 0.3) for c in mode_color)
                cv2.line(overlay, (0, 48), (width, 48), flash_color, 2)
                cv2.line(overlay, (0, 49), (width, 49), flash_color, 1)

        # Swipe hint arrows
        arrow_color = (80, 80, 80)
        # Left arrow
        cv2.arrowedLine(
            overlay, (bar_x - 10, 27), (bar_x - 25, 27),
            arrow_color, 1, cv2.LINE_AA, tipLength=0.4,
        )
        # Right arrow
        cv2.arrowedLine(
            overlay, (bar_x + bar_w + 10, 27), (bar_x + bar_w + 25, 27),
            arrow_color, 1, cv2.LINE_AA, tipLength=0.4,
        )

        return overlay
