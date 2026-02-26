"""Holographic object tagging overlay for detected objects."""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.hud.widgets import HUDMode, HUDState, HUDWidget


class ObjectTags(HUDWidget):
    """Renders holographic floating labels for detected objects.

    Features:
    - Corner bracket bounding boxes
    - Floating label with class name and confidence
    - Connection line from label to object
    - Animated scan line on new detections
    """

    name = "object_tags"
    active_modes = {HUDMode.COMBAT, HUDMode.SCAN}

    def __init__(
        self,
        color: tuple[int, int, int] = (0, 255, 200),
        secondary_color: tuple[int, int, int] = (255, 170, 0),
    ) -> None:
        super().__init__()
        self.color = color
        self.secondary = secondary_color
        self._scan_active = False
        self._scan_y = 0
        self._scan_start_time = 0.0

    def update(self, state: HUDState) -> None:
        # Trigger scan animation on open palm
        from src.gestures.recognizer import GestureType

        if state.gesture_state:
            if state.gesture_state.current_gesture == GestureType.OPEN_PALM:
                if not self._scan_active:
                    self._scan_active = True
                    self._scan_start_time = time.time()
                    self._scan_y = 0

        # Progress scan animation
        if self._scan_active:
            elapsed = time.time() - self._scan_start_time
            _, frame_h = state.frame_size
            self._scan_y = int((elapsed / 2.0) * frame_h)  # 2 second sweep
            if self._scan_y > frame_h:
                self._scan_active = False

    def render(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        # Draw scan line
        if self._scan_active:
            height, width = overlay.shape[:2]
            if 0 <= self._scan_y < height:
                # Main scan line
                cv2.line(overlay, (0, self._scan_y), (width, self._scan_y), self.color, 2)
                # Glow above scan line
                for i in range(1, 20):
                    alpha = max(0, 1.0 - i / 20.0)
                    glow_color = tuple(int(c * alpha * 0.3) for c in self.color)
                    y_pos = self._scan_y - i * 2
                    if 0 <= y_pos < height:
                        cv2.line(overlay, (0, y_pos), (width, y_pos), glow_color, 1)

        # Draw detection tags
        for det in state.detections:
            self._render_detection(overlay, det, state)

        return overlay

    def _render_detection(self, overlay: np.ndarray, det: dict, state: HUDState) -> None:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        conf = det["confidence"]

        # Determine color based on mode
        color = self.secondary if state.mode == HUDMode.COMBAT else self.color

        # Corner brackets (not full rectangle)
        bracket_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        corners = [
            ((x1, y1), (x1 + bracket_len, y1), (x1, y1 + bracket_len)),
            ((x2, y1), (x2 - bracket_len, y1), (x2, y1 + bracket_len)),
            ((x1, y2), (x1 + bracket_len, y2), (x1, y2 - bracket_len)),
            ((x2, y2), (x2 - bracket_len, y2), (x2, y2 - bracket_len)),
        ]

        for corner, h_end, v_end in corners:
            cv2.line(overlay, corner, h_end, color, 2, cv2.LINE_AA)
            cv2.line(overlay, corner, v_end, color, 2, cv2.LINE_AA)

        # Label panel
        label_text = f"{label.upper()} {conf:.0%}"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
        label_w = text_size[0] + 16
        label_h = text_size[1] + 12

        # Position label above the box
        label_x = x1
        label_y = y1 - label_h - 8

        # Keep label in frame
        height, width = overlay.shape[:2]
        if label_y < 0:
            label_y = y2 + 8
        if label_x + label_w > width:
            label_x = width - label_w

        # Label background
        cv2.rectangle(
            overlay,
            (label_x, label_y),
            (label_x + label_w, label_y + label_h),
            (15, 15, 15),
            -1,
        )
        cv2.rectangle(
            overlay,
            (label_x, label_y),
            (label_x + label_w, label_y + label_h),
            color,
            1,
        )

        # Label text
        cv2.putText(
            overlay,
            label_text,
            (label_x + 8, label_y + label_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

        # Connection line from label to box
        line_start = (label_x + label_w // 2, label_y + label_h)
        line_end = ((x1 + x2) // 2, y1)
        if label_y > y1:
            line_start = (label_x + label_w // 2, label_y)
            line_end = ((x1 + x2) // 2, y2)
        cv2.line(overlay, line_start, line_end, color, 1, cv2.LINE_AA)

        # Confidence bar inside the bracket area
        bar_y = y2 - 6
        bar_width = x2 - x1 - 8
        if bar_width > 20:
            fill = int(bar_width * conf)
            bg = (30, 30, 30)
            cv2.rectangle(overlay, (x1 + 4, bar_y), (x1 + 4 + bar_width, bar_y + 3), bg, -1)
            cv2.rectangle(overlay, (x1 + 4, bar_y), (x1 + 4 + fill, bar_y + 3), color, -1)
