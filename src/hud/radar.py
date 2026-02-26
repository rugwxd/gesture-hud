"""Radar minimap HUD widget showing object positions."""

from __future__ import annotations

import math
import time

import cv2
import numpy as np

from src.hud.widgets import HUDMode, HUDState, HUDWidget


class RadarWidget(HUDWidget):
    """Displays a circular radar minimap showing detected object positions.

    Features:
    - Rotating sweep line
    - Blips for detected objects
    - Range rings
    - Cardinal direction markers
    """

    name = "radar"
    active_modes = {HUDMode.COMBAT, HUDMode.SCAN}

    def __init__(
        self,
        color: tuple[int, int, int] = (0, 255, 200),
        radius: int = 80,
    ) -> None:
        super().__init__()
        self.color = color
        self.radius = radius
        self._sweep_angle = 0.0
        self._blips: list[tuple[float, float, float]] = []  # (norm_x, norm_y, age)

    def update(self, state: HUDState) -> None:
        # Rotate sweep
        self._sweep_angle += 3.0
        if self._sweep_angle >= 360:
            self._sweep_angle -= 360

        # Update blips from detections
        now = time.time()
        if state.detections:
            frame_w, frame_h = state.frame_size
            for det in state.detections:
                cx = (det["x1"] + det["x2"]) / 2 / frame_w
                cy = (det["y1"] + det["y2"]) / 2 / frame_h
                self._blips.append((cx, cy, now))

        # Remove old blips (fade after 3 seconds)
        self._blips = [(bx, by, bt) for bx, by, bt in self._blips if now - bt < 3.0]

    def render(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        height, width = overlay.shape[:2]

        # Position: bottom-left
        cx = 30 + self.radius
        cy = height - 30 - self.radius

        # Background circle
        cv2.circle(overlay, (cx, cy), self.radius, (15, 15, 15), -1)
        cv2.circle(overlay, (cx, cy), self.radius, self.color, 1, cv2.LINE_AA)

        # Range rings
        cv2.circle(overlay, (cx, cy), self.radius // 3, self.color, 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 2 * self.radius // 3, self.color, 1, cv2.LINE_AA)

        # Cross lines
        dim_color = tuple(c // 3 for c in self.color)
        cv2.line(
            overlay, (cx - self.radius, cy), (cx + self.radius, cy), dim_color, 1, cv2.LINE_AA
        )
        cv2.line(
            overlay, (cx, cy - self.radius), (cx, cy + self.radius), dim_color, 1, cv2.LINE_AA
        )

        # Cardinal markers
        font = cv2.FONT_HERSHEY_SIMPLEX
        markers = [("N", 0, -1), ("S", 0, 1), ("E", 1, 0), ("W", -1, 0)]
        for label, dx, dy in markers:
            mx = cx + dx * (self.radius + 12)
            my = cy + dy * (self.radius + 12)
            cv2.putText(overlay, label, (mx - 4, my + 4), font, 0.35, self.color, 1, cv2.LINE_AA)

        # Sweep line
        sweep_rad = math.radians(self._sweep_angle)
        sx = int(cx + self.radius * math.cos(sweep_rad))
        sy = int(cy + self.radius * math.sin(sweep_rad))
        cv2.line(overlay, (cx, cy), (sx, sy), self.color, 1, cv2.LINE_AA)

        # Sweep trail (fading arc)
        for i in range(30):
            trail_angle = self._sweep_angle - i * 1.0
            intensity = max(0, 1.0 - i / 30.0)
            trail_color = tuple(int(c * intensity * 0.3) for c in self.color)
            a1 = math.radians(trail_angle)
            p1 = (int(cx + self.radius * math.cos(a1)), int(cy + self.radius * math.sin(a1)))
            cv2.line(overlay, (cx, cy), p1, trail_color, 1)

        # Blips
        now = time.time()
        for bx, by, bt in self._blips:
            age = now - bt
            alpha = max(0.0, 1.0 - age / 3.0)
            # Map normalized frame position to radar position
            rx = int(cx + (bx - 0.5) * 2 * self.radius * 0.8)
            ry = int(cy + (by - 0.5) * 2 * self.radius * 0.8)
            blip_color = tuple(int(c * alpha) for c in self.color)
            cv2.circle(overlay, (rx, ry), 3, blip_color, -1, cv2.LINE_AA)

        # Center dot
        cv2.circle(overlay, (cx, cy), 2, self.color, -1)

        return overlay
