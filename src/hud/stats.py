"""System stats HUD panel showing FPS, CPU, RAM, and time."""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from src.hud.widgets import HUDMode, HUDState, HUDWidget

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class StatsPanel(HUDWidget):
    """Displays real-time system stats in a sci-fi styled panel.

    Shows:
    - FPS counter
    - CPU usage percentage
    - RAM usage percentage
    - Current time
    - Frame counter
    """

    name = "stats"
    active_modes = {HUDMode.COMBAT, HUDMode.SCAN, HUDMode.NAVIGATION}

    def __init__(
        self,
        color: tuple[int, int, int] = (0, 255, 200),
        position: str = "top_right",
    ) -> None:
        super().__init__()
        self.color = color
        self.position = position
        self._fps_history: deque[float] = deque(maxlen=30)
        self._last_time = time.time()
        self._cpu_percent = 0.0
        self._ram_percent = 0.0
        self._update_counter = 0

    def update(self, state: HUDState) -> None:
        # FPS calculation
        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            self._fps_history.append(1.0 / dt)
        self._last_time = now

        # Update system stats every 30 frames (expensive call)
        self._update_counter += 1
        if self._update_counter % 30 == 0 and HAS_PSUTIL:
            self._cpu_percent = psutil.cpu_percent(interval=0)
            self._ram_percent = psutil.virtual_memory().percent

    def render(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        height, width = overlay.shape[:2]
        panel_w = 220
        panel_h = 160

        # Position the panel
        if self.position == "top_right":
            px = width - panel_w - 20
            py = 20
        elif self.position == "top_left":
            px = 20
            py = 20
        else:
            px = width - panel_w - 20
            py = 20

        # Panel background
        bg = overlay[py : py + panel_h, px : px + panel_w]
        bg[:] = (15, 15, 15)

        # Border
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), self.color, 1)

        # Title bar
        cv2.line(overlay, (px, py + 22), (px + panel_w, py + 22), self.color, 1)
        cv2.putText(
            overlay, "SYSTEM STATUS", (px + 10, py + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.color, 1, cv2.LINE_AA,
        )

        # Stats
        avg_fps = sum(self._fps_history) / max(len(self._fps_history), 1)
        stats = [
            (f"FPS: {avg_fps:.1f}", self.color),
            (f"CPU: {self._cpu_percent:.1f}%", self._get_load_color(self._cpu_percent)),
            (f"RAM: {self._ram_percent:.1f}%", self._get_load_color(self._ram_percent)),
            (f"TIME: {datetime.now().strftime('%H:%M:%S')}", self.color),
            (f"FRAME: {state.frame_number}", self.color),
        ]

        for idx, (text, color) in enumerate(stats):
            y_pos = py + 42 + idx * 24
            cv2.putText(
                overlay, text, (px + 12, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
            )

        return overlay

    def _get_load_color(self, percent: float) -> tuple[int, int, int]:
        """Get color based on load percentage."""
        if percent < 50:
            return self.color
        elif percent < 80:
            return (0, 200, 255)  # Yellow-ish
        else:
            return (0, 80, 255)  # Red-ish
