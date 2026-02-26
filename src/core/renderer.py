"""OpenCV rendering pipeline for the HUD overlay."""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from src.config import HUDConfig

logger = logging.getLogger(__name__)


class HUDRenderer:
    """Rendering utilities for drawing HUD elements with OpenCV.

    Provides methods for drawing sci-fi styled shapes, text, and
    overlay compositing with transparency.
    """

    def __init__(self, config: HUDConfig) -> None:
        self.config = config
        self.primary = tuple(config.color_primary)
        self.secondary = tuple(config.color_secondary)
        self.alert = tuple(config.color_alert)
        self.opacity = config.opacity
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.font_scale

    def create_overlay(self, width: int, height: int) -> np.ndarray:
        """Create a transparent overlay frame."""
        return np.zeros((height, width, 3), dtype=np.uint8)

    def composite(self, frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
        """Blend the overlay onto the frame with transparency."""
        mask = np.any(overlay > 0, axis=2)
        result = frame.copy()
        result[mask] = cv2.addWeighted(frame, 1.0 - self.opacity, overlay, self.opacity, 0)[mask]
        return result

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int] | None = None,
        scale: float | None = None,
        thickness: int = 1,
    ) -> None:
        """Draw text with optional glow effect."""
        color = color or self.primary
        scale = scale or self.font_scale
        cv2.putText(frame, text, position, self.font, scale, color, thickness, cv2.LINE_AA)

    def draw_crosshair(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        size: int = 30,
        color: tuple[int, int, int] | None = None,
        thickness: int = 1,
    ) -> None:
        """Draw a targeting crosshair."""
        color = color or self.primary
        cx, cy = center

        # Horizontal lines with gap
        gap = size // 3
        cv2.line(frame, (cx - size, cy), (cx - gap, cy), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx + gap, cy), (cx + size, cy), color, thickness, cv2.LINE_AA)

        # Vertical lines with gap
        cv2.line(frame, (cx, cy - size), (cx, cy - gap), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + gap), (cx, cy + size), color, thickness, cv2.LINE_AA)

    def draw_corner_brackets(
        self,
        frame: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int] | None = None,
        length: int = 20,
        thickness: int = 1,
    ) -> None:
        """Draw corner bracket decorations around a rectangle."""
        color = color or self.primary
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Top-left
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness, cv2.LINE_AA)

        # Top-right
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness, cv2.LINE_AA)

        # Bottom-left
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness, cv2.LINE_AA)

        # Bottom-right
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness, cv2.LINE_AA)

    def draw_arc(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        start_angle: float,
        end_angle: float,
        color: tuple[int, int, int] | None = None,
        thickness: int = 1,
    ) -> None:
        """Draw an arc (partial circle)."""
        color = color or self.primary
        cv2.ellipse(
            frame,
            center,
            (radius, radius),
            0,
            start_angle,
            end_angle,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def draw_progress_bar(
        self,
        frame: np.ndarray,
        position: tuple[int, int],
        width: int,
        height: int,
        progress: float,
        color: tuple[int, int, int] | None = None,
        bg_color: tuple[int, int, int] = (40, 40, 40),
    ) -> None:
        """Draw a horizontal progress bar."""
        color = color or self.primary
        x, y = position
        progress = max(0.0, min(1.0, progress))

        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)

        # Fill
        fill_width = int(width * progress)
        if fill_width > 0:
            cv2.rectangle(frame, (x, y), (x + fill_width, y + height), color, -1)

        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 1)

    def draw_hexagon(
        self,
        frame: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int] | None = None,
        thickness: int = 1,
    ) -> None:
        """Draw a regular hexagon."""
        color = color or self.primary
        cx, cy = center
        points = []
        for i in range(6):
            angle = math.radians(60 * i - 30)
            px = int(cx + radius * math.cos(angle))
            py = int(cy + radius * math.sin(angle))
            points.append([px, py])

        pts = np.array(points, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
