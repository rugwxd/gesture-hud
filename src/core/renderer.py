"""Rendering utilities for the AR Spellcaster overlay."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import Settings
from src.spells.registry import ManaSystem

logger = logging.getLogger(__name__)


class SpellRenderer:
    """Renders HUD elements for the spellcaster: mana bar, spell names,
    hand landmarks, and status indicators.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_mana = settings.spells.show_mana_bar
        self.show_spell = settings.spells.show_spell_name

    def draw_mana_bar(self, frame: np.ndarray, mana: ManaSystem) -> None:
        """Draw mana bar at the bottom of the screen."""
        if not self.show_mana:
            return

        h, w = frame.shape[:2]
        bar_width = int(w * 0.4)
        bar_height = 8
        x = (w - bar_width) // 2
        y = h - 30

        # Background
        cv2.rectangle(
            frame,
            (x - 1, y - 1),
            (x + bar_width + 1, y + bar_height + 1),
            (40, 40, 40),
            -1,
        )

        # Mana fill
        fill_width = int(bar_width * mana.ratio)
        if fill_width > 0:
            # Color transitions from blue to cyan based on mana level
            blue = int(200 + 55 * mana.ratio)
            green = int(100 * mana.ratio)
            color = (blue, green, 0)
            cv2.rectangle(frame, (x, y), (x + fill_width, y + bar_height), color, -1)

        # Border glow
        border_color = (180, 80, 0) if mana.ratio > 0.2 else (0, 0, 200)
        cv2.rectangle(
            frame,
            (x - 1, y - 1),
            (x + bar_width + 1, y + bar_height + 1),
            border_color,
            1,
        )

        # Mana text
        text = f"MANA {int(mana.current_mana)}/{mana.max_mana}"
        text_size = cv2.getTextSize(text, self.font, 0.4, 1)[0]
        tx = x + (bar_width - text_size[0]) // 2
        ty = y - 5
        cv2.putText(frame, text, (tx, ty), self.font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    def draw_spell_name(self, frame: np.ndarray, name: str) -> None:
        """Draw the currently active spell name at top of screen."""
        if not self.show_spell:
            return

        h, w = frame.shape[:2]
        display_name = name.upper().replace("_", " ")
        text_size = cv2.getTextSize(display_name, self.font, 0.8, 2)[0]
        tx = (w - text_size[0]) // 2
        ty = 40

        # Shadow
        cv2.putText(
            frame, display_name, (tx + 1, ty + 1),
            self.font, 0.8, (0, 0, 0), 3, cv2.LINE_AA,
        )
        # Text
        cv2.putText(
            frame, display_name, (tx, ty),
            self.font, 0.8, (0, 200, 255), 2, cv2.LINE_AA,
        )

    def draw_landmarks(self, frame: np.ndarray, hands: list) -> None:
        """Draw subtle hand landmark connections."""
        color = (0, 128, 100)
        height, width = frame.shape[:2]

        for hand in hands:
            points = [
                (int(lm.x * width), int(lm.y * height))
                for lm in hand.landmarks
            ]

            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17),
            ]

            for start, end in connections:
                if start < len(points) and end < len(points):
                    cv2.line(
                        frame, points[start], points[end],
                        color, 1, cv2.LINE_AA,
                    )

            for pt in points:
                cv2.circle(frame, pt, 2, color, -1)
