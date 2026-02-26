"""Bloom/glow post-processing effect."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import EffectsConfig


class GlowEffect:
    """Adds a bloom/glow effect to bright pixels in the frame.

    Extracts bright areas, applies Gaussian blur, and additively
    blends them back onto the original frame.
    """

    def __init__(self, config: EffectsConfig) -> None:
        self.enabled = config.glow_enabled
        self.intensity = config.glow_intensity

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply glow effect to the frame."""
        if not self.enabled or self.intensity <= 0:
            return frame

        # Extract bright areas
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Apply mask to get bright pixels only
        bright = cv2.bitwise_and(frame, frame, mask=bright_mask)

        # Blur the bright areas
        blurred = cv2.GaussianBlur(bright, (21, 21), 0)

        # Additive blend
        result = cv2.addWeighted(frame, 1.0, blurred, self.intensity, 0)

        return np.clip(result, 0, 255).astype(np.uint8)
