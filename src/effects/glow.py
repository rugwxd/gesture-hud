"""Bloom/glow post-processing effect for spell visuals."""

from __future__ import annotations

import cv2
import numpy as np


def apply_glow(
    frame: np.ndarray,
    intensity: float = 0.3,
    threshold: int = 180,
    blur_size: int = 21,
) -> np.ndarray:
    """Apply a bloom/glow effect to bright pixels in the frame.

    Extracts bright areas, applies Gaussian blur, and additively
    blends them back onto the original frame. Makes spell particles
    and effects look more magical.

    Args:
        frame: BGR image to process.
        intensity: Blend strength of the glow (0-1).
        threshold: Brightness threshold for glow extraction.
        blur_size: Gaussian blur kernel size (must be odd).

    Returns:
        Frame with glow effect applied.
    """
    if intensity <= 0:
        return frame

    # Extract bright areas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Apply mask to get bright pixels only
    bright = cv2.bitwise_and(frame, frame, mask=bright_mask)

    # Blur the bright areas
    blurred = cv2.GaussianBlur(bright, (blur_size, blur_size), 0)

    # Additive blend
    result = cv2.addWeighted(frame, 1.0, blurred, intensity, 0)

    return np.clip(result, 0, 255).astype(np.uint8)
