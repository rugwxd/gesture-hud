"""CRT scanline overlay effect."""

from __future__ import annotations

import numpy as np

from src.config import EffectsConfig


class ScanlineEffect:
    """Overlays semi-transparent horizontal scanlines for a CRT/holographic look."""

    def __init__(self, config: EffectsConfig) -> None:
        self.enabled = config.scanlines_enabled
        self.gap = config.scanline_gap
        self._mask: np.ndarray | None = None
        self._last_shape: tuple[int, ...] = ()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply scanline effect to the frame."""
        if not self.enabled or self.gap <= 0:
            return frame

        # Cache the scanline mask for performance
        if frame.shape != self._last_shape:
            self._build_mask(frame.shape)

        # Darken scanline rows
        result = frame.copy()
        result[self._mask] = (result[self._mask] * 0.7).astype(np.uint8)

        return result

    def _build_mask(self, shape: tuple[int, ...]) -> None:
        """Build the scanline mask for the given frame shape."""
        height = shape[0]
        self._mask = np.zeros(height, dtype=bool)
        self._mask[:: self.gap] = True
        self._last_shape = shape
