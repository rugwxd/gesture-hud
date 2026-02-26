"""Holographic visual effects — color shifting, flicker, and distortion."""

from __future__ import annotations

import random
import time

import numpy as np

from src.config import EffectsConfig


class HolographicEffect:
    """Applies holographic-style visual effects.

    Features:
    - Subtle color channel shifting
    - Random flicker (brightness variation)
    - Chromatic aberration on edges
    """

    def __init__(self, config: EffectsConfig) -> None:
        self.enabled = config.holographic_flicker
        self.intensity = config.flicker_intensity
        self._last_flicker = 0.0
        self._flicker_value = 1.0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply holographic effects to the frame."""
        if not self.enabled or self.intensity <= 0:
            return frame

        result = frame.copy()

        # Random flicker
        now = time.time()
        if now - self._last_flicker > 0.05:  # 20 Hz flicker check
            self._flicker_value = 1.0 + random.uniform(-self.intensity, self.intensity)
            self._last_flicker = now

        result = np.clip(result * self._flicker_value, 0, 255).astype(np.uint8)

        # Subtle color channel shift (chromatic aberration)
        if random.random() < 0.1:  # 10% chance per frame
            shift = random.randint(1, 2)
            height, width = result.shape[:2]
            # Shift blue channel slightly
            blue = result[:, :, 0].copy()
            result[:, shift:, 0] = blue[:, :width - shift]

        return result


class EffectsPipeline:
    """Chains all post-processing effects together."""

    def __init__(self, config: EffectsConfig) -> None:
        from src.effects.glow import GlowEffect
        from src.effects.scanlines import ScanlineEffect

        self.glow = GlowEffect(config)
        self.scanlines = ScanlineEffect(config)
        self.holographic = HolographicEffect(config)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all effects in sequence."""
        frame = self.glow.apply(frame)
        frame = self.scanlines.apply(frame)
        frame = self.holographic.apply(frame)
        return frame
