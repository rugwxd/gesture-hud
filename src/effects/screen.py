"""Full-screen post-processing effects for spell impacts."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ScreenShake:
    """Camera shake effect that decays over time.

    Applies random x/y pixel offset to the frame, decaying linearly.
    """

    intensity: float = 0.0
    duration: float = 0.0
    elapsed: float = 0.0
    _offset_x: int = 0
    _offset_y: int = 0

    def trigger(self, intensity: float = 15.0, duration: float = 0.3) -> None:
        """Start a screen shake."""
        self.intensity = intensity
        self.duration = duration
        self.elapsed = 0.0

    @property
    def active(self) -> bool:
        return self.elapsed < self.duration and self.intensity > 0

    def update(self, dt: float) -> None:
        """Update shake offset."""
        if not self.active:
            self._offset_x = 0
            self._offset_y = 0
            return

        self.elapsed += dt
        progress = self.elapsed / self.duration
        current_intensity = self.intensity * max(0, 1.0 - progress)

        self._offset_x = int(random.uniform(-current_intensity, current_intensity))
        self._offset_y = int(random.uniform(-current_intensity, current_intensity))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply shake offset to frame via translation."""
        if not self.active or (self._offset_x == 0 and self._offset_y == 0):
            return frame

        h, w = frame.shape[:2]
        matrix = np.float32([[1, 0, self._offset_x], [0, 1, self._offset_y]])
        return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


@dataclass
class ScreenFlash:
    """Additive color flash overlay that fades out.

    Used for spell impacts — white flash, orange fireball glow, etc.
    """

    color: tuple[int, int, int] = (255, 255, 255)
    intensity: float = 0.0
    duration: float = 0.0
    elapsed: float = 0.0

    def trigger(
        self,
        color: tuple[int, int, int] = (255, 255, 255),
        intensity: float = 0.6,
        duration: float = 0.2,
    ) -> None:
        """Start a screen flash."""
        self.color = color
        self.intensity = intensity
        self.duration = duration
        self.elapsed = 0.0

    @property
    def active(self) -> bool:
        return self.elapsed < self.duration and self.intensity > 0

    def update(self, dt: float) -> None:
        """Advance flash timer."""
        if self.active:
            self.elapsed += dt

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply additive color overlay."""
        if not self.active:
            return frame

        progress = self.elapsed / self.duration
        alpha = self.intensity * max(0, 1.0 - progress * progress)

        if alpha < 0.02:
            return frame

        overlay = np.full_like(frame, self.color, dtype=np.uint8)
        return cv2.addWeighted(overlay, alpha, frame, 1.0, 0)


@dataclass
class ChromaticAberration:
    """RGB channel split effect for glitch/impact moments.

    Shifts red and blue channels in opposite directions.
    """

    shift: int = 0
    duration: float = 0.0
    elapsed: float = 0.0

    def trigger(self, shift: int = 5, duration: float = 0.15) -> None:
        """Start chromatic aberration."""
        self.shift = shift
        self.duration = duration
        self.elapsed = 0.0

    @property
    def active(self) -> bool:
        return self.elapsed < self.duration and self.shift > 0

    def update(self, dt: float) -> None:
        """Advance aberration timer."""
        if self.active:
            self.elapsed += dt

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Shift R and B channels in opposite directions."""
        if not self.active:
            return frame

        progress = self.elapsed / self.duration
        current_shift = int(self.shift * max(0, 1.0 - progress))

        if current_shift < 1:
            return frame

        h, w = frame.shape[:2]
        result = frame.copy()

        # Shift blue channel left, red channel right
        # OpenCV is BGR: B=0, G=1, R=2
        if current_shift < w:
            # Blue channel shifts left
            result[:, :w - current_shift, 0] = frame[:, current_shift:, 0]
            # Red channel shifts right
            result[:, current_shift:, 2] = frame[:, :w - current_shift, 2]

        return result


@dataclass
class ScreenEffects:
    """Manages all screen-level post-processing effects."""

    shake: ScreenShake = field(default_factory=ScreenShake)
    flash: ScreenFlash = field(default_factory=ScreenFlash)
    aberration: ChromaticAberration = field(default_factory=ChromaticAberration)

    @property
    def active(self) -> bool:
        """Whether any screen effect is currently active."""
        return self.shake.active or self.flash.active or self.aberration.active

    def update(self, dt: float) -> None:
        """Update all screen effects."""
        self.shake.update(dt)
        self.flash.update(dt)
        self.aberration.update(dt)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply all active screen effects to the frame."""
        frame = self.shake.apply(frame)
        frame = self.flash.apply(frame)
        frame = self.aberration.apply(frame)
        return frame

    def trigger_impact(
        self,
        shake_intensity: float = 10.0,
        flash_color: tuple[int, int, int] = (255, 255, 255),
        flash_intensity: float = 0.4,
        aberration_shift: int = 4,
    ) -> None:
        """Convenience: trigger shake + flash + aberration together."""
        self.shake.trigger(intensity=shake_intensity, duration=0.3)
        self.flash.trigger(
            color=flash_color, intensity=flash_intensity, duration=0.2
        )
        self.aberration.trigger(shift=aberration_shift, duration=0.15)
