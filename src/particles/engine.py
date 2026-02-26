"""Particle engine — manages particle lifecycle, physics, and rendering."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pre-computed constants for blending
_ALPHA_SCALE = 1.0 / 255.0


@dataclass
class Particle:
    """A single particle with position, velocity, appearance, and lifetime."""

    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    color: tuple[int, int, int] = (0, 255, 200)
    size: float = 3.0
    lifetime: float = 1.0
    age: float = 0.0
    alpha: float = 1.0
    decay_rate: float = 1.0
    size_decay: float = 0.0
    gravity: float = 0.0
    drag: float = 0.0
    shape: str = "circle"  # "circle", "line", "spark"

    @property
    def alive(self) -> bool:
        """Whether this particle is still active."""
        return self.age < self.lifetime and self.alpha > 0.01 and self.size > 0.5

    @property
    def life_ratio(self) -> float:
        """Normalized age (0=born, 1=dead)."""
        return min(self.age / self.lifetime, 1.0) if self.lifetime > 0 else 1.0

    def update(self, dt: float) -> None:
        """Advance particle physics by dt seconds."""
        # Apply drag
        if self.drag > 0:
            drag_factor = max(0, 1.0 - self.drag * dt)
            self.vx *= drag_factor
            self.vy *= drag_factor

        # Apply gravity
        self.vy += self.gravity * dt

        # Apply acceleration
        self.vx += self.ax * dt
        self.vy += self.ay * dt

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Age and decay
        self.age += dt
        self.alpha = max(0, 1.0 - (self.age / self.lifetime) * self.decay_rate)
        self.size = max(0, self.size - self.size_decay * dt)


class ParticleEngine:
    """Manages a collection of particles with batch update and render.

    Optimized for rendering 500+ particles at 30fps using direct
    numpy array manipulation and alpha blending.
    """

    def __init__(self, max_particles: int = 2000) -> None:
        self.max_particles = max_particles
        self._particles: list[Particle] = []

    @property
    def count(self) -> int:
        """Number of active particles."""
        return len(self._particles)

    def emit(self, particles: list[Particle]) -> None:
        """Add particles to the engine."""
        space = self.max_particles - len(self._particles)
        if space <= 0:
            return
        self._particles.extend(particles[:space])

    def update(self, dt: float) -> None:
        """Update all particles and remove dead ones."""
        for p in self._particles:
            p.update(dt)
        self._particles = [p for p in self._particles if p.alive]

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render all particles onto the frame with alpha blending.

        Args:
            frame: BGR image to draw on (modified in place).

        Returns:
            The frame with particles rendered.
        """
        h, w = frame.shape[:2]

        for p in self._particles:
            px = int(p.x * w)
            py = int(p.y * h)

            # Skip if outside frame bounds
            if px < 0 or px >= w or py < 0 or py >= h:
                continue

            alpha = max(0.0, min(1.0, p.alpha))
            size = max(1, int(p.size))
            color = p.color

            if alpha < 0.05:
                continue

            if p.shape == "circle":
                self._draw_circle(frame, px, py, size, color, alpha)
            elif p.shape == "line":
                self._draw_line(frame, px, py, p.vx, p.vy, size, color, alpha, w, h)
            elif p.shape == "spark":
                self._draw_spark(frame, px, py, size, color, alpha)

        return frame

    def _draw_circle(
        self,
        frame: np.ndarray,
        px: int,
        py: int,
        size: int,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Draw an alpha-blended circle particle."""
        if alpha >= 0.9:
            cv2.circle(frame, (px, py), size, color, -1, cv2.LINE_AA)
            return

        # Alpha blending via overlay
        h, w = frame.shape[:2]
        x1 = max(0, px - size - 1)
        y1 = max(0, py - size - 1)
        x2 = min(w, px + size + 2)
        y2 = min(h, py + size + 2)

        if x1 >= x2 or y1 >= y2:
            return

        roi = frame[y1:y2, x1:x2]
        overlay = roi.copy()
        cv2.circle(overlay, (px - x1, py - y1), size, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    def _draw_line(
        self,
        frame: np.ndarray,
        px: int,
        py: int,
        vx: float,
        vy: float,
        size: int,
        color: tuple[int, int, int],
        alpha: float,
        w: int,
        h: int,
    ) -> None:
        """Draw a velocity-aligned line particle (motion trail)."""
        # Line length proportional to velocity
        scale = 15.0
        ex = int(px - vx * w * scale * 0.01)
        ey = int(py - vy * h * scale * 0.01)
        ex = max(0, min(w - 1, ex))
        ey = max(0, min(h - 1, ey))
        thickness = max(1, size // 2)

        if alpha >= 0.9:
            cv2.line(frame, (px, py), (ex, ey), color, thickness, cv2.LINE_AA)
        else:
            x1 = max(0, min(px, ex) - 2)
            y1 = max(0, min(py, ey) - 2)
            x2 = min(w, max(px, ex) + 3)
            y2 = min(h, max(py, ey) + 3)
            if x1 >= x2 or y1 >= y2:
                return
            roi = frame[y1:y2, x1:x2]
            overlay = roi.copy()
            cv2.line(
                overlay,
                (px - x1, py - y1),
                (ex - x1, ey - y1),
                color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    def _draw_spark(
        self,
        frame: np.ndarray,
        px: int,
        py: int,
        size: int,
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        """Draw a small bright spark (cross shape)."""
        h, w = frame.shape[:2]
        x1 = max(0, px - size)
        y1 = max(0, py - size)
        x2 = min(w - 1, px + size)
        y2 = min(h - 1, py + size)

        if alpha >= 0.9:
            cv2.line(frame, (x1, py), (x2, py), color, 1, cv2.LINE_AA)
            cv2.line(frame, (px, y1), (px, y2), color, 1, cv2.LINE_AA)
        else:
            rx1 = max(0, x1 - 1)
            ry1 = max(0, y1 - 1)
            rx2 = min(w, x2 + 2)
            ry2 = min(h, y2 + 2)
            if rx1 >= rx2 or ry1 >= ry2:
                return
            roi = frame[ry1:ry2, rx1:rx2]
            overlay = roi.copy()
            cv2.line(
                overlay,
                (x1 - rx1, py - ry1),
                (x2 - rx1, py - ry1),
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                overlay,
                (px - rx1, y1 - ry1),
                (px - rx1, y2 - ry1),
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)

    def clear(self) -> None:
        """Remove all particles."""
        self._particles.clear()
