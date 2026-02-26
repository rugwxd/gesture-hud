"""Shield spell — glowing hexagonal bubble around the hand."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from src.effects.screen import ScreenEffects
from src.particles.engine import Particle, ParticleEngine
from src.spells.base import Spell, SpellState


class Shield(Spell):
    """Glowing hexagonal force field that forms around the hand.

    Cast by holding a fist. The shield stays active as long as the
    hold gesture is maintained (tracked via update). Hexagonal
    segments pulse and rotate slowly.
    """

    name = "shield"
    mana_cost = 10
    cooldown = 0.5

    def __init__(self) -> None:
        super().__init__()
        self._x: float = 0.5
        self._y: float = 0.5
        self._radius: float = 0.0
        self._target_radius: float = 0.08
        self._rotation: float = 0.0
        self._pulse_phase: float = 0.0
        self._max_duration: float = 5.0
        self._dismissed: bool = False

    def cast(
        self,
        x: float,
        y: float,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
    ) -> None:
        """Activate shield at hand position."""
        self.origin_x = x
        self.origin_y = y
        self._x = x
        self._y = y
        self.state = SpellState.ACTIVE

        # Activation flash
        screen_fx.flash.trigger(
            color=(200, 150, 0),
            intensity=0.2,
            duration=0.1,
        )

    def update(
        self,
        dt: float,
        particles: ParticleEngine,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Expand shield, track hand, emit barrier particles."""
        self.elapsed += dt

        if self.state == SpellState.DONE:
            return

        if self.state == SpellState.ACTIVE:
            # Track hand position
            if hand_x is not None and hand_y is not None:
                self._x += (hand_x - self._x) * 0.3
                self._y += (hand_y - self._y) * 0.3

            # Expand radius to target
            self._radius += (self._target_radius - self._radius) * 5 * dt

            # Slow rotation
            self._rotation += dt * 0.5

            # Pulse effect
            self._pulse_phase += dt * 4.0

            # Emit particles along the barrier edge
            if random.random() < 0.4:
                angle = random.uniform(0, 2 * math.pi)
                px = self._x + math.cos(angle) * self._radius
                py = self._y + math.sin(angle) * self._radius * 0.7
                particles.emit([
                    Particle(
                        x=px,
                        y=py,
                        vx=math.cos(angle) * 0.01,
                        vy=math.sin(angle) * 0.01,
                        color=(200, 150, 0),
                        size=random.uniform(1, 3),
                        lifetime=0.3,
                        drag=2.0,
                        shape="spark",
                    )
                ])

            # Auto-expire
            if self.elapsed >= self._max_duration:
                self.dismiss()

        elif self.state == SpellState.FADING:
            self._radius -= dt * 0.3
            if self._radius <= 0:
                self.state = SpellState.DONE

    def dismiss(self) -> None:
        """End the shield (called when hold gesture ends)."""
        if self.state == SpellState.ACTIVE:
            self._dismissed = True
            self.state = SpellState.FADING

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the hexagonal shield barrier."""
        if self.state == SpellState.DONE or self._radius < 0.005:
            return frame

        h, w = frame.shape[:2]
        cx = int(self._x * w)
        cy = int(self._y * h)
        rx = int(self._radius * w)
        ry = int(self._radius * h * 0.7)

        # Pulsing alpha
        pulse = 0.5 + 0.2 * math.sin(self._pulse_phase)
        alpha = pulse if self.state == SpellState.ACTIVE else pulse * 0.5

        # Draw hexagonal segments
        num_sides = 6
        points = []
        for i in range(num_sides):
            angle = self._rotation + (2 * math.pi * i / num_sides)
            px = cx + int(math.cos(angle) * rx)
            py = cy + int(math.sin(angle) * ry)
            points.append((px, py))

        pts = np.array(points, dtype=np.int32)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (200, 150, 0))
        cv2.addWeighted(overlay, alpha * 0.15, frame, 1 - alpha * 0.15, 0, frame)

        # Glowing edges
        edge_color = (
            int(255 * alpha),
            int(200 * alpha),
            int(50 * alpha),
        )
        cv2.polylines(frame, [pts], True, edge_color, 2, cv2.LINE_AA)

        # Inner hex (smaller, rotated differently)
        inner_pts = []
        for i in range(num_sides):
            angle = -self._rotation * 0.5 + (2 * math.pi * i / num_sides)
            px = cx + int(math.cos(angle) * rx * 0.6)
            py = cy + int(math.sin(angle) * ry * 0.6)
            inner_pts.append((px, py))

        inner = np.array(inner_pts, dtype=np.int32)
        inner_color = (
            int(200 * alpha * 0.6),
            int(150 * alpha * 0.6),
            int(30 * alpha * 0.6),
        )
        cv2.polylines(frame, [inner], True, inner_color, 1, cv2.LINE_AA)

        return frame

    def is_alive(self) -> bool:
        return self.state != SpellState.DONE
