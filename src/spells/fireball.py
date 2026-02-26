"""Fireball spell — flaming orb projectile with ember trail."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from src.effects.screen import ScreenEffects
from src.particles.emitters import BurstEmitter, TrailEmitter
from src.particles.engine import Particle, ParticleEngine
from src.spells.base import Spell, SpellState


class Fireball(Spell):
    """Flaming orb that shoots forward from the hand with ember trail.

    Cast by pushing open palm toward camera. The fireball travels
    forward (grows larger to simulate Z-axis movement), leaving
    a trail of glowing embers, then explodes on reaching the center.
    """

    name = "fireball"
    mana_cost = 20
    cooldown = 1.5

    def __init__(self) -> None:
        super().__init__()
        self._x: float = 0.5
        self._y: float = 0.5
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._orb_size: float = 0.02
        self._max_age: float = 1.2
        self._exploded: bool = False
        self._trail = TrailEmitter(
            count_per_emit=4,
            lifetime=0.5,
            color=(0, 100, 255),  # BGR: orange-red
            size_min=3.0,
            size_max=7.0,
            gravity=0.01,
            drag=1.5,
            size_decay=8.0,
        )
        self._explosion = BurstEmitter(
            count=60,
            speed_min=0.05,
            speed_max=0.25,
            lifetime_min=0.3,
            lifetime_max=0.8,
            color=(0, 140, 255),  # BGR: orange
            size_min=3.0,
            size_max=8.0,
            gravity=0.05,
            drag=0.8,
        )

    def cast(
        self,
        x: float,
        y: float,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
    ) -> None:
        """Launch fireball from hand position toward screen center."""
        self.origin_x = x
        self.origin_y = y
        self._x = x
        self._y = y

        # Aim toward center of screen
        target_x, target_y = 0.5, 0.5
        dx = target_x - x
        dy = target_y - y
        dist = math.sqrt(dx * dx + dy * dy)
        speed = 0.4
        if dist > 0.01:
            self._vx = (dx / dist) * speed
            self._vy = (dy / dist) * speed
        else:
            self._vx = 0.0
            self._vy = -speed

        self.state = SpellState.ACTIVE

        # Initial burst of sparks from hand
        particles.emit(self._trail.emit(x, y))

    def update(
        self,
        dt: float,
        particles: ParticleEngine,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Move the fireball and emit trail particles."""
        self.elapsed += dt

        if self.state == SpellState.DONE:
            return

        if self.state == SpellState.ACTIVE:
            # Move the orb
            self._x += self._vx * dt
            self._y += self._vy * dt

            # Grow orb slightly (simulating coming toward camera)
            self._orb_size = 0.02 + self.elapsed * 0.015

            # Emit trail particles
            particles.emit(self._trail.emit(self._x, self._y))

            # Add some ember sparks
            if random.random() < 0.3:
                particles.emit([
                    Particle(
                        x=self._x + random.gauss(0, 0.01),
                        y=self._y + random.gauss(0, 0.01),
                        vx=random.gauss(0, 0.03),
                        vy=random.gauss(0, 0.03),
                        color=(0, random.randint(80, 200), 255),
                        size=random.uniform(1, 3),
                        lifetime=random.uniform(0.2, 0.4),
                        shape="spark",
                    )
                ])

            # Explode after reaching max age or going off screen
            if (
                self.elapsed >= self._max_age
                or self._x < -0.1
                or self._x > 1.1
                or self._y < -0.1
                or self._y > 1.1
            ):
                self._explode(particles)

        elif self.state == SpellState.FADING:
            if self.elapsed - self._explode_time > 0.8:
                self.state = SpellState.DONE

    def _explode(self, particles: ParticleEngine) -> None:
        """Trigger the fireball explosion."""
        self._exploded = True
        self._explode_time = self.elapsed
        self.state = SpellState.FADING
        particles.emit(self._explosion.emit(self._x, self._y))

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the fireball orb itself (particles handle the trail)."""
        if self.state not in (SpellState.ACTIVE, SpellState.CASTING):
            return frame

        h, w = frame.shape[:2]
        cx = int(self._x * w)
        cy = int(self._y * h)
        radius = max(5, int(self._orb_size * min(w, h)))

        # Inner bright core
        cv2.circle(frame, (cx, cy), radius, (0, 200, 255), -1, cv2.LINE_AA)
        # Outer glow
        cv2.circle(frame, (cx, cy), radius + 3, (0, 140, 255), 2, cv2.LINE_AA)
        # Hot white center
        cv2.circle(frame, (cx, cy), max(2, radius // 2), (200, 255, 255), -1)

        return frame

    def is_alive(self) -> bool:
        return self.state != SpellState.DONE
