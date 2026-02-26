"""Force push spell — expanding shockwave from palm."""

from __future__ import annotations

import math
import random

import cv2
import numpy as np

from src.effects.screen import ScreenEffects
from src.particles.emitters import RingEmitter
from src.particles.engine import ParticleEngine
from src.spells.base import Spell, SpellState


class ForcePush(Spell):
    """Expanding shockwave rings from the open palm.

    Cast by open palm detection. Multiple concentric rings expand
    outward from the hand, creating a visible force push effect.
    """

    name = "force_push"
    mana_cost = 15
    cooldown = 1.0

    def __init__(self) -> None:
        super().__init__()
        self._x: float = 0.5
        self._y: float = 0.5
        self._rings: list[_ShockRing] = []
        self._duration: float = 0.8
        self._ring_emitter = RingEmitter(
            count=30,
            radius=0.01,
            speed=0.2,
            lifetime=0.4,
            color=(200, 200, 0),
            size=2.0,
            drag=0.5,
        )

    def cast(
        self,
        x: float,
        y: float,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
    ) -> None:
        """Trigger force push from hand position."""
        self.origin_x = x
        self.origin_y = y
        self._x = x
        self._y = y
        self.state = SpellState.ACTIVE

        # Spawn 3 rings with staggered timing
        for i in range(3):
            self._rings.append(
                _ShockRing(
                    x=x,
                    y=y,
                    delay=i * 0.1,
                    max_radius=0.15 + i * 0.03,
                    speed=0.4 - i * 0.05,
                )
            )

        # Screen effects
        screen_fx.shake.trigger(intensity=12.0, duration=0.2)
        screen_fx.flash.trigger(
            color=(200, 200, 100),
            intensity=0.3,
            duration=0.1,
        )

        # Ring particles
        particles.emit(self._ring_emitter.emit(x, y))

    def update(
        self,
        dt: float,
        particles: ParticleEngine,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Expand the shockwave rings."""
        self.elapsed += dt

        if self.state == SpellState.DONE:
            return

        for ring in self._rings:
            ring.update(dt)

        if self.elapsed >= self._duration:
            self.state = SpellState.DONE

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw expanding shockwave rings."""
        if self.state == SpellState.DONE:
            return frame

        h, w = frame.shape[:2]

        for ring in self._rings:
            if not ring.visible:
                continue

            cx = int(ring.x * w)
            cy = int(ring.y * h)
            rx = int(ring.radius * w)
            ry = int(ring.radius * h * 0.7)

            alpha = ring.alpha

            # Draw distortion ring
            color = (
                int(200 * alpha),
                int(230 * alpha),
                int(255 * alpha),
            )

            # Outer ring
            cv2.ellipse(
                frame, (cx, cy), (rx, ry), 0, 0, 360,
                color, 2, cv2.LINE_AA,
            )
            # Inner ring (slightly smaller, brighter)
            if rx > 5:
                inner_color = (
                    int(255 * alpha),
                    int(255 * alpha),
                    int(255 * alpha),
                )
                cv2.ellipse(
                    frame, (cx, cy), (rx - 3, ry - 2), 0, 0, 360,
                    inner_color, 1, cv2.LINE_AA,
                )

        return frame

    def is_alive(self) -> bool:
        return self.state != SpellState.DONE


class _ShockRing:
    """A single expanding shockwave ring."""

    def __init__(
        self,
        x: float,
        y: float,
        delay: float,
        max_radius: float,
        speed: float,
    ) -> None:
        self.x = x
        self.y = y
        self.delay = delay
        self.max_radius = max_radius
        self.speed = speed
        self.radius: float = 0.0
        self.elapsed: float = 0.0

    @property
    def visible(self) -> bool:
        return self.elapsed >= self.delay and self.radius < self.max_radius

    @property
    def alpha(self) -> float:
        if not self.visible:
            return 0.0
        progress = self.radius / self.max_radius
        return max(0, 1.0 - progress)

    def update(self, dt: float) -> None:
        self.elapsed += dt
        if self.elapsed >= self.delay:
            self.radius += self.speed * dt
