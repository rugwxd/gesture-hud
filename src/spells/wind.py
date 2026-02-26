"""Wind gust spell — horizontal debris particles across screen."""

from __future__ import annotations

import random

import numpy as np

from src.effects.screen import ScreenEffects
from src.particles.engine import Particle, ParticleEngine
from src.spells.base import Spell, SpellState


class Wind(Spell):
    """Horizontal wind gust that sends debris flying across the screen.

    Cast by swiping left or right. Spawns a wave of particles that
    sweep across the screen in the swipe direction, with trailing
    wisps and leaves.
    """

    name = "wind"
    mana_cost = 10
    cooldown = 0.6

    def __init__(self) -> None:
        super().__init__()
        self._direction: float = 1.0  # 1.0 = right, -1.0 = left
        self._duration: float = 0.8
        self._emit_phase: float = 0.0

    def cast(
        self,
        x: float,
        y: float,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
    ) -> None:
        """Trigger wind gust from hand position."""
        self.origin_x = x
        self.origin_y = y
        self.state = SpellState.ACTIVE

        # Light screen shake in swipe direction
        screen_fx.shake.trigger(intensity=5.0, duration=0.15)

        # Initial burst of debris
        self._emit_debris(particles, count=30)

    def set_direction(self, direction: float) -> None:
        """Set wind direction (positive=right, negative=left)."""
        self._direction = 1.0 if direction >= 0 else -1.0

    def update(
        self,
        dt: float,
        particles: ParticleEngine,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Continuously emit wind particles."""
        self.elapsed += dt

        if self.state == SpellState.DONE:
            return

        self._emit_phase += dt

        # Emit debris particles in waves
        if self.elapsed < self._duration * 0.6 and self._emit_phase > 0.05:
            self._emit_phase = 0.0
            self._emit_debris(particles, count=8)

        if self.elapsed >= self._duration:
            self.state = SpellState.DONE

    def _emit_debris(self, particles: ParticleEngine, count: int) -> None:
        """Spawn wind debris particles across the screen."""
        new_particles: list[Particle] = []

        for _ in range(count):
            # Spawn from the side the wind comes from
            if self._direction > 0:
                x = random.uniform(-0.05, 0.2)
            else:
                x = random.uniform(0.8, 1.05)

            y = random.uniform(0.1, 0.9)

            # Mostly horizontal velocity with some vertical scatter
            speed = random.uniform(0.3, 0.8)
            vx = self._direction * speed
            vy = random.gauss(0, 0.05)

            # Mix of colors: grey dust, green leaves, brown debris
            color = random.choice([
                (180, 180, 180),   # Grey dust
                (100, 180, 100),   # Green leaf
                (80, 130, 160),    # Brown debris
                (200, 200, 200),   # Light dust
                (120, 160, 80),    # Yellow-green
            ])

            shape = random.choice(["circle", "line", "line", "spark"])

            new_particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy,
                    ax=self._direction * 0.1,
                    color=color,
                    size=random.uniform(1.5, 4.0),
                    lifetime=random.uniform(0.5, 1.0),
                    drag=0.3,
                    gravity=random.uniform(0, 0.02),
                    shape=shape,
                )
            )

        particles.emit(new_particles)

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Wind particles are handled by the particle engine."""
        return frame

    def is_alive(self) -> bool:
        return self.state != SpellState.DONE
