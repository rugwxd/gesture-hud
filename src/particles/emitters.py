"""Particle emitters — factories for spawning particles in patterns."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from src.particles.engine import Particle


class Emitter(ABC):
    """Base class for particle emitters."""

    @abstractmethod
    def emit(
        self,
        x: float,
        y: float,
        **kwargs: object,
    ) -> list[Particle]:
        """Spawn particles at the given normalized position.

        Args:
            x: Normalized x coordinate (0-1).
            y: Normalized y coordinate (0-1).
            **kwargs: Emitter-specific parameters.

        Returns:
            List of new particles.
        """


class BurstEmitter(Emitter):
    """Emits particles in a radial burst from a point.

    Used for explosions, impacts, and fireball detonations.
    """

    def __init__(
        self,
        count: int = 30,
        speed_min: float = 0.05,
        speed_max: float = 0.2,
        lifetime_min: float = 0.3,
        lifetime_max: float = 1.0,
        color: tuple[int, int, int] = (0, 200, 255),
        size_min: float = 2.0,
        size_max: float = 6.0,
        gravity: float = 0.0,
        drag: float = 0.5,
        shape: str = "circle",
    ) -> None:
        self.count = count
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.lifetime_min = lifetime_min
        self.lifetime_max = lifetime_max
        self.color = color
        self.size_min = size_min
        self.size_max = size_max
        self.gravity = gravity
        self.drag = drag
        self.shape = shape

    def emit(self, x: float, y: float, **kwargs: object) -> list[Particle]:
        """Emit particles in all directions from (x, y)."""
        particles: list[Particle] = []
        for _ in range(self.count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(self.speed_min, self.speed_max)
            particles.append(
                Particle(
                    x=x + random.gauss(0, 0.005),
                    y=y + random.gauss(0, 0.005),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=self._vary_color(),
                    size=random.uniform(self.size_min, self.size_max),
                    lifetime=random.uniform(self.lifetime_min, self.lifetime_max),
                    gravity=self.gravity,
                    drag=self.drag,
                    shape=self.shape,
                )
            )
        return particles

    def _vary_color(self) -> tuple[int, int, int]:
        """Slightly randomize the base color."""
        return (
            max(0, min(255, self.color[0] + random.randint(-20, 20))),
            max(0, min(255, self.color[1] + random.randint(-20, 20))),
            max(0, min(255, self.color[2] + random.randint(-20, 20))),
        )


class StreamEmitter(Emitter):
    """Emits a continuous stream of particles in a direction.

    Used for fire trails, energy beams, and sustained effects.
    """

    def __init__(
        self,
        count_per_emit: int = 5,
        direction: float = -math.pi / 2,
        spread: float = 0.3,
        speed_min: float = 0.05,
        speed_max: float = 0.15,
        lifetime: float = 0.6,
        color: tuple[int, int, int] = (0, 140, 255),
        size: float = 4.0,
        gravity: float = 0.0,
        drag: float = 0.3,
        shape: str = "circle",
    ) -> None:
        self.count_per_emit = count_per_emit
        self.direction = direction
        self.spread = spread
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.lifetime = lifetime
        self.color = color
        self.size = size
        self.gravity = gravity
        self.drag = drag
        self.shape = shape

    def emit(self, x: float, y: float, **kwargs: object) -> list[Particle]:
        """Emit a burst of particles in the configured direction."""
        particles: list[Particle] = []
        for _ in range(self.count_per_emit):
            angle = self.direction + random.gauss(0, self.spread)
            speed = random.uniform(self.speed_min, self.speed_max)
            particles.append(
                Particle(
                    x=x + random.gauss(0, 0.003),
                    y=y + random.gauss(0, 0.003),
                    vx=math.cos(angle) * speed,
                    vy=math.sin(angle) * speed,
                    color=_vary(self.color, 15),
                    size=self.size + random.uniform(-1, 1),
                    lifetime=self.lifetime + random.uniform(-0.1, 0.1),
                    gravity=self.gravity,
                    drag=self.drag,
                    shape=self.shape,
                )
            )
        return particles


class RingEmitter(Emitter):
    """Emits particles in an expanding ring pattern.

    Used for shockwaves, shield barriers, and force push effects.
    """

    def __init__(
        self,
        count: int = 40,
        radius: float = 0.02,
        speed: float = 0.15,
        lifetime: float = 0.5,
        color: tuple[int, int, int] = (200, 200, 0),
        size: float = 3.0,
        drag: float = 0.8,
        shape: str = "circle",
    ) -> None:
        self.count = count
        self.radius = radius
        self.speed = speed
        self.lifetime = lifetime
        self.color = color
        self.size = size
        self.drag = drag
        self.shape = shape

    def emit(self, x: float, y: float, **kwargs: object) -> list[Particle]:
        """Emit particles arranged in a ring, expanding outward."""
        particles: list[Particle] = []
        for i in range(self.count):
            angle = (2 * math.pi * i) / self.count
            particles.append(
                Particle(
                    x=x + math.cos(angle) * self.radius,
                    y=y + math.sin(angle) * self.radius,
                    vx=math.cos(angle) * self.speed,
                    vy=math.sin(angle) * self.speed,
                    color=_vary(self.color, 10),
                    size=self.size,
                    lifetime=self.lifetime + random.uniform(-0.05, 0.05),
                    drag=self.drag,
                    shape=self.shape,
                )
            )
        return particles


class TrailEmitter(Emitter):
    """Emits particles that trail behind a moving point.

    Used for fireball trails, comet tails, and motion effects.
    Particles spawn with slight random offset and low velocity.
    """

    def __init__(
        self,
        count_per_emit: int = 3,
        lifetime: float = 0.4,
        color: tuple[int, int, int] = (0, 100, 255),
        size_min: float = 2.0,
        size_max: float = 5.0,
        gravity: float = -0.02,
        drag: float = 1.0,
        size_decay: float = 5.0,
        shape: str = "circle",
    ) -> None:
        self.count_per_emit = count_per_emit
        self.lifetime = lifetime
        self.color = color
        self.size_min = size_min
        self.size_max = size_max
        self.gravity = gravity
        self.drag = drag
        self.size_decay = size_decay
        self.shape = shape

    def emit(self, x: float, y: float, **kwargs: object) -> list[Particle]:
        """Emit trail particles at the given position."""
        particles: list[Particle] = []
        for _ in range(self.count_per_emit):
            particles.append(
                Particle(
                    x=x + random.gauss(0, 0.008),
                    y=y + random.gauss(0, 0.008),
                    vx=random.gauss(0, 0.01),
                    vy=random.gauss(0, 0.01),
                    color=_vary(self.color, 25),
                    size=random.uniform(self.size_min, self.size_max),
                    lifetime=self.lifetime + random.uniform(-0.1, 0.1),
                    gravity=self.gravity,
                    drag=self.drag,
                    size_decay=self.size_decay,
                    shape=self.shape,
                )
            )
        return particles


def _vary(color: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    """Slightly randomize a color."""
    return (
        max(0, min(255, color[0] + random.randint(-amount, amount))),
        max(0, min(255, color[1] + random.randint(-amount, amount))),
        max(0, min(255, color[2] + random.randint(-amount, amount))),
    )
