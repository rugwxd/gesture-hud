"""Teleport spell — screen glitch with RGB split and static noise."""

from __future__ import annotations

import random

import cv2
import numpy as np

from src.effects.screen import ScreenEffects
from src.particles.emitters import BurstEmitter
from src.particles.engine import Particle, ParticleEngine
from src.spells.base import Spell, SpellState


class Teleport(Spell):
    """Glitch/teleport effect with RGB split, scan lines, and static.

    Cast by pinch-then-release. The screen tears apart with chromatic
    aberration, horizontal scan line displacement, and digital noise,
    then snaps back.
    """

    name = "teleport"
    mana_cost = 25
    cooldown = 2.0

    def __init__(self) -> None:
        super().__init__()
        self._duration: float = 0.5
        self._glitch_lines: list[tuple[int, int, int]] = []
        self._static_intensity: float = 0.0
        self._burst = BurstEmitter(
            count=40,
            speed_min=0.05,
            speed_max=0.2,
            lifetime_min=0.2,
            lifetime_max=0.5,
            color=(255, 100, 100),
            size_min=1.0,
            size_max=4.0,
            drag=1.0,
            shape="spark",
        )

    def cast(
        self,
        x: float,
        y: float,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
    ) -> None:
        """Trigger teleport glitch at hand position."""
        self.origin_x = x
        self.origin_y = y
        self.state = SpellState.ACTIVE

        # Strong chromatic aberration
        screen_fx.aberration.trigger(shift=8, duration=0.3)
        screen_fx.flash.trigger(
            color=(255, 200, 255),
            intensity=0.6,
            duration=0.08,
        )
        screen_fx.shake.trigger(intensity=6.0, duration=0.1)

        # Digital sparks
        particles.emit(self._burst.emit(x, y))

    def update(
        self,
        dt: float,
        particles: ParticleEngine,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Update glitch effect timing."""
        self.elapsed += dt

        if self.state == SpellState.DONE:
            return

        progress = self.elapsed / self._duration

        # Generate random glitch displacement lines
        if random.random() < 0.6 and progress < 0.8:
            num_lines = random.randint(3, 8)
            self._glitch_lines = []
            for _ in range(num_lines):
                y_pos = random.randint(0, 100)
                height = random.randint(1, 5)
                offset = random.randint(-30, 30)
                self._glitch_lines.append((y_pos, height, offset))
        else:
            self._glitch_lines = []

        # Static noise intensity
        self._static_intensity = max(0, 0.3 * (1 - progress * 1.5))

        # Emit random digital particles
        if random.random() < 0.4 and progress < 0.7:
            particles.emit([
                Particle(
                    x=random.uniform(0.1, 0.9),
                    y=random.uniform(0.1, 0.9),
                    vx=0,
                    vy=0,
                    color=random.choice([
                        (255, 0, 0),
                        (0, 255, 0),
                        (0, 0, 255),
                        (255, 255, 255),
                    ]),
                    size=random.uniform(1, 3),
                    lifetime=0.1,
                    shape="spark",
                )
            ])

        if self.elapsed >= self._duration:
            self.state = SpellState.DONE

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Apply glitch scan-line displacement and static noise."""
        if self.state == SpellState.DONE:
            return frame

        h, w = frame.shape[:2]

        # Horizontal line displacement
        for y_pct, line_h, offset in self._glitch_lines:
            y = int(y_pct * h / 100)
            y_end = min(h, y + line_h)
            if y >= h or y_end <= 0:
                continue

            strip = frame[y:y_end].copy()
            if offset > 0:
                frame[y:y_end, offset:] = strip[:, : w - offset]
                frame[y:y_end, :offset] = strip[:, w - offset :]
            elif offset < 0:
                off = abs(offset)
                frame[y:y_end, : w - off] = strip[:, off:]
                frame[y:y_end, w - off :] = strip[:, :off]

        # Static noise overlay
        if self._static_intensity > 0.02:
            noise = np.random.randint(
                0, int(255 * self._static_intensity),
                (h, w, 1),
                dtype=np.uint8,
            )
            noise_bgr = np.repeat(noise, 3, axis=2)
            frame = cv2.add(frame, noise_bgr)

        return frame

    def is_alive(self) -> bool:
        return self.state != SpellState.DONE
