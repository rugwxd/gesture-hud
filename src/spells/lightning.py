"""Lightning bolt spell — electric bolt from top of screen to fingertip."""

from __future__ import annotations

import random

import cv2
import numpy as np

from src.effects.screen import ScreenEffects
from src.particles.emitters import BurstEmitter
from src.particles.engine import Particle, ParticleEngine
from src.spells.base import Spell, SpellState


class Lightning(Spell):
    """Electric bolt that strikes from the top of the screen to the fingertip.

    Cast by pointing finger up and holding. Generates a branching bolt
    path that flickers each frame. Sparks emit from the strike point.
    """

    name = "lightning"
    mana_cost = 15
    cooldown = 0.8

    def __init__(self) -> None:
        super().__init__()
        self._target_x: float = 0.5
        self._target_y: float = 0.5
        self._bolt_segments: list[tuple[float, float]] = []
        self._branches: list[list[tuple[float, float]]] = []
        self._duration: float = 0.6
        self._strike_spark = BurstEmitter(
            count=15,
            speed_min=0.03,
            speed_max=0.12,
            lifetime_min=0.1,
            lifetime_max=0.3,
            color=(255, 200, 50),  # BGR: cyan-ish
            size_min=1.0,
            size_max=3.0,
            drag=1.5,
            shape="spark",
        )

    def cast(
        self,
        x: float,
        y: float,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
    ) -> None:
        """Strike lightning from top of screen to hand position."""
        self.origin_x = x
        self.origin_y = y
        self._target_x = x
        self._target_y = y
        self.state = SpellState.ACTIVE

        # Generate initial bolt path
        self._regenerate_bolt()

        # Screen effects
        screen_fx.flash.trigger(
            color=(255, 220, 100),
            intensity=0.5,
            duration=0.1,
        )
        screen_fx.shake.trigger(intensity=8.0, duration=0.15)

        # Sparks at strike point
        particles.emit(self._strike_spark.emit(x, y))

    def _regenerate_bolt(self) -> None:
        """Generate a new random bolt path from top to target."""
        self._bolt_segments = _generate_bolt(
            start_x=self._target_x + random.uniform(-0.05, 0.05),
            start_y=0.0,
            end_x=self._target_x,
            end_y=self._target_y,
            segments=12,
            jitter=0.04,
        )

        # Generate 1-3 branches
        self._branches = []
        num_branches = random.randint(1, 3)
        for _ in range(num_branches):
            if len(self._bolt_segments) < 4:
                continue
            branch_idx = random.randint(2, len(self._bolt_segments) - 2)
            bx, by = self._bolt_segments[branch_idx]
            branch_end_x = bx + random.uniform(-0.08, 0.08)
            branch_end_y = by + random.uniform(0.02, 0.08)
            branch = _generate_bolt(bx, by, branch_end_x, branch_end_y, 5, 0.02)
            self._branches.append(branch)

    def update(
        self,
        dt: float,
        particles: ParticleEngine,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Flicker the bolt and emit sparks at strike point."""
        self.elapsed += dt

        if self.state == SpellState.DONE:
            return

        # Track hand if available
        if hand_x is not None and hand_y is not None:
            self._target_x = hand_x
            self._target_y = hand_y

        # Regenerate bolt path every few frames for flicker effect
        if random.random() < 0.7:
            self._regenerate_bolt()

        # Emit sparks at strike point
        if random.random() < 0.5:
            particles.emit([
                Particle(
                    x=self._target_x + random.gauss(0, 0.008),
                    y=self._target_y + random.gauss(0, 0.008),
                    vx=random.gauss(0, 0.05),
                    vy=random.gauss(0, 0.05),
                    color=(
                        255,
                        random.randint(180, 255),
                        random.randint(50, 150),
                    ),
                    size=random.uniform(1, 3),
                    lifetime=0.15,
                    shape="spark",
                )
            ])

        if self.elapsed >= self._duration:
            self.state = SpellState.DONE

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Draw the lightning bolt on the frame."""
        if self.state == SpellState.DONE:
            return frame

        h, w = frame.shape[:2]

        # Fade based on remaining time
        alpha = max(0.3, 1.0 - (self.elapsed / self._duration))

        # Draw main bolt
        _draw_bolt(frame, self._bolt_segments, w, h, alpha, thickness=2)

        # Draw branches (thinner)
        for branch in self._branches:
            _draw_bolt(frame, branch, w, h, alpha * 0.6, thickness=1)

        # Bright glow at strike point
        sx = int(self._target_x * w)
        sy = int(self._target_y * h)
        glow_size = int(15 * alpha)
        if 0 <= sx < w and 0 <= sy < h:
            cv2.circle(frame, (sx, sy), glow_size, (255, 255, 200), -1, cv2.LINE_AA)
            cv2.circle(frame, (sx, sy), glow_size + 4, (255, 200, 50), 2, cv2.LINE_AA)

        return frame

    def is_alive(self) -> bool:
        return self.state != SpellState.DONE


def _generate_bolt(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    segments: int,
    jitter: float,
) -> list[tuple[float, float]]:
    """Generate a jagged lightning bolt path between two points."""
    points: list[tuple[float, float]] = [(start_x, start_y)]

    for i in range(1, segments):
        t = i / segments
        # Linear interpolation with random jitter
        x = start_x + (end_x - start_x) * t + random.gauss(0, jitter)
        y = start_y + (end_y - start_y) * t + random.gauss(0, jitter * 0.5)
        points.append((x, y))

    points.append((end_x, end_y))
    return points


def _draw_bolt(
    frame: np.ndarray,
    segments: list[tuple[float, float]],
    w: int,
    h: int,
    alpha: float,
    thickness: int = 2,
) -> None:
    """Draw a bolt path on the frame."""
    if len(segments) < 2:
        return

    # Core bright line
    core_color = (
        int(255 * alpha),
        int(255 * alpha),
        int(220 * alpha),
    )
    glow_color = (
        int(200 * alpha),
        int(150 * alpha),
        int(50 * alpha),
    )

    for i in range(len(segments) - 1):
        x1 = int(segments[i][0] * w)
        y1 = int(segments[i][1] * h)
        x2 = int(segments[i + 1][0] * w)
        y2 = int(segments[i + 1][1] * h)

        # Outer glow
        cv2.line(frame, (x1, y1), (x2, y2), glow_color, thickness + 3, cv2.LINE_AA)
        # Core
        cv2.line(frame, (x1, y1), (x2, y2), core_color, thickness, cv2.LINE_AA)
