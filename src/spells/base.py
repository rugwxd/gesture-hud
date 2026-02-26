"""Base spell class defining the spell lifecycle."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np

from src.effects.screen import ScreenEffects
from src.particles.engine import ParticleEngine


class SpellState(Enum):
    """Lifecycle states for a spell."""

    CASTING = auto()
    ACTIVE = auto()
    FADING = auto()
    DONE = auto()


class Spell(ABC):
    """Abstract base class for all spells.

    Lifecycle: cast() → update() each frame → render() each frame → is_alive() check.
    When is_alive() returns False, the spell is removed by the engine.
    """

    name: str = "unknown"
    mana_cost: int = 10
    cooldown: float = 1.0

    def __init__(self) -> None:
        self.state = SpellState.CASTING
        self.elapsed: float = 0.0
        self.origin_x: float = 0.5
        self.origin_y: float = 0.5

    @abstractmethod
    def cast(
        self,
        x: float,
        y: float,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
    ) -> None:
        """Initialize the spell at the given hand position.

        Args:
            x: Normalized x coordinate of casting hand.
            y: Normalized y coordinate of casting hand.
            particles: Particle engine for spawning effects.
            screen_fx: Screen effects manager for shake/flash.
        """

    @abstractmethod
    def update(
        self,
        dt: float,
        particles: ParticleEngine,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Advance the spell state by dt seconds.

        Args:
            dt: Time delta in seconds.
            particles: Particle engine for ongoing emissions.
            hand_x: Current hand x position (for tracking spells).
            hand_y: Current hand y position (for tracking spells).
        """

    @abstractmethod
    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render spell-specific overlays (beyond particles).

        Most spells only use particles and return the frame unchanged.
        Some (like lightning, shield) draw additional geometry.

        Args:
            frame: BGR frame to draw on.

        Returns:
            Frame with spell overlays.
        """

    @abstractmethod
    def is_alive(self) -> bool:
        """Whether this spell is still active."""

    def _transition(self, new_state: SpellState) -> None:
        """Transition to a new spell state."""
        self.state = new_state
