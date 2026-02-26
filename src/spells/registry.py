"""Spell registry — maps gestures to spells, manages cooldowns and mana."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from src.audio.player import AudioPlayer
from src.effects.screen import ScreenEffects
from src.gestures.tracker import GestureEvent
from src.particles.engine import ParticleEngine
from src.spells.base import Spell

logger = logging.getLogger(__name__)


@dataclass
class ManaSystem:
    """Simple mana pool with regeneration."""

    max_mana: int = 100
    current_mana: float = 100.0
    regen_rate: float = 8.0  # Mana per second

    def can_cast(self, cost: int) -> bool:
        """Check if there's enough mana."""
        return self.current_mana >= cost

    def spend(self, cost: int) -> bool:
        """Spend mana for a spell. Returns False if insufficient."""
        if self.current_mana < cost:
            return False
        self.current_mana -= cost
        return True

    def regenerate(self, dt: float) -> None:
        """Regenerate mana over time."""
        self.current_mana = min(self.max_mana, self.current_mana + self.regen_rate * dt)

    @property
    def ratio(self) -> float:
        """Current mana as a ratio (0-1)."""
        return self.current_mana / self.max_mana if self.max_mana > 0 else 0


@dataclass
class SpellCooldown:
    """Tracks per-spell cooldown timers."""

    cooldowns: dict[str, float] = field(default_factory=dict)

    def is_ready(self, spell_name: str) -> bool:
        """Check if a spell is off cooldown."""
        return time.time() >= self.cooldowns.get(spell_name, 0)

    def trigger(self, spell_name: str, duration: float) -> None:
        """Put a spell on cooldown."""
        self.cooldowns[spell_name] = time.time() + duration

    def remaining(self, spell_name: str) -> float:
        """Seconds remaining on cooldown (0 if ready)."""
        return max(0, self.cooldowns.get(spell_name, 0) - time.time())


class SpellRegistry:
    """Central manager for spell casting.

    Maps gesture events to spell classes, manages active spells,
    cooldowns, and mana. Orchestrates the cast lifecycle.
    """

    def __init__(
        self,
        particles: ParticleEngine,
        screen_fx: ScreenEffects,
        audio: AudioPlayer,
        max_mana: int = 100,
        mana_regen: float = 8.0,
    ) -> None:
        self.particles = particles
        self.screen_fx = screen_fx
        self.audio = audio
        self.mana = ManaSystem(max_mana=max_mana, regen_rate=mana_regen)
        self.cooldowns = SpellCooldown()
        self._active_spells: list[Spell] = []
        self._gesture_map: dict[str, type[Spell]] = {}

    def register(self, trigger: str, spell_class: type[Spell]) -> None:
        """Register a spell class for a gesture trigger.

        Args:
            trigger: Gesture event key (e.g., 'swipe_left', 'hold_fist').
            spell_class: Spell subclass to instantiate on trigger.
        """
        self._gesture_map[trigger] = spell_class
        logger.debug("Registered spell '%s' for trigger '%s'", spell_class.name, trigger)

    def handle_event(
        self,
        event: GestureEvent,
        hand_x: float,
        hand_y: float,
        gesture_name: str = "",
    ) -> Spell | None:
        """Attempt to cast a spell based on a gesture event.

        Args:
            event: The gesture event that occurred.
            hand_x: Normalized x position of hand.
            hand_y: Normalized y position of hand.
            gesture_name: Additional context (e.g., current gesture type).

        Returns:
            The cast Spell instance, or None if casting failed.
        """
        # Build trigger key from event
        trigger = self._event_to_trigger(event, gesture_name)
        if not trigger:
            return None

        spell_class = self._gesture_map.get(trigger)
        if spell_class is None:
            return None

        # Check cooldown
        if not self.cooldowns.is_ready(spell_class.name):
            remaining = self.cooldowns.remaining(spell_class.name)
            logger.debug(
                "Spell '%s' on cooldown (%.1fs remaining)",
                spell_class.name,
                remaining,
            )
            return None

        # Check mana
        if not self.mana.can_cast(spell_class.mana_cost):
            logger.debug(
                "Not enough mana for '%s' (need %d, have %.0f)",
                spell_class.name,
                spell_class.mana_cost,
                self.mana.current_mana,
            )
            return None

        # Cast the spell
        self.mana.spend(spell_class.mana_cost)
        self.cooldowns.trigger(spell_class.name, spell_class.cooldown)

        spell = spell_class()
        spell.cast(hand_x, hand_y, self.particles, self.screen_fx)
        self._active_spells.append(spell)

        # Play sound
        self.audio.play(spell_class.name)

        logger.info(
            "Cast '%s' at (%.2f, %.2f) | mana: %.0f/%d",
            spell_class.name,
            hand_x,
            hand_y,
            self.mana.current_mana,
            self.mana.max_mana,
        )
        return spell

    def update(
        self,
        dt: float,
        hand_x: float | None = None,
        hand_y: float | None = None,
    ) -> None:
        """Update all active spells and mana regeneration."""
        self.mana.regenerate(dt)

        for spell in self._active_spells:
            spell.update(dt, self.particles, hand_x, hand_y)

        # Remove dead spells
        self._active_spells = [s for s in self._active_spells if s.is_alive()]

    def render(self, frame: "np.ndarray") -> "np.ndarray":
        """Render all active spell overlays."""
        for spell in self._active_spells:
            frame = spell.render(frame)
        return frame

    @property
    def active_spells(self) -> list[Spell]:
        """Currently active spells."""
        return list(self._active_spells)

    @property
    def active_count(self) -> int:
        """Number of active spells."""
        return len(self._active_spells)

    def _event_to_trigger(self, event: GestureEvent, gesture_name: str) -> str:
        """Convert a gesture event to a trigger key."""
        event_map = {
            GestureEvent.SWIPE_LEFT: "swipe_left",
            GestureEvent.SWIPE_RIGHT: "swipe_right",
            GestureEvent.SWIPE_UP: "swipe_up",
            GestureEvent.SWIPE_DOWN: "swipe_down",
            GestureEvent.TAP: "tap",
            GestureEvent.HOLD_START: "hold_start",
            GestureEvent.HOLD_END: "hold_end",
        }

        base = event_map.get(event)
        if not base:
            return ""

        # Combine with gesture type for more specific triggers
        # e.g., "hold_start_fist", "tap_open_palm"
        if gesture_name:
            return f"{base}_{gesture_name}"
        return base

    def clear(self) -> None:
        """Remove all active spells."""
        self._active_spells.clear()
