"""Tests for spell implementations and registry."""

from __future__ import annotations

import numpy as np

from src.gestures.tracker import GestureEvent
from src.spells.base import SpellState
from src.spells.fireball import Fireball
from src.spells.force_push import ForcePush
from src.spells.lightning import Lightning
from src.spells.registry import ManaSystem, SpellCooldown
from src.spells.shield import Shield
from src.spells.teleport import Teleport
from src.spells.wind import Wind


class TestManaSystem:
    def test_initial_full(self):
        mana = ManaSystem(max_mana=100)
        assert mana.current_mana == 100.0
        assert mana.ratio == 1.0

    def test_spend(self):
        mana = ManaSystem(max_mana=100)
        assert mana.spend(30)
        assert mana.current_mana == 70.0

    def test_spend_insufficient(self):
        mana = ManaSystem(max_mana=100, current_mana=10)
        assert not mana.spend(20)
        assert mana.current_mana == 10.0

    def test_can_cast(self):
        mana = ManaSystem(max_mana=100)
        assert mana.can_cast(50)
        assert not mana.can_cast(150)

    def test_regenerate(self):
        mana = ManaSystem(max_mana=100, current_mana=50, regen_rate=10.0)
        mana.regenerate(1.0)
        assert mana.current_mana == 60.0

    def test_regenerate_caps_at_max(self):
        mana = ManaSystem(max_mana=100, current_mana=95, regen_rate=10.0)
        mana.regenerate(1.0)
        assert mana.current_mana == 100.0

    def test_ratio(self):
        mana = ManaSystem(max_mana=100, current_mana=50)
        assert mana.ratio == 0.5


class TestSpellCooldown:
    def test_initially_ready(self):
        cd = SpellCooldown()
        assert cd.is_ready("fireball")

    def test_trigger_cooldown(self):
        cd = SpellCooldown()
        cd.trigger("fireball", 1.0)
        assert not cd.is_ready("fireball")

    def test_remaining(self):
        cd = SpellCooldown()
        cd.trigger("fireball", 10.0)
        assert cd.remaining("fireball") > 0
        assert cd.remaining("lightning") == 0


class TestFireball:
    def test_cast(self, particle_engine, screen_effects):
        spell = Fireball()
        spell.cast(0.3, 0.5, particle_engine, screen_effects)
        assert spell.state == SpellState.ACTIVE
        assert spell.is_alive()

    def test_update_moves_orb(self, particle_engine, screen_effects):
        spell = Fireball()
        spell.cast(0.3, 0.5, particle_engine, screen_effects)
        spell.update(0.1, particle_engine)
        assert spell.is_alive()

    def test_render(self, particle_engine, screen_effects, sample_frame):
        spell = Fireball()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        result = spell.render(sample_frame)
        assert result.shape == sample_frame.shape
        assert np.any(result > 0)

    def test_expires(self, particle_engine, screen_effects):
        spell = Fireball()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        for _ in range(20):
            spell.update(0.1, particle_engine)
        assert not spell.is_alive()

    def test_properties(self):
        assert Fireball.name == "fireball"
        assert Fireball.mana_cost > 0
        assert Fireball.cooldown > 0


class TestLightning:
    def test_cast(self, particle_engine, screen_effects):
        spell = Lightning()
        spell.cast(0.5, 0.4, particle_engine, screen_effects)
        assert spell.state == SpellState.ACTIVE
        assert screen_effects.flash.active

    def test_render_draws_bolt(self, particle_engine, screen_effects, sample_frame):
        spell = Lightning()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        result = spell.render(sample_frame)
        assert np.any(result > 0)

    def test_tracks_hand(self, particle_engine, screen_effects):
        spell = Lightning()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.05, particle_engine, hand_x=0.7, hand_y=0.6)
        assert spell.is_alive()

    def test_expires(self, particle_engine, screen_effects):
        spell = Lightning()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        for _ in range(10):
            spell.update(0.1, particle_engine)
        assert not spell.is_alive()


class TestShield:
    def test_cast(self, particle_engine, screen_effects):
        spell = Shield()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        assert spell.state == SpellState.ACTIVE
        assert spell.is_alive()

    def test_tracks_hand(self, particle_engine, screen_effects):
        spell = Shield()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.1, particle_engine, hand_x=0.7, hand_y=0.3)
        assert spell.is_alive()

    def test_dismiss(self, particle_engine, screen_effects):
        spell = Shield()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.1, particle_engine)
        spell.dismiss()
        assert spell.state == SpellState.FADING

    def test_fading_dies(self, particle_engine, screen_effects):
        spell = Shield()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.1, particle_engine)
        spell.dismiss()
        for _ in range(10):
            spell.update(0.1, particle_engine)
        assert not spell.is_alive()

    def test_render_draws_hex(self, particle_engine, screen_effects, sample_frame):
        spell = Shield()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.2, particle_engine)
        result = spell.render(sample_frame)
        assert np.any(result > 0)


class TestForcePush:
    def test_cast(self, particle_engine, screen_effects):
        spell = ForcePush()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        assert spell.state == SpellState.ACTIVE
        assert screen_effects.shake.active

    def test_render_draws_rings(self, particle_engine, screen_effects, sample_frame):
        spell = ForcePush()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.15, particle_engine)
        result = spell.render(sample_frame)
        assert result.shape == sample_frame.shape

    def test_expires(self, particle_engine, screen_effects):
        spell = ForcePush()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        for _ in range(10):
            spell.update(0.1, particle_engine)
        assert not spell.is_alive()


class TestTeleport:
    def test_cast(self, particle_engine, screen_effects):
        spell = Teleport()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        assert spell.state == SpellState.ACTIVE
        assert screen_effects.aberration.active

    def test_render_glitch(self, particle_engine, screen_effects, sample_frame):
        spell = Teleport()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.1, particle_engine)
        result = spell.render(sample_frame)
        assert result.shape == sample_frame.shape

    def test_expires(self, particle_engine, screen_effects):
        spell = Teleport()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        for _ in range(10):
            spell.update(0.1, particle_engine)
        assert not spell.is_alive()


class TestWind:
    def test_cast(self, particle_engine, screen_effects):
        spell = Wind()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        assert spell.state == SpellState.ACTIVE

    def test_set_direction(self):
        spell = Wind()
        spell.set_direction(-1.0)
        assert spell._direction == -1.0
        spell.set_direction(1.0)
        assert spell._direction == 1.0

    def test_emits_particles(self, particle_engine, screen_effects):
        spell = Wind()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        spell.update(0.1, particle_engine)
        assert particle_engine.count > 0

    def test_expires(self, particle_engine, screen_effects):
        spell = Wind()
        spell.cast(0.5, 0.5, particle_engine, screen_effects)
        for _ in range(10):
            spell.update(0.1, particle_engine)
        assert not spell.is_alive()


class TestSpellRegistry:
    def test_register_and_cast(self, spell_registry):
        spell_registry.register("swipe_up", Fireball)
        spell = spell_registry.handle_event(
            GestureEvent.SWIPE_UP, 0.5, 0.5, "",
        )
        assert spell is not None
        assert isinstance(spell, Fireball)

    def test_mana_deducted(self, spell_registry):
        spell_registry.register("swipe_up", Fireball)
        initial_mana = spell_registry.mana.current_mana
        spell_registry.handle_event(GestureEvent.SWIPE_UP, 0.5, 0.5, "")
        assert spell_registry.mana.current_mana < initial_mana

    def test_cooldown_prevents_recast(self, spell_registry):
        spell_registry.register("swipe_up", Fireball)
        spell_registry.handle_event(GestureEvent.SWIPE_UP, 0.5, 0.5, "")
        second = spell_registry.handle_event(GestureEvent.SWIPE_UP, 0.5, 0.5, "")
        assert second is None

    def test_insufficient_mana(self, spell_registry):
        spell_registry.register("swipe_up", Fireball)
        spell_registry.mana.current_mana = 1.0
        spell = spell_registry.handle_event(GestureEvent.SWIPE_UP, 0.5, 0.5, "")
        assert spell is None

    def test_update_removes_dead_spells(self, spell_registry):
        spell_registry.register("swipe_up", Fireball)
        spell_registry.handle_event(GestureEvent.SWIPE_UP, 0.5, 0.5, "")
        assert spell_registry.active_count == 1
        for _ in range(20):
            spell_registry.update(0.1)
        assert spell_registry.active_count == 0

    def test_mana_regeneration(self, spell_registry):
        spell_registry.mana.current_mana = 50
        spell_registry.update(1.0)
        assert spell_registry.mana.current_mana > 50

    def test_combined_trigger(self, spell_registry):
        spell_registry.register("hold_start_fist", Shield)
        spell = spell_registry.handle_event(
            GestureEvent.HOLD_START, 0.5, 0.5, "fist",
        )
        assert spell is not None
        assert isinstance(spell, Shield)

    def test_none_event_ignored(self, spell_registry):
        spell_registry.register("swipe_up", Fireball)
        spell = spell_registry.handle_event(GestureEvent.NONE, 0.5, 0.5, "")
        assert spell is None

    def test_clear(self, spell_registry):
        spell_registry.register("swipe_up", Fireball)
        spell_registry.handle_event(GestureEvent.SWIPE_UP, 0.5, 0.5, "")
        spell_registry.clear()
        assert spell_registry.active_count == 0

    def test_multiple_active_spells(self, spell_registry):
        spell_registry.register("swipe_left", Wind)
        spell_registry.register("hold_start_fist", Shield)
        spell_registry.handle_event(GestureEvent.SWIPE_LEFT, 0.5, 0.5, "")
        # Reset cooldown for next spell
        spell_registry.cooldowns.cooldowns.clear()
        spell_registry.handle_event(GestureEvent.HOLD_START, 0.5, 0.5, "fist")
        assert spell_registry.active_count == 2
