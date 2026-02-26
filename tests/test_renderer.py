"""Tests for the spell renderer."""

from __future__ import annotations

import numpy as np

from src.core.renderer import SpellRenderer
from src.spells.registry import ManaSystem


class TestSpellRenderer:
    def test_draw_mana_bar(self, test_settings, sample_frame):
        renderer = SpellRenderer(test_settings)
        mana = ManaSystem(max_mana=100, current_mana=75)
        renderer.draw_mana_bar(sample_frame, mana)
        # Should have drawn something on the frame
        assert np.any(sample_frame > 0)

    def test_draw_mana_bar_empty(self, test_settings, sample_frame):
        renderer = SpellRenderer(test_settings)
        mana = ManaSystem(max_mana=100, current_mana=0)
        renderer.draw_mana_bar(sample_frame, mana)
        # Should still draw the border
        assert np.any(sample_frame > 0)

    def test_draw_spell_name(self, test_settings, sample_frame):
        renderer = SpellRenderer(test_settings)
        renderer.draw_spell_name(sample_frame, "fireball")
        assert np.any(sample_frame > 0)

    def test_draw_landmarks(self, test_settings, sample_frame, hand_data_open_palm):
        renderer = SpellRenderer(test_settings)
        renderer.draw_landmarks(sample_frame, [hand_data_open_palm])
        assert np.any(sample_frame > 0)

    def test_draw_landmarks_empty(self, test_settings, sample_frame):
        renderer = SpellRenderer(test_settings)
        original = sample_frame.copy()
        renderer.draw_landmarks(sample_frame, [])
        assert np.array_equal(sample_frame, original)

    def test_mana_bar_disabled(self, test_settings, sample_frame):
        test_settings.spells.show_mana_bar = False
        renderer = SpellRenderer(test_settings)
        original = sample_frame.copy()
        mana = ManaSystem(max_mana=100, current_mana=50)
        renderer.draw_mana_bar(sample_frame, mana)
        assert np.array_equal(sample_frame, original)

    def test_spell_name_disabled(self, test_settings, sample_frame):
        test_settings.spells.show_spell_name = False
        renderer = SpellRenderer(test_settings)
        original = sample_frame.copy()
        renderer.draw_spell_name(sample_frame, "lightning")
        assert np.array_equal(sample_frame, original)
