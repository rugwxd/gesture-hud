"""Tests for HUD engine."""

from __future__ import annotations

from src.core.engine import HUDEngine
from src.hud.widgets import HUDMode


class TestHUDEngine:
    def test_creation(self, test_settings):
        engine = HUDEngine(test_settings)
        assert engine is not None
        assert engine._current_mode == HUDMode.COMBAT

    def test_widget_registry_populated(self, test_settings):
        engine = HUDEngine(test_settings)
        widgets = engine.widget_registry.widgets
        assert len(widgets) > 0

        widget_names = [w.name for w in widgets]
        assert "targeting" in widget_names
        assert "stats" in widget_names
        assert "radar" in widget_names
        assert "object_tags" in widget_names
        assert "mode_menu" in widget_names

    def test_stop(self, test_settings):
        engine = HUDEngine(test_settings)
        engine.stop()
        assert engine._running is False

    def test_mode_menu_registered(self, test_settings):
        engine = HUDEngine(test_settings)
        assert engine.mode_menu is not None
        assert engine.mode_menu.current_mode == HUDMode.COMBAT
