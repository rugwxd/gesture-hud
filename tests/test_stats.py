"""Tests for system stats panel widget."""

from __future__ import annotations

import numpy as np

from src.hud.stats import StatsPanel
from src.hud.widgets import HUDMode, HUDState


class TestStatsPanel:
    def setup_method(self):
        self.panel = StatsPanel(color=(0, 255, 200))

    def test_creation(self):
        assert self.panel.name == "stats"
        assert HUDMode.COMBAT in self.panel.active_modes
        assert HUDMode.NAVIGATION in self.panel.active_modes

    def test_update(self):
        state = HUDState(frame_size=(640, 480))
        self.panel.update(state)
        # Should have at least one FPS reading
        assert len(self.panel._fps_history) >= 1

    def test_render(self):
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(frame_size=(640, 480), frame_number=42)

        self.panel.update(state)
        result = self.panel.render(overlay, state)
        assert np.any(result > 0)

    def test_load_color_normal(self):
        assert self.panel._get_load_color(30) == (0, 255, 200)

    def test_load_color_warning(self):
        color = self.panel._get_load_color(70)
        assert color == (0, 200, 255)

    def test_load_color_critical(self):
        color = self.panel._get_load_color(90)
        assert color == (0, 80, 255)

    def test_position_top_left(self):
        panel = StatsPanel(color=(0, 255, 200), position="top_left")
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(frame_size=(640, 480))
        panel.update(state)
        result = panel.render(overlay, state)
        assert np.any(result > 0)
