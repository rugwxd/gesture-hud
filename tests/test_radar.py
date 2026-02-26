"""Tests for radar minimap widget."""

from __future__ import annotations

import numpy as np

from src.hud.radar import RadarWidget
from src.hud.widgets import HUDMode, HUDState


class TestRadarWidget:
    def setup_method(self):
        self.radar = RadarWidget(color=(0, 255, 200), radius=80)

    def test_creation(self):
        assert self.radar.name == "radar"
        assert HUDMode.COMBAT in self.radar.active_modes
        assert HUDMode.NAVIGATION not in self.radar.active_modes

    def test_update_empty(self):
        state = HUDState(frame_size=(640, 480))
        self.radar.update(state)
        assert self.radar._sweep_angle > 0

    def test_update_with_detections(self, sample_detections):
        state = HUDState(frame_size=(640, 480), detections=sample_detections)
        self.radar.update(state)
        assert len(self.radar._blips) > 0

    def test_render(self):
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        state = HUDState(frame_size=(640, 480))
        self.radar.update(state)
        result = self.radar.render(overlay, state)
        assert np.any(result > 0)

    def test_sweep_rotation(self):
        state = HUDState(frame_size=(640, 480))
        initial_angle = self.radar._sweep_angle
        self.radar.update(state)
        assert self.radar._sweep_angle > initial_angle
