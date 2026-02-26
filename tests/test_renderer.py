"""Tests for HUD renderer."""

from __future__ import annotations

import numpy as np

from src.config import HUDConfig
from src.core.renderer import HUDRenderer


class TestHUDRenderer:
    def setup_method(self):
        config = HUDConfig()
        self.renderer = HUDRenderer(config)

    def test_create_overlay(self):
        overlay = self.renderer.create_overlay(640, 480)
        assert overlay.shape == (480, 640, 3)
        assert overlay.dtype == np.uint8
        assert np.all(overlay == 0)

    def test_composite(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay[100:200, 100:200] = (0, 255, 0)

        result = self.renderer.composite(frame, overlay)
        assert result.shape == frame.shape
        # Blended region should differ from original
        assert not np.array_equal(result[150, 150], frame[150, 150])

    def test_draw_text(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.renderer.draw_text(frame, "TEST", (100, 100))
        assert np.any(frame > 0)

    def test_draw_crosshair(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.renderer.draw_crosshair(frame, (320, 240))
        assert np.any(frame > 0)

    def test_draw_corner_brackets(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.renderer.draw_corner_brackets(frame, (100, 100), (300, 300))
        assert np.any(frame > 0)

    def test_draw_arc(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.renderer.draw_arc(frame, (320, 240), 50, 0, 180)
        assert np.any(frame > 0)

    def test_draw_progress_bar(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.renderer.draw_progress_bar(frame, (100, 100), 200, 20, 0.75)
        assert np.any(frame > 0)

    def test_progress_bar_clamped(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Should clamp to 0.0-1.0 without error
        self.renderer.draw_progress_bar(frame, (100, 100), 200, 20, 1.5)
        self.renderer.draw_progress_bar(frame, (100, 100), 200, 20, -0.5)

    def test_draw_hexagon(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.renderer.draw_hexagon(frame, (320, 240), 50)
        assert np.any(frame > 0)

    def test_custom_colors(self):
        config = HUDConfig(color_primary=[255, 0, 0])
        renderer = HUDRenderer(config)
        assert renderer.primary == (255, 0, 0)
