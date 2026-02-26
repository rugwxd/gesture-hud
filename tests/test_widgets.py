"""Tests for HUD widget system."""

from __future__ import annotations

import numpy as np

from src.hud.widgets import HUDMode, HUDState, HUDWidget, WidgetRegistry


class DummyWidget(HUDWidget):
    name = "dummy"
    active_modes = {HUDMode.COMBAT, HUDMode.SCAN}

    def __init__(self):
        super().__init__()
        self.updated = False
        self.rendered = False

    def update(self, state: HUDState) -> None:
        self.updated = True

    def render(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        self.rendered = True
        return overlay


class TestHUDState:
    def test_creation(self):
        state = HUDState(frame_size=(640, 480))
        assert state.frame_size == (640, 480)
        assert state.mode == HUDMode.COMBAT
        assert state.hands == []
        assert state.detections == []

    def test_with_mode(self):
        state = HUDState(frame_size=(640, 480), mode=HUDMode.SCAN)
        assert state.mode == HUDMode.SCAN


class TestHUDWidget:
    def test_is_active(self):
        widget = DummyWidget()
        assert widget.is_active(HUDMode.COMBAT) is True
        assert widget.is_active(HUDMode.SCAN) is True
        assert widget.is_active(HUDMode.NAVIGATION) is False

    def test_disabled_widget(self):
        widget = DummyWidget()
        widget.enabled = False
        assert widget.is_active(HUDMode.COMBAT) is False


class TestWidgetRegistry:
    def test_register(self):
        registry = WidgetRegistry()
        widget = DummyWidget()
        registry.register(widget)
        assert len(registry.widgets) == 1

    def test_get_active(self):
        registry = WidgetRegistry()
        widget = DummyWidget()
        registry.register(widget)

        active = registry.get_active(HUDMode.COMBAT)
        assert len(active) == 1

        active = registry.get_active(HUDMode.NAVIGATION)
        assert len(active) == 0

    def test_update_all(self):
        registry = WidgetRegistry()
        widget = DummyWidget()
        registry.register(widget)

        state = HUDState(frame_size=(640, 480), mode=HUDMode.COMBAT)
        registry.update_all(state)
        assert widget.updated is True

    def test_render_all(self):
        registry = WidgetRegistry()
        widget = DummyWidget()
        registry.register(widget)

        state = HUDState(frame_size=(640, 480), mode=HUDMode.COMBAT)
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        registry.render_all(overlay, state)
        assert widget.rendered is True

    def test_inactive_not_updated(self):
        registry = WidgetRegistry()
        widget = DummyWidget()
        registry.register(widget)

        state = HUDState(frame_size=(640, 480), mode=HUDMode.NAVIGATION)
        registry.update_all(state)
        assert widget.updated is False
