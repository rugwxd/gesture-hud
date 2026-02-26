"""HUD widget base class and widget registry."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from src.gestures.tracker import GestureState
from src.vision.hands import HandData

logger = logging.getLogger(__name__)


class HUDMode(Enum):
    """HUD display modes."""

    COMBAT = auto()
    SCAN = auto()
    NAVIGATION = auto()


@dataclass
class HUDState:
    """Shared state passed to all HUD widgets each frame."""

    frame_size: tuple[int, int]  # (width, height)
    hands: list[HandData] = field(default_factory=list)
    gesture_state: GestureState | None = None
    detections: list = field(default_factory=list)
    mode: HUDMode = HUDMode.COMBAT
    fps: float = 0.0
    frame_number: int = 0


class HUDWidget(ABC):
    """Base class for all HUD widgets.

    Each widget updates its internal state and renders onto a frame overlay.
    Widgets declare which HUD modes they're active in.
    """

    name: str = "widget"
    active_modes: set[HUDMode] = {HUDMode.COMBAT, HUDMode.SCAN, HUDMode.NAVIGATION}

    def __init__(self) -> None:
        self.enabled = True

    def is_active(self, mode: HUDMode) -> bool:
        """Check if this widget should render in the given mode."""
        return self.enabled and mode in self.active_modes

    @abstractmethod
    def update(self, state: HUDState) -> None:
        """Update widget internal state."""

    @abstractmethod
    def render(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        """Render the widget onto the overlay frame.

        Args:
            overlay: Frame to draw on (may be modified in place).
            state: Current HUD state.

        Returns:
            The overlay frame (same reference or new).
        """


class WidgetRegistry:
    """Registry for managing HUD widgets."""

    def __init__(self) -> None:
        self._widgets: list[HUDWidget] = []

    def register(self, widget: HUDWidget) -> None:
        """Register a widget."""
        self._widgets.append(widget)
        logger.debug("Registered widget: %s", widget.name)

    @property
    def widgets(self) -> list[HUDWidget]:
        return self._widgets

    def get_active(self, mode: HUDMode) -> list[HUDWidget]:
        """Get widgets active in the given mode."""
        return [w for w in self._widgets if w.is_active(mode)]

    def update_all(self, state: HUDState) -> None:
        """Update all active widgets."""
        for widget in self.get_active(state.mode):
            widget.update(state)

    def render_all(self, overlay: np.ndarray, state: HUDState) -> np.ndarray:
        """Render all active widgets onto the overlay."""
        for widget in self.get_active(state.mode):
            overlay = widget.render(overlay, state)
        return overlay
