"""Main spell engine orchestrating camera, tracking, gestures, and spells."""

from __future__ import annotations

import logging
import time

import cv2

from src.audio.player import AudioPlayer
from src.config import Settings
from src.core.renderer import SpellRenderer
from src.effects.glow import apply_glow
from src.effects.screen import ScreenEffects
from src.gestures.recognizer import GestureRecognizer
from src.gestures.tracker import GestureEvent, GestureTracker
from src.particles.engine import ParticleEngine
from src.spells.fireball import Fireball
from src.spells.force_push import ForcePush
from src.spells.lightning import Lightning
from src.spells.registry import SpellRegistry
from src.spells.shield import Shield
from src.spells.teleport import Teleport
from src.spells.wind import Wind
from src.vision.camera import Camera
from src.vision.hands import HandTracker

logger = logging.getLogger(__name__)


class SpellEngine:
    """Main engine that orchestrates camera, hand tracking, gesture
    recognition, spell casting, particle rendering, and screen effects.

    This is the core loop. Webcam → hand tracking → gesture recognition
    → spell registry → particles + screen effects → display.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Vision
        self.camera = Camera(settings.camera)
        self.hand_tracker = HandTracker(settings.hands)

        # Gestures
        self.gesture_recognizer = GestureRecognizer()
        self.gesture_tracker = GestureTracker(settings.gestures)

        # Particles and effects
        self.particles = ParticleEngine(max_particles=settings.particles.max_particles)
        self.screen_fx = ScreenEffects()

        # Audio
        self.audio = AudioPlayer(
            enabled=settings.audio.enabled,
            volume=settings.audio.volume,
        )

        # Spell registry
        self.registry = SpellRegistry(
            particles=self.particles,
            screen_fx=self.screen_fx,
            audio=self.audio,
            max_mana=settings.spells.max_mana,
            mana_regen=settings.spells.mana_regen,
        )

        # Renderer
        self.renderer = SpellRenderer(settings)

        # Register all spells
        self._register_spells()

        # State
        self._running = False
        self._frame_count = 0
        self._last_time = 0.0
        self._active_shield: Shield | None = None
        self._last_gesture_name: str = ""
        self._glow_enabled = settings.particles.glow_enabled
        self._glow_intensity = settings.particles.glow_intensity

    def _register_spells(self) -> None:
        """Register gesture-to-spell mappings."""
        # Fist hold → Shield
        self.registry.register("hold_start_fist", Shield)

        # Open palm hold → Force Push
        self.registry.register("hold_start_open_palm", ForcePush)

        # Point hold → Lightning
        self.registry.register("hold_start_point", Lightning)

        # Swipe left/right → Wind
        self.registry.register("swipe_left", Wind)
        self.registry.register("swipe_right", Wind)

        # Pinch hold → Teleport
        self.registry.register("hold_start_pinch", Teleport)

        # Swipe up → Fireball
        self.registry.register("swipe_up", Fireball)

    def run(self, source: str | None = None) -> None:
        """Run the main spell engine loop.

        Args:
            source: Optional video file path. If None, uses webcam.
        """
        if source:
            self.camera = Camera(self.settings.camera, source=source)

        if not self.camera.open():
            logger.error("Failed to open camera")
            return

        self._running = True
        self._last_time = time.time()
        logger.info("Spell engine started")

        try:
            while self._running:
                self._process_frame()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
        finally:
            self._cleanup()

    def _process_frame(self) -> None:
        """Process a single frame through the full spell pipeline."""
        frame_data = self.camera.read()
        if frame_data is None:
            self._running = False
            return

        frame = frame_data.image
        self._frame_count += 1

        # Delta time
        now = time.time()
        dt = min(now - self._last_time, 0.1)  # Cap at 100ms
        self._last_time = now

        # 1. Hand tracking
        hands = self.hand_tracker.process(frame)

        # 2. Gesture recognition
        gesture_result = None
        gesture_state = None
        hand_x: float | None = None
        hand_y: float | None = None

        if hands:
            gesture_result = self.gesture_recognizer.classify(hands[0])
            gesture_state = self.gesture_tracker.update(gesture_result)
            hand_x = hands[0].center.x
            hand_y = hands[0].center.y
            self._last_gesture_name = gesture_result.gesture.name.lower()
        else:
            gesture_state = self.gesture_tracker.update(None)

        # 3. Handle gesture events → cast spells
        if gesture_state and gesture_state.event != GestureEvent.NONE:
            self._handle_gesture_event(
                gesture_state.event,
                hand_x or 0.5,
                hand_y or 0.5,
            )

        # 4. Handle shield dismissal on hold end
        if gesture_state and gesture_state.event == GestureEvent.HOLD_END:
            self._dismiss_active_shield()

        # 5. Update all spells and particles
        self.registry.update(dt, hand_x, hand_y)
        self.particles.update(dt)
        self.screen_fx.update(dt)

        # 6. Render particles onto frame
        frame = self.particles.render(frame)

        # 7. Render spell overlays (lightning bolts, shield hex, etc.)
        frame = self.registry.render(frame)

        # 8. Apply glow effect
        if self._glow_enabled and self.particles.count > 0:
            frame = apply_glow(frame, intensity=self._glow_intensity)

        # 9. Apply screen effects (shake, flash, aberration)
        frame = self.screen_fx.apply(frame)

        # 10. Draw hand landmarks
        self.renderer.draw_landmarks(frame, hands)

        # 11. Draw mana bar and spell info
        self.renderer.draw_mana_bar(frame, self.registry.mana)
        if self.registry.active_spells:
            spell_name = self.registry.active_spells[-1].name
            self.renderer.draw_spell_name(frame, spell_name)

        # 12. Display
        cv2.imshow("AR Spellcaster", frame)

    def _handle_gesture_event(
        self,
        event: GestureEvent,
        hand_x: float,
        hand_y: float,
    ) -> None:
        """Route gesture events to the spell registry."""
        spell = self.registry.handle_event(
            event, hand_x, hand_y, self._last_gesture_name,
        )

        # Track shield for dismissal
        if spell is not None and isinstance(spell, Shield):
            self._active_shield = spell

        # Set wind direction based on swipe direction
        if spell is not None and isinstance(spell, Wind):
            if event == GestureEvent.SWIPE_LEFT:
                spell.set_direction(-1.0)
            else:
                spell.set_direction(1.0)

    def _dismiss_active_shield(self) -> None:
        """Dismiss the shield when the fist hold ends."""
        if self._active_shield is not None and self._active_shield.is_alive():
            self._active_shield.dismiss()
            self._active_shield = None

    def _cleanup(self) -> None:
        """Release all resources."""
        self._running = False
        self.camera.release()
        self.hand_tracker.release()
        self.audio.stop()
        cv2.destroyAllWindows()
        logger.info("Spell engine stopped")

    def stop(self) -> None:
        """Signal the engine to stop."""
        self._running = False
