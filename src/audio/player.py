"""Audio player with numpy-synthesized spell sounds.

Generates and plays spell sound effects using pure numpy waveform
synthesis. No .wav files needed — all sounds are generated procedurally.
Playback is non-blocking via sounddevice.
"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100


class AudioPlayer:
    """Non-blocking audio player for spell sound effects.

    Synthesizes sounds on first use and caches them. Playback runs
    in a background thread to avoid blocking the render loop.
    """

    def __init__(self, enabled: bool = True, volume: float = 0.5) -> None:
        self.enabled = enabled
        self.volume = max(0.0, min(1.0, volume))
        self._sd = None
        self._initialized = False
        self._lock = threading.Lock()

    def _ensure_init(self) -> bool:
        """Lazy-initialize sounddevice."""
        if self._initialized:
            return self._sd is not None

        self._initialized = True
        try:
            import sounddevice as sd

            self._sd = sd
            logger.info("Audio player initialized (sample_rate=%d)", SAMPLE_RATE)
            return True
        except (ImportError, OSError) as exc:
            logger.warning("Audio disabled: %s", exc)
            self._sd = None
            return False

    def play(self, spell_name: str) -> None:
        """Play the sound for a spell (non-blocking).

        Args:
            spell_name: One of 'fireball', 'lightning', 'shield',
                        'force_push', 'teleport', 'wind'.
        """
        if not self.enabled:
            return

        sound = _get_sound(spell_name)
        if sound is None:
            return

        # Scale by volume
        sound = sound * self.volume

        # Play in background thread
        thread = threading.Thread(target=self._play_array, args=(sound,), daemon=True)
        thread.start()

    def _play_array(self, data: np.ndarray) -> None:
        """Play a numpy array through sounddevice."""
        with self._lock:
            if not self._ensure_init():
                return
            try:
                self._sd.play(data, SAMPLE_RATE)
                self._sd.wait()
            except Exception as exc:
                logger.debug("Audio playback error: %s", exc)

    def stop(self) -> None:
        """Stop any currently playing audio."""
        if self._sd is not None:
            try:
                self._sd.stop()
            except Exception:
                pass


@lru_cache(maxsize=16)
def _get_sound(spell_name: str) -> np.ndarray | None:
    """Get or synthesize the sound for a spell.

    Returns:
        Float32 numpy array normalized to [-1, 1], or None.
    """
    generators = {
        "fireball": _synth_fireball,
        "lightning": _synth_lightning,
        "shield": _synth_shield,
        "force_push": _synth_force_push,
        "teleport": _synth_teleport,
        "wind": _synth_wind,
    }

    gen = generators.get(spell_name)
    if gen is None:
        logger.warning("Unknown spell sound: %s", spell_name)
        return None

    try:
        sound = gen()
        # Normalize
        peak = np.max(np.abs(sound))
        if peak > 0:
            sound = sound / peak * 0.8
        return sound.astype(np.float32)
    except Exception as exc:
        logger.error("Failed to synthesize '%s': %s", spell_name, exc)
        return None


def _envelope(length: int, attack: float = 0.01, release: float = 0.3) -> np.ndarray:
    """Generate an amplitude envelope with attack and release."""
    env = np.ones(length, dtype=np.float64)
    attack_samples = int(attack * SAMPLE_RATE)
    release_samples = int(release * SAMPLE_RATE)

    if attack_samples > 0 and attack_samples < length:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if release_samples > 0 and release_samples < length:
        env[-release_samples:] = np.linspace(1, 0, release_samples)

    return env


def _synth_fireball() -> np.ndarray:
    """Fireball: rising filtered noise sweep with low rumble.

    Whooshing sound that rises in pitch, layered with bass rumble.
    """
    duration = 0.6
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Rising noise sweep
    noise = np.random.randn(len(t)) * 0.3
    freq_sweep = 150 + 600 * (t / duration) ** 2
    carrier = np.sin(2 * np.pi * np.cumsum(freq_sweep) / SAMPLE_RATE)

    # Low rumble
    rumble = np.sin(2 * np.pi * 60 * t) * 0.4

    # Combine with envelope
    env = _envelope(len(t), attack=0.02, release=0.3)
    sound = (carrier * 0.5 + noise * 0.3 + rumble) * env

    return sound


def _synth_lightning() -> np.ndarray:
    """Lightning: sharp white noise crackle with electric buzz.

    Short burst of noise modulated by a high-frequency buzz.
    """
    duration = 0.4
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # White noise crackle
    noise = np.random.randn(len(t))

    # Electric buzz (square wave harmonics)
    buzz = np.sign(np.sin(2 * np.pi * 120 * t)) * 0.3
    buzz += np.sign(np.sin(2 * np.pi * 240 * t)) * 0.15

    # Random amplitude modulation for crackle effect
    rng = np.random.default_rng(42)
    crackle = np.abs(rng.standard_normal(len(t) // 100))
    crackle = np.repeat(crackle, 100)[: len(t)]

    env = _envelope(len(t), attack=0.005, release=0.15)
    sound = (noise * 0.4 * crackle + buzz) * env

    return sound


def _synth_shield() -> np.ndarray:
    """Shield: low resonant hum with harmonic overtones.

    Sustained tone suggesting a force field powering up.
    """
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Fundamental + harmonics
    fundamental = np.sin(2 * np.pi * 110 * t) * 0.5
    harmonic2 = np.sin(2 * np.pi * 220 * t) * 0.25
    harmonic3 = np.sin(2 * np.pi * 330 * t) * 0.15
    harmonic5 = np.sin(2 * np.pi * 550 * t) * 0.1

    # Slight vibrato
    vibrato = np.sin(2 * np.pi * 5 * t) * 0.02
    modulated = fundamental * (1 + vibrato)

    env = _envelope(len(t), attack=0.05, release=0.2)
    sound = (modulated + harmonic2 + harmonic3 + harmonic5) * env

    return sound


def _synth_force_push() -> np.ndarray:
    """Force push: deep bass thud with air displacement.

    Short, punchy bass hit followed by a whoosh.
    """
    duration = 0.35
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Deep bass thud (decaying sine)
    bass_freq = 50 * np.exp(-t * 8)
    bass = np.sin(2 * np.pi * np.cumsum(bass_freq) / SAMPLE_RATE) * 0.7

    # Air whoosh (filtered noise)
    noise = np.random.randn(len(t)) * 0.2
    whoosh_env = np.exp(-((t - 0.05) ** 2) / 0.01)
    whoosh = noise * whoosh_env

    env = _envelope(len(t), attack=0.005, release=0.15)
    sound = (bass + whoosh) * env

    return sound


def _synth_teleport() -> np.ndarray:
    """Teleport: pitch-shifted sweep with digital glitch.

    Descending tone that breaks into digital static.
    """
    duration = 0.5
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Descending pitch sweep
    freq = 2000 * np.exp(-t * 6) + 100
    sweep = np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE) * 0.4

    # Digital glitch: quantized noise
    glitch_noise = np.random.randn(len(t) // 50)
    glitch = np.repeat(glitch_noise, 50)[: len(t)] * 0.3

    # Combine with crossfade: sweep first, then glitch
    crossfade = t / duration
    sound = sweep * (1 - crossfade) + glitch * crossfade

    env = _envelope(len(t), attack=0.01, release=0.1)
    return sound * env


def _synth_wind() -> np.ndarray:
    """Wind: gentle rushing noise with slow modulation.

    Filtered noise with amplitude modulation for organic feel.
    """
    duration = 0.7
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)

    # Base wind noise
    noise = np.random.randn(len(t))

    # Low-frequency amplitude modulation
    mod = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)

    # Simple low-pass via moving average
    kernel_size = 50
    kernel = np.ones(kernel_size) / kernel_size
    filtered = np.convolve(noise, kernel, mode="same")

    env = _envelope(len(t), attack=0.05, release=0.3)
    sound = filtered * mod * env * 0.5

    return sound
