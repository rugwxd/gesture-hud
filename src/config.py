"""Configuration management for AR Spellcaster."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class CameraConfig(BaseModel):
    """Camera capture configuration."""

    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30


class HandsConfig(BaseModel):
    """MediaPipe hand tracking configuration."""

    max_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.5


class GesturesConfig(BaseModel):
    """Gesture recognition configuration."""

    swipe_threshold: float = 0.08
    hold_duration: float = 0.5
    tap_max_duration: float = 0.3
    debounce_frames: int = 5


class ParticlesConfig(BaseModel):
    """Particle system configuration."""

    max_particles: int = 2000
    glow_enabled: bool = True
    glow_intensity: float = 0.3


class SpellsConfig(BaseModel):
    """Spell system configuration."""

    max_mana: int = 100
    mana_regen: float = 8.0
    show_mana_bar: bool = True
    show_spell_name: bool = True


class AudioConfig(BaseModel):
    """Audio configuration."""

    enabled: bool = True
    volume: float = 0.5


class RecordingConfig(BaseModel):
    """Recording and screenshot configuration."""

    output_dir: str = "data/recordings"
    screenshot_key: str = "thumbs_up"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    file: str = "logs/gesture-hud.log"


class Settings(BaseModel):
    """Root configuration for AR Spellcaster."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    hands: HandsConfig = Field(default_factory=HandsConfig)
    gestures: GesturesConfig = Field(default_factory=GesturesConfig)
    particles: ParticlesConfig = Field(default_factory=ParticlesConfig)
    spells: SpellsConfig = Field(default_factory=SpellsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path | None = None) -> Settings:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file. Uses default if not provided.

    Returns:
        Validated Settings instance.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    raw: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", config_path)
    else:
        logger.warning("Config file not found at %s, using defaults", config_path)

    return Settings(**raw)


def setup_logging(config: LoggingConfig) -> None:
    """Configure application-wide logging."""
    log_file = PROJECT_ROOT / config.file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format=config.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
    )

    # Suppress noisy library logs
    for noisy in ("mediapipe", "PIL", "sounddevice"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
