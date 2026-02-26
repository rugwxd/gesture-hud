"""Configuration management for Gesture HUD."""

from __future__ import annotations

import logging
import os
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


class DetectionConfig(BaseModel):
    """YOLOv8 object detection configuration."""

    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    run_every_n_frames: int = 3


class GesturesConfig(BaseModel):
    """Gesture recognition configuration."""

    swipe_threshold: float = 0.15
    hold_duration: float = 0.5
    tap_max_duration: float = 0.3
    debounce_frames: int = 5


class HUDConfig(BaseModel):
    """HUD display configuration."""

    default_mode: str = "combat"
    color_primary: list[int] = Field(default_factory=lambda: [0, 255, 200])
    color_secondary: list[int] = Field(default_factory=lambda: [255, 170, 0])
    color_alert: list[int] = Field(default_factory=lambda: [255, 50, 50])
    opacity: float = 0.8
    font_scale: float = 0.6


class EffectsConfig(BaseModel):
    """Visual effects configuration."""

    glow_enabled: bool = True
    glow_intensity: float = 0.3
    scanlines_enabled: bool = True
    scanline_gap: int = 3
    holographic_flicker: bool = True
    flicker_intensity: float = 0.05


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
    """Root configuration for Gesture HUD."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    hands: HandsConfig = Field(default_factory=HandsConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    gestures: GesturesConfig = Field(default_factory=GesturesConfig)
    hud: HUDConfig = Field(default_factory=HUDConfig)
    effects: EffectsConfig = Field(default_factory=EffectsConfig)
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
    for noisy in ("ultralytics", "mediapipe", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
