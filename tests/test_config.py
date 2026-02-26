"""Tests for configuration management."""

from __future__ import annotations

from src.config import (
    CameraConfig,
    DetectionConfig,
    EffectsConfig,
    GesturesConfig,
    HandsConfig,
    HUDConfig,
    LoggingConfig,
    Settings,
    load_config,
)


class TestCameraConfig:
    def test_defaults(self):
        config = CameraConfig()
        assert config.device_id == 0
        assert config.width == 1280
        assert config.height == 720
        assert config.fps == 30

    def test_custom_values(self):
        config = CameraConfig(device_id=1, width=1920, height=1080, fps=60)
        assert config.device_id == 1
        assert config.width == 1920


class TestHandsConfig:
    def test_defaults(self):
        config = HandsConfig()
        assert config.max_hands == 2
        assert config.min_detection_confidence == 0.7
        assert config.min_tracking_confidence == 0.5


class TestDetectionConfig:
    def test_defaults(self):
        config = DetectionConfig()
        assert config.model == "yolov8n.pt"
        assert config.confidence_threshold == 0.5
        assert config.run_every_n_frames == 3


class TestGesturesConfig:
    def test_defaults(self):
        config = GesturesConfig()
        assert config.swipe_threshold == 0.15
        assert config.hold_duration == 0.5
        assert config.tap_max_duration == 0.3
        assert config.debounce_frames == 5


class TestHUDConfig:
    def test_defaults(self):
        config = HUDConfig()
        assert config.default_mode == "combat"
        assert config.color_primary == [0, 255, 200]
        assert config.opacity == 0.8

    def test_custom_colors(self):
        config = HUDConfig(color_primary=[255, 0, 0])
        assert config.color_primary == [255, 0, 0]


class TestEffectsConfig:
    def test_defaults(self):
        config = EffectsConfig()
        assert config.glow_enabled is True
        assert config.scanlines_enabled is True
        assert config.holographic_flicker is True


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert isinstance(settings.camera, CameraConfig)
        assert isinstance(settings.hands, HandsConfig)
        assert isinstance(settings.detection, DetectionConfig)
        assert isinstance(settings.hud, HUDConfig)
        assert isinstance(settings.effects, EffectsConfig)
        assert isinstance(settings.logging, LoggingConfig)

    def test_nested_override(self):
        settings = Settings(camera=CameraConfig(fps=60))
        assert settings.camera.fps == 60
        assert settings.camera.width == 1280  # Default preserved


class TestLoadConfig:
    def test_load_default(self):
        settings = load_config()
        assert isinstance(settings, Settings)
        assert settings.camera.width == 1280

    def test_load_nonexistent_file(self):
        settings = load_config("/nonexistent/config.yaml")
        assert isinstance(settings, Settings)  # Falls back to defaults

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
camera:
  width: 1920
  height: 1080
  fps: 60

hud:
  default_mode: scan
  opacity: 0.5
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        settings = load_config(str(config_file))
        assert settings.camera.width == 1920
        assert settings.camera.height == 1080
        assert settings.hud.default_mode == "scan"
        assert settings.hud.opacity == 0.5
