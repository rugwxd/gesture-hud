"""Tests for configuration management."""

from __future__ import annotations

from src.config import (
    AudioConfig,
    CameraConfig,
    GesturesConfig,
    HandsConfig,
    LoggingConfig,
    ParticlesConfig,
    Settings,
    SpellsConfig,
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


class TestGesturesConfig:
    def test_defaults(self):
        config = GesturesConfig()
        assert config.swipe_threshold == 0.08
        assert config.hold_duration == 0.5
        assert config.tap_max_duration == 0.3
        assert config.debounce_frames == 5


class TestParticlesConfig:
    def test_defaults(self):
        config = ParticlesConfig()
        assert config.max_particles == 2000
        assert config.glow_enabled is True
        assert config.glow_intensity == 0.3


class TestSpellsConfig:
    def test_defaults(self):
        config = SpellsConfig()
        assert config.max_mana == 100
        assert config.mana_regen == 8.0
        assert config.show_mana_bar is True
        assert config.show_spell_name is True


class TestAudioConfig:
    def test_defaults(self):
        config = AudioConfig()
        assert config.enabled is True
        assert config.volume == 0.5


class TestSettings:
    def test_defaults(self):
        settings = Settings()
        assert isinstance(settings.camera, CameraConfig)
        assert isinstance(settings.hands, HandsConfig)
        assert isinstance(settings.particles, ParticlesConfig)
        assert isinstance(settings.spells, SpellsConfig)
        assert isinstance(settings.audio, AudioConfig)
        assert isinstance(settings.logging, LoggingConfig)

    def test_nested_override(self):
        settings = Settings(camera=CameraConfig(fps=60))
        assert settings.camera.fps == 60
        assert settings.camera.width == 1280


class TestLoadConfig:
    def test_load_default(self):
        settings = load_config()
        assert isinstance(settings, Settings)
        assert settings.camera.width == 1280

    def test_load_nonexistent_file(self):
        settings = load_config("/nonexistent/config.yaml")
        assert isinstance(settings, Settings)

    def test_load_from_yaml(self, tmp_path):
        yaml_content = """
camera:
  width: 1920
  height: 1080
  fps: 60

spells:
  max_mana: 200
  mana_regen: 15.0

audio:
  enabled: false
  volume: 0.3
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        settings = load_config(str(config_file))
        assert settings.camera.width == 1920
        assert settings.camera.height == 1080
        assert settings.spells.max_mana == 200
        assert settings.spells.mana_regen == 15.0
        assert settings.audio.enabled is False
        assert settings.audio.volume == 0.3
