"""Tests for the audio player."""

from __future__ import annotations

from src.audio.player import AudioPlayer, _get_sound


class TestAudioPlayer:
    def test_disabled_player(self):
        player = AudioPlayer(enabled=False)
        # Should not raise
        player.play("fireball")

    def test_volume_clamped(self):
        player = AudioPlayer(volume=2.0)
        assert player.volume == 1.0
        player = AudioPlayer(volume=-1.0)
        assert player.volume == 0.0


class TestSoundSynthesis:
    def test_fireball_sound(self):
        sound = _get_sound("fireball")
        assert sound is not None
        assert len(sound) > 0
        assert sound.dtype.name == "float32"
        assert abs(sound).max() <= 1.0

    def test_lightning_sound(self):
        sound = _get_sound("lightning")
        assert sound is not None
        assert len(sound) > 0

    def test_shield_sound(self):
        sound = _get_sound("shield")
        assert sound is not None
        assert len(sound) > 0

    def test_force_push_sound(self):
        sound = _get_sound("force_push")
        assert sound is not None
        assert len(sound) > 0

    def test_teleport_sound(self):
        sound = _get_sound("teleport")
        assert sound is not None
        assert len(sound) > 0

    def test_wind_sound(self):
        sound = _get_sound("wind")
        assert sound is not None
        assert len(sound) > 0

    def test_unknown_sound(self):
        sound = _get_sound("unknown_spell")
        assert sound is None

    def test_sound_cached(self):
        sound1 = _get_sound("fireball")
        sound2 = _get_sound("fireball")
        assert sound1 is sound2
