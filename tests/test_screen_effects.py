"""Tests for screen effects (shake, flash, chromatic aberration)."""

from __future__ import annotations

import numpy as np

from src.effects.screen import (
    ChromaticAberration,
    ScreenEffects,
    ScreenFlash,
    ScreenShake,
)


class TestScreenShake:
    def test_initial_inactive(self):
        shake = ScreenShake()
        assert not shake.active

    def test_trigger_activates(self):
        shake = ScreenShake()
        shake.trigger(intensity=10.0, duration=0.3)
        assert shake.active

    def test_decays_over_time(self):
        shake = ScreenShake()
        shake.trigger(intensity=10.0, duration=0.3)
        shake.update(0.35)
        assert not shake.active

    def test_apply_no_change_when_inactive(self, sample_frame):
        shake = ScreenShake()
        result = shake.apply(sample_frame)
        assert np.array_equal(result, sample_frame)

    def test_apply_shifts_frame(self):
        shake = ScreenShake()
        shake.trigger(intensity=20.0, duration=1.0)
        shake.update(0.1)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        result = shake.apply(frame)
        assert result.shape == frame.shape


class TestScreenFlash:
    def test_initial_inactive(self):
        flash = ScreenFlash()
        assert not flash.active

    def test_trigger_activates(self):
        flash = ScreenFlash()
        flash.trigger(color=(255, 255, 255), intensity=0.5, duration=0.2)
        assert flash.active

    def test_decays_over_time(self):
        flash = ScreenFlash()
        flash.trigger(intensity=0.5, duration=0.2)
        flash.update(0.25)
        assert not flash.active

    def test_apply_brightens_frame(self, sample_frame):
        flash = ScreenFlash()
        flash.trigger(color=(255, 255, 255), intensity=0.8, duration=1.0)
        result = flash.apply(sample_frame)
        # Flash adds brightness
        assert np.mean(result) >= np.mean(sample_frame)

    def test_apply_no_change_when_inactive(self, sample_frame):
        flash = ScreenFlash()
        result = flash.apply(sample_frame)
        assert np.array_equal(result, sample_frame)


class TestChromaticAberration:
    def test_initial_inactive(self):
        ab = ChromaticAberration()
        assert not ab.active

    def test_trigger_activates(self):
        ab = ChromaticAberration()
        ab.trigger(shift=5, duration=0.15)
        assert ab.active

    def test_decays_over_time(self):
        ab = ChromaticAberration()
        ab.trigger(shift=5, duration=0.15)
        ab.update(0.2)
        assert not ab.active

    def test_apply_shifts_channels(self):
        ab = ChromaticAberration()
        ab.trigger(shift=5, duration=1.0)
        frame = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = ab.apply(frame)
        assert result.shape == frame.shape
        # B and R channels should be shifted differently
        assert not np.array_equal(result[:, :, 0], frame[:, :, 0])

    def test_apply_no_change_when_inactive(self, sample_frame):
        ab = ChromaticAberration()
        result = ab.apply(sample_frame)
        assert np.array_equal(result, sample_frame)


class TestScreenEffects:
    def test_initial_inactive(self):
        fx = ScreenEffects()
        assert not fx.active

    def test_trigger_impact(self, sample_frame):
        fx = ScreenEffects()
        fx.trigger_impact(
            shake_intensity=10.0,
            flash_color=(255, 255, 255),
            flash_intensity=0.4,
            aberration_shift=4,
        )
        assert fx.active
        assert fx.shake.active
        assert fx.flash.active
        assert fx.aberration.active

    def test_update_all(self):
        fx = ScreenEffects()
        fx.trigger_impact()
        fx.update(0.5)
        # After sufficient time, all effects should be done
        assert not fx.active

    def test_apply_all(self, sample_frame):
        fx = ScreenEffects()
        fx.trigger_impact()
        fx.update(0.01)
        result = fx.apply(sample_frame)
        assert result.shape == sample_frame.shape
