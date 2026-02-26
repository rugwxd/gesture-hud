"""Tests for visual effects pipeline."""

from __future__ import annotations

import numpy as np

from src.config import EffectsConfig
from src.effects.glow import GlowEffect
from src.effects.holographic import EffectsPipeline, HolographicEffect
from src.effects.scanlines import ScanlineEffect


class TestGlowEffect:
    def test_apply_to_bright_frame(self, bright_frame):
        config = EffectsConfig(glow_enabled=True, glow_intensity=0.5)
        glow = GlowEffect(config)
        result = glow.apply(bright_frame)
        assert result.shape == bright_frame.shape
        assert result.dtype == np.uint8

    def test_disabled_returns_original(self, sample_frame):
        config = EffectsConfig(glow_enabled=False)
        glow = GlowEffect(config)
        result = glow.apply(sample_frame)
        assert result is sample_frame

    def test_zero_intensity_returns_original(self, sample_frame):
        config = EffectsConfig(glow_enabled=True, glow_intensity=0.0)
        glow = GlowEffect(config)
        result = glow.apply(sample_frame)
        assert result is sample_frame

    def test_glow_brightens_frame(self, bright_frame):
        config = EffectsConfig(glow_enabled=True, glow_intensity=0.5)
        glow = GlowEffect(config)
        result = glow.apply(bright_frame)
        # Result should be at least as bright as input (additive blend)
        assert np.sum(result) >= np.sum(bright_frame)


class TestScanlineEffect:
    def test_apply(self, sample_frame):
        config = EffectsConfig(scanlines_enabled=True, scanline_gap=3)
        scanlines = ScanlineEffect(config)
        result = scanlines.apply(sample_frame)
        assert result.shape == sample_frame.shape

    def test_disabled_returns_original(self, sample_frame):
        config = EffectsConfig(scanlines_enabled=False)
        scanlines = ScanlineEffect(config)
        result = scanlines.apply(sample_frame)
        assert result is sample_frame

    def test_mask_cached(self, sample_frame):
        config = EffectsConfig(scanlines_enabled=True, scanline_gap=3)
        scanlines = ScanlineEffect(config)
        scanlines.apply(sample_frame)
        assert scanlines._mask is not None
        assert scanlines._last_shape == sample_frame.shape

    def test_darkens_scanline_rows(self, bright_frame):
        config = EffectsConfig(scanlines_enabled=True, scanline_gap=2)
        scanlines = ScanlineEffect(config)
        result = scanlines.apply(bright_frame)
        # Scanline rows should be darker
        assert np.sum(result) < np.sum(bright_frame)


class TestHolographicEffect:
    def test_apply(self, sample_frame):
        config = EffectsConfig(holographic_flicker=True, flicker_intensity=0.05)
        holo = HolographicEffect(config)
        result = holo.apply(sample_frame)
        assert result.shape == sample_frame.shape

    def test_disabled_returns_original(self, sample_frame):
        config = EffectsConfig(holographic_flicker=False)
        holo = HolographicEffect(config)
        result = holo.apply(sample_frame)
        assert result is sample_frame


class TestEffectsPipeline:
    def test_pipeline_applies_all(self, bright_frame):
        config = EffectsConfig(
            glow_enabled=True,
            glow_intensity=0.3,
            scanlines_enabled=True,
            scanline_gap=3,
            holographic_flicker=True,
            flicker_intensity=0.05,
        )
        pipeline = EffectsPipeline(config)
        result = pipeline.apply(bright_frame)
        assert result.shape == bright_frame.shape
        assert result.dtype == np.uint8

    def test_pipeline_all_disabled(self, sample_frame):
        config = EffectsConfig(
            glow_enabled=False,
            scanlines_enabled=False,
            holographic_flicker=False,
        )
        pipeline = EffectsPipeline(config)
        result = pipeline.apply(sample_frame)
        assert result.shape == sample_frame.shape
