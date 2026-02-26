"""Tests for the glow/bloom effect."""

from __future__ import annotations

import numpy as np

from src.effects.glow import apply_glow


class TestApplyGlow:
    def test_no_effect_at_zero_intensity(self, sample_frame):
        result = apply_glow(sample_frame, intensity=0.0)
        assert np.array_equal(result, sample_frame)

    def test_brightens_bright_areas(self, bright_frame):
        result = apply_glow(bright_frame, intensity=0.5)
        # Bright areas should be even brighter
        bright_region_original = bright_frame[300:350, 300:350].mean()
        bright_region_result = result[300:350, 300:350].mean()
        assert bright_region_result >= bright_region_original

    def test_preserves_shape(self, sample_frame):
        result = apply_glow(sample_frame, intensity=0.3)
        assert result.shape == sample_frame.shape

    def test_output_valid_range(self, bright_frame):
        result = apply_glow(bright_frame, intensity=0.8)
        assert result.min() >= 0
        assert result.max() <= 255
