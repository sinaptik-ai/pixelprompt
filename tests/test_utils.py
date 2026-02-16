"""Tests for PixelPrompt utility functions."""

from pixelprompt.utils import estimate_compression_ratio, estimate_tokens
from pixelprompt.core import estimate_image_tokens


class TestEstimateTokens:
    """Test token estimation."""

    def test_empty_text(self):
        """Test token count for empty text."""
        assert estimate_tokens("") == 0

    def test_short_text(self):
        """Test token count for short text."""
        tokens = estimate_tokens("Hello")
        assert tokens == 1  # 5 chars / 4 = 1

    def test_longer_text(self):
        """Test token count for longer text."""
        text = "a" * 100
        tokens = estimate_tokens(text)
        assert tokens == 25  # 100 / 4 = 25

    def test_minimum_one_token(self):
        """Test that at least 1 token is returned."""
        tokens = estimate_tokens("a")
        assert tokens >= 1


class TestEstimateCompressionRatio:
    """Test compression ratio estimation."""

    def test_single_image_compression(self):
        """Test compression ratio with single image."""
        text = "a" * 4000  # 1000 tokens
        ratio = estimate_compression_ratio(text, 1, image_width=1568, image_height=300)
        # 1000 text tokens vs (1568*300)/750 = 627 image tokens
        assert ratio > 1

    def test_multiple_images_compression(self):
        """Test compression ratio with multiple images."""
        text = "a" * 4000
        ratio = estimate_compression_ratio(text, 2, image_width=1568, image_height=300)
        assert ratio > 0

    def test_compression_ratio_uses_real_formula(self):
        """Test that compression uses (w*h)/750, not fake constants."""
        text = "a" * 4000  # 1000 tokens
        ratio = estimate_compression_ratio(text, 1, image_width=1568, image_height=100)
        # (1568*100)/750 = 209 image tokens
        expected_image_tokens = estimate_image_tokens(1568, 100)
        expected_ratio = 1000 / expected_image_tokens
        assert abs(ratio - expected_ratio) < 0.01

    def test_dynamic_width_improves_ratio(self):
        """Narrower images should give better compression ratios."""
        text = "a" * 4000
        # Wide image
        ratio_wide = estimate_compression_ratio(text, 1, image_width=1568, image_height=300)
        # Narrow image (same height but less width)
        ratio_narrow = estimate_compression_ratio(text, 1, image_width=500, image_height=300)
        assert ratio_narrow > ratio_wide
