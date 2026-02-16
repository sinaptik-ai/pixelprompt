"""Tests for PixelPrompt utility functions."""

from pixelprompt.utils import estimate_compression_ratio, estimate_tokens


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
        text = "a" * 100
        ratio = estimate_compression_ratio(text, 1)
        # 100 chars = ~25 tokens, 1 image = ~2 tokens
        assert ratio > 1

    def test_multiple_images_compression(self):
        """Test compression ratio with multiple images."""
        text = "a" * 100
        ratio = estimate_compression_ratio(text, 2)
        # 100 chars = ~25 tokens, 2 images = ~4 tokens
        assert ratio > 1

    def test_compression_improves_with_images(self):
        """Test that more images improve compression."""
        text = "a" * 1000
        ratio_1 = estimate_compression_ratio(text, 1)
        ratio_2 = estimate_compression_ratio(text, 2)
        # More images should improve compression ratio
        assert ratio_1 > ratio_2
