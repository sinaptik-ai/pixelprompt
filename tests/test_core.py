"""Tests for PixelPrompt core functionality."""

import pytest

from pixelprompt import PixelPrompt, RenderConfig, RenderedImage
from pixelprompt.core import RenderedImage as RenderedImageClass


class TestRenderConfig:
    """Test RenderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RenderConfig()
        assert config.font_size == 9
        assert config.width == 1568
        assert config.height == 1568
        assert config.background_color == (255, 255, 255)
        assert config.text_color == (0, 0, 0)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RenderConfig(
            font_size=12,
            width=2048,
            background_color=(240, 240, 240),
        )
        assert config.font_size == 12
        assert config.width == 2048
        assert config.background_color == (240, 240, 240)


class TestPixelPrompt:
    """Test PixelPrompt main class."""

    def test_initialization(self):
        """Test PixelPrompt initialization."""
        pxl = PixelPrompt()
        assert pxl.config.font_size == 9
        assert pxl.config.width == 1568

    def test_custom_initialization(self):
        """Test initialization with custom config."""
        config = RenderConfig(font_size=12)
        pxl = PixelPrompt(config=config)
        assert pxl.config.font_size == 12

    def test_invalid_font_size(self):
        """Test that invalid font size raises error."""
        config = RenderConfig(font_size=25)
        with pytest.raises(ValueError, match="font_size must be between 6 and 20"):
            PixelPrompt(config=config)

    def test_invalid_font_family(self):
        """Test that invalid font family raises error."""
        config = RenderConfig(font_family="invalid")
        with pytest.raises(ValueError, match="font_family must be"):
            PixelPrompt(config=config)

    def test_render_simple_text(self):
        """Test rendering simple text."""
        pxl = PixelPrompt()
        images = pxl.render("Hello, World!")
        assert len(images) == 1
        assert isinstance(images[0], RenderedImageClass)

    def test_render_multiline_text(self):
        """Test rendering multiline text."""
        pxl = PixelPrompt()
        text = "Line 1\nLine 2\nLine 3"
        images = pxl.render(text)
        assert len(images) >= 1

    def test_render_empty_text_raises_error(self):
        """Test that empty text raises error."""
        pxl = PixelPrompt()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            pxl.render("")

    def test_render_whitespace_text_raises_error(self):
        """Test that whitespace-only text raises error."""
        pxl = PixelPrompt()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            pxl.render("   \n  ")

    def test_render_long_text_splits(self):
        """Test that long text is split into multiple images."""
        pxl = PixelPrompt()
        # Create text with many lines
        text = "\n".join([f"Line {i}" for i in range(1000)])
        images = pxl.render(text)
        assert len(images) > 1


class TestRenderedImage:
    """Test RenderedImage class."""

    def test_rendered_image_properties(self):
        """Test RenderedImage properties."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        img = images[0]

        assert img.width == 1568
        assert img.height == 1568
        assert img.size_bytes > 0

    def test_png_bytes(self):
        """Test PNG bytes export."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        img = images[0]

        png_bytes = img.png_bytes()
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # PNG signature
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_base64(self):
        """Test base64 encoding."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        img = images[0]

        base64_str = img.base64()
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Should be valid base64
        import base64

        try:
            base64.b64decode(base64_str)
        except Exception as e:
            pytest.fail(f"Invalid base64: {e}")

    def test_save_image(self, tmp_path):
        """Test saving image to file."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        img = images[0]

        output_path = tmp_path / "test.png"
        img.save(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestSplitText:
    """Test text splitting functionality."""

    def test_split_short_text(self):
        """Test that short text is not split."""
        pxl = PixelPrompt()
        text = "Short text"
        chunks = pxl._split_text(text)
        assert len(chunks) == 1

    def test_split_respects_newlines(self):
        """Test that text is split on newlines."""
        pxl = PixelPrompt()
        text = "Line 1\nLine 2\nLine 3"
        chunks = pxl._split_text(text)
        # Should have at least 1 chunk
        assert len(chunks) >= 1
