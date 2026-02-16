"""Tests for PixelPrompt core functionality."""

import base64

import pytest

from pixelprompt import PixelPrompt, RenderConfig, RenderedImage, estimate_image_tokens
from pixelprompt.core import RenderedImage as RenderedImageClass


class TestEstimateImageTokens:
    """Test the image token estimation function."""

    def test_basic_formula(self):
        """Token cost = (w * h) / 750."""
        assert estimate_image_tokens(750, 750) == 750
        assert estimate_image_tokens(1568, 1568) == int(1568 * 1568 / 750)

    def test_small_image(self):
        """Small images should have low token cost."""
        tokens = estimate_image_tokens(200, 100)
        assert tokens == int(200 * 100 / 750)

    def test_dynamic_width_savings(self):
        """Dynamic width images should cost much less than full-size."""
        full_tokens = estimate_image_tokens(1568, 1568)
        narrow_tokens = estimate_image_tokens(1568, 300)
        assert narrow_tokens < full_tokens
        assert narrow_tokens < full_tokens / 3

    def test_scaling_large_images(self):
        """Images larger than 1568 should be scaled down."""
        tokens = estimate_image_tokens(3000, 1000)
        assert tokens < int(3000 * 1000 / 750)

    def test_minimum_one_token(self):
        """Token count should be at least 1."""
        assert estimate_image_tokens(1, 1) >= 1


class TestRenderConfig:
    """Test RenderConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RenderConfig()
        assert config.font_size == 9
        assert config.max_width == 1568
        assert config.max_height == 1568
        assert config.dynamic_width is True
        assert config.dynamic_height is True
        assert config.background_color == (255, 255, 255)
        assert config.text_color == (0, 0, 0)
        assert config.padding == 5
        assert config.line_spacing == 1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RenderConfig(
            font_size=12,
            max_width=2048,
            background_color=(240, 240, 240),
        )
        assert config.font_size == 12
        assert config.max_width == 2048
        assert config.background_color == (240, 240, 240)

    def test_backwards_compat_width_height(self):
        """Test width/height aliases for backwards compatibility."""
        config = RenderConfig(max_width=1024, max_height=512)
        assert config.width == 1024
        assert config.height == 512

    def test_static_sizing(self):
        """Test disabling dynamic sizing."""
        config = RenderConfig(dynamic_width=False, dynamic_height=False)
        assert config.dynamic_width is False
        assert config.dynamic_height is False


class TestPixelPrompt:
    """Test PixelPrompt main class."""

    def test_initialization(self):
        """Test PixelPrompt initialization."""
        pxl = PixelPrompt()
        assert pxl.config.font_size == 9
        assert pxl.config.max_width == 1568

    def test_custom_initialization(self):
        """Test initialization with custom config."""
        config = RenderConfig(font_size=12)
        pxl = PixelPrompt(config=config)
        assert pxl.config.font_size == 12

    def test_char_measurement(self):
        """Test character dimensions are measured."""
        pxl = PixelPrompt()
        assert pxl._char_width > 0
        assert pxl._char_height > 0

    def test_max_chars_per_line(self):
        """Test max chars calculation is reasonable."""
        pxl = PixelPrompt()
        assert pxl._max_chars_per_line > 50
        assert pxl._max_chars_per_line < 500

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
        text = "\n".join(["Line {}".format(i) for i in range(1000)])
        images = pxl.render(text)
        assert len(images) > 1


class TestDynamicSizing:
    """Test dynamic image sizing."""

    def test_narrow_content_narrow_image(self):
        """Short lines should produce narrow images."""
        pxl = PixelPrompt()
        images = pxl.render("Hi")
        img = images[0]
        assert img.width < 1568
        assert img.width < 200  # "Hi" is very short

    def test_short_content_short_image(self):
        """Few lines should produce short images."""
        pxl = PixelPrompt()
        images = pxl.render("Hello\nWorld")
        img = images[0]
        assert img.height < 1568
        assert img.height < 100  # Just 2 lines

    def test_dynamic_saves_tokens(self):
        """Dynamic sizing should use fewer tokens than static."""
        pxl_dynamic = PixelPrompt()
        pxl_static = PixelPrompt(RenderConfig(dynamic_width=False, dynamic_height=False))

        text = "Short text\nJust two lines"
        imgs_dynamic = pxl_dynamic.render(text)
        imgs_static = pxl_static.render(text)

        dynamic_tokens = sum(img.tokens for img in imgs_dynamic)
        static_tokens = sum(img.tokens for img in imgs_static)

        assert dynamic_tokens < static_tokens
        # Should be dramatically less
        assert dynamic_tokens < static_tokens / 10

    def test_static_sizing_uses_full_dimensions(self):
        """Static sizing should use full width/height."""
        pxl = PixelPrompt(RenderConfig(dynamic_width=False, dynamic_height=False))
        images = pxl.render("Tiny text")
        img = images[0]
        assert img.width == 1568
        assert img.height == 1568

    def test_long_line_wraps_and_fits(self):
        """Long lines should be wrapped within max_width."""
        pxl = PixelPrompt()
        long_text = " ".join(["word"] * 200)
        images = pxl.render(long_text)
        for img in images:
            assert img.width <= 1568


class TestRenderedImage:
    """Test RenderedImage class."""

    def test_rendered_image_dynamic_size(self):
        """Test RenderedImage has dynamic size (not always 1568x1568)."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        img = images[0]

        # With dynamic sizing, a single word should be much smaller
        assert img.width < 1568
        assert img.height < 1568

    def test_token_cost_property(self):
        """Test tokens property reflects actual dimensions."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        img = images[0]

        expected = estimate_image_tokens(img.width, img.height)
        assert img.tokens == expected

    def test_token_cost_scales_with_size(self):
        """Larger images should cost more tokens."""
        pxl = PixelPrompt()

        small_imgs = pxl.render("Hi")
        large_imgs = pxl.render("\n".join(["A longer line of text here"] * 50))

        small_tokens = small_imgs[0].tokens
        large_tokens = sum(img.tokens for img in large_imgs)

        assert large_tokens > small_tokens

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
        try:
            base64.b64decode(base64_str)
        except Exception as e:
            pytest.fail("Invalid base64: {}".format(e))

    def test_to_content_block(self):
        """Test Anthropic API content block format."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        block = images[0].to_content_block()

        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/png"
        assert isinstance(block["source"]["data"], str)

    def test_save_image(self, tmp_path):
        """Test saving image to file."""
        pxl = PixelPrompt()
        images = pxl.render("Test")
        img = images[0]

        output_path = tmp_path / "test.png"
        img.save(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0


class TestWordWrapping:
    """Test word wrapping functionality."""

    def test_short_lines_unchanged(self):
        """Short lines should pass through unchanged."""
        pxl = PixelPrompt()
        lines = pxl._wrap_text("Hello World\nFoo Bar")
        assert lines == ["Hello World", "Foo Bar"]

    def test_empty_lines_preserved(self):
        """Empty lines should be preserved."""
        pxl = PixelPrompt()
        lines = pxl._wrap_text("Hello\n\nWorld")
        assert lines == ["Hello", "", "World"]

    def test_long_lines_wrapped(self):
        """Lines longer than max width should be word-wrapped."""
        pxl = PixelPrompt()
        long_line = " ".join(["word"] * 200)
        lines = pxl._wrap_text(long_line)
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= pxl._max_chars_per_line

    def test_very_long_word_split(self):
        """Words longer than max width should be force-split."""
        pxl = PixelPrompt()
        long_word = "x" * (pxl._max_chars_per_line + 50)
        lines = pxl._wrap_text(long_word)
        assert len(lines) >= 2
        for line in lines:
            assert len(line) <= pxl._max_chars_per_line


class TestSplitText:
    """Test text splitting functionality (backwards compat)."""

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
        assert len(chunks) >= 1
