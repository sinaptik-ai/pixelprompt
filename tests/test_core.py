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
        # Use minify=False so newlines are preserved and paginate into multiple pages.
        # With minify=True, lines get joined and may fit on a single wide image.
        pxl = PixelPrompt(RenderConfig(minify=False))
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


class TestMinification:
    """Test text minification for image rendering."""

    def test_default_config_minifies(self):
        """Default config has minify=True."""
        config = RenderConfig()
        assert config.minify is True

    def test_minify_removes_blank_lines(self):
        """Blank lines should be removed."""
        result = PixelPrompt.minify_text("Hello\n\n\nWorld")
        # Non-list, non-indented lines are joined with space
        assert result == "Hello World"

    def test_minify_strips_markdown_headers(self):
        """Markdown ## headers should have prefix removed."""
        result = PixelPrompt.minify_text("## Safety\nDon't do bad things")
        assert result == "Safety Don't do bad things"

    def test_minify_strips_all_header_levels(self):
        """All header levels (# through ######) should be stripped."""
        result = PixelPrompt.minify_text("# H1\n## H2\n### H3\n#### H4\n##### H5\n###### H6")
        assert result == "H1 H2 H3 H4 H5 H6"

    def test_minify_removes_bold_markers(self):
        """Bold ** markers should be removed."""
        result = PixelPrompt.minify_text("This is **bold** text")
        assert result == "This is bold text"

    def test_minify_removes_underscore_bold(self):
        """Bold __ markers should be removed."""
        result = PixelPrompt.minify_text("This is __bold__ text")
        assert result == "This is bold text"

    def test_minify_collapses_multiple_spaces(self):
        """Multiple spaces should collapse to one."""
        result = PixelPrompt.minify_text("Hello    World")
        assert result == "Hello World"

    def test_minify_strips_trailing_whitespace(self):
        """Trailing whitespace should be removed."""
        result = PixelPrompt.minify_text("Hello   \nWorld  ")
        # Joined into continuous text
        assert result == "Hello World"

    def test_minify_preserves_list_indent(self):
        """List item indentation should be preserved."""
        result = PixelPrompt.minify_text("- Item 1\n  - Sub item\n- Item 2")
        assert "  - Sub item" in result

    def test_minify_preserves_list_newlines(self):
        """Newlines before list items should be preserved."""
        result = PixelPrompt.minify_text("Header text\n- Item 1\n- Item 2")
        assert "\n- Item 1" in result
        assert "\n- Item 2" in result

    def test_minify_joins_prose_lines(self):
        """Non-list, non-indented consecutive lines should be joined."""
        result = PixelPrompt.minify_text("First line.\nSecond line.\nThird line.")
        assert result == "First line. Second line. Third line."

    def test_minify_preserves_content(self):
        """Semantic content should be fully preserved."""
        text = "## Config\n- key = value\n- port = 8080"
        result = PixelPrompt.minify_text(text)
        assert "key = value" in result
        assert "port = 8080" in result
        assert "Config" in result

    def test_minify_static_method(self):
        """minify_text should work as a static method."""
        result = PixelPrompt.minify_text("## Hello\n\nWorld")
        # Joined: "Hello World"
        assert result == "Hello World"

    def test_minify_mixed_prose_and_lists(self):
        """Mix of prose and list items should be handled correctly."""
        text = "## Section\nSome intro text.\nMore text.\n- Item 1\n- Item 2\nConclusion here."
        result = PixelPrompt.minify_text(text)
        # Prose joined, list items on own lines
        assert "Section Some intro text. More text." in result
        assert "\n- Item 1" in result
        assert "\n- Item 2" in result
        assert "Conclusion here." in result

    def test_minify_reduces_image_height(self):
        """Minified text should produce shorter images (fewer blank lines)."""
        text = "## Section\n\nLine 1\n\nLine 2\n\nLine 3"
        pxl_minify = PixelPrompt(RenderConfig(minify=True))
        pxl_raw = PixelPrompt(RenderConfig(minify=False))

        imgs_minify = pxl_minify.render(text)
        imgs_raw = pxl_raw.render(text)

        assert imgs_minify[0].height < imgs_raw[0].height

    def test_minify_reduces_tokens(self):
        """Minified rendering should use fewer tokens."""
        text = (
            "## Safety\n\nBe helpful.\n\n## Rules\n\n- Rule 1\n- Rule 2\n\n## Notes\n\nSome text."
        )
        pxl_minify = PixelPrompt(RenderConfig(minify=True))
        pxl_raw = PixelPrompt(RenderConfig(minify=False))

        tokens_minify = sum(i.tokens for i in pxl_minify.render(text))
        tokens_raw = sum(i.tokens for i in pxl_raw.render(text))

        assert tokens_minify < tokens_raw

    def test_minify_false_preserves_formatting(self):
        """With minify=False, blank lines and headers should be preserved."""
        pxl = PixelPrompt(RenderConfig(minify=False))
        text = "## Header\n\nParagraph"
        images = pxl.render(text)
        # Should work (no crash), and preserve blank lines in height
        assert len(images) >= 1

    def test_minify_only_blank_lines_raises(self):
        """Text that becomes empty after minification should raise error."""
        pxl = PixelPrompt(RenderConfig(minify=True))
        with pytest.raises(ValueError, match="empty"):
            pxl.render("\n\n\n")

    def test_minify_realistic_system_prompt(self):
        """Test with realistic system prompt section."""
        section = """## Credential Vault

You have access to an encrypted credential vault for storing and retrieving API keys, tokens, and passwords.

### Tools
- `vault_get(service, key)` — Retrieve a decrypted credential
- `vault_set(service, key, value)` — Store a credential (encrypted at rest)
- `vault_list(service?)` — List stored credentials (names only, never values)
- `vault_delete(service, key)` — Delete a credential

### Usage
- When you need an API key or token, check the vault first with `vault_list`.
- If found, use `vault_get` to retrieve it.
- If the user provides a credential, store it with `vault_set`.

### Security Rules
- NEVER include credential values in your text responses.
- NEVER write credentials to files on disk.
"""
        pxl_minify = PixelPrompt(RenderConfig(minify=True))
        pxl_raw = PixelPrompt(RenderConfig(minify=False))

        tokens_minify = sum(i.tokens for i in pxl_minify.render(section))
        tokens_raw = sum(i.tokens for i in pxl_raw.render(section))

        # Minified should save at least 10%
        savings = (tokens_raw - tokens_minify) / tokens_raw * 100
        assert savings > 10, f"Expected >10% savings, got {savings:.1f}%"

    def test_minify_exported_from_package(self):
        """minify_text should be importable from pixelprompt package."""
        from pixelprompt import minify_text

        result = minify_text("## Hello\n\nWorld")
        assert result == "Hello World"
