"""
Core PixelPrompt implementation for rendering text as optimized images.

Key features (from Pixels Beat Tokens research, benchmark v2):
- Dynamic width/height: images sized to fit content, not fixed dimensions
- Word wrapping: long lines wrapped to fit max width
- Correct token estimation: uses Claude's (w*h)/750 formula
- Menlo font preferred: optimal for Opus 4.6 accuracy
- Content-type presets: optimal font size per content type
- 100% accuracy on 125-question benchmark (Opus 4.6)
- 38-80% net cost savings with optimized prompts
"""

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

# Claude vision token formula (from Anthropic docs, Feb 2026)
TOKENS_PER_PIXEL_DIVISOR = 750
MAX_VISION_DIMENSION = 1568

# Pricing ($/MTok) — Feb 2026
MODEL_PRICING = {
    "claude-opus-4-6": {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
}

# ═══════════════════════════════════════════════════════════
# Content-type presets (from benchmark v2 findings)
# ═══════════════════════════════════════════════════════════

CONTENT_PRESETS = {
    "prose": {
        "font_size": 9,
        "minify": True,
        "description": "Long-form text, articles, documentation",
        "expected_input_savings": 0.71,
        "expected_net_savings": 0.69,
    },
    "json": {
        "font_size": 9,
        "minify": True,  # compact_json: remove whitespace
        "description": "JSON data, API responses, structured data",
        "expected_input_savings": 0.83,
        "expected_net_savings": 0.80,
    },
    "code": {
        "font_size": 7,
        "minify": False,  # never minify code
        "description": "Source code, scripts, functions",
        "expected_input_savings": 0.61,
        "expected_net_savings": 0.59,
    },
    "config": {
        "font_size": 8,
        "minify": False,  # preserve config structure
        "description": "INI, TOML, YAML, env files",
        "expected_input_savings": 0.41,
        "expected_net_savings": 0.39,
    },
}


@dataclass
class RenderConfig:
    """Configuration for text rendering to images."""

    font_size: int = 9
    """Font size in points (range: 6-20). Default: 9 (optimal for Opus 4.6)."""

    font_family: str = "monospace"
    """Font family: 'monospace', 'serif', or 'sans-serif'. Default: 'monospace'."""

    max_width: int = 1568
    """Maximum image width in pixels. Default: 1568 (Claude vision max)."""

    max_height: int = 1568
    """Maximum image height in pixels. Default: 1568 (Claude vision max)."""

    dynamic_width: bool = True
    """If True, image width fits content. If False, always uses max_width."""

    dynamic_height: bool = True
    """If True, image height fits content. If False, always uses max_height."""

    background_color: Tuple[int, int, int] = (255, 255, 255)
    """Background color as (R, G, B) tuple. Default: white (255, 255, 255)."""

    text_color: Tuple[int, int, int] = (0, 0, 0)
    """Text color as (R, G, B) tuple. Default: black (0, 0, 0)."""

    padding: int = 5
    """Padding in pixels from image edges. Default: 5 (minimal for density)."""

    line_spacing: int = 1
    """Extra pixels between lines. Default: 1 (tight spacing)."""

    minify: bool = True
    """If True, strip visual-only formatting before rendering.

    Removes: blank lines, markdown headers (##), bold/italic markers (**/*),
    collapses multiple spaces, strips trailing whitespace. This reduces image
    height and width significantly without losing semantic content.

    Default: True (recommended for LLM context compression).
    Set to False when rendering text that must preserve exact formatting
    (code, config files).
    """

    content_type: Optional[str] = None
    """Content type preset: 'prose', 'json', 'code', 'config', or None.

    When set, overrides font_size and minify with optimal values from
    benchmark v2 research. Set to None to use manual configuration.
    """

    @staticmethod
    def for_content(content_type: str) -> "RenderConfig":
        """Create a RenderConfig with optimal settings for a content type.

        Args:
            content_type: One of 'prose', 'json', 'code', 'config'.

        Returns:
            RenderConfig with optimal font_size and minify settings.

        Raises:
            ValueError: If content_type is not recognized.
        """
        if content_type not in CONTENT_PRESETS:
            valid = ", ".join(sorted(CONTENT_PRESETS.keys()))
            raise ValueError(
                "Unknown content type '{}'. Valid types: {}".format(content_type, valid)
            )
        preset = CONTENT_PRESETS[content_type]
        return RenderConfig(
            font_size=preset["font_size"],
            minify=preset["minify"],
            content_type=content_type,
        )

    # Backwards compatibility aliases
    @property
    def width(self) -> int:
        """Alias for max_width (backwards compatibility)."""
        return self.max_width

    @property
    def height(self) -> int:
        """Alias for max_height (backwards compatibility)."""
        return self.max_height


def estimate_image_tokens(width: int, height: int) -> int:
    """Estimate Claude token cost for an image based on dimensions.

    Uses official Anthropic formula: tokens = (width * height) / 750
    Images larger than 1568px on longest side are scaled down first.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Estimated token count.
    """
    scale = 1.0
    if max(width, height) > MAX_VISION_DIMENSION:
        scale = MAX_VISION_DIMENSION / max(width, height)

    scaled_w = int(width * scale)
    scaled_h = int(height * scale)

    return max(1, int((scaled_w * scaled_h) / TOKENS_PER_PIXEL_DIVISOR))


class RenderedImage:
    """Represents a single rendered image with token cost metadata."""

    def __init__(self, image: Image.Image, token_cost: Optional[int] = None):
        """Initialize with PIL Image.

        Args:
            image: PIL Image object.
            token_cost: Pre-computed token cost. If None, computed from dimensions.
        """
        self._image = image
        self._token_cost = (
            token_cost
            if token_cost is not None
            else estimate_image_tokens(image.width, image.height)
        )

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._image.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._image.height

    @property
    def tokens(self) -> int:
        """Estimated Claude vision token cost for this image."""
        return self._token_cost

    @property
    def size_bytes(self) -> int:
        """Approximate size in bytes (PNG-encoded)."""
        return len(self.png_bytes())

    def png_bytes(self) -> bytes:
        """Get raw PNG bytes."""
        buffer = io.BytesIO()
        self._image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def base64(self) -> str:
        """Get base64-encoded PNG for API integration."""
        return base64.b64encode(self.png_bytes()).decode("utf-8")

    def to_content_block(self) -> dict:
        """Convert to Anthropic API content block format.

        Returns:
            Dict with type, source.type, source.media_type, source.data.
        """
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": self.base64(),
            },
        }

    def save(self, path: str) -> None:
        """Save image to file."""
        self._image.save(path, format="PNG", optimize=True)


class PixelPrompt:
    """
    Renders text content as optimized PNG images for LLM context compression.

    Key improvements over naive text-to-image:
    1. Dynamic width — images are only as wide as the content needs
    2. Dynamic height — images are only as tall as the content needs
    3. Word wrapping — long lines are wrapped to fit max_width
    4. Correct token estimation — uses Claude's (w*h)/750 formula

    Example:
        >>> pxl = PixelPrompt()
        >>> images = pxl.render("Long context here...")
        >>> for img in images:
        ...     print(f"{img.width}x{img.height} = {img.tokens} tokens")
    """

    def __init__(self, config: Optional[RenderConfig] = None):
        """
        Initialize PixelPrompt.

        Args:
            config: RenderConfig object. Uses defaults if None.
        """
        self.config = config or RenderConfig()
        self._validate_config()
        self._load_fonts()
        self._char_width, self._char_height = self._measure_char()
        self._line_height = self._char_height + self.config.line_spacing
        self._max_chars_per_line = max(
            1, (self.config.max_width - 2 * self.config.padding) // max(1, self._char_width)
        )
        self._max_lines_per_image = max(
            1, (self.config.max_height - 2 * self.config.padding) // max(1, self._line_height)
        )

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not 6 <= self.config.font_size <= 20:
            raise ValueError("font_size must be between 6 and 20")
        if self.config.font_family not in ("monospace", "serif", "sans-serif"):
            raise ValueError("font_family must be 'monospace', 'serif', or 'sans-serif'")
        if self.config.max_width < 50 or self.config.max_height < 20:
            raise ValueError("max_width must be >= 50 and max_height must be >= 20")
        if self.config.padding < 0:
            raise ValueError("padding must be non-negative")
        if self.config.line_spacing < 0:
            raise ValueError("line_spacing must be non-negative")

    def _load_fonts(self) -> None:
        """Load available fonts for the system."""
        font_names = {
            "monospace": ["Menlo", "DejaVuSansMono", "Courier New", "Liberation Mono"],
            "sans-serif": ["DejaVuSans", "Arial", "Liberation Sans"],
            "serif": ["DejaVuSerif", "Times New Roman", "Liberation Serif"],
        }

        family_fonts = font_names.get(self.config.font_family, font_names["monospace"])
        self._font = self._find_font(family_fonts)

    def _find_font(self, font_names: List[str]) -> ImageFont.FreeTypeFont:
        """
        Find an available TrueType font from the list.

        Prefers Menlo (optimal per Pixels Beat Tokens research).
        """
        # Common font paths on different systems
        font_paths = [
            # macOS — Menlo preferred (per paper)
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/System/Library/Fonts/SFMono-Regular.otf",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            # Windows
            "/Windows/Fonts/cour.ttf",
            "/Windows/Fonts/consola.ttf",
        ]

        for path in font_paths:
            try:
                return ImageFont.truetype(path, self.config.font_size)
            except (OSError, IOError):
                continue

        # Fallback to default font
        return ImageFont.load_default()

    def _measure_char(self) -> Tuple[int, int]:
        """Measure character dimensions for the loaded monospace font."""
        test_img = Image.new("RGB", (200, 200))
        test_draw = ImageDraw.Draw(test_img)
        bbox = test_draw.textbbox((0, 0), "M", font=self._font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]

    @staticmethod
    def compact_json(text: str) -> str:
        """Compact JSON by removing unnecessary whitespace.

        Parses the JSON and re-serializes with minimal formatting.
        Falls back to regex-based compaction if JSON parsing fails.

        Args:
            text: JSON string (pretty-printed or compact).

        Returns:
            Compact JSON string with no extra whitespace.
        """
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, separators=(",", ":"))
        except (json.JSONDecodeError, ValueError):
            # Fallback: regex-based compaction
            result = re.sub(r"\s+", " ", text)
            result = re.sub(r"\s*([{}:,\[\]])\s*", r"\1", result)
            return result.strip()

    @staticmethod
    def minify_text(text: str) -> str:
        """Strip visual-only formatting to minimize rendered image area.

        Aggressively compresses text for maximum density in rendered images:
        1. Removes blank lines
        2. Removes markdown heading prefixes (##)
        3. Removes bold/italic markers (**/__)
        4. Collapses multiple spaces
        5. Joins consecutive non-list lines into continuous paragraphs

        Line breaks are only preserved before list items (lines starting
        with ``-`` or ``*`` followed by space) and indented content.
        Everything else flows as continuous text — the renderer's word-wrap
        handles line breaking at the optimal width.

        Args:
            text: Raw text, potentially with markdown formatting.

        Returns:
            Minified text optimized for dense image rendering.
        """
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                continue  # Remove blank lines
            # Remove markdown header prefixes (keep the text)
            stripped = re.sub(r"^#{1,6}\s+", "", stripped)
            # Remove bold/italic markers
            stripped = stripped.replace("**", "").replace("__", "")
            # Collapse multiple spaces (but preserve leading indent)
            leading = len(stripped) - len(stripped.lstrip())
            content = re.sub(r"  +", " ", stripped.lstrip())
            stripped = " " * leading + content
            cleaned.append(stripped)

        # Join lines: preserve newlines only before list items and indented lines.
        # Everything else becomes continuous text for optimal word-wrap.
        if not cleaned:
            return ""

        result_parts = [cleaned[0]]
        for line in cleaned[1:]:
            # Preserve line break before list items (- or * followed by space)
            # and indented content (preserves structure)
            if re.match(r"^\s*[-*]\s", line) or line[0:1] == " ":
                result_parts.append("\n" + line)
            else:
                # Join with space — word-wrap handles the rest
                result_parts.append(" " + line)

        return "".join(result_parts)

    def render(self, text: str) -> List[RenderedImage]:
        """
        Render text to one or more optimized PNG images.

        Large texts are automatically split across multiple images.
        Images use dynamic sizing to minimize token cost.
        If ``config.minify`` is True (default), text is minified first
        to strip visual-only formatting and reduce image area.

        Args:
            text: Text content to render.

        Returns:
            List of RenderedImage objects with token cost metadata.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Content-type-specific compaction
        if self.config.content_type == "json":
            text = self.compact_json(text)
        elif self.config.minify:
            text = self.minify_text(text)
            if not text.strip():
                raise ValueError("Text is empty after minification")

        # Word-wrap text into lines
        wrapped_lines = self._wrap_text(text)

        # Paginate into groups that fit one image each
        pages = self._paginate(wrapped_lines)

        # Render each page with dynamic sizing
        images = [self._render_page(page) for page in pages]

        return images

    def _wrap_text(self, text: str) -> List[str]:
        """
        Word-wrap text to fit within max_width.

        Returns a flat list of lines.
        """
        wrapped = []
        for paragraph in text.split("\n"):
            if not paragraph:
                wrapped.append("")
                continue

            # Check if line fits as-is
            if len(paragraph) <= self._max_chars_per_line:
                wrapped.append(paragraph)
                continue

            # Word-wrap long lines
            words = paragraph.split(" ")
            current_line = ""
            for word in words:
                test_line = "{} {}".format(current_line, word).strip() if current_line else word
                if len(test_line) <= self._max_chars_per_line:
                    current_line = test_line
                else:
                    if current_line:
                        wrapped.append(current_line)
                    # Handle words longer than max width
                    while len(word) > self._max_chars_per_line:
                        wrapped.append(word[: self._max_chars_per_line])
                        word = word[self._max_chars_per_line :]
                    current_line = word
            if current_line:
                wrapped.append(current_line)

        return wrapped

    def _paginate(self, lines: List[str]) -> List[List[str]]:
        """Split wrapped lines into pages that fit on a single image."""
        pages = []
        for i in range(0, len(lines), self._max_lines_per_image):
            page = lines[i : i + self._max_lines_per_image]
            pages.append(page)

        return pages if pages else [lines]

    def _render_page(self, lines: List[str]) -> RenderedImage:
        """
        Render a page of lines with dynamic dimensions.

        Width and height are fitted to content when dynamic_width/height
        are enabled, minimizing token cost.
        """
        # Calculate dimensions
        if self.config.dynamic_width:
            longest_line = max((len(line) for line in lines), default=0)
            content_width = longest_line * self._char_width + 2 * self.config.padding
            img_width = min(self.config.max_width, max(content_width, 50))
        else:
            img_width = self.config.max_width

        if self.config.dynamic_height:
            content_height = len(lines) * self._line_height + 2 * self.config.padding
            img_height = min(self.config.max_height, max(content_height, 20))
        else:
            img_height = self.config.max_height

        # Create image
        image = Image.new("RGB", (img_width, img_height), self.config.background_color)
        draw = ImageDraw.Draw(image)

        # Draw text
        y = self.config.padding
        for line in lines:
            draw.text(
                (self.config.padding, y),
                line,
                fill=self.config.text_color,
                font=self._font,
            )
            y += self._line_height

        token_cost = estimate_image_tokens(img_width, img_height)
        return RenderedImage(image, token_cost=token_cost)

    def compare(self, text: str, model: str = "claude-opus-4-6") -> Dict:
        """Compare text vs image token costs for the given content.

        Renders the text and calculates estimated savings.

        Args:
            text: Text content to analyze.
            model: Model name for pricing. Default: "claude-opus-4-6".

        Returns:
            Dict with text_tokens, image_tokens, input_savings_pct,
            estimated_net_savings_pct, text_cost_usd, image_cost_usd.
        """
        images = self.render(text)
        text_tokens = max(1, len(text) // 4)  # ~4 chars/token estimate
        image_tokens = sum(img.tokens for img in images)

        # Look up pricing
        pricing = None
        for key, p in MODEL_PRICING.items():
            if key in model:
                pricing = p
                break
        if pricing is None:
            pricing = MODEL_PRICING["claude-opus-4-6"]

        input_savings = (text_tokens - image_tokens) / text_tokens if text_tokens > 0 else 0

        # Estimate net savings (assumes optimized prompts, ~equal output tokens)
        # Based on benchmark v2: output inflation is ~0% with good prompts
        text_cost = text_tokens * pricing["input"] / 1_000_000
        image_cost = image_tokens * pricing["input"] / 1_000_000

        return {
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "num_images": len(images),
            "image_dimensions": [
                {"width": img.width, "height": img.height} for img in images
            ],
            "input_savings_pct": round(input_savings * 100, 1),
            "text_cost_per_call": round(text_cost, 8),
            "image_cost_per_call": round(image_cost, 8),
            "model": model,
        }

    # ── Backwards compatibility ──────────────────────────────────────────

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks that fit on a single image.

        Deprecated: use render() directly. Kept for backwards compatibility.
        """
        wrapped = self._wrap_text(text)
        pages = self._paginate(wrapped)
        return ["\n".join(page) for page in pages]

    def _render_chunk(self, text: str) -> RenderedImage:
        """
        Render a single text chunk to an image.

        Deprecated: use render() directly. Kept for backwards compatibility.
        """
        lines = text.split("\n")
        return self._render_page(lines)
