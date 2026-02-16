"""
Core PixelPrompt implementation for rendering text as optimized images.
"""

import base64
import io
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


@dataclass
class RenderConfig:
    """Configuration for text rendering to images."""

    font_size: int = 9
    """Font size in points (range: 6-20). Default: 9."""

    font_family: str = "monospace"
    """Font family: 'monospace', 'serif', or 'sans-serif'. Default: 'monospace'."""

    width: int = 1568
    """Image width in pixels. Default: 1568."""

    height: int = 1568
    """Image height in pixels. Default: 1568."""

    background_color: tuple[int, int, int] = (255, 255, 255)
    """Background color as (R, G, B) tuple. Default: white (255, 255, 255)."""

    text_color: tuple[int, int, int] = (0, 0, 0)
    """Text color as (R, G, B) tuple. Default: black (0, 0, 0)."""

    padding: int = 20
    """Padding in pixels from image edges. Default: 20."""

    line_spacing: float = 1.2
    """Line height multiplier. Default: 1.2."""


class RenderedImage:
    """Represents a single rendered image."""

    def __init__(self, image: Image.Image):
        """Initialize with PIL Image."""
        self._image = image

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._image.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._image.height

    @property
    def size_bytes(self) -> int:
        """Approximate size in bytes (PNG-encoded)."""
        return len(self.png_bytes())

    def png_bytes(self) -> bytes:
        """Get raw PNG bytes."""
        buffer = io.BytesIO()
        self._image.save(buffer, format="PNG")
        return buffer.getvalue()

    def base64(self) -> str:
        """Get base64-encoded PNG for API integration."""
        return base64.b64encode(self.png_bytes()).decode("utf-8")

    def save(self, path: str) -> None:
        """Save image to file."""
        self._image.save(path, format="PNG")


class PixelPrompt:
    """
    Renders text content as optimized PNG images for LLM context compression.

    Example:
        >>> pxl = PixelPrompt()
        >>> images = pxl.render("Long context here...")
        >>> for img in images:
        ...     img.save("output.png")
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

    def _validate_config(self) -> None:
        """Validate configuration values."""
        if not 6 <= self.config.font_size <= 20:
            raise ValueError("font_size must be between 6 and 20")
        if self.config.font_family not in ("monospace", "serif", "sans-serif"):
            raise ValueError("font_family must be 'monospace', 'serif', or 'sans-serif'")
        if self.config.width < 256 or self.config.height < 256:
            raise ValueError("width and height must be at least 256 pixels")
        if self.config.padding < 0:
            raise ValueError("padding must be non-negative")
        if self.config.line_spacing <= 0:
            raise ValueError("line_spacing must be positive")

    def _load_fonts(self) -> None:
        """Load available fonts for the system."""
        font_names = {
            "monospace": ["DejaVuSansMono", "Courier New", "Liberation Mono"],
            "sans-serif": ["DejaVuSans", "Arial", "Liberation Sans"],
            "serif": ["DejaVuSerif", "Times New Roman", "Liberation Serif"],
        }

        family_fonts = font_names.get(self.config.font_family, font_names["monospace"])
        self._font = self._find_font(family_fonts)

    def _find_font(self, font_names: list[str]) -> ImageFont.FreeTypeFont:
        """
        Find an available TrueType font from the list.

        Args:
            font_names: List of font names to try.

        Returns:
            Loaded font or default fallback.
        """
        # Common font paths on different systems
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Monaco.ttf",
            "/Windows/Fonts/cour.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]

        for path in font_paths:
            try:
                return ImageFont.truetype(path, self.config.font_size)
            except (OSError, IOError):
                continue

        # Fallback to default font
        return ImageFont.load_default()

    def render(self, text: str) -> list[RenderedImage]:
        """
        Render text to one or more PNG images.

        Large texts are automatically split across multiple images if needed.

        Args:
            text: Text content to render.

        Returns:
            List of RenderedImage objects.

        Raises:
            ValueError: If text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Split text into chunks if needed
        chunks = self._split_text(text)

        # Render each chunk to an image
        images = [self._render_chunk(chunk) for chunk in chunks]

        return images

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks that fit on a single image.

        Args:
            text: Text to split.

        Returns:
            List of text chunks.
        """
        # Calculate how many lines fit on one image
        available_height = self.config.height - 2 * self.config.padding
        line_height = int(self.config.font_size * self.config.line_spacing)

        if line_height == 0:
            line_height = self.config.font_size

        max_lines = max(1, available_height // line_height)

        # Split text into lines
        lines = text.split("\n")

        # Group lines into chunks
        chunks = []
        current_chunk = []

        for line in lines:
            current_chunk.append(line)
            if len(current_chunk) >= max_lines:
                chunks.append("\n".join(current_chunk))
                current_chunk = []

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks if chunks else [text]

    def _render_chunk(self, text: str) -> RenderedImage:
        """
        Render a single text chunk to an image.

        Args:
            text: Text to render.

        Returns:
            RenderedImage object.
        """
        # Create image with background color
        image = Image.new(
            "RGB",
            (self.config.width, self.config.height),
            self.config.background_color,
        )

        draw = ImageDraw.Draw(image)

        # Draw text
        x = self.config.padding
        y = self.config.padding
        line_height = int(self.config.font_size * self.config.line_spacing)

        for line in text.split("\n"):
            draw.text(
                (x, y),
                line,
                fill=self.config.text_color,
                font=self._font,
            )
            y += line_height

        return RenderedImage(image)
