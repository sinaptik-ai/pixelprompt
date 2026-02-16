# PixelPrompt

Compress LLM context by rendering text as optimized images. Based on the research paper *"Pixels Beat Tokens: Multimodal LLMs See Better With Image Sources for Text-Rich VQA"*.

## Why PixelPrompt?

When working with LLMs, token counts directly impact cost and latency. PixelPrompt converts text content into visually optimized PNG images, achieving **4-8x compression** compared to raw text tokens, while maintaining or improving accuracy.

**Key benefits:**
- 🎯 **Significant token savings** — text rendered as images uses fewer tokens
- 📊 **Flexible formatting** — control font size, layout, and visual hierarchy
- 🔄 **Automatic splitting** — large content automatically split across multiple images
- 🎨 **Configurable rendering** — customize fonts, colors, background
- 🚀 **Easy integration** — simple API for any LLM workflow

## Installation

```bash
uv pip install pixelprompt
```

Or with pip:
```bash
pip install pixelprompt
```

## Quick Start

```python
from pixelprompt import PixelPrompt

# Initialize with default settings
pxl = PixelPrompt()

# Render text as image(s)
text = "Your long context here..."
images = pxl.render(text)

# Use with Claude API
from anthropic import Anthropic

client = Anthropic()
message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this document:"
                },
                *[
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img.base64()
                        }
                    }
                    for img in images
                ],
                {
                    "type": "text",
                    "text": "What are the key points?"
                }
            ]
        }
    ]
)

print(message.content[0].text)
```

## Configuration

```python
from pixelprompt import PixelPrompt, RenderConfig

config = RenderConfig(
    font_size=9,  # Default: 9 (range: 6-20)
    font_family="monospace",  # Default: "monospace"
    width=1568,  # Image width in pixels (default: 1568)
    height=1568,  # Image height in pixels (default: 1568)
    background_color=(255, 255, 255),  # RGB tuple (default: white)
    text_color=(0, 0, 0),  # RGB tuple (default: black)
    padding=20,  # Padding in pixels (default: 20)
    line_spacing=1.2,  # Line height multiplier (default: 1.2)
)

pxl = PixelPrompt(config=config)
images = pxl.render(text)
```

## Advanced Usage

### Analyze compression metrics

```python
from pixelprompt import estimate_tokens

text = "Your long context..."
original_tokens = estimate_tokens(text)
compressed_tokens = estimate_tokens(f"[Image with compressed content]")

compression_ratio = original_tokens / compressed_tokens
print(f"Compression: {compression_ratio:.1f}x")
```

### Handle large documents

```python
# Automatically splits into multiple images if content exceeds limits
images = pxl.render(long_document)
print(f"Generated {len(images)} images")

# Access individual images
for i, img in enumerate(images):
    img.save(f"page_{i}.png")
    print(f"Image {i}: {img.width}x{img.height}, size: {img.size_bytes} bytes")
```

### Custom fonts

```python
config = RenderConfig(
    font_family="serif",  # Options: "monospace", "serif", "sans-serif"
    font_size=10,
)
pxl = PixelPrompt(config=config)
```

## API Reference

### `PixelPrompt`

Main class for rendering text to images.

```python
class PixelPrompt:
    def __init__(self, config: RenderConfig | None = None):
        """Initialize with optional configuration."""

    def render(self, text: str) -> list[RenderedImage]:
        """
        Render text to one or more PNG images.

        Args:
            text: Text content to render

        Returns:
            List of RenderedImage objects
        """
```

### `RenderConfig`

Configuration dataclass for rendering parameters.

```python
@dataclass
class RenderConfig:
    font_size: int = 9
    font_family: str = "monospace"
    width: int = 1568
    height: int = 1568
    background_color: tuple[int, int, int] = (255, 255, 255)
    text_color: tuple[int, int, int] = (0, 0, 0)
    padding: int = 20
    line_spacing: float = 1.2
```

### `RenderedImage`

Represents a single rendered image.

```python
class RenderedImage:
    width: int
    height: int
    size_bytes: int

    def png_bytes(self) -> bytes:
        """Get raw PNG bytes."""

    def base64(self) -> str:
        """Get base64-encoded PNG for API integration."""

    def save(self, path: str) -> None:
        """Save to file."""
```

## Performance

Typical compression ratios (depends on content):
- **Code**: 4-6x compression
- **Technical prose**: 5-8x compression
- **JSON/Structured data**: 3-5x compression
- **Natural language**: 4-7x compression

Rendering time: ~100-200ms per image on modern hardware.

## Contributing

Contributions welcome! Please open issues or PRs on GitHub.

## License

MIT License — see LICENSE file for details.

## Citation

If you use PixelPrompt in research, please cite:

```bibtex
@software{pixelprompt,
  author = {Venturi, Gabriele},
  title = {PixelPrompt: Compress LLM Context by Rendering Text as Images},
  year = {2026},
  url = {https://github.com/sinaptik-ai/pixelprompt}
}
```
