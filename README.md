# PixelPrompt

Compress LLM context by rendering text as optimized images. Based on the research paper *"Pixels Beat Tokens"* (Venturi, 2026).

## Why PixelPrompt?

When working with LLMs, token counts directly impact cost and latency. PixelPrompt converts text content into visually optimized PNG images, achieving **38-80% net cost savings** compared to raw text tokens, with **100% accuracy** on a 125-question benchmark (Claude Opus 4.6).

**Key benefits:**
- 💰 **38-80% net cost savings** — input token reduction minus output cost, with optimized prompts
- 🎯 **100% accuracy** — verified across prose, code, JSON, and config content types
- 📊 **Content-type presets** — optimal settings for prose, code, JSON, and config files
- 🔄 **Automatic splitting** — large content automatically split across multiple images
- 📝 **Prompt optimization** — built-in helpers to eliminate output token inflation
- 🚀 **Easy integration** — simple API for any Claude workflow

## Installation

```bash
pip install pixelprompt
```

## Quick Start

```python
from pixelprompt import PixelPrompt, RenderConfig

# Use content-type presets for optimal settings
pxl = PixelPrompt(RenderConfig.for_content("json"))
images = pxl.render(json_data)

# Or use default settings (good for prose)
pxl = PixelPrompt()
images = pxl.render("Your long context here...")

# Use with Claude API
from anthropic import Anthropic

client = Anthropic()
message = client.messages.create(
    model="claude-opus-4-6-20250219",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                *[img.to_content_block() for img in images],
                {
                    "type": "text",
                    "text": "What are the key points? Answer with ONLY the answer value. No explanation, no preamble."
                }
            ]
        }
    ]
)
```

## Content-Type Presets

Different content types have different optimal rendering settings. Use presets for best results:

```python
from pixelprompt import RenderConfig, CONTENT_PRESETS

# Prose: font 9, minify=True — 69% net savings
config = RenderConfig.for_content("prose")

# JSON: font 9, minify=True (compact JSON) — 80% net savings
config = RenderConfig.for_content("json")

# Code: font 7, minify=False — 59% net savings
config = RenderConfig.for_content("code")

# Config (YAML, TOML, INI): font 8, minify=False — 39% net savings
config = RenderConfig.for_content("config")
```

| Content Type | Font Size | Minify | Input Savings | Net Savings |
|:-------------|:---------:|:------:|:-------------:|:-----------:|
| JSON         | 9         | Yes    | 83%           | **80%**     |
| Prose        | 9         | Yes    | 71%           | **69%**     |
| Code         | 7         | No     | 61%           | **59%**     |
| Config       | 8         | No     | 41%           | **39%**     |

## Prompt Optimization

**Critical:** Without prompt optimization, image-mode responses are 2-4x more verbose, eating into your savings. Use the built-in helpers:

```python
from pixelprompt import optimize_prompt, image_query

# Wrap any prompt with output-suppression
prompt = optimize_prompt("What is the server port?")
# → "What is the server port? Answer with ONLY the answer value. No explanation, no preamble."

# Build a complete image-query prompt
prompt = image_query("What functions are defined?", style="concise")
# → "Based on the content shown in the image(s): What functions are defined? Answer with ONLY the answer value. No explanation, no preamble."
```

Available styles: `"concise"` (default), `"extract"`, `"structured"`, `"none"`.

## Cost Comparison

Compare text vs image costs before committing:

```python
pxl = PixelPrompt(RenderConfig.for_content("json"))
result = pxl.compare(json_data, model="claude-opus-4-6")

print(f"Text tokens: {result['text_tokens']}")
print(f"Image tokens: {result['image_tokens']}")
print(f"Input savings: {result['input_savings_pct']}%")
```

## Configuration

```python
from pixelprompt import PixelPrompt, RenderConfig

config = RenderConfig(
    font_size=9,           # Range: 6-20, default: 9
    font_family="monospace",  # "monospace", "serif", "sans-serif"
    max_width=1568,        # Max image width (Claude vision max: 1568)
    max_height=1568,       # Max image height (Claude vision max: 1568)
    dynamic_width=True,    # Fit width to content (saves tokens)
    dynamic_height=True,   # Fit height to content (saves tokens)
    minify=True,           # Strip markdown formatting for density
    content_type=None,     # Or use RenderConfig.for_content()
    padding=5,             # Minimal padding for density
    line_spacing=1,        # Tight line spacing
)

pxl = PixelPrompt(config=config)
```

## API Reference

### `PixelPrompt`

```python
class PixelPrompt:
    def __init__(self, config: RenderConfig | None = None): ...
    def render(self, text: str) -> list[RenderedImage]: ...
    def compare(self, text: str, model: str = "claude-opus-4-6") -> dict: ...

    @staticmethod
    def minify_text(text: str) -> str: ...

    @staticmethod
    def compact_json(text: str) -> str: ...
```

### `RenderConfig`

```python
@dataclass
class RenderConfig:
    font_size: int = 9
    font_family: str = "monospace"
    max_width: int = 1568
    max_height: int = 1568
    dynamic_width: bool = True
    dynamic_height: bool = True
    minify: bool = True
    content_type: str | None = None

    @staticmethod
    def for_content(content_type: str) -> RenderConfig: ...
```

### `RenderedImage`

```python
class RenderedImage:
    width: int          # Image width in pixels
    height: int         # Image height in pixels
    tokens: int         # Estimated Claude vision token cost
    size_bytes: int     # PNG file size in bytes

    def png_bytes(self) -> bytes: ...
    def base64(self) -> str: ...
    def to_content_block(self) -> dict: ...  # Anthropic API format
    def save(self, path: str) -> None: ...
```

### Prompt Helpers

```python
def optimize_prompt(prompt: str, style: str = "concise") -> str: ...
def image_query(question: str, *, style: str = "concise", context_instruction: str | None = None) -> str: ...
```

## Performance

Net cost savings by content type (Opus 4.6, with optimized prompts):

| Content Type | Net Savings | Notes |
|:-------------|:-----------:|:------|
| JSON/Structured | 80% | Compact JSON + high density |
| Long Prose | 69% | Minification + word wrap |
| Source Code | 59% | Smaller font, preserve formatting |
| Config Files | 39% | Moderate density |

Rendering time: ~100-200ms per image on modern hardware.

## License

MIT License — see LICENSE file for details.

## Citation

If you use PixelPrompt in research, please cite:

```bibtex
@software{pixelprompt,
  author = {Venturi, Gabriele},
  title = {PixelPrompt: Compress LLM Context by Rendering Text as Images},
  year = {2026},
  publisher = {Sinaptik GmbH},
  url = {https://github.com/sinaptik-ai/pixelprompt}
}
```
