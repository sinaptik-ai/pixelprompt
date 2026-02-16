"""
PixelPrompt: Compress LLM context by rendering text as optimized images.

Based on the Pixels Beat Tokens research (Venturi, 2026).
"""

__version__ = "0.3.1"
__author__ = "Gabriele Venturi"
__email__ = "gabriele@sinaptik.ai"

from .core import PixelPrompt, RenderConfig, RenderedImage, estimate_image_tokens
from .utils import estimate_tokens

__all__ = [
    "PixelPrompt",
    "RenderConfig",
    "RenderedImage",
    "estimate_image_tokens",
    "estimate_tokens",
    "minify_text",
]

# Convenience alias for standalone use
minify_text = PixelPrompt.minify_text
