"""
PixelPrompt: Compress LLM context by rendering text as optimized images.

Based on the Pixels Beat Tokens research (Venturi, 2026).
Benchmark v2: 100% accuracy, 38-80% net cost savings with optimized prompts.
"""

__version__ = "0.4.0"
__author__ = "Gabriele Venturi"
__email__ = "gabriele@sinaptik.ai"

from .core import (
    CONTENT_PRESETS,
    MODEL_PRICING,
    PixelPrompt,
    RenderConfig,
    RenderedImage,
    estimate_image_tokens,
)
from .prompts import CONCISE_SUFFIX, image_query, optimize_prompt
from .utils import estimate_tokens

__all__ = [
    "PixelPrompt",
    "RenderConfig",
    "RenderedImage",
    "CONTENT_PRESETS",
    "MODEL_PRICING",
    "estimate_image_tokens",
    "estimate_tokens",
    "minify_text",
    "compact_json",
    "optimize_prompt",
    "image_query",
    "CONCISE_SUFFIX",
]

# Convenience aliases for standalone use
minify_text = PixelPrompt.minify_text
compact_json = PixelPrompt.compact_json
