"""
PixelPrompt: Compress LLM context by rendering text as optimized images.

Based on research exploring multimodal LLM capabilities and token efficiency.
"""

__version__ = "0.1.0"
__author__ = "Gabriele Venturi"
__email__ = "gabriele@sinaptik.ai"

from .core import PixelPrompt, RenderConfig, RenderedImage
from .utils import estimate_tokens

__all__ = [
    "PixelPrompt",
    "RenderConfig",
    "RenderedImage",
    "estimate_tokens",
]
