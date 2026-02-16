"""
Utility functions for PixelPrompt.
"""

from .core import estimate_image_tokens


def estimate_tokens(text: str, model: str = "claude-opus-4-6-20250219") -> int:
    """
    Estimate token count for text using Claude's token counting rules.

    Uses a simple approximation: ~4 characters per token on average.
    For precise counts, use the Anthropic token counting API.

    Args:
        text: Text to estimate tokens for.
        model: Model name (for future use with API-based counting).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0

    # Rough approximation: ~4 characters per token
    # This is a conservative estimate for most Claude models
    return max(1, len(text) // 4)


def estimate_compression_ratio(
    original_text: str,
    num_images: int,
    image_width: int = 1568,
    image_height: int = 300,
) -> float:
    """
    Estimate compression ratio when rendering text as images.

    Uses the actual Claude vision token formula: (w*h)/750.

    Args:
        original_text: Original text content.
        num_images: Number of images generated.
        image_width: Average image width in pixels.
        image_height: Average image height in pixels.

    Returns:
        Estimated compression ratio (original tokens / image tokens).
    """
    original_tokens = estimate_tokens(original_text)

    # Use actual Claude vision token formula
    image_tokens = max(1, num_images * estimate_image_tokens(image_width, image_height))

    return original_tokens / image_tokens if image_tokens > 0 else 1.0
