"""
Utility functions for PixelPrompt.
"""


def estimate_tokens(text: str, model: str = "claude-3-5-sonnet-20241022") -> int:
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


def estimate_compression_ratio(original_text: str, num_images: int) -> float:
    """
    Estimate compression ratio when rendering text as images.

    Args:
        original_text: Original text content.
        num_images: Number of images generated.

    Returns:
        Estimated compression ratio (original tokens / image tokens).
    """
    original_tokens = estimate_tokens(original_text)

    # Each image is typically counted as ~1-2 tokens by Claude vision
    image_tokens = max(1, num_images * 2)

    return original_tokens / image_tokens if image_tokens > 0 else 1.0
