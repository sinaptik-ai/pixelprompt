"""
Prompt optimization helpers for image-mode LLM queries.

Key finding from benchmark v2: output token inflation is the #1 threat
to cost savings when using image-mode. Without prompt optimization,
LLM responses in image-mode are 2-4x more verbose, eating into input
savings. With optimized prompts, output inflation drops to ~0%.

These helpers provide prompt wrappers that suppress verbosity.
"""

from typing import Optional

# ═══════════════════════════════════════════════════════════
# Prompt optimization constants (from benchmark v2)
# ═══════════════════════════════════════════════════════════

# This suffix eliminates output token inflation when querying image-mode content.
# Without it, LLM responses are 2-4x longer (benchmark v1: +102% output inflation).
# With it, inflation drops to ~0% (benchmark v2: +16.4% overall, 0% for most types).
CONCISE_SUFFIX = "Answer with ONLY the answer value. No explanation, no preamble."

# Alternative suffixes for different use cases
EXTRACT_SUFFIX = "Extract and return ONLY the requested value. Nothing else."
STRUCTURED_SUFFIX = "Return ONLY the result in the requested format. No commentary."


def optimize_prompt(prompt: str, style: str = "concise") -> str:
    """Add output-suppression suffix to a prompt for image-mode queries.

    When sending text as images to Claude, the model tends to produce
    longer responses (2-4x more verbose). Adding a concise instruction
    suffix eliminates this inflation and preserves cost savings.

    Args:
        prompt: The user's original prompt/question.
        style: Optimization style:
            - "concise" (default): General-purpose, suppresses verbosity
            - "extract": For data extraction tasks
            - "structured": For structured output (JSON, lists, etc.)
            - "none": Return prompt unchanged

    Returns:
        Prompt with optimization suffix appended.

    Example:
        >>> optimize_prompt("What is the main function's return type?")  # doctest: +SKIP
        "What is the main function's return type? Answer with ONLY the answer value. ..."
    """
    if style == "none":
        return prompt

    suffixes = {
        "concise": CONCISE_SUFFIX,
        "extract": EXTRACT_SUFFIX,
        "structured": STRUCTURED_SUFFIX,
    }

    suffix = suffixes.get(style, CONCISE_SUFFIX)

    # Don't double-add if already present
    if suffix in prompt:
        return prompt

    # Add suffix with proper spacing
    prompt = prompt.rstrip()
    if not prompt.endswith((".", "?", "!", ":")):
        prompt += "."
    return "{} {}".format(prompt, suffix)


def image_query(
    question: str,
    *,
    style: str = "concise",
    context_instruction: Optional[str] = None,
) -> str:
    """Build an optimized prompt for querying image-rendered content.

    Creates a complete prompt that:
    1. References the image content
    2. Asks the question
    3. Adds output-suppression suffix

    Args:
        question: The question to ask about the image content.
        style: Optimization style (see optimize_prompt).
        context_instruction: Optional instruction about the image content.
            Default: "Based on the content shown in the image(s):"

    Returns:
        Complete optimized prompt string.

    Example:
        >>> image_query("What port does the server listen on?")  # doctest: +SKIP
        "Based on the content shown in the image(s): What port does the server listen on? ..."
    """
    if context_instruction is None:
        context_instruction = "Based on the content shown in the image(s):"

    full = "{} {}".format(context_instruction.rstrip(), question.strip())
    return optimize_prompt(full, style=style)
