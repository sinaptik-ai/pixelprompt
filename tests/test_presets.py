"""Tests for content-type presets, compact_json, compare, and prompt helpers."""

import json

import pytest

from pixelprompt import (
    CONTENT_PRESETS,
    MODEL_PRICING,
    PixelPrompt,
    RenderConfig,
    compact_json,
    image_query,
    optimize_prompt,
)
from pixelprompt.prompts import CONCISE_SUFFIX, EXTRACT_SUFFIX, STRUCTURED_SUFFIX


# ═══════════════════════════════════════════════════════════
# Content-type presets
# ═══════════════════════════════════════════════════════════


class TestContentPresets:
    """Test CONTENT_PRESETS dictionary and RenderConfig.for_content()."""

    def test_all_preset_types_exist(self):
        """All four content types should be defined."""
        assert "prose" in CONTENT_PRESETS
        assert "json" in CONTENT_PRESETS
        assert "code" in CONTENT_PRESETS
        assert "config" in CONTENT_PRESETS

    def test_preset_keys(self):
        """Each preset should have required keys."""
        required_keys = {"font_size", "minify", "description", "expected_input_savings", "expected_net_savings"}
        for name, preset in CONTENT_PRESETS.items():
            for key in required_keys:
                assert key in preset, "{} preset missing key: {}".format(name, key)

    def test_preset_font_sizes(self):
        """Font sizes should match benchmark v2 findings."""
        assert CONTENT_PRESETS["prose"]["font_size"] == 9
        assert CONTENT_PRESETS["json"]["font_size"] == 9
        assert CONTENT_PRESETS["code"]["font_size"] == 7
        assert CONTENT_PRESETS["config"]["font_size"] == 8

    def test_preset_minify_flags(self):
        """Code and config should NOT minify; prose and json should."""
        assert CONTENT_PRESETS["prose"]["minify"] is True
        assert CONTENT_PRESETS["json"]["minify"] is True
        assert CONTENT_PRESETS["code"]["minify"] is False
        assert CONTENT_PRESETS["config"]["minify"] is False

    def test_for_content_prose(self):
        """for_content('prose') should return correct config."""
        config = RenderConfig.for_content("prose")
        assert config.font_size == 9
        assert config.minify is True
        assert config.content_type == "prose"

    def test_for_content_json(self):
        """for_content('json') should return correct config."""
        config = RenderConfig.for_content("json")
        assert config.font_size == 9
        assert config.minify is True
        assert config.content_type == "json"

    def test_for_content_code(self):
        """for_content('code') should return correct config."""
        config = RenderConfig.for_content("code")
        assert config.font_size == 7
        assert config.minify is False
        assert config.content_type == "code"

    def test_for_content_config(self):
        """for_content('config') should return correct config."""
        config = RenderConfig.for_content("config")
        assert config.font_size == 8
        assert config.minify is False
        assert config.content_type == "config"

    def test_for_content_invalid_type(self):
        """Invalid content type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown content type"):
            RenderConfig.for_content("html")

    def test_for_content_creates_working_config(self):
        """Config from for_content should work with PixelPrompt."""
        for content_type in CONTENT_PRESETS:
            config = RenderConfig.for_content(content_type)
            pxl = PixelPrompt(config=config)
            images = pxl.render("Hello World, this is a test.")
            assert len(images) >= 1


# ═══════════════════════════════════════════════════════════
# compact_json
# ═══════════════════════════════════════════════════════════


class TestCompactJson:
    """Test JSON compaction functionality."""

    def test_compact_pretty_json(self):
        """Pretty-printed JSON should be compacted."""
        pretty = json.dumps({"name": "test", "value": 42}, indent=2)
        result = compact_json(pretty)
        assert result == '{"name":"test","value":42}'

    def test_compact_already_compact(self):
        """Already compact JSON should be unchanged."""
        compact = '{"a":1,"b":2}'
        result = compact_json(compact)
        assert result == compact

    def test_compact_nested_json(self):
        """Nested JSON should be fully compacted."""
        data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}
        pretty = json.dumps(data, indent=4)
        result = compact_json(pretty)
        parsed = json.loads(result)
        assert parsed == data
        assert "\n" not in result
        assert "  " not in result

    def test_compact_invalid_json_fallback(self):
        """Invalid JSON should use regex fallback."""
        text = '{ "key" : "value" , "list" : [ 1 , 2 ] }'
        result = compact_json(text)
        # Should at least remove extra spaces
        assert len(result) < len(text)

    def test_compact_empty_json(self):
        """Empty JSON objects should work."""
        assert compact_json("{}") == "{}"
        assert compact_json("[]") == "[]"

    def test_compact_json_as_static_method(self):
        """compact_json should be callable as PixelPrompt static method."""
        result = PixelPrompt.compact_json('{"a": 1}')
        assert result == '{"a":1}'

    def test_json_preset_uses_compact(self):
        """Rendering with json content_type should compact JSON."""
        data = json.dumps({"key": "value", "number": 42}, indent=2)
        config = RenderConfig.for_content("json")
        pxl = PixelPrompt(config=config)
        # Should not raise — compaction happens internally
        images = pxl.render(data)
        assert len(images) >= 1


# ═══════════════════════════════════════════════════════════
# compare() method
# ═══════════════════════════════════════════════════════════


class TestCompare:
    """Test cost comparison functionality."""

    def test_compare_returns_dict(self):
        """compare() should return a dict with expected keys."""
        pxl = PixelPrompt()
        result = pxl.compare("Hello world, this is a test of the comparison feature.")
        assert isinstance(result, dict)
        assert "text_tokens" in result
        assert "image_tokens" in result
        assert "input_savings_pct" in result
        assert "num_images" in result
        assert "image_dimensions" in result
        assert "model" in result

    def test_compare_shows_savings(self):
        """Longer text should show positive input savings."""
        text = "Lorem ipsum dolor sit amet. " * 100
        pxl = PixelPrompt()
        result = pxl.compare(text)
        assert result["input_savings_pct"] > 0
        assert result["text_tokens"] > result["image_tokens"]

    def test_compare_with_different_models(self):
        """compare() should accept different model names."""
        pxl = PixelPrompt()
        text = "Test content for comparison."
        for model in MODEL_PRICING:
            result = pxl.compare(text, model=model)
            assert result["model"] == model

    def test_compare_unknown_model_uses_opus(self):
        """Unknown model should fall back to Opus pricing."""
        pxl = PixelPrompt()
        result = pxl.compare("Test content.", model="unknown-model")
        assert result["model"] == "unknown-model"
        # Should still return valid results
        assert result["text_tokens"] > 0

    def test_compare_image_dimensions_match(self):
        """Image dimensions in compare result should match num_images."""
        pxl = PixelPrompt()
        result = pxl.compare("Test content for dimensions.")
        assert len(result["image_dimensions"]) == result["num_images"]


# ═══════════════════════════════════════════════════════════
# Prompt optimization
# ═══════════════════════════════════════════════════════════


class TestOptimizePrompt:
    """Test prompt optimization helpers."""

    def test_concise_style(self):
        """Concise style should append CONCISE_SUFFIX."""
        result = optimize_prompt("What is the port?")
        assert CONCISE_SUFFIX in result

    def test_extract_style(self):
        """Extract style should append EXTRACT_SUFFIX."""
        result = optimize_prompt("Get the API key.", style="extract")
        assert EXTRACT_SUFFIX in result

    def test_structured_style(self):
        """Structured style should append STRUCTURED_SUFFIX."""
        result = optimize_prompt("List all endpoints.", style="structured")
        assert STRUCTURED_SUFFIX in result

    def test_none_style(self):
        """None style should return prompt unchanged."""
        original = "What is the meaning of life?"
        result = optimize_prompt(original, style="none")
        assert result == original

    def test_no_double_suffix(self):
        """Should not duplicate suffix if already present."""
        already = "What? " + CONCISE_SUFFIX
        result = optimize_prompt(already)
        assert result.count(CONCISE_SUFFIX) == 1

    def test_adds_period_if_missing(self):
        """Should add period before suffix if prompt doesn't end with punctuation."""
        result = optimize_prompt("What is the port")
        assert ". " + CONCISE_SUFFIX in result

    def test_no_extra_period_if_present(self):
        """Should not add extra period if prompt already ends with punctuation."""
        result = optimize_prompt("What is the port?")
        assert "?." not in result


class TestImageQuery:
    """Test image_query helper."""

    def test_basic_query(self):
        """Should build a complete image query prompt."""
        result = image_query("What port does the server listen on?")
        assert "Based on the content shown in the image(s):" in result
        assert "What port does the server listen on?" in result
        assert CONCISE_SUFFIX in result

    def test_custom_context(self):
        """Should use custom context instruction."""
        result = image_query(
            "What is the error?",
            context_instruction="Looking at this log file:",
        )
        assert "Looking at this log file:" in result
        assert "What is the error?" in result

    def test_query_with_style(self):
        """Should respect style parameter."""
        result = image_query("List all keys.", style="structured")
        assert STRUCTURED_SUFFIX in result


# ═══════════════════════════════════════════════════════════
# Model pricing
# ═══════════════════════════════════════════════════════════


class TestModelPricing:
    """Test MODEL_PRICING dict."""

    def test_all_models_present(self):
        """Should have Opus, Sonnet, and Haiku."""
        assert "claude-opus-4-6" in MODEL_PRICING
        assert "claude-sonnet-4-5" in MODEL_PRICING
        assert "claude-haiku-4-5" in MODEL_PRICING

    def test_pricing_structure(self):
        """Each model should have input and output pricing."""
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing, "{} missing input price".format(model)
            assert "output" in pricing, "{} missing output price".format(model)
            assert pricing["output"] > pricing["input"], "{} output should be > input".format(model)

    def test_opus_most_expensive(self):
        """Opus should be the most expensive model."""
        assert MODEL_PRICING["claude-opus-4-6"]["input"] >= MODEL_PRICING["claude-sonnet-4-5"]["input"]
        assert MODEL_PRICING["claude-opus-4-6"]["input"] >= MODEL_PRICING["claude-haiku-4-5"]["input"]
