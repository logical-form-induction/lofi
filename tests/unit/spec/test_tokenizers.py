"""Tests for tokenizer declarations."""

import didactic.api as dx
import pytest

from lofi.spec.tokenizers import RegexTokenizerSpec


def test_a_regex_tokenizer_requires_nonempty_matches() -> None:
    """Reject expressions that could emit empty tokens."""
    with pytest.raises(dx.ValidationError, match="cannot match empty"):
        RegexTokenizerSpec(pattern=r"\s*")

    with pytest.raises(dx.ValidationError, match="invalid"):
        RegexTokenizerSpec(pattern="[")


def test_a_regex_tokenizer_publishes_its_validated_pattern() -> None:
    """Expose the pattern through the plugin settings boundary."""
    spec = RegexTokenizerSpec(pattern=r"\w+|[^\w\s]")

    assert spec.settings().pattern == r"\w+|[^\w\s]"
