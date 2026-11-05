"""Tests for declared source-text tokenization."""

from lofi.data import ByteSpan
from lofi.data.tokenizers import RegexTokenizer


def test_regex_tokens_retain_utf8_byte_offsets() -> None:
    """Measure offsets in encoded bytes rather than Unicode code points."""
    tokens = RegexTokenizer(r"\w+|[^\w\s]")("A café opens.")

    assert tuple(token.surface for token in tokens) == ("A", "café", "opens", ".")
    assert tuple(token.span for token in tokens) == (
        ByteSpan(0, 1),
        ByteSpan(2, 7),
        ByteSpan(8, 13),
        ByteSpan(13, 14),
    )
