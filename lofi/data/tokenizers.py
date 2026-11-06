"""Declared tokenization of collected source text."""

import re
from typing import Final, Protocol, final

from lofi._frozen import FrozenValue
from lofi.data._values import ByteSpan, Token
from lofi.registry import ComponentPlugin
from lofi.spec.tokenizers import RegexTokenizerSettings, RegexTokenizerSpec

__all__ = ["PLUGIN", "RegexTokenizer", "Tokenizer"]


class Tokenizer(Protocol):
    """Segment source text while retaining UTF-8 byte offsets."""

    def __call__(self, text: str, /) -> tuple[Token, ...]:
        """Return tokens in source order."""
        ...


@final
class RegexTokenizer(FrozenValue):
    """Tokens defined by the nonempty matches of one compiled expression."""

    __slots__ = ("_pattern",)
    _noun = "regex tokenizer"

    def __init__(self, pattern: str) -> None:
        object.__setattr__(self, "_pattern", re.compile(pattern))

    @property
    def pattern(self) -> str:
        """Return the regular expression text."""
        return self._pattern.pattern

    def __call__(self, text: str, /) -> tuple[Token, ...]:
        """Return matched tokens with source-text UTF-8 byte spans."""
        offsets = [0]
        for character in text:
            offsets.append(offsets[-1] + len(character.encode("utf-8")))
        return tuple(
            Token(
                index,
                match.group(),
                ByteSpan(offsets[match.start()], offsets[match.end()]),
            )
            for index, match in enumerate(self._pattern.finditer(text))
        )


def _build(settings: RegexTokenizerSettings, context: None, /) -> RegexTokenizer:
    """Build a regex tokenizer from validated settings."""
    del context
    return RegexTokenizer(settings.pattern)


PLUGIN: Final = ComponentPlugin(spec=RegexTokenizerSpec, build=_build)
"""The regex tokenizer plugin."""
