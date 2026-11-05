"""Declarations for source-text tokenization."""

import re
from typing import Literal, NamedTuple

import didactic.api as dx

from lofi.spec.components.data import TokenizerSpec

__all__ = ["RegexTokenizerSettings", "RegexTokenizerSpec"]


class RegexTokenizerSettings(NamedTuple):
    """The regular expression whose nonempty matches are tokens."""

    pattern: str


class RegexTokenizerSpec(TokenizerSpec):
    """Declare tokens as the nonempty matches of one regular expression."""

    kind: Literal["regex"] = "regex"
    pattern: str = r"\S+"

    @dx.validates("pattern")
    @classmethod
    def _pattern_is_usable(cls, pattern: str) -> str:
        """Return a usable token pattern."""
        if not pattern:
            msg = "a tokenizer needs a pattern"
            raise ValueError(msg)
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            msg = f"a tokenizer pattern is invalid: {error}"
            raise ValueError(msg) from error
        if compiled.match("") is not None:
            msg = "a tokenizer pattern cannot match empty text"
            raise ValueError(msg)
        return pattern

    def settings(self) -> RegexTokenizerSettings:
        """Return the validated tokenizer settings."""
        return RegexTokenizerSettings(self.pattern)
