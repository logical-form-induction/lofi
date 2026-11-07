"""The open grammar declaration root."""

import didactic.api as dx

from lofi.spec.grammar import _lexicon

__all__ = ["GrammarSpec"]


class GrammarSpec(dx.TaggedUnion, discriminator="formalism", extra="forbid"):
    """A complete or concise grammar declaration selected by formalism."""

    lexicon: _lexicon.LexiconSpec
