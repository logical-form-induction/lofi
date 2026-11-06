"""The component family of generators proposing bead lexicons.

A generator declaration names one installed generator, registered under the
``lofi.lexicon_generator`` entry point group, whose built generator proposes
one bead lexicon a judgment item construction fills slots from.
"""

from typing import Final

import didactic.api as dx

from lofi.spec.components.base import ComponentSpec

__all__ = ["GENERATOR_FAMILY", "LexiconGeneratorSpec"]

GENERATOR_FAMILY: Final = "lexicon_generator"
"""The entry point family the installed lexicon generators register under."""


class LexiconGeneratorSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declare a generator proposing one bead lexicon."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry point group lists the generators."""
        return GENERATOR_FAMILY
