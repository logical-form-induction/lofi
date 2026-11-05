"""Chart-weight semirings."""

from collections.abc import Hashable

from lofi._native import (
    Boolean,
    Counting,
    Expectation,
    KBest,
    LogInside,
    PythonCarrier,
    Viterbi,
)

__all__ = [
    "Boolean",
    "Counting",
    "Derivations",
    "Expectation",
    "Expected",
    "Forest",
    "KBest",
    "LogInside",
    "Ranked",
    "Truth",
    "Viterbi",
]

type Truth = bool
"""Boolean derivability weight."""

type Ranked = tuple[float, ...]
"""Stores descending k-best log probabilities."""

type Derivations[E: Hashable] = frozenset[tuple[E, ...]]
"""Stores derivations as edge sequences."""

type Expected = tuple[float, tuple[float, ...]]
"""Stores a probability and accumulated expectations."""


class Forest[E: Hashable]:
    """Tracks every derivation as edge sequences."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the name the algebra is selected by."""
        return "forest"

    @property
    def log_probability(self) -> bool:
        """Report that derivation sets are not log probabilities."""
        return False

    @property
    def zero(self) -> Derivations[E]:
        """Return an empty set for no derivation."""
        return frozenset()

    @property
    def one(self) -> Derivations[E]:
        """Return the derivation that uses no edge."""
        return frozenset({()})

    @property
    def lowering(self) -> PythonCarrier:
        """Select Python-object weights in the native chart kernel."""
        return PythonCarrier()

    def plus(self, a: Derivations[E], b: Derivations[E], /) -> Derivations[E]:
        """Return the derivations of two alternatives together.

        Parameters
        ----------
        a : Derivations[E]
            One set of derivations.
        b : Derivations[E]
            The other.

        Returns
        -------
        Derivations[E]
            Their union.
        """
        return a | b

    def times(self, a: Derivations[E], b: Derivations[E], /) -> Derivations[E]:
        """Return the derivations two antecedents combine into.

        Parameters
        ----------
        a : Derivations[E]
            The left antecedent's derivations.
        b : Derivations[E]
            The right antecedent's derivations.

        Returns
        -------
        Derivations[E]
            Every left derivation followed by every right one.
        """
        return frozenset(x + y for x in a for y in b)

    def is_zero(self, a: Derivations[E], /) -> bool:
        """Return whether an item has no derivation.

        Parameters
        ----------
        a : Derivations[E]
            The weight to test.

        Returns
        -------
        bool
            Whether the set is empty.
        """
        return len(a) == 0
