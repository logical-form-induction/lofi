from collections.abc import MutableMapping, MutableSequence, MutableSet
from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError
from lofi.lexicon import LexiconError
from lofi.lexicon.laws import (
    clamps_are_respected,
    enumeration_is_re_callable,
    positions_are_total,
    reading_does_not_mutate,
)

if TYPE_CHECKING:
    from lofi.lexicon.laws import LexicalSourceFixture

__all__ = [
    "LEXICAL_SOURCE_SUITE_LAWS",
    "LexicalSourceConformance",
    "LexicalSourceSuite",
]

_MUTABLE: Final = (MutableMapping, MutableSequence, MutableSet)

_REFUSAL: Final = "a_clamped_position_is_refused_rather_than_emptied"

_HANDED: Final = "nothing_a_source_hands_out_can_be_mutated_under_it"


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        The counterexample, in that law's own terms.
    law : str
        The law's short name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


class LexicalSourceConformance[T, W](NamedTuple):
    """Inputs for the lexical-source conformance laws.

    Parameters
    ----------
    fixture : LexicalSourceFixture[T, W]
        The builder, the strategy and the example budget the four exported laws take.
    sentence : tuple[str, ...]
        A sentence the source answers without refusing, over which the laws read what an
        accessor hands back.
    unsatisfiable : tuple[str, ...] | None
        A sentence on which a declared clamp covers a position nothing satisfies, or
        None for a source declaring no clamp.
    """

    fixture: LexicalSourceFixture[T, W]
    sentence: tuple[str, ...]
    unsatisfiable: tuple[str, ...] | None = None


class LexicalSourceSuite:
    pytestmark = pytest.mark.conformance

    def test_every_position_indexes_the_sentence_it_came_from[T, W](
        self,
        conformance: LexicalSourceConformance[T, W],
    ) -> None:
        """Check that enumeration is defined everywhere and stays in range.

        Parameters
        ----------
        conformance : LexicalSourceConformance[T, W]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the empty sentence admits an entry, if a position lies outside the
            sentence it came from, or if no generated sentence admitted an entry.
        """
        positions_are_total(conformance.fixture)

    def test_a_source_declaring_itself_re_readable_is_one[T, W](
        self,
        conformance: LexicalSourceConformance[T, W],
    ) -> None:
        """Check that a second reading of a re-readable source agrees.

        Parameters
        ----------
        conformance : LexicalSourceConformance[T, W]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If two readings of one enumeration disagree, or if two calls over one
            sentence disagree.
        """
        enumeration_is_re_callable(conformance.fixture)

    def test_every_declared_clamp_holds_where_it_covers[T, W](
        self,
        conformance: LexicalSourceConformance[T, W],
    ) -> None:
        """Check that an entry at a covered position satisfies the clamp.

        Parameters
        ----------
        conformance : LexicalSourceConformance[T, W]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If an entry comes back at a position a declared clamp covers without
            satisfying it, or if no generated sentence reached such a position.
        """
        clamps_are_respected(conformance.fixture)

    def test_reading_a_source_leaves_it_where_it_was[T, W](
        self,
        conformance: LexicalSourceConformance[T, W],
    ) -> None:
        """Check that what a source enumerates does not depend on its history.

        Parameters
        ----------
        conformance : LexicalSourceConformance[T, W]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If two sources built the same way disagree, or if a reading changes what a
            later one enumerates or what the source declares.
        """
        reading_does_not_mutate(conformance.fixture)

    def test_a_clamped_position_is_refused_rather_than_emptied[T, W](
        self,
        conformance: LexicalSourceConformance[T, W],
    ) -> None:
        """Check that a pin the source cannot meet is reported rather than dropped.

        Parameters
        ----------
        conformance : LexicalSourceConformance[T, W]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the source declares a clamp and the fixture names no sentence to try it
            on, if no clamp reaches a position of the sentence it named, if a covered
            position comes back empty without the call refusing, or if a covered
            position admits an entry every clamp over it satisfies, so that the sentence
            was one the clamps could be met on and the law was never exercised.
        """
        source = conformance.fixture.build()
        clamps = source.clamps

        if not clamps:
            return

        category_matching = conformance.fixture.category_matching
        meaning_matching = conformance.fixture.meaning_matching

        if category_matching is None or meaning_matching is None:
            _fail(
                "a clamped-source fixture declares both clamp congruences",
                law=_REFUSAL,
            )

        tokens = conformance.unsatisfiable

        if tokens is None:
            _fail(
                f"the source declares {clamps!r} and the fixture names no "
                "sentence to try one of them on",
                law=_REFUSAL,
            )

        try:
            admitted = tuple(source.axioms(tokens))

        except LexiconError:
            return

        covered = frozenset(
            position
            for clamp in clamps
            for position in range(clamp.span[0], min(clamp.span[1], len(tokens)))
        )

        if not covered:
            _fail(
                f"no clamp of {clamps!r} reaches a position of {tokens!r}",
                law=_REFUSAL,
            )

        empty = sorted(covered - {position for position, _, _ in admitted})

        if empty:
            _fail(
                f"{tokens!r} came back empty at {empty} where {clamps!r} covers, "
                "and nothing was refused",
                law=_REFUSAL,
            )

        met = tuple(
            (position, entry)
            for position, entry, _ in admitted
            if position in covered
            and all(
                clamp.admits(entry, category_matching, meaning_matching)
                for clamp in clamps
                if clamp.covers(position)
            )
        )

        if met:
            _fail(
                f"{tokens!r} admits {met!r} where {clamps!r} covers, so it is a "
                "sentence the clamps are met on and nothing was refused on it",
                law=_REFUSAL,
            )

    def test_nothing_a_source_hands_out_can_be_mutated_under_it[T, W](
        self,
        conformance: LexicalSourceConformance[T, W],
    ) -> None:
        """Check that an accessor returns nothing the caller can change.

        Parameters
        ----------
        conformance : LexicalSourceConformance[T, W]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the declared clamps or the enumeration come back as a container the
            caller could mutate, which reaches the source's internal state.
        """
        source = conformance.fixture.build()
        handed = (
            ("the clamps", source.clamps),
            ("the enumeration", source.axioms(conformance.sentence)),
        )

        for name, result in handed:
            if isinstance(result, _MUTABLE):
                _fail(
                    f"{name} came back as a {type(result).__name__}, "
                    "which the caller can mutate",
                    law=_HANDED,
                )


LEXICAL_SOURCE_SUITE_LAWS: Final = (
    LexicalSourceSuite.test_every_position_indexes_the_sentence_it_came_from,
    LexicalSourceSuite.test_a_source_declaring_itself_re_readable_is_one,
    LexicalSourceSuite.test_every_declared_clamp_holds_where_it_covers,
    LexicalSourceSuite.test_reading_a_source_leaves_it_where_it_was,
    LexicalSourceSuite.test_a_clamped_position_is_refused_rather_than_emptied,
    LexicalSourceSuite.test_nothing_a_source_hands_out_can_be_mutated_under_it,
)
