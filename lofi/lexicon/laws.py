"""Lexical-source conformance laws."""

from typing import TYPE_CHECKING, Final, NamedTuple

from hypothesis import given, settings

from lofi._checks import LawViolationError, fail_law

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from hypothesis.strategies import SearchStrategy

    from lofi.lexicon._ports import Entry, LexicalSource
    from lofi.terms import Matching

__all__ = [
    "LEXICAL_SOURCE_LAWS",
    "LexicalSourceFixture",
    "check_lexical_source_laws",
    "clamps_are_respected",
    "enumeration_is_re_callable",
    "positions_are_total",
    "reading_does_not_mutate",
]


type _Axioms[T, W] = tuple[tuple[int, Entry[T], W], ...]
"""One lexical-source enumeration."""


class LexicalSourceFixture[T, W](NamedTuple):
    """Inputs for lexical-source law checks.

    Attributes
    ----------
    build : Callable[[], LexicalSource[T, W]]
        Factory for a fresh source.
    tokens : SearchStrategy[tuple[str, ...]]
        Token-sequence strategy.
    max_examples : int
        Maximum Hypothesis examples per law.
    category_matching : Matching[T] | None
        Congruence used to check category clamps, required when clamps are present.
    meaning_matching : Matching[T] | None
        Congruence used to check meaning clamps, required when clamps are present.
    """

    build: Callable[[], LexicalSource[T, W]]
    tokens: SearchStrategy[tuple[str, ...]]
    max_examples: int = 200
    category_matching: Matching[T] | None = None
    meaning_matching: Matching[T] | None = None


def _budget(max_examples: int) -> settings:
    """Return Hypothesis settings for a law.

    Parameters
    ----------
    max_examples : int
        Maximum generated examples.

    Returns
    -------
    settings
        Settings with no per-example deadline.
    """
    # disable per-example deadlines because runtime varies by machine.
    return settings(max_examples=max_examples, deadline=None)


def _call[T, W](
    source: LexicalSource[T, W], tokens: tuple[str, ...], *, law: str
) -> Iterable[tuple[int, Entry[T], W]]:
    """Return a source enumeration and wrap failures as law violations.

    Parameters
    ----------
    source : LexicalSource[T, W]
        Source to call.
    tokens : tuple[str, ...]
        Input tokens.
    law : str
        Current law name.

    Returns
    -------
    Iterable[tuple[int, Entry[T], W]]
        Source enumeration.

    Raises
    ------
    LawViolationError
        If the source raises.
    """
    try:
        return source.axioms(tokens)

    except Exception as exc:
        message = f"{law}: enumerating {tokens!r} raised {exc!r}"

        raise LawViolationError(message) from exc


def _drain[T, W](
    enumeration: Iterable[tuple[int, Entry[T], W]], *, law: str
) -> _Axioms[T, W]:
    """Return one enumeration reading and wrap failures.

    Parameters
    ----------
    enumeration : Iterable[tuple[int, Entry[T], W]]
        Enumeration to consume.
    law : str
        Current law name.

    Returns
    -------
    _Axioms[T, W]
        Materialized enumeration.

    Raises
    ------
    LawViolationError
        If consuming the enumeration raises.
    """
    try:
        return tuple(enumeration)

    except Exception as exc:
        message = f"{law}: reading an enumeration raised {exc!r}"

        raise LawViolationError(message) from exc


def _read[T, W](
    source: LexicalSource[T, W], tokens: tuple[str, ...], *, law: str
) -> _Axioms[T, W]:
    """Return what a source enumerates for one sentence.

    Parameters
    ----------
    source : LexicalSource[T, W]
        Source to read.
    tokens : tuple[str, ...]
        Input tokens.
    law : str
        Current law name.

    Returns
    -------
    _Axioms[T, W]
        Materialized source output.
    """
    return _drain(_call(source, tokens, law=law), law=law)


def _agree[T, W](left: _Axioms[T, W], right: _Axioms[T, W]) -> bool:
    """Return whether two readings contain the same axioms in any order.

    Parameters
    ----------
    left : _Axioms[T, W]
        First reading.
    right : _Axioms[T, W]
        Second reading.

    Returns
    -------
    bool
        Whether the readings are equal as multisets.
    """
    if len(left) != len(right):
        return False

    remaining = list(right)

    for axiom in left:
        for index, other in enumerate(remaining):
            if axiom == other:
                del remaining[index]
                break

        else:
            return False

    return True


def positions_are_total[T, W](fixture: LexicalSourceFixture[T, W]) -> None:
    """Check that enumeration is defined and positions index the sentence.

    Parameters
    ----------
    fixture : LexicalSourceFixture[T, W]
        Source factory and generated inputs.

    Raises
    ------
    LawViolationError
        If enumeration fails or yields an invalid position.
    """
    name = "totality"
    empty = _read(fixture.build(), (), law=name)

    if empty:
        fail_law(f"the empty sentence admitted {empty!r}", law=name)

    seen: list[int] = []

    @given(fixture.tokens)
    @_budget(fixture.max_examples)
    def law(tokens: tuple[str, ...]) -> None:
        span = range(len(tokens))

        for position, entry, _ in _read(fixture.build(), tokens, law=name):
            seen.append(position)

            if position not in span:
                detail = (
                    f"{entry!r} came back at position {position!r} of "
                    f"{tokens!r}, which has {len(tokens)} tokens"
                )

                fail_law(detail, law=name)

    law()

    if not seen:
        detail = (
            "no generated sentence admitted an entry, so totality was never exercised"
        )

        fail_law(detail, law=name)


def enumeration_is_re_callable[T, W](fixture: LexicalSourceFixture[T, W]) -> None:
    """Check that a source declaring itself re-readable is one.

    Parameters
    ----------
    fixture : LexicalSourceFixture[T, W]
        Source factory and generated inputs.

    Raises
    ------
    LawViolationError
        If repeated reads disagree or never exercise the law.
    """
    name = "re-callability"
    source = fixture.build()

    if not source.is_re_callable:
        return

    seen: list[int] = []

    @given(fixture.tokens)
    @_budget(fixture.max_examples)
    def law(tokens: tuple[str, ...]) -> None:
        enumeration = _call(source, tokens, law=name)
        first = _drain(enumeration, law=name)
        second = _drain(enumeration, law=name)
        seen.extend(position for position, _, _ in first)

        if not _agree(first, second):
            detail = (
                f"reading one enumeration of {tokens!r} twice gives {first!r} "
                f"and then {second!r}"
            )

            fail_law(detail, law=name)

        again = _read(source, tokens, law=name)

        if not _agree(first, again):
            detail = f"enumerating {tokens!r} twice gives {first!r} and then {again!r}"

            fail_law(detail, law=name)

    law()

    if not seen:
        detail = (
            "no generated sentence admitted an entry, so re-callability was "
            "never exercised"
        )

        fail_law(detail, law=name)


def clamps_are_respected[T, W](fixture: LexicalSourceFixture[T, W]) -> None:
    """Check that each declared clamp holds where it applies.

    Parameters
    ----------
    fixture : LexicalSourceFixture[T, W]
        Source factory and generated inputs.

    Raises
    ------
    LawViolationError
        If an entry violates a clamp or inputs never exercise clamping.
    """
    name = "clamping"
    clamps = fixture.build().clamps

    if not clamps:
        return

    category_matching = fixture.category_matching
    meaning_matching = fixture.meaning_matching

    if category_matching is None or meaning_matching is None:
        detail = "a clamped-source fixture declares both clamp congruences"
        fail_law(detail, law=name)

    covered: list[int] = []

    @given(fixture.tokens)
    @_budget(fixture.max_examples)
    def law(tokens: tuple[str, ...]) -> None:
        for position, entry, _ in _read(fixture.build(), tokens, law=name):
            for clamp in clamps:
                if not clamp.covers(position):
                    continue

                covered.append(position)

                if not clamp.admits(entry, category_matching, meaning_matching):
                    detail = (
                        f"{clamp!r} covers position {position!r} of {tokens!r}, "
                        f"where {entry!r} came back"
                    )

                    fail_law(detail, law=name)

    law()

    if not covered:
        detail = (
            f"no generated sentence admitted an entry where {clamps!r} covers, "
            f"so clamping was never exercised"
        )

        fail_law(detail, law=name)


def reading_does_not_mutate[T, W](fixture: LexicalSourceFixture[T, W]) -> None:
    """Check that prior reads do not change source enumeration.

    Parameters
    ----------
    fixture : LexicalSourceFixture[T, W]
        Source factory and generated inputs.

    Raises
    ------
    LawViolationError
        If source declarations or enumeration change after a read.
    """
    name = "non-mutation"
    seen: list[int] = []

    @given(fixture.tokens, fixture.tokens)
    @_budget(fixture.max_examples)
    def law(before: tuple[str, ...], tokens: tuple[str, ...]) -> None:
        want = _read(fixture.build(), tokens, law=name)
        got = _read(fixture.build(), tokens, law=name)
        seen.extend(position for position, _, _ in got)

        if not _agree(got, want):
            detail = (
                f"two sources built the same way enumerate {want!r} and "
                f"{got!r} over {tokens!r}"
            )

            fail_law(detail, law=name)

        used = fixture.build()
        declared = (used.is_re_callable, used.clamps)
        _read(used, before, law=name)

        if (used.is_re_callable, used.clamps) != declared:
            detail = (
                f"reading {before!r} changed what the source declares, from "
                f"{declared!r} to {(used.is_re_callable, used.clamps)!r}"
            )

            fail_law(detail, law=name)

        if not used.is_re_callable:
            return

        after = _read(used, tokens, law=name)

        if not _agree(after, want):
            detail = (
                f"reading {before!r} first changes what {tokens!r} enumerates, "
                f"from {want!r} to {after!r}"
            )

            fail_law(detail, law=name)

    law()

    if not seen:
        detail = (
            "no generated sentence admitted an entry, so non-mutation was "
            "never exercised"
        )

        fail_law(detail, law=name)


LEXICAL_SOURCE_LAWS: Final = (
    positions_are_total,
    enumeration_is_re_callable,
    clamps_are_respected,
    reading_does_not_mutate,
)
"""Lexical-source laws in conformance order."""


def check_lexical_source_laws[T, W](fixture: LexicalSourceFixture[T, W]) -> None:
    """Run every lexical-source law against one source.

    Parameters
    ----------
    fixture : LexicalSourceFixture[T, W]
        Source factory and generated inputs.

    Raises
    ------
    LawViolationError
        If any lexical-source law fails.
    """
    for law in LEXICAL_SOURCE_LAWS:
        law(fixture)
