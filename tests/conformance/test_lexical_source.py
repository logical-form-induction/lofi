from typing import TYPE_CHECKING, Final, override

import pytest
from hypothesis import strategies as st
from tests.conformance.lexical_source import (
    LEXICAL_SOURCE_SUITE_LAWS,
    LexicalSourceConformance,
    LexicalSourceSuite,
)

from lofi._checks import LawViolationError
from lofi.lexicon import Clamp, Entry, LexiconError
from lofi.lexicon.laws import LexicalSourceFixture

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

FORMS: Final = ("the", "cat", "sleeps", "aardvark")

TABLE: Mapping[str, tuple[Entry[str], ...]] = {
    "the": (Entry("the", "NP/N", weight=-0.1),),
    "cat": (Entry("cat", "N", "cat", -0.5), Entry("cat", "NP", "cat", -3.0)),
    "sleeps": (Entry("sleeps", "S\\NP", "sleep", -0.2),),
}

PINNED: Final = 1

NOUN: Clamp[str] = Clamp((PINNED, PINNED + 1), category={"N"})

SENTENCE: Final = ("the", "cat", "sleeps")

UNSATISFIABLE: Final = ("the", "the", "sleeps")

EXAMPLES: Final = 50


class Equality[T]:
    """Value equality used by the string-valued fixtures."""

    name = "equality"
    has_key = True

    def key(self, value: T, /) -> T:
        """Return the value as its representative."""
        return value

    def equivalent(self, left: T, right: T, /) -> bool:
        """Return whether two values compare equal."""
        return left == right


EQUALITY: Final = Equality[str]()


def admits(clamp: Clamp[str], entry: Entry[str]) -> bool:
    """Return whether the fixture clamp admits the entry under value equality."""
    return clamp.admits(entry, EQUALITY, EQUALITY)


SENTENCES = st.tuples(
    st.sampled_from(FORMS),
    st.just("cat"),
    st.lists(st.sampled_from(FORMS), max_size=2).map(tuple),
).map(lambda drawn: (drawn[0], drawn[1], *drawn[2]))


class Narrowed:
    __slots__ = ()

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[str], ...]:
        return (NOUN,)

    def _admits(self, position: int, entry: Entry[str]) -> bool:
        return all(
            admits(clamp, entry) for clamp in self.clamps if clamp.covers(position)
        )

    def _record(self, tokens: Sequence[str]) -> None:
        del tokens

    def _weight(self, entry: Entry[str]) -> float:
        return entry.weight

    def _extra(
        self, admitted: tuple[tuple[int, Entry[str], float], ...]
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        return admitted

    def _refuse(
        self,
        tokens: Sequence[str],
        admitted: tuple[tuple[int, Entry[str], float], ...],
    ) -> None:
        """Raise where a clamp covers a position nothing survived at.

        Raises
        ------
        LexiconError
            On the first covered position holding no admitted entry.
        """
        filled = frozenset(position for position, _, _ in admitted)

        for clamp in self.clamps:
            for position in range(clamp.span[0], min(clamp.span[1], len(tokens))):
                if position not in filled:
                    msg = f"{clamp!r} is met by nothing at position {position}"

                    raise LexiconError(msg)

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[str], float]]:
        self._record(tokens)
        admitted = tuple(
            (position, entry, self._weight(entry))
            for position, token in enumerate(tokens)
            for entry in TABLE.get(token, ())
            if self._admits(position, entry)
        )
        self._refuse(tokens, admitted)

        return self._extra(admitted)


class Misplaced(Narrowed):
    __slots__ = ()

    @override
    def _extra(
        self, admitted: tuple[tuple[int, Entry[str], float], ...]
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        if not admitted:
            return admitted

        _, entry, weight = admitted[0]

        return (*admitted, (-1, entry, weight))


class BareGenerator(Narrowed):
    __slots__ = ()

    @override
    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterator[tuple[int, Entry[str], float]]:
        yield from super().axioms(tokens)


class IgnoresItsClamp(Narrowed):
    __slots__ = ()

    @override
    def _admits(self, position: int, entry: Entry[str]) -> bool:
        del position, entry

        return True


class Growing(Narrowed):
    __slots__ = ("_misses",)

    def __init__(self) -> None:
        self._misses: set[str] = set()

    @override
    def _record(self, tokens: Sequence[str]) -> None:
        self._misses.update(token for token in tokens if token not in TABLE)

    @override
    def _weight(self, entry: Entry[str]) -> float:
        return entry.weight / (len(TABLE) + len(self._misses))


class Emptying(Narrowed):
    __slots__ = ()

    @override
    def _refuse(
        self,
        tokens: Sequence[str],
        admitted: tuple[tuple[int, Entry[str], float], ...],
    ) -> None:
        del tokens, admitted


class Leaking(Narrowed):
    __slots__ = ()

    @override
    def axioms(self, tokens: Sequence[str], /) -> list[tuple[int, Entry[str], float]]:
        return list(super().axioms(tokens))


def _against(build: type[Narrowed]) -> LexicalSourceConformance[str, float]:
    """Return the conformance the laws run one kind of source through.

    Parameters
    ----------
    build
        The source to build.

    Returns
    -------
    LexicalSourceConformance
        The fixture, over the sentences above and the clamp every subject declares.
    """
    return LexicalSourceConformance(
        fixture=LexicalSourceFixture(build, SENTENCES, EXAMPLES, EQUALITY, EQUALITY),
        sentence=SENTENCE,
        unsatisfiable=UNSATISFIABLE,
    )


def _red(conformance: LexicalSourceConformance[str, float]) -> frozenset[str]:
    """Return the name of every law that fails against a subject.

    Parameters
    ----------
    conformance
        The subject and what the laws need to run against it.

    Returns
    -------
    frozenset of str
        The names of the failing laws.
    """
    suite = LexicalSourceSuite()
    red: list[str] = []

    for law in LEXICAL_SOURCE_SUITE_LAWS:
        try:
            law(suite, conformance)

        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


_BROKEN: Final[tuple[tuple[str, type[Narrowed]], ...]] = (
    ("test_every_position_indexes_the_sentence_it_came_from", Misplaced),
    ("test_a_source_declaring_itself_re_readable_is_one", BareGenerator),
    ("test_every_declared_clamp_holds_where_it_covers", IgnoresItsClamp),
    ("test_reading_a_source_leaves_it_where_it_was", Growing),
    ("test_a_clamped_position_is_refused_rather_than_emptied", Emptying),
    ("test_nothing_a_source_hands_out_can_be_mutated_under_it", Leaking),
)


class TestASourceThatRefusesWhatItCannotAdmit(LexicalSourceSuite):
    @pytest.fixture
    def conformance(self) -> LexicalSourceConformance[str, float]:
        return _against(Narrowed)


def test_every_law_holds_of_a_source_that_refuses_what_it_cannot_admit() -> None:
    assert _red(_against(Narrowed)) == frozenset()


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN,
    ids=[law.removeprefix("test_") for law, _ in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(law: str, build: type[Narrowed]) -> None:
    assert _red(_against(build)) == frozenset({law})


def test_the_sentence_named_unsatisfiable_is_one_the_good_subject_refuses() -> None:
    with pytest.raises(LexiconError, match="met by nothing at position 1"):
        Narrowed().axioms(UNSATISFIABLE)


def test_the_refusal_law_refuses_a_sentence_the_clamp_is_met_on() -> None:
    conformance = LexicalSourceConformance(
        fixture=LexicalSourceFixture(Emptying, SENTENCES, EXAMPLES, EQUALITY, EQUALITY),
        sentence=SENTENCE,
        unsatisfiable=SENTENCE,
    )

    with pytest.raises(LawViolationError, match="nothing was refused on it"):
        LexicalSourceSuite().test_a_clamped_position_is_refused_rather_than_emptied(
            conformance
        )


def test_the_refusal_law_needs_a_sentence_to_try_the_clamp_on() -> None:
    conformance = LexicalSourceConformance(
        fixture=LexicalSourceFixture(Narrowed, SENTENCES, EXAMPLES, EQUALITY, EQUALITY),
        sentence=SENTENCE,
    )

    with pytest.raises(LawViolationError, match="names no sentence"):
        LexicalSourceSuite().test_a_clamped_position_is_refused_rather_than_emptied(
            conformance
        )


def test_the_refusal_law_needs_a_sentence_a_clamp_reaches() -> None:
    conformance = LexicalSourceConformance(
        fixture=LexicalSourceFixture(Narrowed, SENTENCES, EXAMPLES, EQUALITY, EQUALITY),
        sentence=SENTENCE,
        unsatisfiable=("the",),
    )

    with pytest.raises(LawViolationError, match="reaches a position"):
        LexicalSourceSuite().test_a_clamped_position_is_refused_rather_than_emptied(
            conformance
        )
