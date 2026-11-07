from typing import TYPE_CHECKING, Final, override

import pytest
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi.lexicon._ports import Clamp, Entry
from lofi.lexicon.laws import (
    LEXICAL_SOURCE_LAWS,
    LexicalSourceFixture,
    check_lexical_source_laws,
    clamps_are_respected,
    enumeration_is_re_callable,
    positions_are_total,
    reading_does_not_mutate,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from lofi.lexicon._ports import LexicalSource

TABLE: Mapping[str, tuple[Entry[str], ...]] = {
    "the": (Entry("the", "NP/N", weight=-0.1),),
    "cat": (Entry("cat", "N", weight=-0.5), Entry("cat", "NP", weight=-3.0)),
    "sleeps": (Entry("sleeps", "S\\NP", "sleep", -0.2),),
    "dog": (Entry("dog", "N", weight=-0.4),),
    "runs": (Entry("runs", "S\\NP", "run", -0.3),),
}

KNOWN: Final = ("the", "cat", "sleeps", "dog", "runs")
UNKNOWN: Final = ("aardvark", "zebra")
CLAMP: Clamp[str] = Clamp((1, 2), category={"N"})
NEGATED: Clamp[str] = Clamp((1, 2), category={"N"}, negate=True)
MEANING: Clamp[str] = Clamp((1, 2), meaning="sleep")
WIDE: Clamp[str] = Clamp((1, 2), category={"N", "NP"})
EXAMPLES: Final = 100


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

SENTENCES = st.lists(st.sampled_from([*KNOWN, *UNKNOWN]), max_size=4).map(tuple)

VERB_SECOND = st.tuples(
    st.sampled_from(KNOWN),
    st.sampled_from(("sleeps", "runs")),
    st.lists(st.sampled_from([*KNOWN, *UNKNOWN]), max_size=2).map(tuple),
).map(lambda drawn: (drawn[0], drawn[1], *drawn[2]))

KNOWN_SECOND = st.tuples(
    st.sampled_from(KNOWN),
    st.sampled_from(("cat", "the", "sleeps")),
    st.lists(st.sampled_from([*KNOWN, *UNKNOWN]), max_size=2).map(tuple),
).map(lambda drawn: (drawn[0], drawn[1], *drawn[2]))


class Table:
    """Fixed entries for each surface form."""

    def __init__(self, clamps: tuple[Clamp[str], ...] = ()) -> None:
        self._entries = dict(TABLE)
        self._clamps = clamps

    @property
    def is_re_callable(self) -> bool:
        """Return whether the table supports repeated reads."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[str], ...]:
        """Return the clamps this table enforces."""
        return self._clamps

    def admits(self, position: int, entry: Entry[str]) -> bool:
        """Return whether every clamp covering a position accepts an entry."""
        for clamp in self._clamps:
            if not clamp.covers(position):
                continue

            named = (not clamp.category or entry.category in clamp.category) and (
                clamp.meaning is None or entry.meaning == clamp.meaning
            )

            if named is clamp.negate:
                return False

        return True

    def weight(self, entry: Entry[str]) -> float:
        """Return the weight the enumeration reports for an entry."""
        return entry.weight

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        """Return every entry the tokens admit, each beside its position."""
        return tuple(
            (position, entry, self.weight(entry))
            for position, token in enumerate(tokens)
            for entry in self._entries.get(token, ())
            if self.admits(position, entry)
        )


class Draining:
    """A single-use lexical enumeration."""

    def __init__(self, clamps: tuple[Clamp[str], ...] = ()) -> None:
        self._under = Table(clamps)

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration supports repeated reads."""
        return False

    @property
    def clamps(self) -> tuple[Clamp[str], ...]:
        """Return the clamps the source beneath enforces."""
        return self._under.clamps

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterator[tuple[int, Entry[str], float]]:
        """Yield every entry the tokens admit, each beside its position."""
        yield from self._under.axioms(tokens)


class BareGenerator(Draining):
    """A one-shot source that claims to support repeated enumeration."""

    @property
    @override
    def is_re_callable(self) -> bool:
        """Claim that the enumeration may be read again."""
        return True


class Silent(Table):
    """A lexical source with no entries."""

    @override
    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        """Return no entries."""
        del tokens

        return ()


class Misplaced(Table):
    """Reports an entry outside the sentence."""

    @override
    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        """Return the admitted entries, and the first of them again at minus one."""
        axioms = super().axioms(tokens)

        if not axioms:
            return axioms

        _, entry, weight = axioms[0]

        return (*axioms, (-1, entry, weight))


class IgnoresItsClamp(Table):
    """Declares a clamp without enforcing it."""

    @override
    def admits(self, position: int, entry: Entry[str]) -> bool:
        """Accept every entry."""
        del position, entry

        return True


class Growing(Table):
    """Inserts missed forms into the source."""

    def __init__(self, clamps: tuple[Clamp[str], ...] = ()) -> None:
        super().__init__(clamps)
        self._misses: set[str] = set()

    @override
    def weight(self, entry: Entry[str]) -> float:
        """Return a state-dependent weight."""
        return entry.weight / (len(self._entries) + len(self._misses))

    @override
    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        """Return the admitted entries, having recorded every form missed."""
        self._misses.update(token for token in tokens if token not in self._entries)

        return super().axioms(tokens)


class Refusing(Table):
    """Rejects forms absent from the source."""

    @override
    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        """Reject sentences containing unknown forms.

        Raises
        ------
        KeyError
            At the first form absent from the table.
        """
        for token in tokens:
            if token not in self._entries:
                raise KeyError(token)

        return super().axioms(tokens)


def fixture_over(
    build: Callable[[], LexicalSource[str, float]],
) -> LexicalSourceFixture[str, float]:
    return LexicalSourceFixture(build, SENTENCES, EXAMPLES, EQUALITY, EQUALITY)


def clamped() -> LexicalSource[str, float]:
    return Table((CLAMP,))


def laws_failed(build: Callable[[], LexicalSource[str, float]]) -> list[str]:
    failed: list[str] = []

    for law in LEXICAL_SOURCE_LAWS:
        try:
            law(fixture_over(build))

        except LawViolationError:
            failed.append(law.__name__)

    return failed


@pytest.mark.parametrize("law", LEXICAL_SOURCE_LAWS, ids=lambda law: law.__name__)
def test_each_law_collects_and_runs_on_its_own(
    law: Callable[[LexicalSourceFixture[str, float]], None],
) -> None:
    law(fixture_over(clamped))


def test_a_table_holds_to_every_law() -> None:
    check_lexical_source_laws(fixture_over(Table))


def test_a_clamped_table_holds_to_every_law() -> None:
    check_lexical_source_laws(fixture_over(clamped))


def test_a_source_readable_once_holds_to_every_law_it_declares() -> None:
    check_lexical_source_laws(fixture_over(Draining))


@pytest.mark.parametrize("law", LEXICAL_SOURCE_LAWS, ids=lambda law: law.__name__)
def test_a_source_enumerating_nothing_fails_every_law(
    law: Callable[[LexicalSourceFixture[str, float]], None],
) -> None:
    def build() -> LexicalSource[str, float]:
        return Silent((CLAMP,))

    with pytest.raises(LawViolationError, match="never exercised"):
        law(fixture_over(build))


def test_the_entry_point_reports_the_law_that_failed() -> None:
    def build() -> LexicalSource[str, float]:
        return Silent((CLAMP,))

    with pytest.raises(LawViolationError, match="totality"):
        check_lexical_source_laws(fixture_over(build))


def test_totality_catches_a_position_outside_the_sentence() -> None:
    with pytest.raises(LawViolationError, match="totality"):
        positions_are_total(fixture_over(Misplaced))


def test_totality_catches_a_source_that_raises_rather_than_answering() -> None:
    with pytest.raises(LawViolationError, match="totality"):
        positions_are_total(fixture_over(Refusing))


def test_re_callability_catches_a_generator_behind_the_annotation() -> None:
    with pytest.raises(LawViolationError, match="re-callability"):
        enumeration_is_re_callable(fixture_over(BareGenerator))


def test_re_callability_returns_without_running_on_a_source_readable_once() -> None:
    enumeration_is_re_callable(fixture_over(Draining))


def test_clamping_catches_a_source_that_declares_a_clamp_and_ignores_it() -> None:
    def build() -> LexicalSource[str, float]:
        return IgnoresItsClamp((CLAMP,))

    with pytest.raises(LawViolationError, match="clamping"):
        clamps_are_respected(fixture_over(build))


def test_clamping_holds_of_a_source_honoring_a_negated_clamp() -> None:
    def build() -> LexicalSource[str, float]:
        return Table((NEGATED,))

    clamps_are_respected(fixture_over(build))


def test_clamping_catches_a_source_ignoring_a_negated_clamp() -> None:
    def build() -> LexicalSource[str, float]:
        return IgnoresItsClamp((NEGATED,))

    with pytest.raises(LawViolationError, match="clamping"):
        clamps_are_respected(fixture_over(build))


def test_clamping_holds_of_a_source_admitting_every_member_of_a_wide_clamp() -> None:
    def build() -> LexicalSource[str, float]:
        return Table((WIDE,))

    clamps_are_respected(
        LexicalSourceFixture(build, KNOWN_SECOND, EXAMPLES, EQUALITY, EQUALITY)
    )


def test_clamping_catches_a_source_ignoring_a_clamp_naming_several_categories() -> None:
    def build() -> LexicalSource[str, float]:
        return IgnoresItsClamp((WIDE,))

    with pytest.raises(LawViolationError, match="clamping"):
        clamps_are_respected(
            LexicalSourceFixture(build, KNOWN_SECOND, EXAMPLES, EQUALITY, EQUALITY)
        )


def test_clamping_holds_of_a_source_honoring_a_clamp_naming_only_a_meaning() -> None:
    def build() -> LexicalSource[str, float]:
        return Table((MEANING,))

    clamps_are_respected(
        LexicalSourceFixture(build, VERB_SECOND, EXAMPLES, EQUALITY, EQUALITY)
    )


def test_clamping_catches_a_source_ignoring_a_clamp_naming_only_a_meaning() -> None:
    def build() -> LexicalSource[str, float]:
        return IgnoresItsClamp((MEANING,))

    with pytest.raises(LawViolationError, match="clamping"):
        clamps_are_respected(
            LexicalSourceFixture(build, VERB_SECOND, EXAMPLES, EQUALITY, EQUALITY)
        )


def test_clamping_returns_without_running_on_a_source_declaring_none() -> None:
    clamps_are_respected(fixture_over(Table))


def test_non_mutation_catches_a_lookup_whose_miss_inserts() -> None:
    with pytest.raises(LawViolationError, match="non-mutation"):
        reading_does_not_mutate(fixture_over(Growing))


def test_a_source_whose_miss_inserts_fails_the_fourth_law_and_no_other() -> None:
    def build() -> LexicalSource[str, float]:
        return Growing((CLAMP,))

    assert laws_failed(build) == ["reading_does_not_mutate"]


def test_a_source_reporting_a_position_outside_the_sentence_fails_only_totality() -> (
    None
):
    def build() -> LexicalSource[str, float]:
        return Misplaced((CLAMP,))

    assert laws_failed(build) == ["positions_are_total"]


def test_a_generator_source_fails_only_re_callability() -> None:
    def build() -> LexicalSource[str, float]:
        return BareGenerator((CLAMP,))

    assert laws_failed(build) == ["enumeration_is_re_callable"]


def test_a_source_ignoring_its_clamp_fails_only_clamping() -> None:
    def build() -> LexicalSource[str, float]:
        return IgnoresItsClamp((CLAMP,))

    assert laws_failed(build) == ["clamps_are_respected"]


def test_a_law_refuses_a_strategy_that_admits_no_entry() -> None:
    unknown = LexicalSourceFixture(Table, st.just(UNKNOWN), EXAMPLES)

    with pytest.raises(LawViolationError, match="never exercised"):
        positions_are_total(unknown)


def test_a_law_refuses_a_strategy_that_never_reaches_a_clamped_position() -> None:
    short = LexicalSourceFixture(
        clamped, st.just(("the",)), EXAMPLES, EQUALITY, EQUALITY
    )

    with pytest.raises(LawViolationError, match="never exercised"):
        clamps_are_respected(short)
