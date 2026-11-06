from math import log
from typing import TYPE_CHECKING, Final, override

import pytest
from hypothesis import strategies as st
from tests.conformance.lexical_source import (
    LexicalSourceConformance,
    LexicalSourceSuite,
)

from lofi.lexicon._clamped import (
    ClampedSource,
    DropClampPolicy,
    ErrorPolicy,
    SkipItemPolicy,
)
from lofi.lexicon._errors import UnsatisfiableClampError
from lofi.lexicon._ports import Clamp, ClampMode, ClampScope, Entry, LexicalSource
from lofi.lexicon._static import StaticLexicon
from lofi.lexicon.laws import LexicalSourceFixture

if TYPE_CHECKING:
    from collections.abc import Sequence

TABLE: Final = (
    Entry("the", "NP/N", weight=log(4.0)),
    Entry("cat", "N", "cat", log(6.0)),
    Entry("cat", "NP", "cat", log(2.0)),
    Entry("sleeps", "S\\NP", "sleep", log(3.0)),
    Entry("runs", "S\\NP", "run", log(1.0)),
)

KNOWN: Final = ("the", "cat", "sleeps", "runs")
UNKNOWN: Final = "aardvark"
PINNED: Final = 1
NOUN: Final[Clamp[str]] = Clamp((PINNED, PINNED + 1), category={"N"})
NOUNS: Final[Clamp[str]] = Clamp((PINNED, PINNED + 1), category={"N", "NP"})
EXAMPLES: Final = 50


class Equality[T]:
    """Value equality used by the string-valued fixture."""

    name = "equality"
    has_key = True

    def key(self, value: T, /) -> T:
        """Return the value as its representative."""
        return value

    def equivalent(self, left: T, right: T, /) -> bool:
        """Return whether two values compare equal."""
        return left == right


EQUALITY: Final = Equality[str]()

SENTENCES = st.tuples(
    st.sampled_from([*KNOWN, UNKNOWN]),
    st.just("cat"),
    st.lists(st.sampled_from([*KNOWN, UNKNOWN]), max_size=2).map(tuple),
).map(lambda drawn: (drawn[0], drawn[1], *drawn[2]))


def table() -> StaticLexicon[str]:
    """Return the static lexicon narrowed by the fixtures.

    Returns
    -------
    StaticLexicon
        Entries weighted by category given form.
    """
    return StaticLexicon(TABLE)


def clamped(*clamps: Clamp[str]) -> ClampedSource[str]:
    """Return the static lexicon narrowed by one or more clamps.

    Parameters
    ----------
    clamps
        The clamps to enforce.

    Returns
    -------
    ClampedSource
        The narrowed source.
    """
    return ClampedSource(table(), clamps, EQUALITY, EQUALITY, ErrorPolicy())


def built() -> LexicalSource[str, float]:
    """Return the source used by the lexical-source laws.

    Returns
    -------
    LexicalSource
        The table narrowed to a noun at the pinned position.
    """
    return clamped(NOUN)


def built_over_a_set() -> LexicalSource[str, float]:
    """Return a source whose pinned position admits both noun categories.

    Returns
    -------
    LexicalSource
        The table narrowed to either noun category at the pinned position.
    """
    return clamped(NOUNS)


def by_position(
    axioms: Sequence[tuple[int, Entry[str], float]],
) -> dict[int, list[tuple[str, str | None, float]]]:
    """Return enumerated entries grouped by position.

    Parameters
    ----------
    axioms
        One ordered enumeration from a lexical source.

    Returns
    -------
    dict
        Category, meaning, and weight triples in enumeration order at each position.
    """
    found: dict[int, list[tuple[str, str | None, float]]] = {}

    for position, entry, weight in axioms:
        found.setdefault(position, []).append((entry.category, entry.meaning, weight))

    return found


def test_a_clamped_position_keeps_only_what_the_clamp_names() -> None:
    tokens = ("the", "cat", "sleeps")

    found = by_position(clamped(NOUN).axioms(tokens))

    assert [category for category, _, _ in found[PINNED]] == ["N"]


def test_every_other_position_is_the_unclamped_source_entry_for_entry() -> None:
    tokens = ("the", "cat", "sleeps", UNKNOWN, "runs")
    plain = by_position(table().axioms(tokens))
    narrowed = by_position(clamped(NOUN).axioms(tokens))

    for position in range(len(tokens)):
        if position == PINNED:
            continue

        assert narrowed.get(position, []) == plain.get(position, [])


def test_the_clamped_position_keeps_the_weight_the_source_reported() -> None:
    tokens = ("the", "cat")
    plain = by_position(table().axioms(tokens))
    narrowed = by_position(clamped(NOUN).axioms(tokens))
    kept = [reading for reading in plain[PINNED] if reading[0] == "N"]

    assert narrowed[PINNED] == kept


def test_a_soft_clamp_penalizes_incompatible_entries_without_filtering_them() -> None:
    tokens = ("the", "cat")
    preferred: Clamp[str] = Clamp(
        (PINNED, PINNED + 1),
        category={"N"},
        mode=ClampMode.SOFT,
        penalty=0.5,
    )

    found = by_position(clamped(preferred).axioms(tokens))

    assert [category for category, _, _ in found[PINNED]] == ["N", "NP"]
    assert found[PINNED][0][2] == pytest.approx(log(0.75))
    assert found[PINNED][1][2] == pytest.approx(log(0.25) - 0.5)


def test_a_category_clamp_uses_the_supplied_congruence() -> None:
    class Casefold(Equality[str]):
        """Case-insensitive equality for fixture strings."""

        name = "casefold"

        @override
        def equivalent(self, left: str, right: str, /) -> bool:
            """Compare case-folded strings."""
            return left.casefold() == right.casefold()

    matching = Casefold()
    source = ClampedSource(
        table(),
        (Clamp((PINNED, PINNED + 1), category={"n"}),),
        matching,
        EQUALITY,
        ErrorPolicy(),
    )

    found = by_position(source.axioms(("the", "cat")))

    assert [category for category, _, _ in found[PINNED]] == ["N"]


def test_a_clamp_naming_a_category_the_source_does_not_have_is_refused() -> None:
    verb: Clamp[str] = Clamp((PINNED, PINNED + 1), category={"S\\NP"})

    with pytest.raises(UnsatisfiableClampError) as refused:
        clamped(verb).axioms(("the", "cat"))

    assert refused.value.clamp == verb
    assert refused.value.position == PINNED
    assert refused.value.form == "cat"
    assert set(refused.value.offered) == {"N", "NP"}
    assert "[1, 2)" in str(refused.value)
    assert "'S\\\\NP'" in str(refused.value)


def test_drop_clamp_continues_and_reports_each_unsatisfiable_clamp() -> None:
    policy = DropClampPolicy()
    source = ClampedSource(
        table(),
        (Clamp((PINNED, PINNED + 1), category={"S\\NP"}),),
        EQUALITY,
        EQUALITY,
        policy,
    )

    found = by_position(source.axioms(("the", "cat")))

    assert [category for category, _, _ in found[PINNED]] == ["N", "NP"]
    assert source.report.dropped_clamps == 1
    assert source.report.skipped_items == 0


def test_skip_item_emits_nothing_and_reports_the_item() -> None:
    source = ClampedSource(
        table(),
        (Clamp((PINNED, PINNED + 1), category={"S\\NP"}),),
        EQUALITY,
        EQUALITY,
        SkipItemPolicy(),
    )

    assert source.axioms(("the", "cat")) == ()
    assert source.report.dropped_clamps == 0
    assert source.report.skipped_items == 1


def test_a_clamp_over_a_position_the_source_knows_nothing_of_is_refused() -> None:
    with pytest.raises(UnsatisfiableClampError) as refused:
        clamped(Clamp((0, 1), category={"N"})).axioms((UNKNOWN,))

    assert refused.value.offered == ()
    assert "offers nothing" in str(refused.value)


def test_a_clamp_naming_a_meaning_the_position_does_not_have_is_refused() -> None:
    with pytest.raises(UnsatisfiableClampError, match="'dog'"):
        clamped(Clamp((PINNED, PINNED + 1), meaning="dog")).axioms(("the", "cat"))


def test_a_clamp_naming_a_meaning_the_position_has_narrows_to_it() -> None:
    tokens = ("the", "cat")

    found = by_position(
        clamped(Clamp((PINNED, PINNED + 1), meaning="cat")).axioms(tokens)
    )

    assert len(found[PINNED]) == 2


def test_a_set_of_categories_keeps_every_entry_bearing_one_of_them() -> None:
    tokens = ("the", "cat")
    both: Clamp[str] = Clamp((PINNED, PINNED + 1), category=("N", "NP"))

    found = by_position(clamped(both).axioms(tokens))

    assert sorted(category for category, _, _ in found[PINNED]) == ["N", "NP"]


def test_a_satisfiable_set_admits_at_every_position_the_clamp_covers() -> None:
    tokens = ("cat", "cat", "sleeps")
    both: Clamp[str] = Clamp((0, 2), category=("N", "NP"), scope=ClampScope.OCCURRENCE)

    found = by_position(clamped(both).axioms(tokens))

    assert sorted(category for category, _, _ in found[0]) == ["N", "NP"]
    assert sorted(category for category, _, _ in found[1]) == ["N", "NP"]
    assert [category for category, _, _ in found[2]] == ["S\\NP"]


def test_a_set_the_position_holds_none_of_is_refused_naming_every_member() -> None:
    absent: Clamp[str] = Clamp((PINNED, PINNED + 1), category=("S\\NP", "NP/N"))

    with pytest.raises(UnsatisfiableClampError) as refused:
        clamped(absent).axioms(("the", "cat"))

    message = str(refused.value)

    assert refused.value.clamp == absent
    assert "[1, 2)" in message
    assert "'S\\\\NP'" in message
    assert "'NP/N'" in message


def test_a_negated_set_forbidding_all_a_position_has_is_refused() -> None:
    forbidden: Clamp[str] = Clamp(
        (PINNED, PINNED + 1), category=("N", "NP"), negate=True
    )

    with pytest.raises(UnsatisfiableClampError, match="forbids"):
        clamped(forbidden).axioms(("the", "cat"))


def test_a_negated_clamp_keeps_everything_it_does_not_name() -> None:
    tokens = ("the", "cat")
    forbidden: Clamp[str] = Clamp((PINNED, PINNED + 1), category={"N"}, negate=True)

    found = by_position(clamped(forbidden).axioms(tokens))

    assert [category for category, _, _ in found[PINNED]] == ["NP"]


def test_a_negated_clamp_forbidding_all_there_is_is_refused() -> None:
    forbidden: Clamp[str] = Clamp((0, 1), category={"NP/N"}, negate=True)

    with pytest.raises(UnsatisfiableClampError, match="forbids"):
        clamped(forbidden).axioms(("the", "cat"))


def test_two_clamps_over_one_position_are_enforced_together() -> None:
    tokens = ("the", "cat")
    noun = Clamp((PINNED, PINNED + 1), category={"N"})
    meaning: Clamp[str] = Clamp((PINNED, PINNED + 1), meaning="cat")

    found = by_position(clamped(noun, meaning).axioms(tokens))

    assert [category for category, _, _ in found[PINNED]] == ["N"]


def test_two_clamps_that_cannot_both_hold_name_the_one_that_emptied_it() -> None:
    noun = Clamp((PINNED, PINNED + 1), category={"N"})
    absent: Clamp[str] = Clamp((PINNED, PINNED + 1), category={"S\\NP"})

    with pytest.raises(UnsatisfiableClampError) as refused:
        clamped(noun, absent).axioms(("the", "cat"))

    assert refused.value.clamp == absent


def test_two_clamps_each_satisfiable_alone_name_the_first_of_them() -> None:
    noun = Clamp((PINNED, PINNED + 1), category={"N"})
    phrase = Clamp((PINNED, PINNED + 1), category={"NP"})

    with pytest.raises(UnsatisfiableClampError) as refused:
        clamped(noun, phrase).axioms(("the", "cat"))

    assert refused.value.clamp == noun


def test_a_clamp_reaching_past_the_sentence_covers_only_what_is_there() -> None:
    reaching: Clamp[str] = Clamp(
        (PINNED, 40), category={"N"}, scope=ClampScope.OCCURRENCE
    )

    found = by_position(clamped(reaching).axioms(("the", "cat")))

    assert [category for category, _, _ in found[PINNED]] == ["N"]


def test_the_empty_sentence_admits_nothing_and_refuses_nothing() -> None:
    assert clamped(NOUN).axioms(()) == ()


def test_a_sentence_too_short_to_reach_the_clamp_is_left_alone() -> None:
    tokens = ("the",)

    assert by_position(clamped(NOUN).axioms(tokens)) == by_position(
        table().axioms(tokens)
    )


def test_the_source_declares_its_own_clamps_after_the_ones_beneath_it() -> None:
    inner = clamped(Clamp((0, 1), category={"NP/N"}))
    outer = ClampedSource(inner, (NOUN,), EQUALITY, EQUALITY, ErrorPolicy())

    assert outer.clamps == (Clamp((0, 1), category={"NP/N"}), NOUN)
    assert outer.enforced == (NOUN,)
    assert outer.is_re_callable


def test_a_source_readable_once_is_declared_as_one() -> None:
    class Draining:
        """A single-use lexical enumeration."""

        @property
        def is_re_callable(self) -> bool:
            """Return whether the enumeration supports repeated reads."""
            return False

        @property
        def clamps(self) -> tuple[Clamp[str], ...]:
            """Return the clamps enforced by this source."""
            return ()

        def axioms(
            self, tokens: Sequence[str], /
        ) -> tuple[tuple[int, Entry[str], float], ...]:
            """Return a noun wherever the sentence holds one."""
            return tuple(
                (position, Entry(form, "N", form), 0.0)
                for position, form in enumerate(tokens)
                if form == "cat"
            )

    source: ClampedSource[str] = ClampedSource(
        Draining(), (NOUN,), EQUALITY, EQUALITY, ErrorPolicy()
    )

    assert not source.is_re_callable
    assert len(source.axioms(("the", "cat"))) == 1


@pytest.mark.parametrize(
    ("clamps", "message"),
    [
        ((), "no clamp"),
        ((Clamp((2, 2), category={"N"}),), "covers no position"),
        ((Clamp((3, 1), category={"N"}),), "covers no position"),
        ((Clamp((1, 2)),), "names the category or the meaning"),
        ((Clamp((1, 2), negate=True),), "names the category or the meaning"),
    ],
)
def test_a_clamp_that_could_narrow_nothing_is_refused_where_it_is_built(
    clamps: tuple[Clamp[str], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ClampedSource(table(), clamps, EQUALITY, EQUALITY, ErrorPolicy())


def test_a_span_running_below_zero_is_refused_where_it_is_built() -> None:
    with pytest.raises(ValueError, match="negative"):
        ClampedSource(
            table(),
            (Clamp((-1, 2), category={"N"}),),
            EQUALITY,
            EQUALITY,
            ErrorPolicy(),
        )


def test_the_clamped_source_renders_what_it_narrows_and_how_much() -> None:
    assert repr(clamped(NOUN)) == (
        "ClampedSource(StaticLexicon(forms=4, entries=5), clamps=1)"
    )


class TestAClampedSource(LexicalSourceSuite):
    """Tests clamped sources against lexical-source laws."""

    @pytest.fixture
    def conformance(self) -> LexicalSourceConformance[str, float]:
        """Return the conformance the laws run the narrowed source through."""
        return LexicalSourceConformance(
            fixture=LexicalSourceFixture(
                build=built,
                tokens=SENTENCES,
                max_examples=EXAMPLES,
                category_matching=EQUALITY,
                meaning_matching=EQUALITY,
            ),
            sentence=("the", "cat", "sleeps"),
            unsatisfiable=("the", "the", "sleeps"),
        )


class TestAClampedSourceOverASetOfCategories(LexicalSourceSuite):
    """Tests clamps with multiple allowed categories."""

    @pytest.fixture
    def conformance(self) -> LexicalSourceConformance[str, float]:
        """Return the conformance the laws run the widely clamped source through."""
        return LexicalSourceConformance(
            fixture=LexicalSourceFixture(
                build=built_over_a_set,
                tokens=SENTENCES,
                max_examples=EXAMPLES,
                category_matching=EQUALITY,
                meaning_matching=EQUALITY,
            ),
            sentence=("the", "cat", "sleeps"),
            unsatisfiable=("the", "the", "sleeps"),
        )
