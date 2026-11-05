from copy import copy, deepcopy
from typing import TYPE_CHECKING, Final

import pytest
from hypothesis import given
from hypothesis import strategies as st
from tests.conformance._suite import Conformance, ConformanceSuite

from lofi.lexicon._ports import Clamp, ClampMode, ClampScope, Entry, LexicalSource

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

TABLE: Mapping[str, tuple[Entry[str], ...]] = {
    "the": (Entry("the", "NP/N", weight=-0.1),),
    "cat": (Entry("cat", "N"), Entry("cat", "NP", weight=-3.0)),
    "sleeps": (Entry("sleeps", "S\\NP", "sleep", -0.2),),
}

TOKENS = ("the", "cat", "sleeps")

CATEGORIES: Final = ("N", "NP", "S", "S\\NP", "NP/N")
MEANINGS: Final = (None, "cat", "sleep")


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


SETS = st.lists(st.sampled_from(CATEGORIES), min_size=1, unique=True).map(frozenset)
ORDERINGS = st.lists(st.sampled_from(CATEGORIES), min_size=1, unique=True)
ENTRIES = st.builds(
    Entry,
    st.sampled_from(("cat", "dog")),
    st.sampled_from(CATEGORIES),
    st.sampled_from(MEANINGS),
)


class Table:
    """Fixed entries for each surface form."""

    def __init__(self, entries: Mapping[str, tuple[Entry[str], ...]]) -> None:
        self._entries = dict(entries)

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration can be read more than once."""
        return True

    @property
    def clamps(self) -> tuple[Clamp[str], ...]:
        """Return no clamps."""
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        """Return every entry the tokens admit, each beside its position."""
        return tuple(
            (position, entry, entry.weight)
            for position, token in enumerate(tokens)
            for entry in self._entries.get(token, ())
        )


class Draining:
    """A single-use lexical enumeration."""

    def __init__(self, under: Table) -> None:
        self._under = under

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration can be read more than once."""
        return False

    @property
    def clamps(self) -> tuple[Clamp[str], ...]:
        """Return no clamps."""
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterator[tuple[int, Entry[str], float]]:
        """Yield every entry the tokens admit, each beside its position."""
        yield from self._under.axioms(tokens)


class Clamped:
    """Filters entries at clamped positions."""

    def __init__(self, under: Table, clamps: tuple[Clamp[str], ...]) -> None:
        self._under = under
        self._clamps = clamps

    @property
    def is_re_callable(self) -> bool:
        """Return whether the filtered source can be read more than once."""
        return self._under.is_re_callable

    @property
    def clamps(self) -> tuple[Clamp[str], ...]:
        """Return the clamps this source enforces."""
        return self._clamps

    def _admits(self, position: int, entry: Entry[str]) -> bool:
        """Return whether every clamp covering a position accepts an entry."""
        return all(
            admits(clamp, entry) for clamp in self._clamps if clamp.covers(position)
        )

    def axioms(
        self, tokens: Sequence[str], /
    ) -> tuple[tuple[int, Entry[str], float], ...]:
        """Return the entries surviving every clamp that covers their position."""
        return tuple(
            axiom
            for axiom in self._under.axioms(tokens)
            if self._admits(axiom[0], axiom[1])
        )


class Unclamped:
    """Omits the required `clamps` property."""

    @property
    def is_re_callable(self) -> bool:
        """Return whether the enumeration can be read more than once."""
        return True

    def axioms(self, tokens: Sequence[str], /) -> Iterable[tuple[int, Entry[str], int]]:
        """Return no entries at all."""
        del tokens

        return ()


def test_entry_reads_back_the_four_fields_it_was_built_from() -> None:
    entry = Entry("sleeps", "S\\NP", "sleep", -0.2)

    assert entry.form == "sleeps"
    assert entry.category == "S\\NP"
    assert entry.meaning == "sleep"
    assert entry.weight == -0.2


def test_entry_accepts_distinct_category_and_meaning_types() -> None:
    entry: Entry[int, str] = Entry("one", 1, "one")

    assert entry.category == 1
    assert entry.meaning == "one"


def test_an_entry_with_no_meaning_carries_the_slot_empty() -> None:
    entry = Entry("cat", "N")

    assert entry.meaning is None
    assert entry.weight == 0.0


def test_reweighting_an_entry_leaves_it_the_same_entry() -> None:
    light = Entry("cat", "N", weight=0.0)
    heavy = Entry("cat", "N", weight=-3.0)

    assert light == heavy
    assert hash(light) == hash(heavy)
    assert len({light, heavy}) == 1
    assert light.weight != heavy.weight


def test_entries_differing_in_form_category_or_meaning_are_distinct() -> None:
    base = Entry("cat", "N", "cat_")

    assert base != Entry("cats", "N", "cat_")
    assert base != Entry("cat", "NP", "cat_")
    assert base != Entry("cat", "N", "feline_")
    assert base != Entry("dog", "N", "cat_")
    assert base != Entry("cat", "S", "cat_")
    assert base != Entry("cat", "N", "dog_")
    assert base != Entry("cat", "N")
    assert len({base, Entry("cat", "N")}) == 2


def test_an_entry_is_unequal_to_a_value_that_is_not_one() -> None:
    assert Entry("cat", "N") != "cat"
    assert Entry("cat", "N") != ("cat", "N", None, 0.0)


def test_an_entry_cannot_be_rewritten_in_place() -> None:
    entry = Entry("cat", "N")

    for field, value in (
        ("_weight", -1.0),
        ("_category", "NP"),
        ("_form", "dog"),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(entry, field, value)

    assert not hasattr(entry, "__dict__")
    assert entry.category == "N"


def test_an_entry_renders_its_form_category_and_weight() -> None:
    assert repr(Entry("cat", "N", weight=-3.0)) == (
        "Entry(form='cat', category='N', weight=-3.0)"
    )


def test_a_clamp_covers_the_positions_of_its_half_open_span() -> None:
    clamp: Clamp[str] = Clamp((1, 3), category={"N"})

    assert not clamp.covers(0)
    assert clamp.covers(1)
    assert clamp.covers(2)
    assert not clamp.covers(3)


def test_an_empty_span_covers_nothing() -> None:
    clamp: Clamp[str] = Clamp((2, 2), category={"N"})

    assert not any(clamp.covers(position) for position in range(4))


def test_a_clamp_reads_back_what_it_names() -> None:
    clamp: Clamp[str] = Clamp((0, 1), category={"NP"}, meaning="john_")

    assert clamp.span == (0, 1)
    assert clamp.category == frozenset({"NP"})
    assert clamp.meaning == "john_"
    assert clamp.mode is ClampMode.HARD
    assert clamp.penalty is None
    assert clamp.scope is ClampScope.CONSTITUENT
    assert clamp.require_constituent
    assert not clamp.negate


def test_a_soft_clamp_declares_its_positive_penalty() -> None:
    clamp: Clamp[str] = Clamp((0, 1), category={"NP"}, mode=ClampMode.SOFT, penalty=0.5)

    assert clamp.mode is ClampMode.SOFT
    assert clamp.penalty == 0.5


@pytest.mark.parametrize("penalty", [None, 0.0, -1.0, float("inf"), float("nan")])
def test_a_soft_clamp_refuses_a_missing_or_nonpositive_penalty(
    penalty: float | None,
) -> None:
    error = TypeError if penalty is None else ValueError

    with pytest.raises(error, match="penalty"):
        Clamp(
            (0, 1),
            category={"NP"},
            mode=ClampMode.SOFT,
            penalty=penalty,
        )


def test_a_hard_clamp_refuses_a_penalty() -> None:
    with pytest.raises(ValueError, match="hard clamp has no penalty"):
        Clamp((0, 1), category={"NP"}, penalty=0.5)


def test_a_clamp_may_leave_the_category_or_the_meaning_free() -> None:
    only_meaning: Clamp[str] = Clamp((0, 1), meaning="john_")

    assert only_meaning.category == frozenset()
    assert only_meaning.meaning == "john_"


def test_a_category_handed_over_bare_rather_than_in_a_set_is_refused() -> None:
    with pytest.raises(TypeError, match="rather than one of them"):
        Clamp((0, 1), "NP")

    with pytest.raises(TypeError, match="rather than one of them"):
        Clamp((0, 1), b"NP")


def test_negating_a_clamp_makes_a_different_clamp() -> None:
    required: Clamp[str] = Clamp((1, 2), category={"N"})
    forbidden: Clamp[str] = Clamp((1, 2), category={"N"}, negate=True)

    assert required != forbidden
    assert len({required, forbidden}) == 2
    assert required == Clamp((1, 2), category={"N"})
    assert hash(required) == hash(Clamp((1, 2), category={"N"}))


@pytest.mark.parametrize(
    ("clamp", "entry"),
    [
        (Clamp((0, 1), category={"N"}), Entry("cat", "N", "cat")),
        (Clamp((0, 1), meaning="cat"), Entry("cat", "NP", "cat")),
        (Clamp((0, 1), category={"N"}, meaning="cat"), Entry("cat", "N", "cat")),
        (Clamp((0, 1), category={"N"}, negate=True), Entry("cat", "NP")),
        (Clamp((0, 1)), Entry("cat", "N")),
    ],
)
def test_a_clamp_admits_an_entry_matching_everything_it_names(
    clamp: Clamp[str],
    entry: Entry[str],
) -> None:
    assert admits(clamp, entry)


@pytest.mark.parametrize(
    ("clamp", "entry"),
    [
        (Clamp((0, 1), category={"N"}), Entry("cat", "NP", "cat")),
        (Clamp((0, 1), meaning="cat"), Entry("cat", "NP", "dog")),
        (Clamp((0, 1), category={"N"}, meaning="cat"), Entry("cat", "N", "dog")),
        (Clamp((0, 1), category={"N"}, negate=True), Entry("cat", "N")),
        (Clamp((0, 1), negate=True), Entry("cat", "N")),
    ],
)
def test_a_clamp_refuses_an_entry_missing_one_of_the_slots_it_names(
    clamp: Clamp[str],
    entry: Entry[str],
) -> None:
    assert not admits(clamp, entry)


def test_a_clamp_naming_a_meaning_says_nothing_about_a_category() -> None:
    clamp: Clamp[str] = Clamp((0, 1), meaning="cat")

    assert admits(clamp, Entry("cat", "N", "cat"))
    assert admits(clamp, Entry("kitten", "NP", "cat"))
    assert not admits(clamp, Entry("cat", "N"))


def one_category_admits(
    category: str,
    meaning: str | None,
    entry: Entry[str],
    *,
    negate: bool,
) -> bool:
    """Return whether a single-category clamp admits an entry.

    Parameters
    ----------
    category
        The required category.
    meaning
        The required meaning, or ``None`` to leave meaning unrestricted.
    entry
        The entry to test.
    negate
        Whether the clamp excludes rather than requires the match.
    """
    named = entry.category == category and (meaning is None or entry.meaning == meaning)

    return named is not negate


def test_a_clamp_admits_an_entry_bearing_any_category_its_set_holds() -> None:
    clamp: Clamp[str] = Clamp((0, 1), category=("N", "NP"))

    assert admits(clamp, Entry("cat", "N"))
    assert admits(clamp, Entry("cat", "NP"))
    assert not admits(clamp, Entry("cat", "S"))


def test_a_negated_clamp_refuses_every_category_its_set_holds() -> None:
    clamp: Clamp[str] = Clamp((0, 1), category=("N", "NP"), negate=True)

    assert not admits(clamp, Entry("cat", "N"))
    assert not admits(clamp, Entry("cat", "NP"))
    assert admits(clamp, Entry("cat", "S"))


def test_a_set_and_a_meaning_are_both_required_of_an_entry() -> None:
    clamp: Clamp[str] = Clamp((0, 1), category=("N", "NP"), meaning="cat")

    assert admits(clamp, Entry("cat", "NP", "cat"))
    assert not admits(clamp, Entry("cat", "S", "cat"))
    assert not admits(clamp, Entry("cat", "NP", "dog"))


@given(categories=SETS, entry=ENTRIES)
def test_a_clamp_admits_exactly_the_entries_its_set_holds_the_category_of(
    categories: frozenset[str],
    entry: Entry[str],
) -> None:
    clamp: Clamp[str] = Clamp((0, 1), category=categories)

    assert admits(clamp, entry) is (entry.category in categories)


@given(categories=SETS, entry=ENTRIES)
def test_a_negated_clamp_admits_exactly_the_entries_its_set_leaves_out(
    categories: frozenset[str],
    entry: Entry[str],
) -> None:
    clamp: Clamp[str] = Clamp((0, 1), category=categories, negate=True)

    assert admits(clamp, entry) is (entry.category not in categories)


@given(
    category=st.sampled_from(CATEGORIES),
    meaning=st.sampled_from(MEANINGS),
    entry=ENTRIES,
    negate=st.booleans(),
)
def test_a_singleton_set_admits_what_naming_that_one_category_admitted(
    category: str,
    meaning: str | None,
    entry: Entry[str],
    *,
    negate: bool,
) -> None:
    clamp: Clamp[str] = Clamp(
        (0, 1), category={category}, meaning=meaning, negate=negate
    )

    assert admits(clamp, entry) is one_category_admits(
        category, meaning, entry, negate=negate
    )


@given(categories=ORDERINGS)
def test_two_clamps_over_one_set_agree_whatever_order_it_was_given_in(
    categories: list[str],
) -> None:
    forward: Clamp[str] = Clamp((1, 2), category=categories)
    backward: Clamp[str] = Clamp((1, 2), category=reversed(categories))

    assert forward == backward
    assert hash(forward) == hash(backward)
    assert len({forward, backward}) == 1


def test_a_member_given_twice_is_one_member() -> None:
    once: Clamp[str] = Clamp((1, 2), category=("N", "NP"))
    twice: Clamp[str] = Clamp((1, 2), category=("NP", "N", "N"))

    assert once.category == frozenset({"N", "NP"})
    assert once == twice
    assert hash(once) == hash(twice)


def test_clamps_over_different_sets_are_different_clamps() -> None:
    narrow: Clamp[str] = Clamp((1, 2), category=("N",))
    other: Clamp[str] = Clamp((1, 2), category=("NP",))
    wide: Clamp[str] = Clamp((1, 2), category=("N", "NP"))
    free: Clamp[str] = Clamp((1, 2), meaning="cat")

    assert narrow != other
    assert narrow != wide
    assert narrow != free
    assert len({narrow, other, wide, free}) == 4
    assert len({hash(narrow), hash(other), hash(wide)}) == 3


def test_a_clamp_cannot_be_rewritten_in_place() -> None:
    clamp: Clamp[str] = Clamp((1, 2), category={"N"})

    for field, value in (
        ("_span", (0, 5)),
        ("_category", frozenset({"NP"})),
        ("_negate", True),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(clamp, field, value)

    assert not hasattr(clamp, "__dict__")
    assert clamp.span == (1, 2)


def test_a_clamp_is_shared_by_copy_and_refuses_field_deletion() -> None:
    clamp: Clamp[str] = Clamp((1, 2), category={"N"})

    assert copy(clamp) is clamp
    assert deepcopy(clamp) is clamp

    with pytest.raises(AttributeError, match="immutable"):
        clamp.__delattr__("_span")

    assert clamp.span == (1, 2)


def test_a_clamp_renders_its_span_terms_and_direction() -> None:
    clamp: Clamp[str] = Clamp((1, 2), category={"N"})

    assert repr(clamp) == (
        "Clamp(span=(1, 2), category=frozenset({'N'}), meaning=None, "
        "mode='hard', penalty=None, scope='constituent', "
        "require_constituent=True, negate=False)"
    )


def test_a_table_enumerates_every_admitted_entry_beside_its_position() -> None:
    source = Table(TABLE)

    assert isinstance(source, LexicalSource)

    axioms = tuple(source.axioms(TOKENS))
    positions = [position for position, _, _ in axioms]

    assert positions == [0, 1, 1, 2]
    assert all(0 <= position < len(TOKENS) for position in positions)
    assert set(positions) == set(range(len(TOKENS)))
    assert [entry.category for _, entry, _ in axioms] == ["NP/N", "N", "NP", "S\\NP"]
    assert [weight for _, _, weight in axioms] == [-0.1, 0.0, -3.0, -0.2]


def test_a_position_admitting_nothing_leaves_the_others_where_they_are() -> None:
    source = Table(TABLE)
    axioms = tuple(source.axioms(("the", "dog", "sleeps")))

    assert [position for position, _, _ in axioms] == [0, 2]


def test_a_re_callable_source_reads_twice_and_agrees() -> None:
    source = Table(TABLE)

    assert source.is_re_callable

    enumeration = source.axioms(TOKENS)
    first = tuple(enumeration)
    second = tuple(enumeration)

    assert len(first) == 4
    assert second == first
    assert tuple(source.axioms(TOKENS)) == first


def test_a_draining_source_says_so_and_empties_on_a_second_reading() -> None:
    source = Draining(Table(TABLE))

    assert isinstance(source, LexicalSource)
    assert not source.is_re_callable

    enumeration = source.axioms(TOKENS)

    assert len(tuple(enumeration)) == 4
    assert tuple(enumeration) == ()
    assert len(tuple(source.axioms(TOKENS))) == 4


def test_reading_a_source_does_not_change_what_it_yields_next_time() -> None:
    source = Table(TABLE)
    first = tuple(source.axioms(TOKENS))

    assert tuple(source.axioms(("dog",))) == ()
    assert tuple(source.axioms(TOKENS)) == first


def test_a_clamped_source_declares_its_clamps_and_narrows_where_they_cover() -> None:
    clamp: Clamp[str] = Clamp((1, 2), category={"N"})
    source = Clamped(Table(TABLE), (clamp,))

    assert isinstance(source, LexicalSource)
    assert source.clamps == (clamp,)

    axioms = tuple(source.axioms(TOKENS))
    pinned = [entry.category for position, entry, _ in axioms if clamp.covers(position)]
    elsewhere = [
        (position, entry.category)
        for position, entry, _ in axioms
        if not clamp.covers(position)
    ]

    assert pinned == ["N"]
    assert elsewhere == [(0, "NP/N"), (2, "S\\NP")]


def test_a_negated_clamp_removes_what_it_names_and_keeps_the_rest() -> None:
    clamp: Clamp[str] = Clamp((1, 2), category={"N"}, negate=True)
    source = Clamped(Table(TABLE), (clamp,))
    axioms = tuple(source.axioms(TOKENS))

    assert [entry.category for position, entry, _ in axioms if position == 1] == ["NP"]


def test_a_clamp_naming_only_a_meaning_narrows_a_source_by_the_meaning() -> None:
    clamp: Clamp[str] = Clamp((2, 3), meaning="sleep")
    source = Clamped(Table(TABLE), (clamp,))
    axioms = tuple(source.axioms(TOKENS))

    assert [entry.category for position, entry, _ in axioms if position == 2] == [
        "S\\NP"
    ]
    assert [entry.category for position, entry, _ in axioms if position == 1] == [
        "N",
        "NP",
    ]


def test_a_clamp_naming_several_categories_keeps_every_one_of_them() -> None:
    clamp: Clamp[str] = Clamp((1, 2), category=("N", "NP"))
    source = Clamped(Table(TABLE), (clamp,))
    axioms = tuple(source.axioms(TOKENS))

    assert [entry.category for position, entry, _ in axioms if position == 1] == [
        "N",
        "NP",
    ]


def test_a_source_that_declares_no_clamps_is_not_a_lexical_source() -> None:
    assert not isinstance(Unclamped(), LexicalSource)


class TestAnEntry(ConformanceSuite):
    """Tests entries against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Entry[str]]:
        """Return the conformance the laws run an entry through."""
        return Conformance(
            build=lambda: Entry("cat", "N", "sleep", -0.5),
            unequal=Entry("cat", "NP", "sleep", -0.5),
            operations=(
                ("form", lambda entry: entry.form),
                ("category", lambda entry: entry.category),
                ("meaning", lambda entry: entry.meaning),
                ("weight", lambda entry: entry.weight),
            ),
        )


class TestAClamp(ConformanceSuite):
    """Tests clamps against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[Clamp[str]]:
        """Return the conformance the laws run a clamp through."""
        return Conformance(
            build=lambda: Clamp((1, 2), category=("N", "NP")),
            unequal=Clamp((1, 2), category=("N",)),
            operations=(
                ("span", lambda clamp: clamp.span),
                ("category", lambda clamp: clamp.category),
                ("meaning", lambda clamp: clamp.meaning),
                ("mode", lambda clamp: clamp.mode),
                ("penalty", lambda clamp: clamp.penalty),
                ("scope", lambda clamp: clamp.scope),
                ("require_constituent", lambda clamp: clamp.require_constituent),
                ("negate", lambda clamp: clamp.negate),
                ("covers", lambda clamp: clamp.covers(1)),
                ("admits", lambda clamp: admits(clamp, Entry("cat", "N"))),
            ),
        )
