from math import exp, inf, isclose, log
from typing import Final

import pytest
from hypothesis import strategies as st
from tests.conformance.lexical_source import (
    LexicalSourceConformance,
    LexicalSourceSuite,
)

from lofi.lexicon._ports import Entry, LexicalSource
from lofi.lexicon._static import StaticLexicon
from lofi.lexicon.laws import LexicalSourceFixture


def fresh(category: str) -> str:
    return category[:1] + category[1:]


# three log-count readings with distinct category objects for interning checks.
BANK: Final = (
    Entry("bank", fresh("NP"), "riverside", log(6.0)),
    Entry("bank", fresh("NP"), "vault", log(3.0)),
    Entry("bank", fresh("S\\NP/NP"), "deposit", log(1.0)),
)

TABLE: Final = (
    *BANK,
    Entry("the", fresh("NP/N"), weight=log(4.0)),
    Entry("river", fresh("NP"), "stream", log(2.0)),
)

KNOWN: Final = ("bank", "the", "river")
UNKNOWN: Final = ("aardvark", "zebra")
EXAMPLES: Final = 50

SENTENCES = st.lists(st.sampled_from([*KNOWN, *UNKNOWN]), max_size=4).map(tuple)


def built() -> StaticLexicon[str]:
    return StaticLexicon(TABLE)


def total_mass(lexicon: StaticLexicon[str], form: str) -> float:
    return sum(exp(entry.weight) for entry in lexicon.entries(form))


def test_a_form_with_three_readings_keeps_three_entries() -> None:
    lexicon = StaticLexicon(BANK)

    assert len(lexicon.entries("bank")) == 3
    assert len(lexicon) == 3
    assert lexicon.forms == ("bank",)


def test_the_entries_of_one_form_carry_a_probability_distribution() -> None:
    lexicon = built()

    for form in KNOWN:
        assert abs(total_mass(lexicon, form) - 1.0) < 1e-9


def test_the_entry_count_is_the_number_of_records_read() -> None:
    lexicon = built()

    assert len(lexicon) == len(TABLE)
    assert sum(len(lexicon.entries(form)) for form in lexicon.forms) == len(TABLE)


def test_the_weights_are_the_category_given_the_word() -> None:
    lexicon = StaticLexicon(BANK)
    riverside, vault, ditransitive = lexicon.entries("bank")

    assert isclose(exp(riverside.weight), 0.6)
    assert isclose(exp(vault.weight), 0.3)
    assert isclose(exp(ditransitive.weight), 0.1)


def test_the_mass_of_a_form_is_the_total_its_entries_were_built_with() -> None:
    lexicon = built()

    assert isclose(exp(lexicon.log_mass("bank")), 10.0)
    assert isclose(exp(lexicon.log_mass("the")), 4.0)
    assert isclose(exp(lexicon.log_mass("river")), 2.0)
    assert lexicon.log_mass("aardvark") == -inf

    for form in KNOWN:
        for entry in lexicon.entries(form):
            original = next(
                held
                for held in TABLE
                if held.form == form
                and held.category == entry.category
                and held.meaning == entry.meaning
            )

            assert isclose(lexicon.log_mass(form) + entry.weight, original.weight)


def test_a_table_carrying_no_weights_is_uniform_over_a_form() -> None:
    lexicon = StaticLexicon(
        (Entry("run", "NP"), Entry("run", "S\\NP"), Entry("run", "S/NP"))
    )

    for entry in lexicon.entries("run"):
        assert isclose(exp(entry.weight), 1 / 3)


def test_one_object_stands_for_every_category_equal_to_it() -> None:
    lexicon = built()
    stored = (
        lexicon.entries("bank")[0].category,
        lexicon.entries("river")[0].category,
    )

    assert BANK[0].category is not TABLE[-1].category
    assert stored[0] == stored[1] == "NP"
    assert stored[0] is stored[1]


def test_a_category_appearing_twice_under_one_form_stays_two_entries() -> None:
    lexicon = StaticLexicon(BANK)
    categories = [entry.category for entry in lexicon.entries("bank")]

    assert BANK[0].category is not BANK[1].category
    assert categories[0] == categories[1] == "NP"
    assert categories[0] is categories[1]
    assert len(lexicon.entries("bank")) == 3


def test_a_repeated_entry_gathers_its_mass_rather_than_doubling() -> None:
    lexicon = StaticLexicon(
        (
            Entry("bank", fresh("NP"), "riverside", log(1.0)),
            Entry("bank", fresh("NP"), "riverside", log(2.0)),
            Entry("bank", fresh("S\\NP/NP"), "deposit", log(1.0)),
        )
    )

    assert len(lexicon) == 2
    assert isclose(exp(lexicon.entries("bank")[0].weight), 0.75)


def test_a_scored_lookup_finds_the_category_the_table_stored() -> None:
    lexicon = built()

    assert isclose(exp(lexicon.log_probability("bank", fresh("NP"))), 0.9)
    assert isclose(exp(lexicon.log_probability("bank", "S\\NP/NP")), 0.1)
    assert isclose(exp(lexicon.log_probability("the", "NP/N")), 1.0)


def test_a_category_a_form_does_not_bear_weighs_log_zero() -> None:
    lexicon = built()

    assert lexicon.log_probability("bank", "PP") == -inf
    assert lexicon.log_probability("aardvark", "NP") == -inf


def test_a_form_the_table_does_not_hold_admits_nothing() -> None:
    lexicon = built()

    assert lexicon.entries("aardvark") == ()
    assert lexicon.axioms(("aardvark",)) == ()


def test_a_miss_writes_nothing_into_the_table() -> None:
    lexicon = built()
    before = (lexicon.forms, len(lexicon), lexicon.log_probability("bank", "NP"))

    lexicon.entries("aardvark")
    lexicon.log_probability("aardvark", "NP")
    lexicon.axioms(("aardvark", "zebra", "aardvark"))

    after = (lexicon.forms, len(lexicon), lexicon.log_probability("bank", "NP"))

    assert before == after
    assert "aardvark" not in lexicon.forms


def test_a_denominator_survives_every_reading_that_missed() -> None:
    lexicon = built()
    want = total_mass(lexicon, "bank")

    for _ in range(20):
        lexicon.axioms(("aardvark", "bank"))

    assert isclose(total_mass(lexicon, "bank"), want)
    assert abs(want - 1.0) < 1e-9


def test_the_axioms_of_a_sentence_carry_the_position_and_the_weight() -> None:
    lexicon = built()
    drawn = lexicon.axioms(("the", "bank"))

    assert [position for position, _, _ in drawn] == [0, 1, 1, 1]
    assert all(weight == entry.weight for _, entry, weight in drawn)
    assert {entry.form for _, entry, _ in drawn} == {"the", "bank"}


def test_an_unknown_token_leaves_the_positions_of_the_others_alone() -> None:
    lexicon = built()
    drawn = lexicon.axioms(("aardvark", "the", "zebra", "the"))

    assert [position for position, _, _ in drawn] == [1, 3]


def test_the_empty_sentence_admits_nothing() -> None:
    assert built().axioms(()) == ()


def test_an_empty_table_holds_nothing() -> None:
    lexicon = StaticLexicon(())

    assert len(lexicon) == 0
    assert lexicon.forms == ()
    assert lexicon.axioms(("the",)) == ()
    assert lexicon.log_probability("the", "NP/N") == -inf


def test_the_table_declares_itself_re_readable_and_unclamped() -> None:
    lexicon = built()

    assert isinstance(lexicon, LexicalSource)
    assert lexicon.is_re_callable
    assert lexicon.clamps == ()


def test_two_readings_of_one_sentence_agree() -> None:
    lexicon = built()

    assert lexicon.axioms(("the", "bank")) == lexicon.axioms(("the", "bank"))


def test_reweighting_a_table_leaves_it_the_same_size() -> None:
    heavier = tuple(
        Entry(entry.form, entry.category, entry.meaning, entry.weight + 3.0)
        for entry in TABLE
    )
    lexicon = StaticLexicon(heavier)
    plain = built()

    assert len(lexicon) == len(plain)
    assert lexicon.forms == plain.forms

    for form in plain.forms:
        assert isclose(
            lexicon.log_probability(form, "NP"), plain.log_probability(form, "NP")
        )


@pytest.mark.parametrize(
    ("broken", "raised", "message"),
    [
        ((Entry("", "NP"),), ValueError, "empty"),
        ((Entry("the", "NP/N", weight=inf),), ValueError, "finite"),
        ((Entry("the", "NP/N", weight=float("nan")),), ValueError, "finite"),
    ],
)
def test_a_record_the_table_refuses_is_named_in_the_message(
    broken: tuple[Entry[str], ...],
    raised: type[Exception],
    message: str,
) -> None:
    with pytest.raises(raised, match=message):
        StaticLexicon(broken)


def test_the_table_renders_its_two_counts() -> None:
    assert repr(built()) == "StaticLexicon(forms=3, entries=5)"


class TestAStaticLexicon(LexicalSourceSuite):
    """Tests static lexicons against lexical-source laws."""

    @pytest.fixture
    def conformance(self) -> LexicalSourceConformance[str, float]:
        """Return the conformance the laws run the table through."""
        return LexicalSourceConformance(
            fixture=LexicalSourceFixture(
                build=built,
                tokens=SENTENCES,
                max_examples=EXAMPLES,
            ),
            sentence=("the", "bank", "river"),
        )
