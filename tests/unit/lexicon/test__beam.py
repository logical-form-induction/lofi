from math import log
from typing import Final, cast

import pytest
from hypothesis import strategies as st
from tests.conformance.lexical_source import (
    LexicalSourceConformance,
    LexicalSourceSuite,
)

from lofi.lexicon._beam import BeamSource, GrowingBeamSource
from lofi.lexicon._errors import ExhaustedBeamError
from lofi.lexicon._ports import Entry, LexicalSource
from lofi.lexicon.laws import LexicalSourceFixture


def fresh(category: str) -> str:
    return category[:1] + category[1:]


# two log-scored beams requiring two widenings for a spanning parse.
DOGS: Final = (
    Entry("dogs", fresh("S"), weight=log(0.5)),
    Entry("dogs", fresh("S\\NP"), weight=log(0.3)),
    Entry("dogs", fresh("NP"), weight=log(0.2)),
)

BARK: Final = (Entry("bark", fresh("S\\NP"), weight=log(1.0)),)

BEAMS: Final = (DOGS, BARK)
SENTENCE: Final = ("dogs", "bark")
EXAMPLES: Final = 50

SENTENCES = st.lists(st.sampled_from(("dogs", "bark", "cat")), max_size=3).map(
    lambda rest: ("dogs", *rest)
)


def built() -> BeamSource[str]:
    return BeamSource(BEAMS)


def seeded() -> GrowingBeamSource[str]:
    return GrowingBeamSource(built())


def categories(source: LexicalSource[str, float], tokens: tuple[str, ...]) -> list[str]:
    return [entry.category for _, entry, _ in source.axioms(tokens)]


def _applies(left: str, right: str) -> str | None:
    """Return the result of applying either adjacent category to the other.

    Parameters
    ----------
    left
        The left category.
    right
        The right category.

    Returns
    -------
    str or None
        The application result, or ``None`` if neither direction applies.
    """
    if right.endswith("\\" + left):
        return right[: -len(left) - 1]

    if left.endswith("/" + right):
        return left[: -len(right) - 1]

    return None


def spanning(source: LexicalSource[str, float], tokens: tuple[str, ...]) -> set[str]:
    """Return categories spanning the complete sentence under application.

    Parameters
    ----------
    source
        The lexical source for the sentence.
    tokens
        The sentence to parse.

    Returns
    -------
    set
        Every category spanning the sentence.
    """
    cells: dict[tuple[int, int], set[str]] = {
        (start, start + 1): set() for start in range(len(tokens))
    }

    for position, entry, _ in source.axioms(tokens):
        cells[position, position + 1].add(entry.category)

    for width in range(2, len(tokens) + 1):
        for start in range(len(tokens) - width + 1):
            end = start + width
            made: set[str] = set()

            for cut in range(start + 1, end):
                for left in cells[start, cut]:
                    for right in cells[cut, end]:
                        found = _applies(left, right)

                        if found is not None:
                            made.add(found)

            cells[start, end] = made

    return cells[0, len(tokens)]


def widen_until_spanning(
    source: GrowingBeamSource[str], tokens: tuple[str, ...]
) -> tuple[GrowingBeamSource[str], int]:
    """Widen the best available position until a parse spans the sentence.

    Parameters
    ----------
    source
        The seeded source to widen.
    tokens
        The sentence to parse.

    Returns
    -------
    tuple
        The final source and the number of widening calls.
    """
    calls = 0

    while "S" not in spanning(source, tokens):
        offers = [
            (candidate.weight, position)
            for position in range(len(tokens))
            if (candidate := source.next_best(position)) is not None
        ]

        if not offers:
            break

        source = source.widen(max(offers)[1])
        calls += 1

    return (source, calls)


def test_a_beam_comes_back_in_descending_weight_order() -> None:
    source = built()

    assert [entry.category for entry in source.candidates(0)] == ["S", "S\\NP", "NP"]
    assert [entry.weight for entry in source.candidates(0)] == sorted(
        (entry.weight for entry in DOGS), reverse=True
    )


def test_candidates_a_tagger_scored_equally_keep_the_order_it_gave() -> None:
    tied = BeamSource(
        (
            (
                Entry("cat", "N", weight=-1.0),
                Entry("cat", "NP", weight=-1.0),
                Entry("cat", "S", weight=-0.5),
            ),
        )
    )

    assert [entry.category for entry in tied.candidates(0)] == ["S", "N", "NP"]


def test_one_category_is_stored_once_across_the_whole_sentence() -> None:
    source = built()

    assert source.candidates(0)[1].category is source.candidates(1)[0].category
    assert source.candidates(0)[1].category == "S\\NP"


def test_the_widths_and_the_beams_cannot_disagree() -> None:
    source = built()

    assert source.widths == tuple(len(beam) for beam in source.beams)
    assert source.positions == len(source.beams)
    assert len(source) == 4


def test_a_beam_holding_entries_over_two_forms_is_refused() -> None:
    with pytest.raises(ValueError, match="position 0"):
        BeamSource(((Entry("dogs", "NP"), Entry("cats", "NP")),))


def test_a_beam_holding_one_candidate_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        BeamSource(((Entry("dogs", "NP", weight=-1.0), Entry("dogs", "NP")),))


def test_a_beam_over_an_empty_form_or_an_infinite_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="form"):
        BeamSource(((Entry("", "NP"),),))

    with pytest.raises(ValueError, match="weight"):
        BeamSource(((Entry("dogs", "NP", weight=float("-inf")),),))


def test_a_position_the_beams_do_not_cover_is_refused() -> None:
    source = built()

    with pytest.raises(IndexError, match="position 2"):
        source.candidates(2)

    with pytest.raises(IndexError, match="position -1"):
        source.candidates(-1)


def test_the_full_source_admits_every_candidate_of_its_own_sentence() -> None:
    assert categories(built(), SENTENCE) == ["S", "S\\NP", "NP", "S\\NP"]


def test_a_beam_admits_nothing_where_another_token_stands() -> None:
    assert categories(built(), ("cat", "bark")) == ["S\\NP"]
    assert categories(built(), ("cat", "cat")) == []


def test_a_sentence_shorter_than_the_beams_reaches_only_what_it_covers() -> None:
    assert categories(built(), ("dogs",)) == ["S", "S\\NP", "NP"]
    assert categories(built(), ()) == []


def test_a_sentence_longer_than_the_beams_reaches_only_what_they_cover() -> None:
    admitted = built().axioms(("dogs", "bark", "loudly"))

    assert {position for position, _, _ in admitted} == {0, 1}


def test_the_weights_are_the_ones_the_tagger_reported() -> None:
    weights = [weight for _, _, weight in built().axioms(("dogs",))]

    assert weights == [log(0.5), log(0.3), log(0.2)]


def test_a_seeded_source_admits_the_best_candidate_of_every_position() -> None:
    source = seeded()

    assert source.positions == 2
    assert source.frontier == (1, 1)
    assert source.widenings == 0
    assert categories(source, SENTENCE) == ["S", "S\\NP"]


def test_a_position_offering_nothing_is_seeded_at_nothing() -> None:
    source = GrowingBeamSource(BeamSource((DOGS, ())))

    assert source.frontier == (1, 0)
    assert source.widenings == 0


def test_widening_admits_the_next_best_candidate_and_reports_the_count() -> None:
    once = seeded().widen(0)

    assert once.frontier == (2, 1)
    assert once.widenings == 1
    assert categories(once, SENTENCE) == ["S", "S\\NP", "S\\NP"]

    twice = once.widen(0)

    assert twice.frontier == (3, 1)
    assert twice.widenings == 2
    assert categories(twice, SENTENCE) == ["S", "S\\NP", "NP", "S\\NP"]


def test_widening_leaves_the_source_it_was_called_on_where_it_was() -> None:
    source = seeded()
    before = source.axioms(SENTENCE)

    widened = source.widen(0)

    assert widened is not source
    assert source.frontier == (1, 1)
    assert source.widenings == 0
    assert source.axioms(SENTENCE) == before


def test_the_next_best_candidate_is_the_one_a_widening_admits() -> None:
    source = seeded()
    offered = source.next_best(0)

    assert offered is not None
    assert offered.category == "S\\NP"
    assert offered in [entry for _, entry, _ in source.widen(0).axioms(SENTENCE)]
    assert source.next_best(1) is None


def test_widening_an_exhausted_position_is_refused_naming_what_it_offered() -> None:
    source = seeded()

    with pytest.raises(ExhaustedBeamError) as refusal:
        source.widen(1)

    assert refusal.value.position == 1
    assert refusal.value.width == 1
    assert "its one candidate" in str(refusal.value)


def test_widening_a_position_offering_nothing_is_refused() -> None:
    source = GrowingBeamSource(BeamSource((DOGS, ())))

    with pytest.raises(ExhaustedBeamError, match="no candidate at all"):
        source.widen(1)


def test_widening_a_position_the_beams_do_not_cover_is_refused() -> None:
    with pytest.raises(IndexError, match="position 7"):
        seeded().widen(7)


def test_a_frontier_that_is_not_one_entry_per_position_is_refused() -> None:
    with pytest.raises(ValueError, match="one entry per position"):
        GrowingBeamSource(built(), frontier=(1,))


def test_a_frontier_wider_than_the_beam_it_counts_into_is_refused() -> None:
    with pytest.raises(ValueError, match="position 1"):
        GrowingBeamSource(built(), frontier=(1, 2))


def test_a_frontier_admitting_nothing_where_a_beam_offers_something_is_refused() -> (
    None
):
    with pytest.raises(ValueError, match="admits none"):
        GrowingBeamSource(built(), frontier=(0, 1))


def test_a_frontier_entry_that_is_not_a_count_is_refused() -> None:
    with pytest.raises(TypeError, match="frontier entry"):
        GrowingBeamSource(built(), frontier=cast("tuple[int, ...]", (1.0, 1)))


def test_the_only_spanning_parse_is_reached_in_exactly_two_widenings() -> None:
    source = seeded()

    assert "S" not in spanning(source, SENTENCE)
    assert "S" not in spanning(source.widen(0), SENTENCE)

    reached, calls = widen_until_spanning(source, SENTENCE)

    assert "S" in spanning(reached, SENTENCE)
    assert calls == 2
    assert reached.widenings == 2


def test_the_search_leaves_the_source_it_searched_over_where_it_was() -> None:
    source = seeded()
    before = (source.frontier, source.widenings, source.axioms(SENTENCE))

    widen_until_spanning(source, SENTENCE)

    assert (source.frontier, source.widenings, source.axioms(SENTENCE)) == before
    assert source.is_re_callable


def test_the_widened_source_admits_everything_the_one_before_it_did() -> None:
    source = seeded()
    reached, _ = widen_until_spanning(source, SENTENCE)

    assert set(categories(source, SENTENCE)) <= set(categories(reached, SENTENCE))
    assert set(categories(reached, SENTENCE)) <= set(categories(built(), SENTENCE))


def test_both_sources_present_the_lexical_source_surface() -> None:
    for source in (built(), seeded()):
        assert isinstance(source, LexicalSource)
        assert source.is_re_callable
        assert source.clamps == ()


def test_the_sources_render_what_they_hold() -> None:
    assert repr(built()) == "BeamSource(positions=2, candidates=4)"
    assert repr(seeded().widen(0)) == (
        "GrowingBeamSource(BeamSource(positions=2, candidates=4), widenings=1)"
    )


class TestABeamSource(LexicalSourceSuite):
    """Tests fixed beams against lexical-source laws."""

    @pytest.fixture
    def conformance(self) -> LexicalSourceConformance[str, float]:
        """Return the conformance the laws run the whole beam through."""
        return LexicalSourceConformance(
            fixture=LexicalSourceFixture(
                build=built,
                tokens=SENTENCES,
                max_examples=EXAMPLES,
            ),
            sentence=SENTENCE,
        )


class TestAGrowingBeamSource(LexicalSourceSuite):
    """Tests growing beams against lexical-source laws."""

    @pytest.fixture
    def conformance(self) -> LexicalSourceConformance[str, float]:
        """Return the conformance the laws run the seeded beam through."""
        return LexicalSourceConformance(
            fixture=LexicalSourceFixture(
                build=seeded,
                tokens=SENTENCES,
                max_examples=EXAMPLES,
            ),
            sentence=SENTENCE,
        )
