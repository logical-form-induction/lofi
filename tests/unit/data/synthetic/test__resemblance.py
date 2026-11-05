from collections import Counter
from math import inf, log
from typing import TYPE_CHECKING

import pytest
from tests.unit.data.synthetic.conftest import (
    SampleSemanticSource,
    sample_digest,
    sample_rows,
    semantic_sample_digest,
    semantic_sample_rows,
    semantic_source,
)

from lofi.data.synthetic import Meaning, logical_classes
from lofi.data.synthetic._resemblance import (
    LexiconEstimate,
    SemanticLexiconEstimate,
    estimate,
    semantic_estimate,
)
from lofi.lexicon import Entry, StaticLexicon

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.unit.data.synthetic.conftest import SemanticRow

    from lofi.data.synthetic import LogicalClass

SAMPLE_SIZES = {
    "NP/N": 3,
    "N": 18,
    "NP": 4,
    "S\\NP/NP": 4,
    "N/N": 5,
    "S\\NP": 4,
}


def _sample() -> StaticLexicon[str]:
    return StaticLexicon(
        Entry(form, category, None, log(count))
        for form, category, count in sample_rows()
    )


def _zipf(exponent: float, forms: int) -> StaticLexicon[str]:
    return StaticLexicon(
        Entry(f"w{rank}", "N" if rank % 2 else "NP", None, -exponent * log(rank))
        for rank in range(1, forms + 1)
    )


def _estimate() -> LexiconEstimate:
    return estimate(_sample(), str, sample_digest())


def _sizes(rows: tuple[tuple[str, int, float], ...]) -> dict[str, int]:
    return {category: size for category, size, _ in rows}


def test_the_sample_is_read_as_class_sizes_shares_and_a_form_count() -> None:
    measured = _estimate()

    assert measured.digest == sample_digest()
    assert measured.forms == 34
    assert _sizes(measured.rows) == SAMPLE_SIZES
    assert sum(share for _, _, share in measured.rows) == pytest.approx(1.0)


def test_the_categories_are_ordered_by_descending_share() -> None:
    measured = _estimate()
    shares = [share for _, _, share in measured.rows]

    assert shares == sorted(shares, reverse=True)
    assert measured.rows[0][0] == "NP/N"


def test_a_shared_share_orders_the_categories_by_their_text() -> None:
    measured = estimate(
        StaticLexicon(
            (
                Entry("w0", "NP", None, 0.0),
                Entry("w1", "N", None, 0.0),
                Entry("w2", "S", None, 0.0),
            )
        ),
        str,
        "shared",
    )

    assert [category for category, _, _ in measured.rows] == ["N", "NP", "S"]


def test_a_form_bearing_two_categories_is_one_form_of_each_class() -> None:
    measured = estimate(
        StaticLexicon(
            (
                Entry("runs", "N", None, log(2.0)),
                Entry("runs", "S\\NP", None, log(6.0)),
                Entry("dog", "N", None, log(8.0)),
            )
        ),
        str,
        "shared",
    )

    assert _sizes(measured.rows) == {"N": 2, "S\\NP": 1}
    assert {category: share for category, _, share in measured.rows} == pytest.approx(
        {"N": 0.625, "S\\NP": 0.375}
    )


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0, 1.75])
def test_the_exponent_recovers_the_rank_law_the_masses_fall_along(
    exponent: float,
) -> None:
    measured = estimate(_zipf(exponent, 60), str, "zipf")

    assert measured.exponent == pytest.approx(exponent)


def test_the_sample_falls_along_a_decaying_rank_law() -> None:
    assert _estimate().exponent == pytest.approx(0.916, abs=1e-3)


def test_a_smaller_inventory_subsamples_every_class_in_proportion() -> None:
    measured = _estimate()
    scaled = measured.at(11)

    assert _sizes(scaled) == {
        "NP/N": 1,
        "N": 6,
        "NP": 1,
        "S\\NP/NP": 1,
        "N/N": 2,
        "S\\NP": 1,
    }
    assert [share for _, _, share in scaled] == [share for _, _, share in measured.rows]


def test_a_larger_inventory_extrapolates_every_class_in_proportion() -> None:
    assert _sizes(_estimate().at(71)) == {
        "NP/N": 6,
        "N": 38,
        "NP": 8,
        "S\\NP/NP": 8,
        "N/N": 10,
        "S\\NP": 8,
    }


def test_the_source_size_returns_the_source_joint() -> None:
    measured = _estimate()

    assert measured.at(measured.forms) == measured.rows


def test_a_class_that_scales_to_nothing_leaves_the_inventory_a_proper_subset() -> None:
    measured = _estimate()
    scaled = {category for category, _, _ in measured.at(5)}

    assert scaled < set(SAMPLE_SIZES)
    assert "NP/N" not in scaled
    assert "N" in scaled


def test_an_inventory_too_small_for_any_class_is_refused() -> None:
    sparse = LexiconEstimate("sparse", (("N", 4, 0.6), ("NP", 3, 0.4)), 10, 1.0)

    with pytest.raises(ValueError, match="too small to resemble a source of 10"):
        sparse.at(1)


def test_a_non_positive_inventory_is_refused() -> None:
    with pytest.raises(ValueError, match="a scaled inventory holds no form"):
        _estimate().at(0)


def test_a_source_of_one_form_has_no_rank_frequency_curve() -> None:
    with pytest.raises(ValueError, match="needs two forms, not 1"):
        estimate(StaticLexicon((Entry("dog", "N", None, 0.0),)), str, "one")


def test_a_source_of_no_form_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a form"):
        estimate(StaticLexicon(()), str, "none")


def test_a_form_under_no_category_is_refused() -> None:
    class Gapped:
        @property
        def forms(self) -> tuple[str, ...]:
            return ("cat", "dog")

        def entries(self, form: str) -> tuple[Entry[str], ...]:
            return () if form == "cat" else (Entry(form, "N", None, 0.0),)

        def log_mass(self, form: str) -> float:
            del form

            return 0.0

    with pytest.raises(ValueError, match="the form 'cat' under no category"):
        estimate(Gapped(), str, "gap")


def test_the_categories_are_recorded_under_the_text_the_renderer_gives() -> None:
    measured = estimate(_sample(), lambda category: category.lower(), sample_digest())

    assert set(_sizes(measured.rows)) == {category.lower() for category in SAMPLE_SIZES}


@pytest.mark.parametrize(
    ("rows", "refusal"),
    [
        ((), "needs a category"),
        ((("N", 0, 1.0),), "bears 0 of the source's 3 forms"),
        ((("N", 3, 1.0), ("N", 1, 1.0)), "names a category twice"),
        ((("N", 4, 1.0),), "bears 4 of the source's 3 forms"),
        ((("", 1, 1.0),), "an estimated category is empty"),
        ((("N", 1, 0.0),), "carries no token mass"),
        ((("N", 1, -1.0),), "carries no token mass"),
    ],
)
def test_an_estimate_refuses_a_joint_it_cannot_scale(
    rows: tuple[tuple[str, int, float], ...], refusal: str
) -> None:
    with pytest.raises(ValueError, match=refusal):
        LexiconEstimate("digest", rows, 3, 1.0)


@pytest.mark.parametrize(
    ("digest", "forms", "exponent", "refusal"),
    [
        ("", 3, 1.0, "a source digest is empty"),
        ("digest", 0, 1.0, "a source lexicon holds no form"),
        ("digest", 3, float("inf"), "an estimated exponent is not finite"),
    ],
)
def test_an_estimate_refuses_a_source_it_cannot_name(
    digest: str, forms: int, exponent: float, refusal: str
) -> None:
    with pytest.raises(ValueError, match=refusal):
        LexiconEstimate(digest, (("N", 1, 1.0),), forms, exponent)


def test_two_estimates_agree_when_the_source_and_the_joint_agree() -> None:
    measured = _estimate()
    same = estimate(_sample(), str, sample_digest())
    elsewhere = estimate(_sample(), str, "another-source")

    assert measured == same
    assert hash(measured) == hash(same)
    assert measured != elsewhere
    assert measured != measured.rows
    assert repr(measured).startswith(f"LexiconEstimate({sample_digest()!r}, (")


SAMPLE_CLASSES = {
    "quantifier": 2,
    "connective": 4,
    "predicate": 24,
    "identity": 1,
    "constant": 5,
}

SAMPLE_ARITIES = ((1, 16), (2, 6), (3, 2))


def _semantic() -> SemanticLexiconEstimate:
    return semantic_estimate(semantic_source(), semantic_sample_digest())


def _source(rows: Sequence[SemanticRow]) -> SampleSemanticSource:
    return SampleSemanticSource(rows)


def _semantic_zipf(exponent: float, constants: int) -> SampleSemanticSource:
    return _source(
        [
            (
                f"w{rank}",
                "NP",
                "constant",
                f"c{rank}",
                None,
                round(rank**-exponent * 10**6),
            )
            for rank in range(1, constants + 1)
        ]
    )


def _class_sizes(rows: tuple[tuple[str, int, float], ...]) -> dict[str, int]:
    return {logical_class: size for logical_class, size, _ in rows}


def test_the_semantic_sample_is_read_as_class_sizes_shares_and_an_inventory() -> None:
    measured = _semantic()

    assert measured.digest == semantic_sample_digest()
    assert _class_sizes(measured.rows) == SAMPLE_CLASSES
    assert measured.constants == 36
    assert sum(share for _, _, share in measured.rows) == pytest.approx(1.0)


def test_a_class_s_share_is_its_token_mass_and_not_its_item_count() -> None:
    rows = semantic_sample_rows()
    mass: Counter[str] = Counter()
    items: Counter[str] = Counter()

    for _, _, logical_class, _, _, count in rows:
        mass[logical_class] += count
        items[logical_class] += 1

    total = sum(mass.values())
    measured = {logical_class: share for logical_class, _, share in _semantic().rows}

    assert measured == pytest.approx(
        {logical_class: held / total for logical_class, held in mass.items()}
    )
    assert measured != pytest.approx(
        {logical_class: held / len(rows) for logical_class, held in items.items()}
    )


def test_a_class_is_sized_by_its_constants_and_not_by_its_items() -> None:
    measured = semantic_estimate(
        _source(
            [
                ("dogs", "NP", "predicate", "dog", 1, 26),
                ("dog", "N", "predicate", "dog", 1, 155),
                ("hound", "N", "predicate", "dog", 1, 9),
                ("cat", "N", "predicate", "cat", 1, 131),
                ("john", "NP", "constant", "john", None, 210),
            ]
        ),
        "repeated",
    )

    assert _class_sizes(measured.rows) == {"predicate": 2, "constant": 1}
    assert sum(items for _, _, _, items in measured.assignment) == 5


def test_the_classes_are_read_in_report_order() -> None:
    assert [logical_class for logical_class, _, _ in _semantic().rows] == list(
        logical_classes()
    )


def test_a_constant_borne_at_two_categories_is_one_constant_of_its_class() -> None:
    measured = _source(
        [
            ("dogs", "NP", "predicate", "dog", 1, 26),
            ("dog", "N", "predicate", "dog", 1, 155),
            ("cat", "N", "predicate", "cat", 1, 131),
        ]
    )
    read = semantic_estimate(measured, "shared")

    assert _class_sizes(read.rows) == {"predicate": 2}
    assert read.arities == ((1, 2),)
    assert read.assignment == (
        ("N", "predicate", 1, 2),
        ("NP", "predicate", 1, 1),
    )


def test_the_arity_rows_place_every_predicate_constant() -> None:
    measured = _semantic()

    assert measured.arities == SAMPLE_ARITIES
    assert sum(count for _, count in measured.arities) == SAMPLE_CLASSES["predicate"]


def test_the_closed_classes_name_the_constants_the_source_uses() -> None:
    measured = _semantic()

    assert measured.quantifiers == ("exists", "forall")
    assert measured.connectives == ("and", "if", "not", "or")
    assert len(measured.quantifiers) == SAMPLE_CLASSES["quantifier"]
    assert len(measured.connectives) == SAMPLE_CLASSES["connective"]


def test_the_assignment_counts_the_items_of_each_category_class_and_arity() -> None:
    assignment = {
        (category, logical_class, arity): items
        for category, logical_class, arity, items in _semantic().assignment
    }

    assert assignment["NP", "constant", None] == 5
    assert assignment["NP", "predicate", 1] == 2
    assert assignment["N", "predicate", 1] == 8
    assert assignment["N", "predicate", 2] == 2
    assert assignment["N", "constant", None] == 2
    assert assignment["NP/N", "quantifier", None] == 5
    assert sum(assignment.values()) == len(semantic_sample_rows())


@pytest.mark.parametrize("exponent", [0.0, 0.5, 1.0, 1.75])
def test_the_semantic_exponent_recovers_the_rank_law_the_masses_fall_along(
    exponent: float,
) -> None:
    measured = semantic_estimate(_semantic_zipf(exponent, 60), "zipf")

    assert measured.exponent == pytest.approx(exponent, abs=1e-3)


def test_the_semantic_sample_falls_along_a_decaying_rank_law() -> None:
    assert _semantic().exponent == pytest.approx(0.892, abs=1e-3)


def test_a_smaller_inventory_subsamples_every_logical_class_in_proportion() -> None:
    measured = _semantic()
    scaled = measured.at(24)

    assert _class_sizes(scaled) == {
        "quantifier": 1,
        "connective": 3,
        "predicate": 16,
        "identity": 1,
        "constant": 3,
    }
    assert sum(size for _, size, _ in scaled) == 24
    assert [share for _, _, share in scaled] == [share for _, _, share in measured.rows]


def test_a_larger_inventory_holds_the_closed_classes_and_grows_the_open_ones() -> None:
    measured = _semantic()
    scaled = measured.at(72)

    assert _class_sizes(scaled) == {
        "quantifier": 2,
        "connective": 4,
        "predicate": 54,
        "identity": 1,
        "constant": 11,
    }
    assert sum(size for _, size, _ in scaled) == 72
    assert all(
        size <= SAMPLE_CLASSES[logical_class]
        for logical_class, size, _ in scaled
        if logical_class in {"quantifier", "connective", "identity"}
    )


def test_the_source_inventory_returns_the_source_joint() -> None:
    measured = _semantic()

    assert measured.at(measured.constants) == measured.rows


def test_a_logical_class_that_scales_to_nothing_leaves_a_proper_subset() -> None:
    scaled = {logical_class for logical_class, _, _ in _semantic().at(3)}

    assert scaled < set(SAMPLE_CLASSES)
    assert "predicate" in scaled


def test_a_source_of_closed_classes_alone_cannot_be_grown() -> None:
    closed = SemanticLexiconEstimate(
        "closed",
        (("quantifier", 2, 0.7), ("connective", 2, 0.3)),
        (),
        (("NP/N", "quantifier", None, 4), ("(S\\S)/S", "connective", None, 2)),
        quantifiers=("exists", "forall"),
        connectives=("and", "or"),
        exponent=1.0,
    )

    assert closed.at(4) == closed.rows

    with pytest.raises(ValueError, match="are all closed classes"):
        closed.at(9)


def test_a_non_positive_semantic_inventory_is_refused() -> None:
    with pytest.raises(ValueError, match="a scaled inventory holds no constant"):
        _semantic().at(0)


def test_a_source_of_one_constant_has_no_rank_frequency_curve() -> None:
    with pytest.raises(ValueError, match="needs two logical constants, not 1"):
        semantic_estimate(_source([("john", "NP", "constant", "john", None, 3)]), "one")


def test_a_semantic_source_of_no_item_is_refused() -> None:
    with pytest.raises(ValueError, match="needs an item"):
        semantic_estimate(_source([]), "none")


def test_an_item_under_no_logical_form_is_refused() -> None:
    class Gapped:
        @property
        def items(self) -> tuple[tuple[str, str], ...]:
            return (("NP", "john"), ("NP", "mary"))

        def meaning(self, item: tuple[str, str]) -> Meaning | None:
            _, form = item

            return None if form == "john" else Meaning("constant", form, text=form)

        def log_mass(self, item: tuple[str, str]) -> float:
            del item

            return 0.0

    with pytest.raises(ValueError, match=r"the item \('NP', 'john'\) under no logical"):
        semantic_estimate(Gapped(), "gap")


def test_an_item_with_no_token_mass_is_refused() -> None:
    class Weightless:
        @property
        def items(self) -> tuple[tuple[str, str], ...]:
            return (("NP", "john"), ("NP", "mary"))

        def meaning(self, item: tuple[str, str]) -> Meaning | None:
            _, form = item

            return Meaning("constant", form, text=form)

        def log_mass(self, item: tuple[str, str]) -> float:
            _, form = item

            return 0.0 if form == "mary" else -inf

    with pytest.raises(ValueError, match=r"the item \('NP', 'john'\) with no token"):
        semantic_estimate(Weightless(), "weightless")


@pytest.mark.parametrize(
    ("rows", "arities", "assignment", "refusal"),
    [
        ((), (), (("N", "constant", None, 1),), "needs a class"),
        (
            (("constant", 1, 1.0), ("constant", 2, 1.0)),
            (),
            (("N", "constant", None, 1),),
            "names the constant class twice",
        ),
        (
            (("constant", 0, 1.0),),
            (),
            (("N", "constant", None, 1),),
            "class holds no constant",
        ),
        (
            (("constant", 1, 0.0),),
            (),
            (("N", "constant", None, 1),),
            "carries no token mass",
        ),
        (
            (("identity", 2, 1.0),),
            (),
            (("N", "identity", None, 1),),
            "identity class holds one constant, not 2",
        ),
        (
            (("predicate", 3, 1.0),),
            ((1, 2),),
            (("N", "predicate", 1, 1),),
            "place 2 predicate constants",
        ),
        (
            (("predicate", 2, 1.0),),
            ((1, 1), (1, 1)),
            (("N", "predicate", 1, 1),),
            "names the arity 1 twice",
        ),
        (
            (("predicate", 1, 1.0),),
            ((0, 1),),
            (("N", "predicate", 0, 1),),
            "an estimated arity is not positive",
        ),
        ((("constant", 1, 1.0),), (), (), "needs an assigned item"),
        (
            (("constant", 1, 1.0),),
            (),
            (("N", "predicate", 1, 1),),
            "which the estimate does not size",
        ),
        (
            (("constant", 1, 1.0),),
            (),
            (("N", "constant", 1, 1),),
            "only a predicate bears one",
        ),
        (
            (("predicate", 1, 1.0),),
            ((1, 1),),
            (("N", "predicate", None, 1),),
            "only a predicate bears one",
        ),
        (
            (("predicate", 1, 1.0),),
            ((1, 1),),
            (("N", "predicate", 2, 1),),
            "the unplaced arity 2",
        ),
        (
            (("constant", 1, 1.0),),
            (),
            (("N", "constant", None, 0),),
            "no item of category 'N'",
        ),
        (
            (("constant", 1, 1.0),),
            (),
            (("", "constant", None, 1),),
            "an assigned category is empty",
        ),
        (
            (("constant", 1, 1.0),),
            (),
            (("N", "constant", None, 1), ("N", "constant", None, 2)),
            "class at arity None twice",
        ),
    ],
)
def test_a_semantic_estimate_refuses_a_joint_it_cannot_scale(
    rows: tuple[tuple[LogicalClass, int, float], ...],
    arities: tuple[tuple[int, int], ...],
    assignment: tuple[tuple[str, LogicalClass, int | None, int], ...],
    refusal: str,
) -> None:
    with pytest.raises(ValueError, match=refusal):
        SemanticLexiconEstimate("digest", rows, arities, assignment, exponent=1.0)


@pytest.mark.parametrize(
    ("quantifiers", "connectives", "refusal"),
    [
        (
            ("exists",),
            (),
            "quantifier class holds 0 constants but the estimate names 1",
        ),
        ((), ("and",), "connective class holds 0 constants but the estimate names 1"),
        (("",), (), "an estimated quantifier is empty"),
    ],
)
def test_a_semantic_estimate_refuses_a_closed_class_it_cannot_name(
    quantifiers: tuple[str, ...], connectives: tuple[str, ...], refusal: str
) -> None:
    with pytest.raises(ValueError, match=refusal):
        SemanticLexiconEstimate(
            "digest",
            (("constant", 1, 1.0),),
            (),
            (("N", "constant", None, 1),),
            quantifiers=quantifiers,
            connectives=connectives,
            exponent=1.0,
        )


def test_a_semantic_estimate_refuses_a_quantifier_named_twice() -> None:
    with pytest.raises(ValueError, match="names a quantifier twice"):
        SemanticLexiconEstimate(
            "digest",
            (("quantifier", 2, 1.0),),
            (),
            (("NP/N", "quantifier", None, 1),),
            quantifiers=("exists", "exists"),
            exponent=1.0,
        )


@pytest.mark.parametrize(
    ("digest", "exponent", "refusal"),
    [
        ("", 1.0, "a source digest is empty"),
        ("digest", float("inf"), "an estimated exponent is not finite"),
    ],
)
def test_a_semantic_estimate_refuses_a_source_it_cannot_name(
    digest: str, exponent: float, refusal: str
) -> None:
    with pytest.raises(ValueError, match=refusal):
        SemanticLexiconEstimate(
            digest,
            (("constant", 1, 1.0),),
            (),
            (("N", "constant", None, 1),),
            exponent=exponent,
        )


def test_two_semantic_estimates_agree_when_the_source_and_the_joint_agree() -> None:
    measured = _semantic()
    same = semantic_estimate(semantic_source(), semantic_sample_digest())
    elsewhere = semantic_estimate(semantic_source(), "another-source")

    assert measured == same
    assert hash(measured) == hash(same)
    assert measured != elsewhere
    assert measured != measured.rows
    assert repr(measured).startswith(
        f"SemanticLexiconEstimate({semantic_sample_digest()!r}, ("
    )
