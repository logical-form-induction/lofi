from copy import copy, deepcopy
from pickle import dumps, loads
from typing import TYPE_CHECKING, NamedTuple, cast

import pytest

from lofi._types import ItemId
from lofi.data.corpora.ccgbank import PargDependency, PargSentence
from lofi.data.corpora.ccgbank._parg import PargCodec
from lofi.data.deps import LexicalHead, ProjectedDependency
from lofi.eval._dependencies import (
    DEPENDENCY_MATCH_MODES,
    CompatibilityPattern,
    DependencyArc,
    DependencyCompatibilityFilter,
    DependencyMatchMode,
    DependencyPrediction,
    DependencyScorer,
    DependencyScoreRow,
    DependencyToken,
    parg_gold,
    projected_prediction,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _arc(
    predicate: int,
    argument: int,
    *,
    forms: tuple[str, str] = ("head", "dependent"),
    label: tuple[str | None, int | None, str | None] = (
        "S\\NP",
        1,
        "application",
    ),
) -> DependencyArc:
    return DependencyArc(
        DependencyToken(predicate, forms[0]),
        DependencyToken(argument, forms[1]),
        label[0],
        label[1],
        label[2],
    )


def _prediction(*arcs: DependencyArc) -> DependencyPrediction:
    return DependencyPrediction(arcs)


def _row(
    scorer: DependencyScorer,
    mode: DependencyMatchMode,
    coverage: str,
    metric: str,
) -> DependencyScoreRow:
    name = f"dependency:{mode}:{coverage}:{metric}"
    return next(row for row in scorer.rows() if row.name == name)


class _DenominatorCase(NamedTuple):
    prediction: DependencyPrediction | None
    gold: DependencyPrediction
    predicted: int
    expected_gold: int
    exact: float
    coverage: float


def test_rows_cross_modes_policies_and_metrics_once_in_stable_order() -> None:
    scorer = DependencyScorer()
    rows = scorer.rows()
    metrics = (
        "micro-precision",
        "micro-recall",
        "micro-f-score",
        "macro-precision",
        "macro-recall",
        "macro-f-score",
        "sentence-exact",
        "coverage",
    )
    expected = tuple(
        f"dependency:{mode}:{coverage}:{metric}"
        for mode in DEPENDENCY_MATCH_MODES
        for coverage in ("ignore", "zero")
        for metric in metrics
    )

    assert len(rows) == 64
    assert tuple(row.name for row in rows) == expected
    assert len(set(expected)) == 64
    assert all(row.result.value is None and row.result.n == 0 for row in rows)

    for row in rows:
        factors = row.conventions
        assert factors.punctuation == "keep"
        assert factors.unary_chains == "keep"
        assert factors.trivial_spans == "keep"
        assert factors.minimum_length == 0
        assert factors.coverage in ("ignore", "zero")
        assert factors.self_arcs == "forbid"
        assert factors.gold_derivation == "canonical"
        expected_aggregation = (
            "sentence"
            if any(
                metric in row.name
                for metric in ("macro-", "sentence-exact", ":coverage")
            )
            else "corpus"
        )
        assert factors.aggregation == expected_aggregation


def test_four_modes_preserve_multiplicity_position_and_direction() -> None:
    repeated = _arc(
        0,
        1,
        forms=("same", "same"),
        label=("C", 1, "application"),
    )
    gold = _prediction(
        repeated,
        repeated,
        _arc(
            2,
            3,
            forms=("same", "other"),
            label=("D", 2, "application"),
        ),
    )
    prediction = _prediction(
        repeated,
        _arc(
            1,
            0,
            forms=("same", "same"),
            label=("C", 1, "application"),
        ),
        _arc(
            2,
            3,
            forms=("same", "other"),
            label=("X", 2, "application"),
        ),
        _arc(
            4,
            5,
            forms=("same", "same"),
            label=("C", 1, "application"),
        ),
    )
    scorer = DependencyScorer()
    scorer.update(prediction, gold, key=ItemId("modes"))
    expected_matches = {
        "labeled-directed": 1.0,
        "labeled-undirected": 2.0,
        "unlabeled-directed": 2.0,
        "unlabeled-undirected": 3.0,
    }

    for mode, matched in expected_matches.items():
        row = _row(scorer, cast("DependencyMatchMode", mode), "ignore", "micro-f-score")
        assert row.result.components == {
            "matched": matched,
            "predicted": 4.0,
            "gold": 3.0,
        }
        precision = matched / 4
        recall = matched / 3
        assert row.result.value == pytest.approx(
            2 * precision * recall / (precision + recall)
        )


def test_micro_and_macro_aggregation_use_their_exact_denominators() -> None:
    scorer = DependencyScorer()
    one = _arc(0, 1)
    scorer.update(_prediction(one), _prediction(one), key=ItemId("short"))
    scorer.update(
        _prediction(_arc(2, 3)),
        _prediction(_arc(2, 3), _arc(4, 5), _arc(6, 7)),
        key=ItemId("long"),
    )
    mode: DependencyMatchMode = "labeled-directed"

    assert _row(scorer, mode, "ignore", "micro-precision").result.value == 1.0
    assert _row(scorer, mode, "ignore", "micro-recall").result.value == 0.5
    assert _row(scorer, mode, "ignore", "micro-f-score").result.value == pytest.approx(
        2 / 3
    )
    assert _row(scorer, mode, "ignore", "macro-precision").result.value == 1.0
    assert _row(scorer, mode, "ignore", "macro-recall").result.value == pytest.approx(
        2 / 3
    )
    assert _row(scorer, mode, "ignore", "macro-f-score").result.value == 0.75


def test_failures_and_successful_empty_predictions_have_distinct_denominators() -> None:
    scorer = DependencyScorer()
    empty = DependencyPrediction(())
    one = _prediction(_arc(0, 1))
    scorer.update(None, one, key=ItemId("failed-with-gold"))
    scorer.update(empty, one, key=ItemId("empty-with-gold"))
    scorer.update(None, empty, key=ItemId("failed-empty"))
    scorer.update(empty, empty, key=ItemId("empty-empty"))
    mode: DependencyMatchMode = "labeled-directed"

    ignore_exact = _row(scorer, mode, "ignore", "sentence-exact")
    zero_exact = _row(scorer, mode, "zero", "sentence-exact")
    ignore_recall = _row(scorer, mode, "ignore", "micro-recall")
    zero_recall = _row(scorer, mode, "zero", "micro-recall")

    assert ignore_exact.result.value == 0.5
    assert ignore_exact.result.n == 2
    assert zero_exact.result.value == 0.25
    assert zero_exact.result.n == 4
    assert ignore_recall.result.value == 0.0
    assert ignore_recall.result.components["gold"] == 1.0
    assert zero_recall.result.value == 0.0
    assert zero_recall.result.components["gold"] == 2.0

    for coverage in ("ignore", "zero"):
        coverage_row = _row(scorer, mode, coverage, "coverage")
        assert coverage_row.result.value == 0.5
        assert coverage_row.result.n == 4
        assert coverage_row.result.components == {
            "parsed": 2.0,
            "sentences": 4.0,
        }

    assert ignore_exact.result.components == {"exact": 1.0, "sentences": 2.0}
    assert zero_exact.result.components == {"exact": 1.0, "sentences": 4.0}


def test_all_failures_are_undefined_when_ignored_and_zero_when_penalized() -> None:
    scorer = DependencyScorer()
    scorer.update(None, _prediction(_arc(0, 1)), key=ItemId("failure"))
    mode: DependencyMatchMode = "labeled-directed"

    for metric in (
        "micro-precision",
        "micro-recall",
        "micro-f-score",
        "macro-precision",
        "macro-recall",
        "macro-f-score",
        "sentence-exact",
    ):
        ignored = _row(scorer, mode, "ignore", metric).result
        penalized = _row(scorer, mode, "zero", metric).result
        assert ignored.value is None
        assert ignored.n == 0
        assert penalized.value == 0.0
        assert penalized.n == 1


@pytest.mark.parametrize(
    "case",
    [
        _DenominatorCase(
            DependencyPrediction(()), DependencyPrediction(()), 0, 0, 1.0, 1.0
        ),
        _DenominatorCase(
            DependencyPrediction(()),
            _prediction(_arc(0, 1)),
            0,
            1,
            0.0,
            1.0,
        ),
        _DenominatorCase(
            _prediction(_arc(0, 1)),
            DependencyPrediction(()),
            1,
            0,
            0.0,
            1.0,
        ),
        _DenominatorCase(None, DependencyPrediction(()), 0, 0, 0.0, 0.0),
    ],
    ids=("empty-empty", "empty-nonempty", "nonempty-empty", "failure-empty"),
)
def test_empty_and_failure_denominator_matrix(case: _DenominatorCase) -> None:
    scorer = DependencyScorer()
    scorer.update(case.prediction, case.gold, key=ItemId("matrix"))
    mode: DependencyMatchMode = "labeled-directed"
    metrics = (
        "micro-precision",
        "micro-recall",
        "micro-f-score",
        "macro-precision",
        "macro-recall",
        "macro-f-score",
    )

    for policy in ("ignore", "zero"):
        eligible = case.prediction is not None or policy == "zero"
        for metric in metrics:
            result = _row(scorer, mode, policy, metric).result
            assert result.value == (0.0 if eligible else None)
            assert result.n == int(eligible)
            assert result.components == {
                "matched": 0.0,
                "predicted": float(case.predicted if eligible else 0),
                "gold": float(case.expected_gold if eligible else 0),
            }

        exact_result = _row(scorer, mode, policy, "sentence-exact").result
        assert exact_result.value == (case.exact if eligible else None)
        assert exact_result.n == int(eligible)
        assert exact_result.components == {
            "exact": case.exact if eligible else 0.0,
            "sentences": float(eligible),
        }

        coverage_result = _row(scorer, mode, policy, "coverage").result
        assert coverage_result.value == case.coverage
        assert coverage_result.n == 1
        assert coverage_result.components == {
            "parsed": case.coverage,
            "sentences": 1.0,
        }


def test_updates_are_idempotent_order_independent_and_resettable() -> None:
    first = (
        _prediction(_arc(0, 1), _arc(2, 3)),
        _prediction(_arc(0, 1), _arc(2, 3)),
    )
    second = (_prediction(_arc(2, 3)), _prediction(_arc(2, 4)))
    forward = DependencyScorer()
    reverse = DependencyScorer()

    reordered_first = _prediction(*reversed(first[0].arcs))
    forward.update(reordered_first, first[1], key=ItemId("a"))
    forward.update(*second, key=ItemId("b"))
    forward.update(*first, key=ItemId("a"))
    reverse.update(*second, key=ItemId("b"))
    reverse.update(*first, key=ItemId("a"))

    assert forward.name == "dependency"
    assert forward.order_invariant
    assert forward.rows() == reverse.rows()
    assert reordered_first == first[0]
    assert hash(reordered_first) == hash(first[0])

    with pytest.raises(ValueError, match="already scored"):
        forward.update(second[0], first[1], key=ItemId("a"))

    forward.reset()
    assert forward.rows() == DependencyScorer().rows()


def test_self_arcs_are_removed_before_symmetric_validation_and_scoring() -> None:
    invalid_self = _arc(0, 0, label=(None, None, "application"))
    ordinary = _arc(1, 2)
    scorer = DependencyScorer()
    scorer.update(
        _prediction(invalid_self, ordinary),
        _prediction(invalid_self, ordinary),
        key=ItemId("self"),
    )

    row = _row(scorer, "labeled-directed", "ignore", "micro-f-score")
    assert row.result.value == 1.0
    assert row.result.components["predicted"] == 1.0
    assert row.result.components["gold"] == 1.0


def test_labeled_rows_require_categories_and_positive_slots() -> None:
    scorer = DependencyScorer()

    with pytest.raises(ValueError, match="no category"):
        scorer.update(
            _prediction(_arc(0, 1, label=(None, 1, "application"))),
            _prediction(),
            key=ItemId("category"),
        )

    with pytest.raises(ValueError, match="no argument slot"):
        scorer.update(
            _prediction(_arc(0, 1, label=("S", None, "application"))),
            _prediction(),
            key=ItemId("missing-slot"),
        )

    with pytest.raises(ValueError, match="nonpositive slot"):
        scorer.update(
            _prediction(_arc(0, 1, label=("S", 0, "application"))),
            _prediction(),
            key=ItemId("zero-slot"),
        )


def test_parg_adapter_reads_codec_annotations_directly() -> None:
    source = (
        '<s id="clean.room"> 2\n'
        "0 1 (S[dcl]\\NP)/NP 2 theme saw\n"
        "1 1 S\\NP 1 saw saw\n"
        "<\\s>\n"
    )
    codec = PargCodec()
    sentence = codec.iso.backward(codec.decode(source))[0]
    gold = parg_gold(sentence)

    assert gold == _prediction(
        _arc(
            1,
            0,
            forms=("saw", "theme"),
            label=("(S[dcl]\\NP)/NP", 2, None),
        )
    )


def test_parg_adapter_validates_features_and_drops_self_arcs() -> None:
    self_arc = PargDependency(0, 0, "N", 1, "word", "word")
    assert parg_gold(PargSentence("self", 0, (self_arc,))).arcs == ()

    with pytest.raises(ValueError, match="nonpositive slot"):
        parg_gold(
            PargSentence(
                "slot",
                1,
                (PargDependency(0, 1, "S", 0, "a", "b"),),
            )
        )

    with pytest.raises(ValueError, match="exceeds last index"):
        parg_gold(
            PargSentence(
                "index",
                1,
                (PargDependency(0, 2, "S", 1, "a", "b"),),
            )
        )


def test_raw_compatibility_filter_runs_before_category_rendering() -> None:
    omitted_category = ("S", "omit-markup")
    kept_category = ("S", "keep-markup")
    omitted = ProjectedDependency(
        LexicalHead(0, "skip", omitted_category),
        LexicalHead(1, "argument", ("NP", "raw")),
        2,
        "composition",
    )
    kept = ProjectedDependency(
        LexicalHead(2, "keep", kept_category),
        LexicalHead(3, "argument", ("NP", "raw")),
        1,
        "application",
    )
    compatibility = DependencyCompatibilityFilter(
        (
            CompatibilityPattern(
                predicate_form="skip",
                argument_form="argument",
                raw_category=omitted_category,
                slot=2,
                rule_id="composition",
            ),
        ),
        "independent clean-room fixture",
    )
    rendered: list[tuple[str, str]] = []

    def category_text(category: tuple[str, str]) -> str:
        rendered.append(category)
        if category == omitted_category:
            msg = "excluded raw category was rendered"
            raise AssertionError(msg)
        return category[0]

    prediction = projected_prediction((omitted, kept), compatibility, category_text)

    assert rendered == [kept_category]
    assert prediction == _prediction(
        _arc(
            2,
            3,
            forms=("keep", "argument"),
            label=("S", 1, "application"),
        )
    )


def test_compatibility_patterns_are_conjunctive_and_caller_owned() -> None:
    dependency = ProjectedDependency(
        LexicalHead(0, "head", "S[feature]"),
        LexicalHead(1, "dependent", "NP"),
        2,
        "composition",
    )
    matching = CompatibilityPattern("head", "dependent", "S[feature]", 2, "composition")
    mismatch = CompatibilityPattern(raw_category="S")
    source = [matching]
    compatibility = DependencyCompatibilityFilter(source, "fixture-v1")
    source.clear()

    assert matching.matches(dependency)
    assert not mismatch.matches(dependency)
    assert not compatibility.keeps(dependency)
    assert compatibility.patterns == (matching,)
    assert compatibility.provenance == "fixture-v1"


def test_projected_adapter_preserves_position_form_slot_and_rule_provenance() -> None:
    raw = ProjectedDependency(
        LexicalHead(4, "head", "S[feature]"),
        LexicalHead(7, "dependent", "NP"),
        3,
        "forward-application",
    )
    compatibility = DependencyCompatibilityFilter[str]((), "no exclusions")
    prediction = projected_prediction((raw,), compatibility, lambda _category: "S")
    arc = prediction.arcs[0]

    assert arc.predicate_position == 4
    assert arc.predicate_form == "head"
    assert arc.argument_position == 7
    assert arc.argument_form == "dependent"
    assert arc.category == "S"
    assert arc.slot == 3
    assert arc.rule_id == "forward-application"


def test_projected_adapter_drops_self_arcs_without_rendering() -> None:
    raw = ProjectedDependency(
        LexicalHead(1, "head", "S"),
        LexicalHead(1, "dependent", "NP"),
        1,
        "application",
    )
    compatibility = DependencyCompatibilityFilter[str]((), "no exclusions")

    def unexpected(_category: str) -> str:
        msg = "self arc was rendered"
        raise AssertionError(msg)

    assert projected_prediction((raw,), compatibility, unexpected).arcs == ()


def test_public_values_are_frozen_hashable_copyable_and_pickleable() -> None:
    token = DependencyToken(0, "head")
    arc = _arc(0, 1)
    prediction = _prediction(arc)
    pattern = CompatibilityPattern[str](raw_category="S")
    compatibility = DependencyCompatibilityFilter((pattern,), "fixture")
    scorer = DependencyScorer()
    scorer.update(prediction, prediction, key=ItemId("row"))
    row = scorer.rows()[0]
    values = (token, arc, prediction, pattern, compatibility, row)

    for value in values:
        assert copy(value) is value
        assert deepcopy(value) is value
        assert loads(dumps(value)) == value
        assert hash(loads(dumps(value))) == hash(value)
        field = "_changed"
        with pytest.raises(AttributeError, match="immutable"):
            setattr(value, field, True)
        with pytest.raises(AttributeError, match="immutable"):
            delattr(value, field)


def test_public_value_validation_rejects_malformed_inputs() -> None:
    with pytest.raises(ValueError, match="negative"):
        DependencyToken(-1, "word")
    with pytest.raises(ValueError, match="form is empty"):
        DependencyToken(0, "")
    with pytest.raises(TypeError, match="predicate is not a token"):
        DependencyArc(
            cast("DependencyToken", "head"),
            DependencyToken(1, "dependent"),
            "S",
            1,
            None,
        )
    boolean = True
    with pytest.raises(TypeError, match="slot is not an integer"):
        _arc(0, 1, label=("S", cast("int", boolean), None))
    with pytest.raises(TypeError, match="arc 0 is not one"):
        DependencyPrediction(cast("tuple[DependencyArc, ...]", ("arc",)))
    with pytest.raises(ValueError, match="no constraints"):
        CompatibilityPattern[str]()
    with pytest.raises(ValueError, match="not positive"):
        CompatibilityPattern[str](slot=0)
    with pytest.raises(TypeError, match="pattern 0 is not one"):
        DependencyCompatibilityFilter(
            cast("tuple[CompatibilityPattern[str], ...]", ("pattern",)),
            "fixture",
        )
    with pytest.raises(ValueError, match=r"provenance.*empty"):
        DependencyCompatibilityFilter[str]((), "")


def test_adapter_and_scorer_validation_rejects_wrong_value_kinds() -> None:
    compatibility = DependencyCompatibilityFilter[str]((), "fixture")

    with pytest.raises(TypeError, match="PARG sentence is not one"):
        parg_gold(cast("PargSentence", "sentence"))
    with pytest.raises(TypeError, match="projected dependency 0 is not one"):
        projected_prediction(
            cast("tuple[ProjectedDependency[str], ...]", ("dependency",)),
            compatibility,
            str,
        )
    with pytest.raises(TypeError, match="filter is not one"):
        projected_prediction(
            (),
            cast("DependencyCompatibilityFilter[str]", "filter"),
            str,
        )
    with pytest.raises(TypeError, match="not callable"):
        projected_prediction(
            (), compatibility, cast("Callable[[str], str]", "renderer")
        )
    with pytest.raises(TypeError, match="prediction is not one"):
        DependencyScorer().update(
            cast("DependencyPrediction", "prediction"),
            DependencyPrediction(()),
            key=ItemId("bad"),
        )
