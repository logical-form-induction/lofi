from copy import copy, deepcopy
from pickle import dumps, loads
from typing import TYPE_CHECKING, cast

import pytest
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutSubject,
    packaged_held_out_pass,
)

from lofi._types import ItemId
from lofi.data import (
    Derivation,
    DerivationNode,
    LabelBlocker,
    ReachabilityReport,
    RuleBlocker,
    TokenSpan,
    UnrecoverableStep,
    reachability_report,
)
from lofi.eval import (
    BLOCKER_KINDS,
    BlockerCell,
    ConventionVector,
    HeldOutEvaluationContext,
    InconsistentReachabilityError,
    MetricResult,
    ReachabilityRow,
    blocker_cell,
    reachability_table,
)
from lofi.eval._reachability import PLUGIN, ReachabilityEvaluator
from lofi.registry import plugin_for
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.reachability import (
    REACHABILITY_ROWS,
    ReachabilityEvaluatorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.eval import BlockerKind


class Labels:
    __slots__ = ("_labels",)

    def __init__(self, *labels: str) -> None:
        self._labels = frozenset(labels)

    def contains(self, label: str, /) -> bool:
        return label in self._labels


class Steps:
    __slots__ = ("_steps",)

    def __init__(self, *steps: tuple[tuple[str, ...], str]) -> None:
        self._steps = frozenset(steps)

    def recover(self, inputs: Sequence[str], output: str, /) -> tuple[str, ...]:
        return ("step",) if (tuple(inputs), output) in self._steps else ()


ALL_LABELS = Labels("a", "b", "c", "d", "e", "x")


def binary(
    output: str, left: str, right: str, *, rule: str | None = None
) -> Derivation[str]:
    return Derivation(
        0,
        {
            0: DerivationNode(TokenSpan(0, 2), output, rule=rule, children=(1, 2)),
            1: DerivationNode(TokenSpan(0, 1), left),
            2: DerivationNode(TokenSpan(1, 2), right),
        },
    )


def lexical(label: str) -> Derivation[str]:
    return Derivation(0, {0: DerivationNode(TokenSpan(0, 1), label)})


def two_failures() -> Derivation[str]:
    """Return a tree whose root and inner step both fail under ``Steps()``."""
    return Derivation(
        0,
        {
            0: DerivationNode(TokenSpan(0, 3), "c", children=(1, 4)),
            1: DerivationNode(TokenSpan(0, 2), "b", children=(2, 3)),
            2: DerivationNode(TokenSpan(0, 1), "a"),
            3: DerivationNode(TokenSpan(1, 2), "a"),
            4: DerivationNode(TokenSpan(2, 3), "d"),
        },
    )


def row(
    report: ReachabilityReport[str],
    *,
    inventory: str = "full",
    corpus: str = "bank",
    language: str = "en",
    training_size: int = 7,
) -> ReachabilityRow[str]:
    return ReachabilityRow(
        report,
        inventory=inventory,
        corpus=corpus,
        language=language,
        training_size=training_size,
    )


def test_a_tree_with_several_failed_steps_is_counted_once() -> None:
    report = reachability_report(ALL_LABELS, Steps(), (two_failures(),))
    counted = row(report)

    assert report.checked_steps == 2
    assert report.reachable_steps == 0
    assert counted.unreachable_derivations == 1
    assert counted.histogram == (
        (BlockerCell("construction", inputs=("b", "d"), output="c"), 1),
    )
    assert sum(count for _, count in counted.histogram) == 1


def test_histogram_cells_sum_to_the_unreachable_derivation_count() -> None:
    derivations = (
        two_failures(),
        binary("c", "a", "b"),
        binary("c", "a", "x"),
        binary("e", "a", "b", rule="fa"),
        two_failures(),
    )
    steps = Steps((("a", "b"), "c"), (("a", "a"), "b"))
    report = reachability_report(Labels("a", "b", "c", "d", "e"), steps, derivations)
    counted = row(report)

    assert counted.checked_derivations == 5
    assert counted.reachable_derivations == 1
    assert counted.unreachable_derivations == 4
    assert sum(count for _, count in counted.histogram) == 4
    assert sum(counted.blocked(kind) for kind in BLOCKER_KINDS) == 4
    assert counted.blocked("label") == 1
    assert counted.blocked("rule") == 1
    assert counted.blocked("construction") == 2


def test_a_missing_label_takes_precedence_over_a_failed_rule() -> None:
    report = reachability_report(Labels("a", "b", "d"), Steps(), (two_failures(),))

    assert row(report).histogram == ((BlockerCell("label", label="c"), 1),)


def test_a_recorded_rule_name_is_the_cell_and_an_unnamed_step_is_its_shape() -> None:
    derivations = (
        binary("c", "a", "b", rule="fa"),
        binary("c", "b", "a", rule="fa"),
        binary("c", "a", "b"),
    )
    report = reachability_report(ALL_LABELS, Steps(), derivations)
    counted = row(report)

    assert counted.histogram == (
        (BlockerCell("rule", rule="fa"), 2),
        (BlockerCell("construction", inputs=("a", "b"), output="c"), 1),
    )
    assert counted.blocked("rule") == 2
    assert counted.blocked("construction") == 1
    assert counted.blocked("label") == 0


def test_blocker_cells_drop_the_node_identifier() -> None:
    assert blocker_cell(LabelBlocker(3, "x")) == blocker_cell(LabelBlocker(9, "x"))
    assert blocker_cell(RuleBlocker(1, ("a",), "b")) == blocker_cell(
        RuleBlocker(5, ("a",), "b")
    )
    assert blocker_cell(RuleBlocker(1, ("a",), "b", "r")) == BlockerCell(
        "rule", rule="r"
    )
    assert blocker_cell(RuleBlocker(1, ("a",), "b")) != blocker_cell(
        RuleBlocker(1, ("b",), "a")
    )


def test_the_histogram_is_most_frequent_first_and_ties_keep_first_observation() -> None:
    derivations = (
        binary("c", "a", "e"),
        binary("c", "a", "d"),
        binary("c", "a", "b"),
        binary("c", "a", "b"),
        binary("c", "a", "d"),
        binary("c", "a", "b"),
        binary("c", "b", "e"),
        binary("c", "b", "b"),
    )
    report = reachability_report(ALL_LABELS, Steps(), derivations)

    assert [(cell.inputs, count) for cell, count in row(report).histogram] == [
        (("a", "b"), 3),
        (("a", "d"), 2),
        (("a", "e"), 1),
        (("b", "e"), 1),
        (("b", "b"), 1),
    ]


def test_counts_and_rates_are_read_from_the_report() -> None:
    derivations = (binary("c", "a", "b"), two_failures())
    report = reachability_report(ALL_LABELS, Steps((("a", "b"), "c")), derivations)
    counted = row(report)

    assert counted.report is report
    assert counted.checked_steps == 3
    assert counted.reachable_steps == 1
    assert counted.checked_derivations == 2
    assert counted.reachable_derivations == 1
    assert counted.unreachable_derivations == 1
    assert counted.step_reachability == pytest.approx(1 / 3)
    assert counted.derivation_reachability == pytest.approx(1 / 2)


def test_an_empty_report_has_absent_rates_and_an_empty_histogram() -> None:
    counted = row(reachability_report(ALL_LABELS, Steps(), ()))

    assert counted.step_reachability is None
    assert counted.derivation_reachability is None
    assert counted.histogram == ()
    assert counted.unreachable_derivations == 0


def test_a_row_is_keyed_by_inventory_corpus_and_language() -> None:
    counted = row(
        reachability_report(ALL_LABELS, Steps(), ()),
        inventory="application",
        corpus="pmb",
        language="nl",
        training_size=12,
    )

    assert counted.key == ("application", "pmb", "nl")
    assert (counted.inventory, counted.corpus, counted.language) == (
        "application",
        "pmb",
        "nl",
    )
    assert counted.training_size == 12


def test_the_specification_names_every_count_and_cell() -> None:
    derivations = (binary("c", "a", "b"), binary("c", "a", "d"), binary("x", "a", "b"))
    report = reachability_report(
        Labels("a", "b", "c", "d"), Steps((("a", "b"), "c")), derivations
    )

    assert row(report).to_spec(str.upper) == {
        "inventory": "full",
        "corpus": "bank",
        "language": "en",
        "training_size": 7,
        "checked_steps": 3,
        "reachable_steps": 1,
        "checked_derivations": 3,
        "reachable_derivations": 1,
        "step_reachability": pytest.approx(1 / 3),
        "derivation_reachability": pytest.approx(1 / 3),
        "blocked": {"label": 1, "rule": 0, "construction": 1},
        "histogram": [
            {
                "cell": {"kind": "construction", "inputs": ["A", "D"], "output": "C"},
                "derivations": 1,
            },
            {"cell": {"kind": "label", "label": "X"}, "derivations": 1},
        ],
    }
    assert BlockerCell("rule", rule="fa").to_spec(str.upper) == {
        "kind": "rule",
        "rule": "fa",
    }


def test_an_unknown_blocker_kind_is_refused() -> None:
    counted = row(reachability_report(ALL_LABELS, Steps(), ()))

    with pytest.raises(ValueError, match="unknown blocker kind"):
        counted.blocked(cast("BlockerKind", "lexical"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("inventory", "", "inventory name"),
        ("corpus", "", "corpus name"),
        ("language", "", "language"),
    ],
)
def test_empty_names_are_refused(field: str, value: str, message: str) -> None:
    names = {"inventory": "full", "corpus": "bank", "language": "en", field: value}

    with pytest.raises(ValueError, match=message):
        ReachabilityRow(
            reachability_report(ALL_LABELS, Steps(), ()),
            inventory=names["inventory"],
            corpus=names["corpus"],
            language=names["language"],
            training_size=0,
        )


def test_a_negative_training_size_and_a_foreign_report_are_refused() -> None:
    report = reachability_report(ALL_LABELS, Steps(), ())

    with pytest.raises(ValueError, match="training-set size"):
        row(report, training_size=-1)

    with pytest.raises(TypeError, match="reachability report"):
        ReachabilityRow(
            cast("ReachabilityReport[str]", (1, 1)),
            inventory="full",
            corpus="bank",
            language="en",
            training_size=0,
        )


def test_rows_are_frozen_hashable_copyable_and_pickleable() -> None:
    report = reachability_report(ALL_LABELS, Steps(), (binary("c", "a", "b"),))
    counted = row(report)

    assert counted == row(report)
    assert hash(counted) == hash(row(report))
    assert counted != row(report, inventory="application")
    assert counted != row(report, corpus="pmb")
    assert counted != row(report, language="de")
    assert counted != row(report, training_size=8)
    assert counted != row(reachability_report(ALL_LABELS, Steps(), ()))
    assert copy(counted) is counted
    assert deepcopy(counted) is counted
    assert loads(dumps(counted)) == counted
    assert hash(loads(dumps(counted))) == hash(counted)
    assert repr(counted) == (
        f"ReachabilityRow({report!r}, inventory='full', corpus='bank', "
        "language='en', training_size=7)"
    )

    for field in ("_inventory", "_report"):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(counted, field, "other")

        with pytest.raises(AttributeError, match="immutable"):
            delattr(counted, field)


def full_and_reduced() -> tuple[ReachabilityReport[str], ReachabilityReport[str]]:
    derivations = (binary("c", "a", "b"), binary("c", "a", "d"))
    full = reachability_report(
        ALL_LABELS, Steps((("a", "b"), "c"), (("a", "d"), "c")), derivations
    )
    reduced = reachability_report(ALL_LABELS, Steps((("a", "b"), "c")), derivations)

    return full, reduced


def test_a_table_keeps_rows_in_order_across_inventories_and_languages() -> None:
    full, reduced = full_and_reduced()
    rows = (
        row(full, inventory="full"),
        row(reduced, inventory="reduced"),
        row(reduced, inventory="full", language="de", training_size=3),
    )

    assert reachability_table(iter(rows)) == rows


def test_a_table_refuses_two_rows_under_one_key() -> None:
    full, reduced = full_and_reduced()

    with pytest.raises(InconsistentReachabilityError, match="two reachability rows"):
        reachability_table((row(full), row(reduced)))


@pytest.mark.parametrize(
    "disagreeing",
    ["training_size", "derivations", "steps"],
)
def test_a_table_refuses_inventories_compared_over_different_material(
    disagreeing: str,
) -> None:
    full, _ = full_and_reduced()
    derivations = (binary("c", "a", "b"), binary("c", "a", "d"))
    other = {
        "training_size": full,
        "derivations": reachability_report(
            ALL_LABELS,
            Steps((("a", "b"), "c"), (("a", "d"), "c")),
            (*derivations, lexical("a")),
        ),
        "steps": ReachabilityReport(
            3,
            1,
            2,
            0,
            (UnrecoverableStep(("a",), "b", 2),),
            (
                RuleBlocker(0, ("a",), "b"),
                RuleBlocker(0, ("a",), "b"),
            ),
        ),
    }[disagreeing]
    training = 8 if disagreeing == "training_size" else 7
    differs = {
        "training_size": training != row(full).training_size,
        "derivations": other.checked_derivations != full.checked_derivations,
        "steps": other.checked_steps != full.checked_steps,
    }

    assert differs == {name: name == disagreeing for name in differs}

    with pytest.raises(InconsistentReachabilityError, match="disagree"):
        reachability_table(
            (row(full), row(other, inventory="reduced", training_size=training))
        )


CONVENTIONS = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="corpus",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)


def evaluator(context: HeldOutEvaluationContext | None = None) -> ReachabilityEvaluator:
    """Return reachability rows over the packaged slash grammar's inventory."""
    return ReachabilityEvaluator(
        packaged_held_out_pass().context if context is None else context,
        inventory="slash",
        corpus="synthetic",
        language="en",
        conventions=CONVENTIONS,
    )


class TestReachabilityObeysHeldOutEvaluatorLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(self) -> HeldOutSubject[None, Derivation[str]]:
        return HeldOutSubject(evaluator(), packaged_held_out_pass())


def scored_rows() -> dict[str, MetricResult]:
    """Return every row of the evaluator over the packaged gold derivations."""
    built = evaluator()
    held_out = packaged_held_out_pass()
    rows = {}
    for metric in built.metrics:
        for item, derivation in held_out.gold:
            metric.update(None, built.annotate(item, derivation), key=item.item_id)
        rows[metric.name] = metric.compute()
    return rows


def test_the_grammars_own_derivations_are_all_reachable() -> None:
    rows = scored_rows()
    held_out = packaged_held_out_pass()
    derivations = rows["reachability-derivations"]

    assert tuple(rows) == REACHABILITY_ROWS
    assert rows["reachability-steps"].value == 1.0
    assert derivations.value == 1.0
    assert derivations.components["checked-derivations"] == len(held_out.gold)
    assert derivations.components["training-size"] == held_out.context.training_items
    assert all(
        rows[f"reachability-blocked-{kind}"].value == 0.0 for kind in BLOCKER_KINDS
    )


def test_a_derivation_outside_the_inventory_is_blocked_by_its_label() -> None:
    built = evaluator()
    item, derivation = packaged_held_out_pass().gold[0]
    foreign = lexical("((")
    blocked = {metric.name: metric for metric in built.metrics}
    label = blocked["reachability-blocked-label"]
    label.update(None, derivation, key=item.item_id)
    label.update(None, foreign, key=ItemId("foreign"))
    result = label.compute()

    assert result.value == 1.0
    assert result.components == {"label:((": 1.0}
    assert label.per_item()[ItemId("foreign")].value == 1.0


def test_the_declared_names_are_report_conventions() -> None:
    declaration = EvaluatorSpec.model_validate(
        {
            "kind": "reachability",
            "interval": {"replicates": 20, "level": 0.95},
            "inventory": "slash",
            "corpus": "synthetic",
            "language": "en",
            "conventions": {
                "punctuation": "keep",
                "unary_chains": "keep",
                "trivial_spans": "keep",
                "minimum_length": 0,
                "aggregation": "corpus",
                "coverage": "zero",
                "self_arcs": "allow",
                "gold_derivation": "canonical",
            },
        }
    )
    assert type(declaration) is ReachabilityEvaluatorSpec
    built = PLUGIN.construct(declaration, packaged_held_out_pass().context)

    assert plugin_for(declaration) is PLUGIN
    assert isinstance(built, ReachabilityEvaluator)
    assert tuple(metric.name for metric in built.metrics) == declaration.metrics()
    assert built.conventions.factors["reachability_inventory"] == "slash"
    assert built.conventions.factors["reachability_language"] == "en"


def test_a_context_without_a_training_partition_is_refused() -> None:
    context = packaged_held_out_pass().context
    bare = HeldOutEvaluationContext(
        context.grammar,
        interpretation=None,
        forms={},
        coherence={},
        target_sets={},
        lexical_training={},
        parse=context.parse,
        training_items=None,
    )

    with pytest.raises(ValueError, match="training partition"):
        evaluator(bare)
