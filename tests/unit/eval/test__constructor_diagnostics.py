from typing import Final

import pytest
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutSubject,
    packaged_held_out_pass,
)

from lofi._types import ItemId
from lofi.eval import BeamNormalization, ConventionVector
from lofi.eval._constructor_diagnostics import (
    ConstructorDiagnosticEvaluator,
    ConstructorDiagnosticGold,
    ConstructorDiagnosticPrediction,
    _constructor_rows,
    _ConstructorMetric,
    _ReportContext,
)
from lofi.spec.evaluators.constructor_diagnostics import ConstructorViewSettings
from lofi.terms import (
    DeclaredConstructors,
    Operator,
    SExpSyntax,
    Signature,
    TermStore,
)

CONVENTIONS: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=0,
    aggregation="sentence",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)


def evaluator() -> ConstructorDiagnosticEvaluator:
    """Return diagnostics over the packaged slash-pair grammar."""
    grammar = packaged_held_out_pass().grammar
    return ConstructorDiagnosticEvaluator(
        grammar,
        readout="category-terms",
        view=ConstructorViewSettings(
            "NP",
            (("/", ("result", "argument")), ("\\", ("result", "argument"))),
        ),
        normalization="none",
        widths=(1, 2),
        depths=(1, 2),
        collapse_margin=0.5,
        conventions=CONVENTIONS,
    )


class TestConstructorDiagnosticsObeyHeldOutEvaluatorLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(
        self,
    ) -> HeldOutSubject[ConstructorDiagnosticPrediction, ConstructorDiagnosticGold]:
        return HeldOutSubject(evaluator(), packaged_held_out_pass())


def test_an_arrow_product_view_emits_every_declared_row() -> None:
    signature = Signature(
        "arrow-product-diagnostics",
        ("Cat",),
        (
            Operator("n", "Cat"),
            Operator("np", "Cat"),
            Operator("s", "Cat"),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("product", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )
    store = TermStore(signature)
    syntax = SExpSyntax(signature)
    constructors = DeclaredConstructors(
        store,
        {
            "arrow": ("result", "argument"),
            "product": ("left-factor", "right-factor"),
        },
    )
    n = store.op(signature.by_name("n"))
    np = store.op(signature.by_name("np"))
    s = store.op(signature.by_name("s"))
    gold_arrow = constructors.make("arrow", (s, np))
    gold_product = constructors.make("product", (np, s))
    prediction = ConstructorDiagnosticPrediction(
        ("f", "pair"),
        (
            ((constructors.make("arrow", (n, np)), 0.0), (gold_arrow, -1.0)),
            (
                (constructors.make("product", (np, n)), 0.0),
                (gold_product, -1.0),
            ),
        ),
    )
    gold = ConstructorDiagnosticGold(("f", "pair"), (gold_arrow, gold_product))
    context = _ReportContext(
        store,
        syntax,
        constructors,
        n,
        ("n", "np", "s"),
        ("result", "argument", "left-factor", "right-factor"),
    )
    names = (
        "beam-accuracy@2",
        "typeless-accuracy@2",
        "truncated-accuracy@1",
        "mean-reciprocal-rank",
        "primitive-induced-count",
        "primitive-gold-count",
        "primitive-inventory-exact",
        "primitive-f1",
        "constructor-induced-count",
        "constructor-gold-count",
        "constructor-inventory-exact",
        "constructor-role-information",
        "lexical-type-f1",
        "lexical-token-accuracy",
        "depth-gap",
        "depth-collapse",
    )
    results = {}

    rows = dict(_constructor_rows((2,), (1,), 0.5, context, BeamNormalization.NONE))

    assert tuple(rows) == names

    for name, reading in rows.items():
        metric = _ConstructorMetric(name, reading, CONVENTIONS)
        metric.update(prediction, gold, key=ItemId("a-times"))
        results[name] = metric.compute().value

    assert set(results) == set(names)
    assert all(value is not None for value in results.values())
    assert results["beam-accuracy@2"] == 1.0
    assert results["typeless-accuracy@2"] == 1.0
    assert results["truncated-accuracy@1"] == 1.0
    assert results["primitive-induced-count"] == 2.0
    assert results["primitive-gold-count"] == 2.0
    assert results["primitive-inventory-exact"] == 0.0
    primitive_f1 = results["primitive-f1"]
    assert primitive_f1 is not None
    assert primitive_f1 < 1.0
    assert results["constructor-induced-count"] == 2.0
    assert results["constructor-gold-count"] == 2.0
    assert results["constructor-inventory-exact"] == 1.0
    assert results["lexical-token-accuracy"] == 0.0
    assert results["depth-gap"] == 0.0
    assert results["depth-collapse"] == 0.0


def shallow_context() -> tuple[_ReportContext, DeclaredConstructors, TermStore]:
    """Return a one-constructor view over three primitive categories."""
    signature = Signature(
        "shallow-collapse",
        ("Cat",),
        (
            Operator("n", "Cat"),
            Operator("np", "Cat"),
            Operator("s", "Cat"),
            Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )
    store = TermStore(signature)
    constructors = DeclaredConstructors(store, {"arrow": ("result", "argument")})
    context = _ReportContext(
        store,
        SExpSyntax(signature),
        constructors,
        store.op(signature.by_name("n")),
        ("n", "np", "s"),
        ("result", "argument"),
    )
    return context, constructors, store


@pytest.mark.parametrize(("margin", "collapsed"), [(0.25, 1.0), (0.5, 0.0)])
def test_a_shallow_model_drives_the_depth_gap_negative_and_warns_of_collapse(
    margin: float, collapsed: float
) -> None:
    context, constructors, store = shallow_context()
    signature = store.signature
    np = store.op(signature.by_name("np"))
    s = store.op(signature.by_name("s"))
    gold_arrow = constructors.make("arrow", (s, np))
    prediction = ConstructorDiagnosticPrediction(
        ("runs", "dog"), (((np, 0.0), (gold_arrow, -2.0)), ((np, 0.0),))
    )
    gold = ConstructorDiagnosticGold(("runs", "dog"), (gold_arrow, np))
    rows = dict(_constructor_rows((1,), (1,), margin, context, BeamNormalization.NONE))
    results = {}

    for name in ("depth-gap", "depth-collapse"):
        metric = _ConstructorMetric(name, rows[name], CONVENTIONS)
        metric.update(prediction, gold, key=ItemId("shallow"))
        results[name] = metric.compute()

    assert results["depth-gap"].value == -0.5
    assert results["depth-gap"].components == {
        "predicted-depth": 1.0,
        "gold-depth": 1.5,
    }
    assert results["depth-collapse"].value == collapsed


def test_the_depth_rows_are_absent_without_a_scored_candidate() -> None:
    context, _, store = shallow_context()
    np = store.op(store.signature.by_name("np"))
    rows = dict(_constructor_rows((1,), (1,), 0.0, context, BeamNormalization.NONE))
    metric = _ConstructorMetric("depth-gap", rows["depth-gap"], CONVENTIONS)
    metric.update(
        ConstructorDiagnosticPrediction(("dog",), ((),)),
        ConstructorDiagnosticGold(("dog",), (np,)),
        key=ItemId("empty"),
    )

    assert metric.compute().value is None


def test_reconstruction_rows_retain_per_primitive_and_per_word_values() -> None:
    diagnostics = evaluator()
    held_out = packaged_held_out_pass()
    predictions = {}
    for outputs in held_out.published:
        predictions.update(diagnostics.predict(outputs))
    gold = {
        item.item_id: diagnostics.annotate(item, derivation)
        for item, derivation in held_out.gold
    }
    rows = {}
    for metric in diagnostics.metrics:
        for key, prediction in predictions.items():
            metric.update(prediction, gold[key], key=key)
        rows[metric.name] = metric.compute()

    assert any(
        name.startswith("primitive:") for name in rows["primitive-f1"].components
    )
    assert "primitive:NP" in rows["primitive-f1"].components
    assert "primitive:#const[Cat]" not in rows["primitive-f1"].components
    assert any(name.startswith("word:") for name in rows["lexical-type-f1"].components)


def test_constructor_view_refuses_an_undeclared_or_wrong_arity_constructor() -> None:
    grammar = packaged_held_out_pass().grammar

    with pytest.raises((KeyError, ValueError)):
        ConstructorDiagnosticEvaluator(
            grammar,
            readout="category-terms",
            view=ConstructorViewSettings(
                "NP",
                (("/", ("only-one-role",)), ("product", ("left", "right"))),
            ),
            normalization="none",
            widths=(1,),
            depths=(1,),
            collapse_margin=0.5,
            conventions=CONVENTIONS,
        )


@pytest.mark.parametrize("normalization", ["none", "depth", "nodes", "terminals"])
def test_every_beam_normalization_is_a_declared_report_factor(
    normalization: str,
) -> None:
    grammar = packaged_held_out_pass().grammar
    diagnostics = ConstructorDiagnosticEvaluator(
        grammar,
        readout="category-terms",
        view=ConstructorViewSettings(
            "NP",
            (("/", ("result", "argument")), ("\\", ("result", "argument"))),
        ),
        normalization=normalization,
        widths=(1, 2),
        depths=(1, 2),
        collapse_margin=0.25,
        conventions=CONVENTIONS,
    )

    assert diagnostics.conventions.factors["collapse_margin"] == "0.25"
    assert diagnostics.conventions.factors["beam_normalization"] == normalization
    assert diagnostics.conventions.factors["beam_widths"] == "[1,2]"
    assert diagnostics.conventions.factors["projection_depths"] == "[1,2]"
