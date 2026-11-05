from typing import TYPE_CHECKING, Final

import pytest
from tests.acceptance.driver import composed
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutPass,
    HeldOutSubject,
    held_out_run,
)

from lofi._types import ItemId
from lofi.eval import ConventionVector, HeldOutEvaluationContext
from lofi.eval._logical_form_evaluator import (
    PLUGIN,
    LogicalFormEvaluator,
    stable_logical_form_alignments,
)
from lofi.nn.loss._values import ObjectiveOutputs
from lofi.registry import plugin_for
from lofi.spec.evaluators.alignment import AlignmentKind, AlignmentSettings
from lofi.spec.evaluators.logical_form import LogicalFormEvaluatorSpec
from lofi.terms import Nominal

if TYPE_CHECKING:
    from lofi.data import Derivation, Item

CONVENTIONS: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=1,
    aggregation="corpus",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
ALIGNMENT: Final = AlignmentSettings(
    kind="identity",
    tie_break=None,
    fit_on="dev",
    score_on="test",
    floor_replicates=20,
    floor_level=0.95,
    stability_replicates=20,
    stability_level=0.95,
)


@pytest.fixture(scope="module")
def interpreted() -> HeldOutPass:
    return held_out_run(composed("meaning-evaluator.yaml"))


def evaluator(
    interpreted: HeldOutPass,
    *,
    permutation_limit: int = 5_040,
) -> LogicalFormEvaluator:
    built = LogicalFormEvaluator(
        interpreted.context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
        restarts=4,
        permutation_limit=permutation_limit,
    )
    gold = {
        item.item_id: built.annotate(item, derivation)
        for item, derivation in interpreted.gold
    }
    built.fit_alignment(gold, gold)
    return built


class TestTheLogicalFormEvaluatorObeysTheLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(
        self, interpreted: HeldOutPass
    ) -> HeldOutSubject[Nominal, Nominal]:
        return HeldOutSubject(evaluator(interpreted), interpreted)


def test_gold_forms_are_read_from_the_corpus_boundary(
    interpreted: HeldOutPass,
) -> None:
    built = evaluator(interpreted)

    for item, derivation in interpreted.gold:
        gold = built.annotate(item, derivation)

        assert isinstance(gold, Nominal)
        assert item.item_id in interpreted.context.forms


def test_predictions_select_one_complete_root_form_per_item(
    interpreted: HeldOutPass,
) -> None:
    built = evaluator(interpreted)

    for outputs in interpreted.published:
        predictions = built.predict(outputs)

        assert set(predictions) == set(outputs.batch.item_ids)
        assert all(isinstance(form, Nominal) for form in predictions.values())


def test_a_missing_complete_term_readout_is_refused(
    interpreted: HeldOutPass,
) -> None:
    built = evaluator(interpreted)
    published = interpreted.published[0]
    bare = ObjectiveOutputs(
        spans=published.spans,
        chart=published.chart,
        batch=published.batch,
    )

    with pytest.raises(ValueError, match="decoded-forms"):
        built.predict(bare)


def test_an_uninterpreted_context_is_refused(interpreted: HeldOutPass) -> None:
    with pytest.raises(ValueError, match="declared interpretation"):
        LogicalFormEvaluator(
            HeldOutEvaluationContext(
                interpreted.grammar,
                interpretation=None,
                forms={},
                coherence={},
                target_sets={},
                lexical_training={},
                parse=interpreted.context.parse,
                training_items=interpreted.context.training_items,
            ),
            readout="decoded-forms",
            measure="exact",
            conventions=CONVENTIONS,
            alignment=ALIGNMENT,
            restarts=4,
            permutation_limit=5_040,
        )


def test_a_gold_item_absent_from_the_form_mapping_is_refused(
    interpreted: HeldOutPass,
) -> None:
    context = HeldOutEvaluationContext(
        interpreted.grammar,
        interpretation=interpreted.context.interpretation,
        forms={next(iter(interpreted.context.forms)): "john"},
        coherence={},
        target_sets={},
        lexical_training={},
        parse=interpreted.context.parse,
        training_items=interpreted.context.training_items,
    )
    built = LogicalFormEvaluator(
        context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
        restarts=4,
        permutation_limit=5_040,
    )
    missing = next(
        (pair for pair in interpreted.gold if pair[0].item_id not in context.forms),
        None,
    )

    if missing is None:
        pytest.skip("the held-out split contains only the retained item")

    item: Item
    derivation: Derivation[str]
    item, derivation = missing

    with pytest.raises(ValueError, match="no gold logical form"):
        built.annotate(item, derivation)


def test_the_plugin_builds_the_selected_metric(interpreted: HeldOutPass) -> None:
    declaration = LogicalFormEvaluatorSpec.model_validate(
        {
            "kind": "logical-form",
            "interval": {"replicates": 20, "level": 0.95},
            "readout": "decoded-forms",
            "measure": "exact",
            "conventions": dict(CONVENTIONS.factors),
            "alignment": {
                "kind": "identity",
                "fit_on": "dev",
                "score_on": "test",
                "random_floor": {"replicates": 20, "level": 0.95},
                "stability": {"replicates": 20, "level": 0.95},
            },
        }
    )
    built = PLUGIN.construct(declaration, interpreted.context)
    gold = {
        item.item_id: built.annotate(item, derivation)
        for item, derivation in interpreted.gold
    }
    built.fit_alignment(gold, gold)

    assert tuple(metric.name for metric in built.metrics) == ("logical-form-exact",)
    assert built.alignment == ALIGNMENT
    assert built.metrics[0].conventions == built.conventions
    assert built.conventions.factors["alignment"] == "identity"
    assert built.conventions.factors["smatch_restarts"] == 4
    assert declaration.metrics()[:1] == tuple(metric.name for metric in built.metrics)
    assert plugin_for(declaration) is PLUGIN


def test_alignment_scoring_reports_the_fiber_controls(
    interpreted: HeldOutPass,
) -> None:
    built = evaluator(interpreted, permutation_limit=2)
    gold = {
        item.item_id: built.annotate(item, derivation)
        for item, derivation in interpreted.gold
    }

    scored = built.score_alignment(gold, gold, seed=19)
    assert tuple(scored.rows) == (
        "identity-accuracy",
        "random-relabeling-floor",
        "fiber-relative-recovery",
        "fixed-vocabulary-accuracy",
        "fiber-floor",
        "fiber-size",
    )
    assert scored.rows["fiber-relative-recovery"].value == 1.0
    assert scored.rows["fixed-vocabulary-accuracy"].value == 1.0
    floor_value = scored.rows["fiber-floor"].value
    assert floor_value is not None
    assert 0.0 <= floor_value <= 1.0
    assert scored.rows["fiber-size"].value == 12.0
    assert scored.rows["fiber-size"].components == {
        "gold_labels": 3.0,
        "group_order": 12.0,
        "stabilizer_size": 1.0,
    }


def test_a_thin_permuted_symbol_control_lands_at_the_fiber_floor(
    interpreted: HeldOutPass,
) -> None:
    settings = AlignmentSettings(
        kind="one-to-one",
        tie_break="least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=20,
        floor_level=0.95,
        stability_replicates=20,
        stability_level=0.95,
    )
    built = LogicalFormEvaluator(
        interpreted.context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=settings,
        restarts=4,
        permutation_limit=5_040,
    )
    item, derivation = interpreted.gold[0]
    form = built.annotate(item, derivation)
    by_name = {
        node.payload: node for node in _constant_nodes(form) if node.payload is not None
    }
    constants = tuple(by_name[name] for name in sorted(by_name))
    assert len(constants) == 3
    target, planted = constants[:2]
    built.fit_alignment(
        {ItemId("dev-permuted"): planted},
        {ItemId("dev-permuted"): target},
    )
    gold = {ItemId(f"test-{position}"): node for position, node in enumerate(constants)}
    predictions = dict.fromkeys(gold, planted)

    scored = built.score_alignment(predictions, gold, seed=31)
    recovered = scored.rows["fiber-relative-recovery"]
    floor = scored.rows["fiber-floor"]

    assert recovered.value == pytest.approx(1 / 3)
    assert recovered.value == pytest.approx(floor.value)
    assert recovered.components["induced_labels"] == 1.0
    assert recovered.components["gold_labels"] == 3.0
    assert scored.rows["fixed-vocabulary-accuracy"].value == 0.0


def _constants(node: Nominal) -> tuple[str, ...]:
    """Return constant payloads in one named form."""
    here = () if node.op.payload_sort != "constant" else (node.payload,)
    return tuple(value for value in here if value is not None) + tuple(
        value for child in node.args for value in _constants(child)
    )


def _constant_nodes(node: Nominal) -> tuple[Nominal, ...]:
    """Return constant nodes in one named form."""
    here = () if node.op.payload_sort != "constant" else (node,)
    return here + tuple(
        value for child in node.args for value in _constant_nodes(child)
    )


def _rename(node: Nominal, mapping: dict[str, str]) -> Nominal:
    """Return one planted constant relabeling of a named form."""
    payload = node.payload
    if node.op.payload_sort == "constant" and payload is not None:
        payload = mapping[payload]
    return Nominal(
        node.op,
        tuple(_rename(child, mapping) for child in node.args),
        payload=payload,
        binders=node.binders,
    )


def _spine(node: Nominal, application: str) -> tuple[Nominal, tuple[Nominal, ...]]:
    """Return a nominal application head and its arguments."""
    head = node
    arguments: list[Nominal] = []
    while head.op.name == application:
        head, argument = head.args
        arguments.append(argument)
    arguments.reverse()
    return head, tuple(arguments)


def _binary_form(forms: tuple[Nominal, ...], application: str) -> tuple[Nominal, str]:
    """Return a form and the first binary predicate it contains."""
    pending = list(forms)
    while pending:
        node = pending.pop()
        head, arguments = _spine(node, application)
        if (
            len(arguments) == 2
            and head.op.payload_sort == "constant"
            and head.payload is not None
        ):
            return node, head.payload
        pending.extend(reversed(node.args))
    msg = "the logical-form fixture contains no binary predicate"
    raise ValueError(msg)


def _swap_predicate(node: Nominal, application: str, symbol: str) -> Nominal:
    """Swap every complete binary spine headed by one predicate."""
    head, arguments = _spine(node, application)
    if head.payload == symbol and len(arguments) == 2:
        held = head
        for argument in reversed(arguments):
            held = Nominal(
                node.op,
                (_swap_predicate(held, application, symbol), argument),
            )
        return held
    return Nominal(
        node.op,
        tuple(_swap_predicate(child, application, symbol) for child in node.args),
        payload=node.payload,
        binders=node.binders,
    )


def test_a_consistent_observed_argument_swap_recovers_only_up_to_the_fiber(
    interpreted: HeldOutPass,
) -> None:
    built = LogicalFormEvaluator(
        interpreted.context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
        restarts=4,
        permutation_limit=5_040,
    )
    forms = tuple(
        built.annotate(item, derivation) for item, derivation in interpreted.gold
    )
    calculus = interpreted.context.interpretation
    assert calculus is not None
    gold, symbol = _binary_form(forms, calculus.calculus.operators.app)
    swapped = _swap_predicate(gold, calculus.calculus.operators.app, symbol)
    built.fit_alignment({ItemId("dev"): swapped}, {ItemId("dev"): gold})

    scored = built.score_alignment(
        {ItemId("test"): swapped},
        {ItemId("test"): gold},
        seed=37,
    )

    assert scored.rows["fiber-relative-recovery"].value == 1.0
    fixed = scored.rows["fixed-vocabulary-accuracy"].value
    assert fixed is not None
    assert fixed < 1.0


def test_inconsistent_observed_argument_orders_do_not_define_a_predicate(
    interpreted: HeldOutPass,
) -> None:
    built = LogicalFormEvaluator(
        interpreted.context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
        restarts=4,
        permutation_limit=5_040,
    )
    forms = tuple(
        built.annotate(item, derivation) for item, derivation in interpreted.gold
    )
    calculus = interpreted.context.interpretation
    assert calculus is not None
    gold, symbol = _binary_form(forms, calculus.calculus.operators.app)
    swapped = _swap_predicate(gold, calculus.calculus.operators.app, symbol)
    built.fit_alignment(
        {ItemId("dev-identity"): gold, ItemId("dev-swapped"): swapped},
        {ItemId("dev-identity"): gold, ItemId("dev-swapped"): gold},
    )

    scored = built.score_alignment(
        {ItemId("test"): gold},
        {ItemId("test"): gold},
        seed=41,
    )

    recovered = scored.rows["fiber-relative-recovery"].value
    assert recovered is not None
    assert recovered < 1.0


@pytest.mark.parametrize("kind", ["many-to-one", "one-to-one"])
def test_logical_symbol_alignment_is_fitted_on_dev_and_applied_on_test(
    interpreted: HeldOutPass,
    kind: AlignmentKind,
) -> None:
    settings = AlignmentSettings(
        kind=kind,
        tie_break="least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=40,
        floor_level=0.95,
        stability_replicates=40,
        stability_level=0.95,
    )
    built = LogicalFormEvaluator(
        interpreted.context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=settings,
        restarts=4,
        permutation_limit=5_040,
    )
    gold = {
        item.item_id: built.annotate(item, derivation)
        for item, derivation in interpreted.gold
    }
    labels = tuple(
        sorted({label for form in gold.values() for label in _constants(form)})
    )
    assert len(labels) > 1
    rotated = labels[1:] + labels[:1]
    permutation = dict(zip(labels, rotated, strict=True))
    inverse = {target: source for source, target in permutation.items()}
    predictions = {
        item_id: _rename(form, permutation) for item_id, form in gold.items()
    }

    fitted = built.fit_alignment(predictions, gold)
    scored = built.score_alignment(predictions, gold, seed=7)
    for item_id in sorted(gold):
        built.metrics[0].update(predictions[item_id], gold[item_id], key=item_id)

    assert dict(fitted.fitted.alignment.mapping) == inverse
    assert built.metrics[0].compute().value == 1.0
    assert scored.observed.value == 1.0
    assert scored.rows["fiber-relative-recovery"].value == 1.0
    assert scored.rows["fixed-vocabulary-accuracy"].value == 0.0
    observed = scored.observed.value
    assert observed is not None
    assert observed > scored.floor.value
    assert "random-relabeling-floor" in scored.rows
    assert built.relabeling_base(ambiguity_budget=1).group_size > 1
    stability = stable_logical_form_alignments(
        (fitted, fitted, fitted, fitted), seed=11
    )
    assert stability.agreement.value == 1.0


def test_zero_coverage_scores_a_symbol_absent_from_development_as_zero(
    interpreted: HeldOutPass,
) -> None:
    settings = AlignmentSettings(
        kind="many-to-one",
        tie_break="least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=40,
        floor_level=0.95,
        stability_replicates=40,
        stability_level=0.95,
    )
    built = LogicalFormEvaluator(
        interpreted.context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=settings,
        restarts=4,
        permutation_limit=5_040,
    )
    item, derivation = interpreted.gold[0]
    constants = _constant_nodes(built.annotate(item, derivation))
    assert len(constants) > 1
    development, held_out = constants[:2]
    fitted = built.fit_alignment(
        {item.item_id: development},
        {item.item_id: development},
    )
    assert held_out.payload not in fitted.fitted.alignment.mapping

    built.metrics[0].update(held_out, held_out, key=item.item_id)

    assert built.metrics[0].compute().value == 0.0


def test_identity_alignment_preserves_a_symbol_absent_from_development(
    interpreted: HeldOutPass,
) -> None:
    built = LogicalFormEvaluator(
        interpreted.context,
        readout="decoded-forms",
        measure="exact",
        conventions=CONVENTIONS,
        alignment=ALIGNMENT,
        restarts=4,
        permutation_limit=5_040,
    )
    item, derivation = interpreted.gold[0]
    constants = _constant_nodes(built.annotate(item, derivation))
    assert len(constants) > 1
    development, held_out = constants[:2]
    fitted = built.fit_alignment(
        {item.item_id: development},
        {item.item_id: development},
    )
    assert held_out.payload not in fitted.fitted.alignment.mapping

    built.metrics[0].update(held_out, held_out, key=item.item_id)

    assert built.metrics[0].compute().value == 1.0
