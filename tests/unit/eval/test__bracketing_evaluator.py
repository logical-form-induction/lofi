from typing import TYPE_CHECKING, Final, Literal

import pytest
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutSubject,
    packaged_held_out_pass,
)

from lofi._types import ItemId
from lofi.eval import Bracketing, Constituent, ConventionVector, GoldBracketing
from lofi.eval._bracketing_evaluator import PLUGIN, BracketingEvaluator
from lofi.nn.loss._values import ObjectiveOutputs
from lofi.registry import plugin_for
from lofi.spec.evaluators.bracketing import BracketingEvaluatorSpec

if TYPE_CHECKING:
    from lofi.nn.loss._parser import SpanMarginals

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


def evaluator(
    conventions: ConventionVector = CONVENTIONS,
    *,
    punctuation_source: Literal["none", "surface"] = "none",
    punctuation_forms: tuple[str, ...] = (),
    decoder: Literal["minimum-risk", "viterbi"] = "minimum-risk",
) -> BracketingEvaluator:
    return BracketingEvaluator(
        conventions,
        punctuation_source=punctuation_source,
        punctuation_forms=punctuation_forms,
        decoder=decoder,
    )


class TestTheBracketingEvaluatorObeysTheLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(self) -> HeldOutSubject[Bracketing | None, GoldBracketing]:
        return HeldOutSubject(evaluator(), packaged_held_out_pass())


def binary_trees(start: int, end: int) -> list[frozenset[tuple[int, int]]]:
    """Return the spans of two tokens or more of every binary tree over a range."""
    if end - start == 1:
        return [frozenset()]

    return [
        left | right | {(start, end)}
        for cut in range(start + 1, end)
        for left in binary_trees(start, cut)
        for right in binary_trees(cut, end)
    ]


def mass(marginals: SpanMarginals, row: int, tree: frozenset[tuple[int, int]]) -> float:
    """Return the constituent marginal mass a tree's spans hold for one item."""
    return sum(float(marginals.at(row, start, end)) for start, end in tree)


def depth(parents: dict[int, int], node_id: int) -> int:
    """Return how many ancestors a derivation node has."""
    return 0 if node_id not in parents else 1 + depth(parents, parents[node_id])


def test_each_prediction_is_the_tree_with_the_most_constituent_mass() -> None:
    built = evaluator()

    for outputs in packaged_held_out_pass().published:
        marginals = outputs.marginals
        assert marginals is not None
        predicted = built.predict(outputs)

        for row, (item_id, words) in enumerate(
            zip(outputs.batch.item_ids, outputs.batch.words, strict=True)
        ):
            length = len(words)
            best = max(mass(marginals, row, tree) for tree in binary_trees(0, length))
            found = predicted[item_id]

            assert found is not None
            leaves = {(start, start + 1) for start in range(length)}
            assert leaves <= set(found.spans)
            assert (0, length) in found.spans
            assert len(found.spans) == 2 * length - 1
            assert mass(
                marginals, row, frozenset(set(found.spans) - leaves)
            ) == pytest.approx(best)


def test_viterbi_decoding_reads_the_declared_split_posterior() -> None:
    built = evaluator(decoder="viterbi")

    for outputs in packaged_held_out_pass().published:
        assert outputs.splits is not None
        predicted = built.predict(outputs)

        assert set(predicted) == set(outputs.batch.item_ids)
        assert all(entry is not None for entry in predicted.values())


def test_punctuation_source_changes_the_applicable_score_and_conventions() -> None:
    stripping = ConventionVector(
        punctuation="strip",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="sentence",
        coverage="zero",
        self_arcs="allow",
        gold_derivation="canonical",
    )
    plain = evaluator(stripping)
    marked = evaluator(
        stripping,
        punctuation_source="surface",
        punctuation_forms=(",",),
        decoder="viterbi",
    )
    prediction = Bracketing(
        3,
        ((0, 1), (1, 2), (2, 3), (1, 3), (0, 3)),
    )
    constituents = (
        Constituent(0, 1, label="A", depth=2),
        Constituent(1, 2, label="B", depth=2),
        Constituent(2, 3, label="C", depth=1),
        Constituent(0, 2, label="AB", depth=1),
        Constituent(0, 3, label="ABC", depth=0),
    )

    plain.metrics[0].update(
        prediction,
        GoldBracketing(3, constituents, punctuation=()),
        key=ItemId("plain"),
    )
    marked.metrics[0].update(
        prediction,
        GoldBracketing(3, constituents, punctuation=(0,)),
        key=ItemId("marked"),
    )

    assert plain.metrics[0].compute().value == pytest.approx(0.8)
    assert marked.metrics[0].compute().value == pytest.approx(6 / 7)
    assert plain.conventions != marked.conventions
    assert plain.conventions.factors["decoder"] == "minimum-risk"
    assert marked.conventions.factors["decoder"] == "viterbi"


def test_the_gold_bracketing_holds_every_node_at_its_depth() -> None:
    built = evaluator()

    for item, derivation in packaged_held_out_pass().gold:
        gold = built.annotate(item, derivation)
        parents = {
            child: node_id
            for node_id in derivation.nodes
            for child in derivation.node(node_id).children
        }

        assert gold.length == len(item.tokens)
        assert gold.punctuation == frozenset()
        assert sorted(
            (entry.start, entry.end, entry.label, entry.depth)
            for entry in gold.constituents
        ) == sorted(
            (node.span.start, node.span.end, node.label, depth(parents, node_id))
            for node_id, node in derivation.nodes.items()
        )


def test_surface_punctuation_is_read_from_the_declared_inventory() -> None:
    item, derivation = packaged_held_out_pass().gold[0]
    form = item.tokens[0].surface
    built = evaluator(
        punctuation_source="surface",
        punctuation_forms=(form,),
    )

    assert built.annotate(item, derivation).punctuation == frozenset({0})


def test_a_pass_without_span_marginals_is_refused() -> None:
    published = packaged_held_out_pass().published[0]
    bare = ObjectiveOutputs(
        spans=published.spans, chart=published.chart, batch=published.batch
    )

    with pytest.raises(ValueError, match="span marginals"):
        evaluator().predict(bare)


def test_the_plugin_scores_under_the_declared_conventions() -> None:
    declared = ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=3,
        aggregation="corpus",
        coverage="ignore",
        self_arcs="forbid",
        gold_derivation="oracle",
    )
    declaration = BracketingEvaluatorSpec.model_validate(
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": dict(declared.factors),
            "punctuation": {"source": "surface", "forms": [",", "."]},
            "decoder": "viterbi",
        }
    )
    built = PLUGIN.construct(declaration, packaged_held_out_pass().context)

    assert built.conventions.factors["punctuation_source"] == "surface"
    assert built.conventions.factors["decoder"] == "viterbi"
    assert built.metrics[0].conventions == built.conventions
    assert built.metrics[0].name == "unlabeled-bracketing-f1"
    assert declaration.metrics() == tuple(metric.name for metric in built.metrics)
    assert "decoder='viterbi'" in repr(built)


def test_the_evaluator_family_lists_this_plugin_under_its_kind() -> None:
    declaration = BracketingEvaluatorSpec.model_validate(
        {
            "kind": "bracketing",
            "interval": {"replicates": 50, "level": 0.95},
            "conventions": dict(CONVENTIONS.factors),
            "punctuation": {"source": "none", "forms": []},
            "decoder": "minimum-risk",
        }
    )

    assert plugin_for(declaration) is PLUGIN


def test_a_perfect_prediction_scores_one() -> None:
    built = evaluator()
    (metric,) = built.metrics

    for item, derivation in packaged_held_out_pass().gold:
        gold = built.annotate(item, derivation)
        metric.update(
            Bracketing(gold.length, (entry.span for entry in gold.constituents)),
            gold,
            key=item.item_id,
        )

    assert metric.compute().value == pytest.approx(1.0)
