from math import log
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import patch

import pytest
import torch
from didactic.settings import load_document
from tests.conformance.evaluator import (
    HeldOutEvaluatorSuite,
    HeldOutPass,
    HeldOutSubject,
    held_out_pass,
    held_out_run,
    packaged_held_out_pass,
)

from lofi._types import ReadoutId
from lofi.deduce import (
    FREE_ROOTS,
    GOAL_ROOTS,
    LINEAR_WIDENING,
    SUPERTAG_BEAM_PROPOSER,
    SUPERTAG_OUTSIDE,
    ZERO_HEURISTIC,
    AStarContext,
    AStarOptions,
    AStarProposerFactory,
    AStarResult,
    LogInside,
    ParsePolicy,
    Proposer,
    Reachability,
    Retain,
    Viterbi,
    parse,
    supertag_outside,
)
from lofi.eval import (
    ConventionVector,
    HeldOutEvaluationContext,
    HeldOutItems,
    InducedBracketing,
    LabeledDerivationEvaluator,
)
from lofi.eval._decode_evaluator import (
    PLUGIN,
    DecodeEvaluator,
    DecodePrediction,
    _source,
    _UnitScorer,
)
from lofi.formalism import DeductionProfile
from lofi.interpret import ConstituentUtility
from lofi.lexicon import BeamSource, GrowingBeamSource
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._values import ObjectiveOutputs, SpanTerms
from lofi.registry import plugin_for
from lofi.run import section_root
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.spec.evaluators.decode import (
    DecodeEvaluatorSpec,
    DerivationDecision,
    MbrDecision,
    ViterbiDecision,
    derivation_nodes,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import ItemId, JsonValue
    from lofi.data import Derivation, Item
    from lofi.deduce import Derivation as ForestDerivation
    from lofi.deduce import ItemKey, ParseSettings
    from lofi.terms import Term


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
VITERBI: Final = ViterbiDecision()
CODEC: Final = RunSpecCodec(
    search_path=(PACKAGED_CONFIG_ROOT / "run", PACKAGED_CONFIG_ROOT)
)


def _options(
    *,
    zero: bool = False,
    proposer: AStarProposerFactory | None = None,
) -> AStarOptions:
    return AStarOptions(
        LINEAR_WIDENING,
        GOAL_ROOTS,
        ZERO_HEURISTIC if zero else SUPERTAG_OUTSIDE,
        proposer,
    )


def _lexical_output(
    held_out: HeldOutPass,
    labels: tuple[tuple[str, ...], ...] | None = None,
) -> ObjectiveOutputs[HeldOutItems]:
    """Return one fixture pass with explicit decodable lexical categories."""
    outputs = held_out.published[0]
    item_id = outputs.batch.item_ids[0]
    derivation = next(
        derivation for item, derivation in held_out.gold if item.item_id == item_id
    )
    choices = (
        tuple((leaf.label,) for leaf in derivation.leaves())
        if labels is None
        else labels
    )
    cells: dict[tuple[int, int], WeightedTerms] = {}

    for position, category_labels in enumerate(choices):
        terms = tuple(
            held_out.grammar.syntax.parse(held_out.grammar.store, label)
            for label in category_labels
        )
        span = outputs.spans.span_id(position, position + 1)
        cells[0, span] = WeightedTerms(
            terms,
            torch.full((len(terms),), -log(len(terms))),
            torch.tensor(-torch.inf),
            Estimate.EXACT,
        )

    return ObjectiveOutputs(
        spans=outputs.spans,
        chart=outputs.chart,
        batch=outputs.batch,
        terms={
            ReadoutId("category-terms"): SpanTerms(outputs.spans, cells),
        },
    )


def _evaluator(
    *,
    astar: AStarOptions | None = None,
    decode: DerivationDecision | None = None,
) -> DecodeEvaluator:
    held_out = packaged_held_out_pass()

    return DecodeEvaluator(
        held_out.context,
        readout="category-terms",
        output="lexical-types",
        astar=astar,
        decode=decode,
        conventions=CONVENTIONS,
    )


class TestTheDecodeEvaluatorObeysTheLaws(HeldOutEvaluatorSuite):
    @pytest.fixture
    def held_out_subject(self) -> HeldOutSubject[DecodePrediction, ItemId]:
        return HeldOutSubject(_evaluator(), packaged_held_out_pass())


def test_a_plain_decode_returns_the_shared_strategy_beam_unchanged() -> None:
    outputs = packaged_held_out_pass().published[0]
    decoded = outputs.terms[ReadoutId("category-terms")]
    prediction = next(iter(_evaluator().predict(outputs).values()))
    root = outputs.spans.span_id(0, len(outputs.batch.words[0]))
    published = decoded.at(0, root)

    assert prediction.beams == (published,)
    assert prediction.beams[0] is published
    assert prediction.beams[0].log_weights is published.log_weights
    assert prediction.search is None
    assert prediction.derivation is None


def _edge_signature(
    derivation: ForestDerivation[Term, float] | None,
) -> tuple[str, ...]:
    assert derivation is not None
    return tuple(repr(node.edge) for node in derivation_nodes(derivation))


def test_both_declared_heuristics_match_exhaustive_cky_and_viterbi() -> None:
    held_out = packaged_held_out_pass()
    outputs = _lexical_output(held_out)
    words = tuple(outputs.batch.words[0])
    decoded = outputs.terms[ReadoutId("category-terms")]
    source, published_beams = _source(words, decoded, 0)
    grammar = held_out.grammar.grammar
    exhaustive = parse(
        words,
        rules=grammar.rules,
        source=source,
        semiring=LogInside(),
        scorer=_UnitScorer(),
        goal=grammar.goal,
        unary_depth=grammar.budget.unary_rounds,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    roots = tuple(grammar.goal.roots(exhaustive))
    complete_source = GrowingBeamSource(
        source,
        frontier=tuple(
            len(source.candidates(position)) for position in range(source.positions)
        ),
    )
    exhaustive_result = AStarResult(
        exhaustive,
        roots,
        complete_source,
        supertag_outside(source),
        1,
        _options(),
    )
    context = AStarContext(
        grammar.rules,
        _UnitScorer(),
        grammar.goal,
        grammar.budget.unary_rounds,
        grammar.budget.threads,
        cast("ParseSettings[float]", held_out.context.parse),
    )
    expected = _edge_signature(VITERBI.select(exhaustive_result, context))
    predictions = []

    for zero in (False, True):
        prediction = next(
            iter(
                _evaluator(astar=_options(zero=zero), decode=VITERBI)
                .predict(outputs)
                .values()
            )
        )
        search = prediction.search

        assert search is not None
        assert search.parsed
        assert search.roots == roots
        assert prediction.beams == published_beams
        assert _edge_signature(prediction.derivation) == expected
        assert search.attempts > 0
        assert search.widenings >= 0
        predictions.append(prediction)

    assert predictions[0].search is not None
    assert predictions[1].search is not None
    assert predictions[0].search.roots == predictions[1].search.roots
    assert _edge_signature(predictions[0].derivation) == _edge_signature(
        predictions[1].derivation
    )


def _uniform_ambiguous_output() -> ObjectiveOutputs[HeldOutItems]:
    return _lexical_output(
        packaged_held_out_pass(),
        (
            ("S/NP/NP", "NP", "NP/NP"),
            ("S\\NP/NP", "NP", "NP/NP"),
            ("NP",),
        ),
    )


def test_sixteen_candidate_mbr_searches_an_ambiguous_forest() -> None:
    outputs = _uniform_ambiguous_output()
    viterbi = next(
        iter(_evaluator(astar=_options(), decode=VITERBI).predict(outputs).values())
    )
    mbr = next(
        iter(
            _evaluator(
                astar=_options(),
                decode=MbrDecision(ConstituentUtility(), 16),
            )
            .predict(outputs)
            .values()
        )
    )

    assert viterbi.derivation is not None
    assert mbr.derivation is not None
    assert mbr.search is not None
    assert sum(len(mbr.search.chart.edges(root)) for root in mbr.search.roots) > 1


class _CountingUtility:
    """Record how many globally budgeted MBR candidates are compared."""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def gain[T, W](
        self,
        derivation: ForestDerivation[T, W],
        /,
        *,
        marginals: Mapping[ItemKey[T], float],
    ) -> float:
        del derivation, marginals
        self.calls += 1
        return 0.0


def test_mbr_candidate_count_is_global_across_free_roots() -> None:
    outputs = _uniform_ambiguous_output()
    utility = _CountingUtility()
    held_out = packaged_held_out_pass()
    grammar = held_out.grammar.grammar
    words = tuple(outputs.batch.words[0])
    decoded = outputs.terms[ReadoutId("category-terms")]
    source, _ = _source(words, decoded, 0)
    settings = cast("ParseSettings[float]", held_out.context.parse)
    context = AStarContext(
        grammar.rules,
        _UnitScorer(),
        grammar.goal,
        grammar.budget.unary_rounds,
        grammar.budget.threads,
        settings,
    )
    astar = AStarOptions(
        LINEAR_WIDENING,
        FREE_ROOTS,
        SUPERTAG_OUTSIDE,
        None,
    )
    chart = parse(
        words,
        rules=grammar.rules,
        source=source,
        semiring=settings.semiring,
        scorer=context.scorer,
        goal=grammar.goal,
        unary_depth=grammar.budget.unary_rounds,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    roots = tuple(FREE_ROOTS.goal(grammar.goal).roots(chart))
    complete_source = GrowingBeamSource(
        source,
        frontier=tuple(
            len(source.candidates(position)) for position in range(source.positions)
        ),
    )
    result = AStarResult(
        chart,
        roots,
        complete_source,
        supertag_outside(source),
        1,
        astar,
    )
    selected = MbrDecision(utility, 3).select(result, context)

    assert len(roots) > 1
    assert selected is not None
    assert utility.calls == 3


class _NonLexicalGrammar:
    profile = DeductionProfile(
        1,
        frozenset({2}),
        unary_closure=False,
        category_set="finite",
        lexicalized=False,
    )


class _NonLexicalAssembly:
    grammar = _NonLexicalGrammar()


class _NonLexicalContext:
    grammar = _NonLexicalAssembly()


def test_astar_is_refused_when_the_grammar_is_not_lexicalized() -> None:
    with pytest.raises(ValueError, match="requires a lexicalized profile"):
        DecodeEvaluator(
            cast("HeldOutEvaluationContext", _NonLexicalContext()),
            readout="category-terms",
            output="lexical-types",
            astar=_options(),
            decode=VITERBI,
            conventions=CONVENTIONS,
        )


def test_a_missing_complete_term_readout_is_refused() -> None:
    outputs = packaged_held_out_pass().published[0]
    bare = ObjectiveOutputs(
        spans=outputs.spans,
        chart=outputs.chart,
        batch=outputs.batch,
    )

    with pytest.raises(ValueError, match="category-terms"):
        _evaluator().predict(bare)


def test_astar_metrics_report_each_sentence_search() -> None:
    outputs = packaged_held_out_pass().published[0]
    evaluator = _evaluator(astar=_options(), decode=VITERBI)
    prediction = next(iter(evaluator.predict(outputs).values()))
    search = prediction.search
    item_id = outputs.batch.item_ids[0]

    assert search is not None

    for metric in evaluator.metrics:
        metric.update(prediction, item_id, key=item_id)

    assert evaluator.metrics[1].per_item()[item_id].value == float(search.parsed)
    assert evaluator.metrics[2].per_item()[item_id].value == float(not search.parsed)
    assert evaluator.metrics[3].per_item()[item_id].value == search.attempts
    assert evaluator.metrics[4].per_item()[item_id].value == search.widenings


class _RecordingProposerFactory:
    """Record each sentence source used to construct a checked proposer."""

    name = "recording"

    def __init__(self) -> None:
        self.sources: list[BeamSource[Term] | GrowingBeamSource[Term]] = []

    def build[T](
        self,
        source: BeamSource[T] | GrowingBeamSource[T],
        /,
    ) -> Proposer[T]:
        """Record and delegate to the real supertag-beam proposer."""
        self.sources.append(cast("BeamSource[Term] | GrowingBeamSource[Term]", source))
        return SUPERTAG_BEAM_PROPOSER.build(source)


def test_evaluation_builds_checked_proposers_from_active_search_frontiers() -> None:
    outputs = packaged_held_out_pass().published[0]
    factory = _RecordingProposerFactory()
    prediction = next(
        iter(
            _evaluator(astar=_options(proposer=factory), decode=VITERBI)
            .predict(outputs)
            .values()
        )
    )

    assert prediction.search is not None
    assert factory.sources
    assert all(isinstance(source, GrowingBeamSource) for source in factory.sources)
    assert all(
        source.positions == len(outputs.batch.words[0]) for source in factory.sources
    )
    assert factory.sources[-1] is prediction.search.source


def test_the_declared_decode_choices_are_part_of_the_convention_vector() -> None:
    evaluator = _evaluator(astar=_options(zero=True), decode=VITERBI)

    assert dict(evaluator.conventions.factors) == {
        "punctuation": "keep",
        "unary_chains": "keep",
        "trivial_spans": "keep",
        "minimum_length": 0,
        "aggregation": "sentence",
        "coverage": "zero",
        "self_arcs": "allow",
        "gold_derivation": "canonical",
        "astar_heuristic": "zero",
        "astar_proposer": None,
        "astar_root_mode": "goal",
        "astar_strategy": "linear",
        "decode_candidates": 1,
        "decode_method": "viterbi",
        "decode_output": "lexical-types",
        "decode_readout": "category-terms",
        "decode_utility": None,
        "parse_semiring": "log-inside",
    }


def test_the_plugin_builds_the_declared_decode_evaluator() -> None:
    declaration = DecodeEvaluatorSpec.model_validate(
        {
            "kind": "decode",
            "interval": {"replicates": 50, "level": 0.95},
            "readout": "category-terms",
            "output": "lexical-types",
            "astar": {"heuristic": {"kind": "zero"}},
            "decode": {"kind": "viterbi"},
            "conventions": dict(CONVENTIONS.factors),
        }
    )
    built = PLUGIN.construct(declaration, packaged_held_out_pass().context)

    assert plugin_for(declaration) is PLUGIN
    assert tuple(metric.name for metric in built.metrics) == declaration.metrics()
    assert all(
        built.conventions.factors[name] == value
        for name, value in CONVENTIONS.factors.items()
    )
    assert built.conventions.factors["astar_heuristic"] == "zero"
    assert "readout='category-terms'" in repr(built)


@pytest.mark.parametrize(
    ("document", "heuristic"),
    [
        ("synthetic-decode-zero.yaml", "zero"),
        ("synthetic-decode-supertag.yaml", "supertag-outside"),
    ],
)
def test_a_packaged_decode_document_runs_its_declared_astar(
    document: str,
    heuristic: str,
) -> None:
    path = Path(__file__).parents[3] / "lofi/configs/run" / document
    held_out = held_out_pass(path)
    declaration = cast(
        "DecodeEvaluatorSpec",
        CODEC(path).evaluators["decoded-types"],
    )
    built = cast(
        "DecodeEvaluator",
        plugin_for(declaration).construct(declaration, held_out.context),
    )
    prediction = next(iter(built.predict(_lexical_output(held_out)).values()))

    assert prediction.search is not None
    assert prediction.search.options.heuristic.name == heuristic
    assert prediction.search.parsed
    assert prediction.derivation is not None


def test_a_decode_evaluator_labels_each_span_of_its_selected_derivation() -> None:
    path = Path(__file__).parents[3] / "lofi/configs/run/synthetic-decode-supertag.yaml"
    held_out = held_out_pass(path)
    declaration = cast(
        "DecodeEvaluatorSpec",
        CODEC(path).evaluators["decoded-types"],
    )
    built = cast(
        "DecodeEvaluator",
        plugin_for(declaration).construct(declaration, held_out.context),
    )
    outputs = _lexical_output(held_out)
    item_id, prediction = next(iter(built.predict(outputs).items()))
    gold = next(
        derivation for item, derivation in held_out.gold if item.item_id == item_id
    )
    expected: dict[tuple[int, int], str] = {}

    for node_id in gold.preorder:
        node = gold.node(node_id)
        expected.setdefault((node.span.start, node.span.end), node.label)

    assert isinstance(built, LabeledDerivationEvaluator)
    assert built.labeled_spans(prediction) == InducedBracketing(
        len(outputs.batch.words[0]), expected
    )


def test_a_decode_without_a_derivation_labels_no_span() -> None:
    outputs = packaged_held_out_pass().published[0]
    evaluator = _evaluator()
    prediction = next(iter(evaluator.predict(outputs).values()))

    assert prediction.derivation is None
    assert evaluator.labeled_spans(prediction) is None


@pytest.mark.parametrize(
    ("parse_name", "semiring"),
    [("exhaustive", LogInside), ("viterbi-beam", Viterbi)],
)
def test_the_run_parse_declaration_controls_the_decoding_parse(
    parse_name: str,
    semiring: type[LogInside | Viterbi],
) -> None:
    path = Path(__file__).parents[3] / "lofi/configs/run/synthetic-decode-zero.yaml"
    run_spec = CODEC(
        path,
        extra=cast(
            "Mapping[str, JsonValue]",
            {"parse": load_document(section_root("parse") / f"{parse_name}.yaml")},
        ),
    )
    held_out = held_out_run(run_spec)
    declaration = cast("DecodeEvaluatorSpec", run_spec.evaluators["decoded-types"])
    built = cast(
        "DecodeEvaluator",
        plugin_for(declaration).construct(declaration, held_out.context),
    )
    prediction = next(iter(built.predict(_lexical_output(held_out)).values()))

    assert prediction.search is not None
    assert type(prediction.search.chart.semiring) is semiring
    assert (held_out.context.parse.pruner is None) == (parse_name == "exhaustive")
    assert built.conventions.factors["parse_semiring"] == (
        held_out.context.parse.semiring.name
    )


def test_the_grammar_s_declared_threads_reach_the_decoding_parse() -> None:
    path = Path(__file__).parents[3] / "lofi/configs/run/synthetic-decode-zero.yaml"
    run_spec = CODEC(
        path,
        overrides=("data.datasets.train.dataset.grammar.budget.threads=2",),
    )
    held_out = held_out_run(run_spec)
    declaration = cast("DecodeEvaluatorSpec", run_spec.evaluators["decoded-types"])
    built = cast(
        "DecodeEvaluator",
        plugin_for(declaration).construct(declaration, held_out.context),
    )

    with patch("lofi.deduce._astar.parse", wraps=parse) as parser:
        built.predict(_lexical_output(held_out))

    assert parser.call_args is not None
    assert parser.call_args.kwargs["threads"] == 2


def test_annotation_uses_the_sentence_identifier() -> None:
    held_out = packaged_held_out_pass()
    item: Item
    derivation: Derivation[str]
    item, derivation = held_out.gold[0]

    assert _evaluator().annotate(item, derivation) == item.item_id
