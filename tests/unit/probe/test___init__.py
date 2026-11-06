from typing import Final

import pytest
import torch
from torch.nn import Module

import lofi.probe
from lofi.deduce import Counting, Edge, ItemKey, PackedForest, SpanIndex
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap
from lofi.probe import (
    _acdc,
    _attribution,
    _boundary,
    _bucketing,
    _catalog,
    _causal,
    _causal_probe,
    _chart,
    _circuit_probe,
    _circuitry,
    _circuits,
    _classifier_probe,
    _constructors,
    _controller_choice,
    _controller_relations,
    _controls,
    _counterfactuals,
    _das,
    _description,
    _determinism,
    _diagnose,
    _diagnostics_probe,
    _differentiation,
    _diora,
    _dtypes,
    _erasure,
    _external,
    _faithfulness,
    _filtering,
    _finite,
    _grading,
    _graph,
    _instrument,
    _intervention_probe,
    _interventions,
    _joint,
    _margin,
    _marginals,
    _node_vocabulary,
    _observation_groups,
    _ports,
    _pricing,
    _probing,
    _provenance,
    _purity,
    _recurrent,
    _recursion,
    _roles,
    _sae,
    _scores,
    _selectivity,
    _sensitivity,
    _structure,
    _superposition,
    _sweep,
    _transformer,
    _values,
    _variables,
)
from lofi.probe._das import AlignmentObjective
from lofi.stats import Bootstrap, LabelPermutation, p_value_against_replicates

PARSER = ComponentId("cky-parser")

MODULES: Final = (
    _acdc,
    _attribution,
    _boundary,
    _bucketing,
    _catalog,
    _causal,
    _causal_probe,
    _chart,
    _circuits,
    _circuitry,
    _circuit_probe,
    _classifier_probe,
    _constructors,
    _controller_choice,
    _controller_relations,
    _controls,
    _counterfactuals,
    _das,
    _description,
    _determinism,
    _diagnose,
    _diagnostics_probe,
    _differentiation,
    _diora,
    _dtypes,
    _erasure,
    _external,
    _faithfulness,
    _filtering,
    _finite,
    _grading,
    _graph,
    _instrument,
    _intervention_probe,
    _intervention_probe,
    _interventions,
    _joint,
    _margin,
    _marginals,
    _node_vocabulary,
    _observation_groups,
    _ports,
    _pricing,
    _probing,
    _provenance,
    _purity,
    _recurrent,
    _recursion,
    _roles,
    _sae,
    _scores,
    _selectivity,
    _sensitivity,
    _structure,
    _superposition,
    _sweep,
    _transformer,
    _values,
    _variables,
)


def test_the_published_surface_carries_a_circuit_and_the_evidence_for_an_edge() -> None:
    left = lofi.probe.Unit(Site(PARSER, Role.CHART_CELL, (0, 2)), {"width": 3})
    whole = lofi.probe.Unit(Site(PARSER, Role.CHART_CELL, (0, 4)), {"width": 3})
    found = lofi.probe.Circuit([left, whole])
    edge = lofi.probe.CircuitEdge(
        left,
        whole,
        lofi.probe.AttributionEvidence(
            0.75, lofi.probe.Provenance("eap-ig", "ccgbank-dev", 11)
        ),
    )

    assert found == lofi.probe.Circuit([whole, left])
    assert found.id == lofi.probe.Circuit([whole, left]).id
    assert lofi.probe.evidence_from_json(edge.evidence.to_json()) == edge.evidence


def test_the_published_surface_buckets_and_aligns_probe_results() -> None:
    provenance = lofi.probe.Provenance("eap-ig", "held-out", 11)
    features = lofi.probe.AbstractionFeatures(
        sentence_length=2,
        derivation_depth=1,
        ambiguity=0,
        span_length=1,
    )
    bucketed = lofi.probe.bucket_abstraction(
        (lofi.probe.AbstractionCase(features, matched=True),),
        well_cutoff=1.0,
        min_bucket_size=1,
        min_accuracy_gap=0.0,
    )

    varied = lofi.probe.TypeAt((1, 2))
    base: dict[lofi.probe.HighLevelVariable, str | int] = {varied: "Sdcl"}
    variant: dict[lofi.probe.HighLevelVariable, str | int] = {varied: "Sng"}
    pair = lofi.probe.GeneratedPair(
        ("really", "sleeps"),
        ("S/S", "Sdcl"),
        ("S/S", "Sng"),
        base,
        variant,
        varied,
        "toy:1",
        provenance,
    )
    policy = lofi.probe.AlignmentPolicy("all", "exclude", contiguous=True)
    alignment = lofi.probe.TokenizerAlignment("word-tokenizer", ((0,), (1,)), (), 2)
    aligned = lofi.probe.AlignedInput(pair.tokens, (alignment,))
    refusal = lofi.probe.AlignmentRefusal(
        pair.id,
        alignment.tokenizer,
        pair.span,
        "noncontiguous-model-row-span",
    )
    manifest = lofi.probe.AlignmentManifest(
        policy,
        aligned.tokenizers,
        1,
        0,
    )
    generation = lofi.probe.CounterfactualGeneration((), {}, (refusal,), manifest)

    assert bucketed.buckets[0].label == "well"
    assert generation.refusals == (refusal,)
    assert generation.manifest.policy is policy


def test_the_published_surface_carries_a_graph_built_from_a_forest() -> None:
    forest = PackedForest[str, int](SpanIndex(2), Counting())
    determiner = ItemKey((0, 1), "NP/N", "lex")
    noun = ItemKey((1, 2), "N", "lex")
    phrase = ItemKey((0, 2), "NP", "step")
    forest.add(determiner, Edge("lex", (), "NP/N", 1))
    forest.add(noun, Edge("lex", (), "N", 1))
    forest.add(phrase, Edge("fa", (determiner, noun), "NP", 1, split=1))
    sites = {
        key: Site(PARSER, Role.CHART_CELL, key.span)
        for key in (determiner, noun, phrase)
    }
    graph = lofi.probe.ComputationGraph.from_forest(forest, sites)
    whole = lofi.probe.GraphNode(PARSER, phrase)

    assert len(graph.nodes) == 3
    assert graph.edges == {
        lofi.probe.GraphEdge(
            lofi.probe.GraphNode(PARSER, determiner), whole, Role.CHART_CELL
        ),
        lofi.probe.GraphEdge(
            lofi.probe.GraphNode(PARSER, noun), whole, Role.CHART_CELL
        ),
    }
    assert graph.site_of(whole) == Site(PARSER, Role.CHART_CELL, (0, 2))


def test_the_published_surface_carries_a_counterfactual_pair_and_its_result() -> None:
    category = lofi.probe.TypeAt((0, 2))
    split = lofi.probe.SplitAt((0, 4))
    pair = lofi.probe.CounterfactualPair(
        {category: "NP", split: 2},
        {category: "S", split: 2},
        category,
        (0, 4),
        lofi.probe.Provenance("grammar-counterfactuals", "ccgbank-dev", 7),
    )
    tested = p_value_against_replicates(
        0.9,
        [0.4, 0.5, 0.55],
        tail="greater",
        null="cell-permutation",
        family="permutation",
        exact=False,
        seed=3,
    )
    result = lofi.probe.IIAResult(0.9, pair.varied, 8, tested.provenance)

    assert pair.varied == category
    assert result.variable == pair.varied
    assert result.null.family == "permutation"


def test_the_published_surface_names_the_model_ready_pair_boundary() -> None:
    assert lofi.probe.ModelReadyCausalPair is _ports.ModelReadyCausalPair


def test_the_published_surface_reads_a_point_and_then_intervenes_on_it() -> None:
    cell = Site(PARSER, Role.CHART_CELL, (0, 2))
    reading = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    observed = lofi.probe.Observation(cell, reading, axes=("item", "width"))

    trace = lofi.probe.Trace()
    trace.record(observed)
    trace.close()

    assert torch.equal(trace[cell].select(item=1).tensor, reading[1])
    assert torch.equal(lofi.probe.Zero().stands_in_for(observed), torch.zeros(3, 4))
    assert torch.equal(
        lofi.probe.PatchFrom(trace, cell).stands_in_for(
            lofi.probe.Observation(cell, torch.zeros(3, 4), axes=("item", "width"))
        ),
        reading,
    )


def test_the_published_surface_reports_ablation_semantics_sensitivity() -> None:
    cells = [
        lofi.probe.AblationSensitivityCell(semantics, metric, 0.5, conclusion=True)
        for semantics in (
            lofi.probe.AblationSemantics.RESAMPLE,
            lofi.probe.AblationSemantics.MEAN,
        )
        for metric in lofi.probe.AblationMetric
    ]
    report = lofi.probe.AblationSensitivityReport(cells)

    assert report.default_intervention is lofi.probe.AblationSemantics.RESAMPLE
    assert not report.flip
    assert report.conclusion is True
    assert report.status == "reported"


def test_the_published_surface_fits_and_reports_linear_concept_erasure() -> None:
    representations = torch.tensor([[2.0, 1.0], [2.0, -1.0], [-2.0, 1.0], [-2.0, -1.0]])
    concepts = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]])
    erasure = lofi.probe.fit_leace(representations, concepts)
    report = lofi.probe.ErasureReport(
        ((1.0, 0.9, 0.8, 0.7), (0.5, 0.95, 0.4, 0.6)),
        ((1.0, 0.9, 0.8, 0.7), (0.5, 0.4, 0.3, 0.6)),
        ((0.8, 0.7, 0.6, 0.5), (0.79, 0.71, 0.59, 0.51)),
        behavioral_direction="higher",
        decodability_direction="higher",
        control_direction="higher",
        bootstrap=Bootstrap(unit="item", replicates=19, seed=2, level=0.95),
    )

    assert isinstance(erasure, lofi.probe.LinearErasure)
    assert erasure.tag == "linear-erasure"
    assert report.behavioral_drop.interval.unit == "item"


def test_the_published_surface_carries_a_site_map_read_from_a_declaration() -> None:
    declared = lofi.probe.SiteMapSpec(
        component="cky-parser",
        kind="custom",
        sites=(
            lofi.probe.SiteDecl(
                role="OUTSIDE_SCORE",
                key=(0, 2),
                axes=("item", "width"),
                dtype_class="real",
                differentiable=True,
                access=("OBSERVE",),
            ),
        ),
    )

    published = lofi.probe.SiteMapIso().forward(declared)
    point = Site(PARSER, Role.OUTSIDE_SCORE, (0, 2))

    assert published[point].position("width") == 1
    assert published[point].dtype_class is DTypeClass.REAL
    assert Access.PATCH not in published[point].access


def test_the_published_surface_resolves_a_diagnostic_by_the_roles_it_needs() -> None:
    published = SiteMap.custom(
        PARSER,
        {
            Site(PARSER, Role.COMBINATOR_LOGITS): lofi.probe.SiteMapIso().forward(
                lofi.probe.SiteMapSpec(
                    component="cky-parser",
                    kind="custom",
                    sites=(
                        lofi.probe.SiteDecl(
                            role="COMBINATOR_LOGITS",
                            key=(),
                            axes=("cell", "action"),
                            dtype_class="log-probability",
                            differentiable=True,
                            access=("OBSERVE",),
                        ),
                    ),
                )
            )[Site(PARSER, Role.COMBINATOR_LOGITS)]
        },
    )
    context = lofi.probe.DiagnosticContext(published)
    resolved = lofi.probe.installed_diagnostics().matching(context)

    assert [one.tag for one in resolved] == [
        "action-margin",
        "controller-choice",
        "controller-relations",
        "finite-readings",
    ]
    assert lofi.probe.ActionMargin.requires <= context.roles


def test_the_published_surface_probes_a_model_from_a_declaration_alone() -> None:
    declared = lofi.probe.SiteMapSpec(
        component="toy-controller",
        kind="custom",
        sites=(
            lofi.probe.SiteDecl(
                role="COMBINATOR_LOGITS",
                key=(0,),
                axes=("span", "action"),
                dtype_class="log-probability",
                differentiable=True,
                access=("OBSERVE", "PATCH"),
            ),
        ),
    )
    published = lofi.probe.SiteMapIso().forward(declared)
    (point,) = published
    weighed = torch.tensor([[0.0, -1.0, -2.0], [-3.0, -0.5, -4.0]])

    trace = lofi.probe.Trace()
    trace.record(lofi.probe.Observation(point, weighed, axes=published[point].axes))
    trace.close()

    believed = lofi.probe.SpanWeighting(torch.tensor([0.75, 0.25]), axes=("span",))
    context = lofi.probe.DiagnosticContext(
        published,
        relations={
            point: lofi.probe.ControllerRelation(
                2,
                sequences=((0, 1),),
                similar_inputs=((0, 1),),
            )
        },
        weighting=believed,
    )
    measured = {
        one.tag: one.run(trace, context=context)
        for one in lofi.probe.installed_diagnostics().matching(context)
    }
    margin = measured["action-margin"].aggregate(point, "margin")

    # gaps of 1.0 and 2.5, weighed 0.75 and 0.25 or alike
    assert set(measured) == {
        "action-margin",
        "controller-choice",
        "controller-relations",
        "finite-readings",
    }
    assert margin == lofi.probe.SpanAggregate(1.375, 1.75, 2)
    assert measured["action-margin"].value(point, "rows") == 2.0
    assert measured["finite-readings"].value(point, "nonfinite-share") == 0.0


def test_the_published_surface_compares_complete_node_vocabulary_sweeps() -> None:
    variable = lofi.probe.TypeAt((0, 2))
    search = lofi.probe.FixedRankSearch(
        rank=1,
        schedule=lofi.probe.Schedule(epochs=1, learning_rate=0.1, batch_size=2, seed=0),
        null=LabelPermutation(replicates=3, seed=0),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    costs = lofi.probe.CostModel(
        lofi.probe.UnitPrice(
            unit,
            seconds_per_run=0.01,
            seconds_fastest=0.005,
            warmup=1,
            repeats=2,
            device="cpu",
        )
        for unit in (
            lofi.probe.ProbeUnit.FIXED_ALIGNMENT_FIT,
            lofi.probe.ProbeUnit.FIXED_PERMUTATION_REFIT,
        )
    )
    tested = p_value_against_replicates(
        0.5,
        (0.1, 0.3, 0.7),
        tail="greater",
        null="cell permutation",
        family="permutation",
        exact=False,
        seed=11,
    )
    rows = []

    for index, arm in enumerate(lofi.probe.NodeVocabularyArm):
        site = Site(
            ComponentId(f"node-vocabulary-{index}"),
            Role.RESIDUAL_STREAM,
            (0, 0),
        )
        subgrid = lofi.probe.Subgrid((variable,), (0,), (0,), {(0, 0): site})
        preregistration = lofi.probe.Preregistration(
            subgrid, search, costs=costs, budget_hours=1.0
        )
        cell = subgrid.cells[0]
        accuracy = 0.2 + 0.1 * index
        result = lofi.probe.IIAResult(accuracy, variable, 1, tested.provenance)
        report = lofi.probe.SweepReport(
            preregistration,
            (
                lofi.probe.CellResult(
                    cell,
                    site,
                    lofi.probe.Assessment(result, tested),
                    preregistration.digest,
                ),
            ),
        )
        rows.append(
            lofi.probe.node_vocabulary_row(
                arm,
                report,
                ("pair-b", "pair-a") if index % 2 else ("pair-a", "pair-b"),
            )
        )

    comparison = lofi.probe.NodeVocabularyComparison(reversed(rows))

    assert lofi.probe.NodeVocabularyArm is _node_vocabulary.NodeVocabularyArm
    assert lofi.probe.NodeVocabularyRow is _node_vocabulary.NodeVocabularyRow
    assert (
        lofi.probe.NodeVocabularyComparison is _node_vocabulary.NodeVocabularyComparison
    )
    assert tuple(row.arm for row in comparison.rows) == tuple(
        lofi.probe.NodeVocabularyArm
    )
    assert [row["mean_iia"] for row in comparison.to_json()] == pytest.approx(
        [0.2, 0.3, 0.4, 0.5]
    )
    assert len({row["pair_digest"] for row in comparison.to_json()}) == 1


def test_the_package_publishes_every_name_its_modules_declare() -> None:
    declared = set()

    for module in MODULES:
        declared |= set(module.__all__)

    assert set(lofi.probe.__all__) == declared

    for name in lofi.probe.__all__:
        assert getattr(lofi.probe, name) is not None


def test_the_published_surface_costs_a_grid_from_measured_prices() -> None:
    price = lofi.probe.UnitPrice(
        lofi.probe.ProbeUnit.FIXED_ALIGNMENT_FIT,
        seconds_per_run=36.0,
        seconds_fastest=30.0,
        warmup=1,
        repeats=3,
        device="cpu",
    )
    model = lofi.probe.CostModel((price,))

    grid = lofi.probe.GridSpec(
        (("layer", 100),),
        fit_unit=lofi.probe.ProbeUnit.FIXED_ALIGNMENT_FIT,
        refit_unit=lofi.probe.ProbeUnit.FIXED_PERMUTATION_REFIT,
    )
    assert model.hours(grid) == pytest.approx(1.0)

    with pytest.raises(lofi.probe.UnpricedUnitError):
        model.hours(
            lofi.probe.GridSpec(
                (("layer", 100),),
                fit_unit=lofi.probe.ProbeUnit.FIXED_ALIGNMENT_FIT,
                refit_unit=lofi.probe.ProbeUnit.FIXED_PERMUTATION_REFIT,
                replicates=1,
            )
        )


def test_the_published_surface_exposes_only_the_current_search_strategies() -> None:
    schedule = lofi.probe.Schedule(epochs=2, learning_rate=0.1, batch_size=4, seed=0)
    null = LabelPermutation(replicates=1, seed=0)
    fixed = lofi.probe.FixedRankSearch(
        rank=2,
        schedule=schedule,
        null=null,
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    boundless = lofi.probe.BoundlessSearch(
        boundaries=(0.0, 2.0),
        margin=1e-6,
        annealing=lofi.probe.Annealing(initial=1.0, final=0.1),
        schedule=schedule,
        null=null,
        objective=AlignmentObjective.CROSS_ENTROPY,
    )

    assert fixed.rank == 2
    assert boundless.initial_width == 2
    assert not hasattr(lofi.probe, "Search")


def test_no_class_the_package_declares_holds_submodules_of_its_own() -> None:
    held = [
        f"{module.__name__}.{name}"
        for module in MODULES
        for name, value in vars(module).items()
        if isinstance(value, type)
        and issubclass(value, Module)
        and value.__module__ == module.__name__
    ]

    assert held == []
