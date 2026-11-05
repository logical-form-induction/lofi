import math
from typing import Final, cast

import pytest
import torch

from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders.rnn import TreeRnnEncoder
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
from lofi.probe._instrument import instrument
from lofi.probe._recurrent import (
    GatePatterns,
    HiddenStateEvolution,
    MemoryCellDynamics,
    RecurrentConnectionAnalysis,
)
from lofi.probe._values import Observation, Trace
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

ENCODER: Final = ComponentId("recurrent-encoder")
HIDDEN: Final = Site(ENCODER, Role.RECURRENT_HIDDEN, ("upper",))
HIDDEN_OTHER: Final = Site(ENCODER, Role.RECURRENT_HIDDEN, ("lower",))
CELL: Final = Site(ENCODER, Role.RECURRENT_CELL, ("upper",))
GATE: Final = Site(ENCODER, Role.GATE, (9, "arbitrary"))
GATE_OTHER: Final = Site(ENCODER, Role.GATE, ("not-a-gate-name", 4))

REAL: Final = SiteSpec(
    ("term", "slot", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)
PROBABILITY: Final = SiteSpec(
    ("term", "slot", "width"),
    DTypeClass.PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
PARENTS: Final = (Encoding.NO_PARENT, 0, 0)

RECURRENT_TYPES: Final = frozenset(
    {
        GatePatterns,
        HiddenStateEvolution,
        MemoryCellDynamics,
        RecurrentConnectionAnalysis,
    }
)


def observed(
    site: Site,
    values: torch.Tensor,
    *,
    axes: tuple[str, ...] = ("term", "slot", "width"),
    mask: torch.Tensor | None = None,
) -> Observation:
    return Observation(site, values, axes=axes, mask=mask)


def traced(*readings: Observation) -> Trace:
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def result_for_hidden(
    diagnostic: HiddenStateEvolution | RecurrentConnectionAnalysis,
    values: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> DiagnosticResult:
    return diagnostic.run(
        traced(observed(HIDDEN, values, mask=mask)),
        context=DiagnosticContext({HIDDEN: REAL}),
    )


def test_each_diagnostic_declares_exactly_the_roles_it_needs() -> None:
    assert HiddenStateEvolution.requires == frozenset({Role.RECURRENT_HIDDEN})
    assert GatePatterns.requires == frozenset({Role.GATE})
    assert MemoryCellDynamics.requires == frozenset(
        {Role.RECURRENT_HIDDEN, Role.RECURRENT_CELL, Role.GATE}
    )
    assert RecurrentConnectionAnalysis.requires == frozenset({Role.RECURRENT_HIDDEN})


@pytest.mark.parametrize(
    ("published", "expected"),
    [
        (
            {HIDDEN: REAL},
            {HiddenStateEvolution, RecurrentConnectionAnalysis},
        ),
        (
            {HIDDEN: REAL, GATE: PROBABILITY},
            {GatePatterns, HiddenStateEvolution, RecurrentConnectionAnalysis},
        ),
        (
            {HIDDEN: REAL, CELL: REAL, GATE: PROBABILITY},
            RECURRENT_TYPES,
        ),
    ],
    ids=("vanilla", "gru", "lstm"),
)
def test_role_capabilities_resolve_only_the_recurrent_diagnostics_they_support(
    published: dict[Site, SiteSpec], expected: set[type] | frozenset[type]
) -> None:
    resolved = {
        type(diagnostic)
        for diagnostic in installed_diagnostics().matching(DiagnosticContext(published))
    }

    assert resolved & RECURRENT_TYPES == expected


def test_hidden_state_evolution_is_normalized_child_to_parent_displacement() -> None:
    values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]])
    result = result_for_hidden(HiddenStateEvolution(PARENTS), values)
    expected = ((1.0 / 3.0) + (1.0 / (1.0 + math.sqrt(2.0)))) / 2.0

    assert result.value(HIDDEN, "edges") == 2.0
    assert result.value(HIDDEN, "displacements") == 2.0
    assert result.value(HIDDEN, "normalized-displacement") == pytest.approx(expected)


def test_recurrent_connection_analysis_is_child_to_parent_activation_alignment() -> (
    None
):
    values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]])
    result = result_for_hidden(RecurrentConnectionAnalysis(PARENTS), values)

    assert result.value(HIDDEN, "edges") == 2.0
    assert result.value(HIDDEN, "alignments") == 2.0
    assert result.value(HIDDEN, "directional-alignment") == pytest.approx(
        (1.0 + 1.0 / math.sqrt(2.0)) / 2.0
    )


def test_both_endpoints_must_be_completely_live_for_an_edge_to_be_measured() -> None:
    values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]])
    mask = torch.tensor([[[True, True], [True, True], [True, False]]], dtype=torch.bool)
    evolution = result_for_hidden(HiddenStateEvolution(PARENTS), values, mask=mask)
    connection = result_for_hidden(
        RecurrentConnectionAnalysis(PARENTS), values, mask=mask
    )

    assert evolution.value(HIDDEN, "edges") == 1.0
    assert connection.value(HIDDEN, "edges") == 1.0
    assert evolution.value(HIDDEN, "normalized-displacement") == pytest.approx(
        1.0 / 3.0
    )
    assert connection.value(HIDDEN, "directional-alignment") == pytest.approx(1.0)


def test_zero_no_edge_and_all_masked_populations_report_only_counts() -> None:
    zeros = torch.zeros(1, 3, 2)
    evolution = result_for_hidden(HiddenStateEvolution(PARENTS), zeros)
    connection = result_for_hidden(RecurrentConnectionAnalysis(PARENTS), zeros)

    assert evolution.value(HIDDEN, "edges") == 2.0
    assert evolution.value(HIDDEN, "displacements") == 0.0
    assert connection.value(HIDDEN, "edges") == 2.0
    assert connection.value(HIDDEN, "alignments") == 0.0

    for result, name in (
        (evolution, "normalized-displacement"),
        (connection, "directional-alignment"),
    ):
        with pytest.raises(KeyError):
            result.value(HIDDEN, name)

    no_edges = result_for_hidden(
        HiddenStateEvolution((Encoding.NO_PARENT,)), torch.ones(1, 1, 2)
    )
    all_masked = result_for_hidden(
        RecurrentConnectionAnalysis(PARENTS),
        torch.ones(1, 3, 2),
        mask=torch.zeros(1, 3, dtype=torch.bool),
    )

    assert no_edges.value(HIDDEN, "edges") == 0.0
    assert no_edges.value(HIDDEN, "displacements") == 0.0
    assert all_masked.value(HIDDEN, "edges") == 0.0
    assert all_masked.value(HIDDEN, "alignments") == 0.0


@pytest.mark.parametrize("scale", [1e-6, 1e6, 4.0])
def test_hidden_displacement_and_alignment_are_positive_scale_invariant(
    scale: float,
) -> None:
    values = torch.tensor([[[1.0, 2.0], [3.0, 1.0], [-2.0, 4.0]]])

    for diagnostic, name in (
        (HiddenStateEvolution(PARENTS), "normalized-displacement"),
        (RecurrentConnectionAnalysis(PARENTS), "directional-alignment"),
    ):
        first = result_for_hidden(diagnostic, values).value(HIDDEN, name)
        scaled = result_for_hidden(diagnostic, values * scale).value(HIDDEN, name)

        assert scaled == pytest.approx(first, abs=1e-7)


def test_gate_patterns_measure_live_values_without_interpreting_site_keys() -> None:
    first = torch.tensor([[[0.0, 1.0], [8.0, 9.0]]])
    second = torch.tensor([[[-1.0, 2.0], [7.0, 6.0]]])
    mask = torch.tensor([[True, False]])
    context = DiagnosticContext({GATE: PROBABILITY, GATE_OTHER: REAL})
    result = GatePatterns().run(
        traced(
            observed(GATE_OTHER, second, mask=mask),
            observed(GATE, first, mask=mask),
        ),
        context=context,
    )

    assert result.sites == (GATE, GATE_OTHER)
    assert result.value(GATE, "values") == 2.0
    assert result.value(GATE, "mean") == pytest.approx(0.5)
    assert result.value(GATE, "variance") == pytest.approx(0.25)
    assert result.value(GATE_OTHER, "values") == 2.0
    assert result.value(GATE_OTHER, "mean") == pytest.approx(0.5)
    assert result.value(GATE_OTHER, "variance") == pytest.approx(2.25)


def test_an_all_masked_gate_reports_its_zero_count_and_no_moments() -> None:
    result = GatePatterns().run(
        traced(
            observed(
                GATE,
                torch.ones(1, 2, 3),
                mask=torch.zeros(1, 2, dtype=torch.bool),
            )
        ),
        context=DiagnosticContext({GATE: PROBABILITY}),
    )

    assert result.value(GATE, "values") == 0.0

    for name in ("mean", "variance"):
        with pytest.raises(KeyError):
            result.value(GATE, name)


def test_memory_cell_dynamics_reports_displacement_and_cosine_retention() -> None:
    cells = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]])
    context = DiagnosticContext({HIDDEN: REAL, CELL: REAL, GATE: PROBABILITY})
    result = MemoryCellDynamics(PARENTS).run(
        traced(observed(CELL, cells)), context=context
    )

    expected = ((1.0 / 3.0) + (math.sqrt(2.0) / 2.0)) / 2.0
    assert result.value(CELL, "edges") == 2.0
    assert result.value(CELL, "displacements") == 2.0
    assert result.value(CELL, "retentions") == 2.0
    assert result.value(CELL, "normalized-displacement") == pytest.approx(expected)
    assert result.value(CELL, "cosine-retention") == pytest.approx(0.5)

    scaled = MemoryCellDynamics(PARENTS).run(
        traced(observed(CELL, cells * 1e6)), context=context
    )
    assert scaled.value(CELL, "normalized-displacement") == pytest.approx(expected)
    assert scaled.value(CELL, "cosine-retention") == pytest.approx(0.5)


def test_undefined_memory_cell_populations_report_counts_without_summaries() -> None:
    context = DiagnosticContext({HIDDEN: REAL, CELL: REAL, GATE: PROBABILITY})
    result = MemoryCellDynamics(PARENTS).run(
        traced(observed(CELL, torch.zeros(1, 3, 2))), context=context
    )

    assert result.value(CELL, "edges") == 2.0
    assert result.value(CELL, "displacements") == 0.0
    assert result.value(CELL, "retentions") == 0.0

    for name in ("normalized-displacement", "cosine-retention"):
        with pytest.raises(KeyError):
            result.value(CELL, name)

    no_edges = MemoryCellDynamics((Encoding.NO_PARENT,)).run(
        traced(observed(CELL, torch.ones(1, 1, 2))), context=context
    )
    all_masked = MemoryCellDynamics(PARENTS).run(
        traced(
            observed(
                CELL,
                torch.ones(1, 3, 2),
                mask=torch.zeros(1, 3, dtype=torch.bool),
            )
        ),
        context=context,
    )

    for empty in (no_edges, all_masked):
        assert empty.value(CELL, "edges") == 0.0
        assert empty.value(CELL, "displacements") == 0.0
        assert empty.value(CELL, "retentions") == 0.0


def test_a_diagnostic_without_its_declared_capability_returns_no_error_record() -> None:
    trace = traced()
    vanilla = DiagnosticContext({HIDDEN: REAL})

    assert GatePatterns().run(trace, context=vanilla).measurements == ()
    assert MemoryCellDynamics(PARENTS).run(trace, context=vanilla).measurements == ()


def test_parent_structure_and_recurrent_observation_shapes_are_named_on_refusal() -> (
    None
):
    with pytest.raises(TypeError, match="parent of slot 1"):
        HiddenStateEvolution(cast("tuple[int, ...]", (Encoding.NO_PARENT, 0.5)))

    with pytest.raises(ValueError, match="own parent"):
        HiddenStateEvolution((Encoding.NO_PARENT, 1))

    with pytest.raises(ValueError, match="2 parents"):
        result_for_hidden(
            HiddenStateEvolution((Encoding.NO_PARENT, 0)), torch.ones(1, 3, 2)
        )

    with pytest.raises(ValueError, match="runs over"):
        HiddenStateEvolution(PARENTS).run(
            traced(
                observed(
                    HIDDEN,
                    torch.ones(1, 3, 2),
                    axes=("slot", "term", "width"),
                )
            ),
            context=DiagnosticContext({HIDDEN: REAL}),
        )


def test_a_real_gru_trace_feeds_diagnostics_only_through_published_observations() -> (
    None
):
    atom = Operator("a", "T", family="primitive")
    other = Operator("b", "T", family="primitive")
    pair = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
    signature = Signature("recurrent-probe", ("T",), (atom, other, pair))
    index = OperatorIndex(signature)
    layout = CompleteTree(signature, signature.operators, order="bf", depth=2)
    codec = TermCodec(index, layout)
    store = TermStore(signature)
    terms = codec.encode_batch((store.op(pair, store.op(atom), store.op(other)),))
    encoder = TreeRnnEncoder(
        codec=NeuralCodecView(codec),
        choices=ChoiceIndex(signature, depth_bounds={"T": layout.depth}),
        component=ComponentId("observed-gru"),
        scheme=ArityHeadScheme(width=4, reached=(pair,)),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )
    recurrent = frozenset(
        site
        for site in encoder.site_map()
        if site.role in {Role.RECURRENT_HIDDEN, Role.GATE}
    )

    with instrument(encoder, observe=recurrent) as region:
        region(lambda: encoder.encode(terms))

    context = DiagnosticContext.observing(encoder)
    parents = layout.skeleton.parents
    evolution = HiddenStateEvolution(parents).run(region.trace, context=context)
    gates = GatePatterns().run(region.trace, context=context)
    connection = RecurrentConnectionAnalysis(parents).run(region.trace, context=context)

    assert evolution.sites == context.sites_for(Role.RECURRENT_HIDDEN)
    assert gates.sites == context.sites_for(Role.GATE)
    assert connection.sites == context.sites_for(Role.RECURRENT_HIDDEN)
    assert all(evolution.value(site, "edges") == 2.0 for site in evolution.sites)
