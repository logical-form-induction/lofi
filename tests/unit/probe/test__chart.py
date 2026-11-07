from math import inf
from typing import Final, cast

import pytest
import torch

from lofi.deduce import SpanIndex
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._chart import CutpointEntropy, SpanLengthBias
from lofi.probe._diagnose import DiagnosticContext, DiagnosticResult
from lofi.probe._marginals import SpanWeighting
from lofi.probe._values import Observation, Trace

PARSER: Final = ComponentId("chart")
SPLITS: Final = Site(PARSER, Role.SPLIT_DISTRIBUTION, ("alpha",))
MARGINALS: Final = Site(PARSER, Role.CONSTITUENT_MARGINAL, ("mu",))
CELL: Final = Site(PARSER, Role.CHART_CELL, ("cells",))

SPLIT_SPEC: Final = SiteSpec(
    ("item", "span", "split"),
    DTypeClass.LOG_PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
MARGINAL_SPEC: Final = SiteSpec(
    ("item", "span"),
    DTypeClass.PROBABILITY,
    differentiable=True,
    access=Access.OBSERVE,
)
CELL_SPEC: Final = SiteSpec(
    ("item", "span", "width"),
    DTypeClass.REAL,
    differentiable=True,
    access=Access.OBSERVE,
)

INDEX: Final = SpanIndex(4)


def a_trace(*readings: Observation) -> Trace:
    """Return a closed trace holding the readings in order.

    Parameters
    ----------
    *readings : Observation
        The readings to record.
    """
    trace = Trace()

    for reading in readings:
        trace.record(reading)

    trace.close()

    return trace


def split_reading(
    *,
    lengths: tuple[int, ...] = (4,),
    rows: dict[tuple[int, int, int], tuple[float, ...]] | None = None,
    index: SpanIndex = INDEX,
) -> Observation:
    """Return exact split distributions for a possibly ragged batch.

    Parameters
    ----------
    lengths : tuple[int, ...]
        The token count of each item.
    rows : dict[tuple[int, int, int], tuple[float, ...]] | None
        Replacement probability rows keyed by item, start and end, or None.
    index : SpanIndex
        The shared chart addressing.
    """
    tensor = torch.full(
        (len(lengths), index.num_spans, index.max_splits),
        -inf,
        dtype=torch.float64,
    )
    mask = torch.zeros_like(tensor, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in index.spans_by_length(min_length=2):
            if end > length:
                continue

            span = index.span_id(start, end)
            choices = end - start - 1
            probabilities = (1.0 / choices,) * choices

            if rows is not None:
                probabilities = rows.get((item, start, end), probabilities)

            tensor[item, span, :choices] = torch.tensor(probabilities).log()
            mask[item, span, :choices] = True

    return Observation(SPLITS, tensor, axes=SPLIT_SPEC.axes, mask=mask)


def marginal_reading(
    weights: dict[tuple[int, int, int], float] | None = None,
    *,
    lengths: tuple[int, ...] = (4,),
    index: SpanIndex = INDEX,
) -> Observation:
    """Return exact constituent marginals for a possibly ragged batch.

    Parameters
    ----------
    weights : dict[tuple[int, int, int], float] | None
        Replacement marginals keyed by item, start and end, or None.
    lengths : tuple[int, ...]
        The token count of each item.
    index : SpanIndex
        The shared chart addressing.
    """
    tensor = torch.zeros(len(lengths), index.num_spans, dtype=torch.float64)
    mask = torch.zeros_like(tensor, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in index.spans_by_length():
            if end > length:
                continue

            span = index.span_id(start, end)
            default = 1.0 if end - start == 1 or (start, end) == (0, length) else 0.5
            tensor[item, span] = (
                default if weights is None else weights.get((item, start, end), default)
            )
            mask[item, span] = True

    return Observation(MARGINALS, tensor, axes=MARGINAL_SPEC.axes, mask=mask)


def split_context(
    weighting: torch.Tensor,
    *,
    index: SpanIndex = INDEX,
    spec: SiteSpec = SPLIT_SPEC,
) -> DiagnosticContext:
    """Return a split context under one span weighting.

    Parameters
    ----------
    weighting : torch.Tensor
        One constituent weight per item and span.
    index : SpanIndex
        The chart addressing.
    spec : SiteSpec
        What the split site publishes.
    """
    return DiagnosticContext(
        {SPLITS: spec},
        span_index=index,
        weighting=SpanWeighting(weighting, axes=("item", "span")),
    )


def marginal_context(
    *, index: SpanIndex = INDEX, spec: SiteSpec = MARGINAL_SPEC
) -> DiagnosticContext:
    """Return a constituent-marginal context.

    Parameters
    ----------
    index : SpanIndex
        The chart addressing.
    spec : SiteSpec
        What the marginal site publishes.
    """
    return DiagnosticContext({MARGINALS: spec}, span_index=index)


def aggregate(
    result: DiagnosticResult, site: Site, name: str
) -> tuple[float, float, int]:
    """Return both means and the count of one aggregate.

    Parameters
    ----------
    result : DiagnosticResult
        The result to read.
    site : Site
        The source site.
    name : str
        The aggregate's name.
    """
    found = result.aggregate(site, name)

    return found.weighted, found.uniform, found.count


def test_cutpoint_entropy_matches_uniform_and_peaked_rows_under_planted_weights() -> (
    None
):
    reading = split_reading(rows={(0, 0, 3): (0.5, 0.5), (0, 1, 4): (1.0, 0.0)})
    weights = torch.ones(1, INDEX.num_spans, dtype=torch.float64)
    weights[0, INDEX.span_id(0, 3)] = 0.25
    weights[0, INDEX.span_id(1, 4)] = 0.75
    result = CutpointEntropy().run(a_trace(reading), context=split_context(weights))

    assert aggregate(result, SPLITS, "normalized-cutpoint-entropy") == pytest.approx(
        (0.25, 0.5, 2)
    )
    assert result.value(SPLITS, "spans") == 2.0


def test_cutpoint_entropy_excludes_forced_rows_and_the_item_root() -> None:
    weights = torch.ones(1, INDEX.num_spans, dtype=torch.float64)
    weights[0, INDEX.span_id(0, 2)] = 1e6
    weights[0, INDEX.span_id(0, 4)] = 1e6
    first = split_reading(
        rows={
            (0, 0, 2): (1.0,),
            (0, 0, 3): (0.5, 0.5),
            (0, 1, 4): (1.0, 0.0),
            (0, 0, 4): (1.0 / 3.0,) * 3,
        }
    )
    second = split_reading(
        rows={
            (0, 0, 2): (1.0,),
            (0, 0, 3): (0.5, 0.5),
            (0, 1, 4): (1.0, 0.0),
            (0, 0, 4): (1.0, 0.0, 0.0),
        }
    )
    context = split_context(weights)
    before = CutpointEntropy().run(a_trace(first), context=context)
    after = CutpointEntropy().run(a_trace(second), context=context)

    assert before == after


def test_cutpoint_entropy_obeys_ragged_split_masks() -> None:
    reading = split_reading(
        lengths=(4, 3),
        rows={
            (0, 0, 3): (0.5, 0.5),
            (0, 1, 4): (1.0, 0.0),
            (1, 0, 3): (0.5, 0.5),
        },
    )
    weights = torch.ones(2, INDEX.num_spans, dtype=torch.float64)
    weights[1, INDEX.span_id(0, 3)] = 1e6
    result = CutpointEntropy().run(a_trace(reading), context=split_context(weights))

    assert aggregate(result, SPLITS, "normalized-cutpoint-entropy") == pytest.approx(
        (0.5, 0.5, 2)
    )


def test_span_length_bias_uses_width_over_item_length() -> None:
    weights = {
        (0, 0, 2): 0.1,
        (0, 1, 3): 0.1,
        (0, 2, 4): 0.1,
        (0, 0, 3): 0.6,
        (0, 1, 4): 0.6,
    }
    result = SpanLengthBias().run(
        a_trace(marginal_reading(weights)), context=marginal_context()
    )

    assert aggregate(result, MARGINALS, "normalized-span-length") == pytest.approx(
        (0.7, 0.6, 5)
    )
    assert result.value(MARGINALS, "spans") == 5.0


def test_moving_marginal_mass_from_short_to_long_changes_only_the_weighted_length() -> (
    None
):
    short = {
        (0, 0, 2): 0.8,
        (0, 1, 3): 0.8,
        (0, 2, 4): 0.8,
        (0, 0, 3): 0.2,
        (0, 1, 4): 0.2,
    }
    long = {
        (0, 0, 2): 0.2,
        (0, 1, 3): 0.2,
        (0, 2, 4): 0.2,
        (0, 0, 3): 0.8,
        (0, 1, 4): 0.8,
    }
    context = marginal_context()
    first = (
        SpanLengthBias()
        .run(a_trace(marginal_reading(short)), context=context)
        .aggregate(MARGINALS, "normalized-span-length")
    )
    second = (
        SpanLengthBias()
        .run(a_trace(marginal_reading(long)), context=context)
        .aggregate(MARGINALS, "normalized-span-length")
    )

    assert first.weighted < second.weighted
    assert first.uniform == pytest.approx(second.uniform)
    assert first.count == second.count


def test_span_length_bias_excludes_leaves_roots_and_ragged_padding() -> None:
    reading = marginal_reading(lengths=(4, 3))
    result = SpanLengthBias().run(a_trace(reading), context=marginal_context())
    changed = reading.tensor.clone()

    for item, length in enumerate((4, 3)):
        changed[item, :length] = 0.1
        changed[item, INDEX.span_id(0, length)] = 0.1

    same = SpanLengthBias().run(
        a_trace(
            Observation(
                MARGINALS,
                changed,
                axes=MARGINAL_SPEC.axes,
                mask=reading.mask,
            )
        ),
        context=marginal_context(),
    )

    assert result == same
    assert aggregate(result, MARGINALS, "normalized-span-length") == pytest.approx(
        (13.0 / 21.0, 13.0 / 21.0, 7)
    )


@pytest.mark.parametrize(
    ("diagnostic", "site", "spec"),
    [
        (CutpointEntropy(), SPLITS, SPLIT_SPEC),
        (SpanLengthBias(), MARGINALS, MARGINAL_SPEC),
    ],
)
def test_a_published_chart_site_that_was_not_read_is_named(
    diagnostic: CutpointEntropy | SpanLengthBias,
    site: Site,
    spec: SiteSpec,
) -> None:
    context = DiagnosticContext({site: spec}, span_index=INDEX)

    with pytest.raises(ValueError, match=site.role.name):
        diagnostic.run(a_trace(), context=context)


def test_chart_diagnostics_require_caller_supplied_span_addressing() -> None:
    split_weights = torch.ones(1, INDEX.num_spans)
    split_context_without = DiagnosticContext(
        {SPLITS: SPLIT_SPEC},
        weighting=SpanWeighting(split_weights, axes=("item", "span")),
    )

    with pytest.raises(ValueError, match="no SpanIndex"):
        CutpointEntropy().run(a_trace(split_reading()), context=split_context_without)

    with pytest.raises(ValueError, match="no SpanIndex"):
        SpanLengthBias().run(
            a_trace(marginal_reading()),
            context=DiagnosticContext({MARGINALS: MARGINAL_SPEC}),
        )


def test_an_index_that_does_not_address_the_observation_is_named() -> None:
    other = SpanIndex(3)
    weights = torch.ones(1, INDEX.num_spans)

    with pytest.raises(ValueError, match=r"SpanIndex\(3\).*addresses"):
        CutpointEntropy().run(
            a_trace(split_reading()), context=split_context(weights, index=other)
        )

    with pytest.raises(ValueError, match=r"SpanIndex\(3\).*addresses"):
        SpanLengthBias().run(
            a_trace(marginal_reading()), context=marginal_context(index=other)
        )


def test_wrong_chart_axes_and_numeric_declarations_are_named() -> None:
    wrong_axes = Observation(
        SPLITS,
        split_reading().tensor,
        axes=("item", "cell", "cut"),
        mask=split_reading().mask,
    )
    weights = torch.ones(1, INDEX.num_spans)
    real_spec = SiteSpec(
        SPLIT_SPEC.axes,
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE,
    )

    with pytest.raises(ValueError, match=r"item.*span.*split"):
        CutpointEntropy().run(a_trace(wrong_axes), context=split_context(weights))

    with pytest.raises(ValueError, match="log-probability"):
        CutpointEntropy().run(
            a_trace(split_reading()),
            context=split_context(weights, spec=real_spec),
        )


def test_malformed_chart_masks_are_refused_by_source_site() -> None:
    splits = split_reading()
    split_values = splits.tensor.clone()
    split_mask = cast("torch.Tensor", splits.mask).clone()
    split_values[0, INDEX.span_id(0, 3), 1] = -inf
    split_mask[0, INDEX.span_id(0, 3), 1] = False
    weights = torch.ones(1, INDEX.num_spans)

    with pytest.raises(ValueError, match=r"split mask.*SPLIT_DISTRIBUTION"):
        CutpointEntropy().run(
            a_trace(
                Observation(
                    SPLITS,
                    split_values,
                    axes=SPLIT_SPEC.axes,
                    mask=split_mask,
                )
            ),
            context=split_context(weights),
        )

    marginals = marginal_reading()
    marginal_mask = cast("torch.Tensor", marginals.mask).clone()
    marginal_mask[0, INDEX.span_id(1, 3)] = False

    with pytest.raises(ValueError, match=r"span mask.*CONSTITUENT_MARGINAL"):
        SpanLengthBias().run(
            a_trace(
                Observation(
                    MARGINALS,
                    marginals.tensor.masked_fill(~marginal_mask, 0.0),
                    axes=MARGINAL_SPEC.axes,
                    mask=marginal_mask,
                )
            ),
            context=marginal_context(),
        )


def test_empty_eligible_support_and_zero_eligible_mass_are_refused() -> None:
    split_values = torch.full(
        (1, INDEX.num_spans, INDEX.max_splits), -inf, dtype=torch.float64
    )
    split_mask = torch.zeros_like(split_values, dtype=torch.bool)
    weights = torch.ones(1, INDEX.num_spans)

    with pytest.raises(ValueError, match="every span is masked off"):
        CutpointEntropy().run(
            a_trace(
                Observation(
                    SPLITS,
                    split_values,
                    axes=SPLIT_SPEC.axes,
                    mask=split_mask,
                )
            ),
            context=split_context(weights),
        )

    no_mass = marginal_reading(
        {
            (0, 0, 2): 0.0,
            (0, 1, 3): 0.0,
            (0, 2, 4): 0.0,
            (0, 0, 3): 0.0,
            (0, 1, 4): 0.0,
        }
    )

    with pytest.raises(ValueError, match="no mass on any live span"):
        SpanLengthBias().run(a_trace(no_mass), context=marginal_context())


def test_invalid_marginal_values_are_refused_by_source_site() -> None:
    reading = marginal_reading()
    values = reading.tensor.clone()
    values[0, INDEX.span_id(0, 2)] = -0.1

    with pytest.raises(ValueError, match=r"CONSTITUENT_MARGINAL.*unit interval"):
        SpanLengthBias().run(
            a_trace(
                Observation(
                    MARGINALS,
                    values,
                    axes=MARGINAL_SPEC.axes,
                    mask=reading.mask,
                )
            ),
            context=marginal_context(),
        )


def test_models_without_chart_roles_produce_no_chart_measurement() -> None:
    context = DiagnosticContext({CELL: CELL_SPEC}, span_index=INDEX)
    cell = Observation(CELL, torch.zeros(1, INDEX.num_spans, 2), axes=CELL_SPEC.axes)

    assert CutpointEntropy().run(a_trace(cell), context=context).sites == ()
    assert SpanLengthBias().run(a_trace(cell), context=context).sites == ()


def test_chart_diagnostics_declare_their_roles_scores_tags_and_representations() -> (
    None
):
    assert CutpointEntropy.tag == "cutpoint-entropy"
    assert CutpointEntropy.requires == frozenset({Role.SPLIT_DISTRIBUTION})
    assert CutpointEntropy.scores == frozenset({"normalized-cutpoint-entropy"})
    assert repr(CutpointEntropy()) == "CutpointEntropy()"
    assert SpanLengthBias.tag == "span-length-bias"
    assert SpanLengthBias.requires == frozenset({Role.CONSTITUENT_MARGINAL})
    assert SpanLengthBias.scores == frozenset({"normalized-span-length"})
    assert repr(SpanLengthBias()) == "SpanLengthBias()"
    assert installed_diagnostics().kind("cutpoint-entropy") is CutpointEntropy
    assert installed_diagnostics().kind("span-length-bias") is SpanLengthBias
