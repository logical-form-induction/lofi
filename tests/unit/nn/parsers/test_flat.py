from importlib.metadata import EntryPoint
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.chart_scorer import (
    ChartScorerConformance,
    ChartScorerSuite,
    published,
)
from tests.conformance.component_plugin import plugin_context

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded
from lofi.nn.build import build_component
from lofi.nn.parsers import flat
from lofi.nn.parsers.flat import FlatSpanEncoder
from lofi.nn.policy import DevicePolicy
from lofi.nn.ports import ChartScorer, TermEncoder
from lofi.nn.sites import Access, ComponentId, ComponentKind, DTypeClass, Role, Site
from lofi.probe import ensure_publishes
from lofi.registry import entry_point_group, plugin_for
from lofi.spec.parsers.flat import (
    FlatParserSpec,
    FlatProjectionInitialization,
    FlatSpanFeature,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.nn._values import LogDistribution
    from lofi.nn.ports import SubwordPooler

INPUT_WIDTH: Final = 4
WIDTH: Final = 4
ITEMS: Final = 2
WORDS: Final = 5
SEED: Final = 11
SPAN: Final = ComponentId("span-encoder")

LEAF_AXES: Final = ("item", "word", "width")
CELL_AXES: Final = ("item", "span", "width")

AXES: Final[Mapping[str, tuple[str, ...]]] = {
    "inside": ("item", "span"),
    "outside": ("item", "span"),
    "alpha": ("item", "span", "split"),
    "mu": ("item", "span"),
    "log_z": ("item",),
    "identity": ("item", "span", "split"),
    "cells": CELL_AXES,
}

TOLERANCE: Final = 1e-10

FAMILY: Final = "chart"
TAG: Final = "flat"
ENTRY_POINT: Final = "lofi.nn.parsers.flat:PLUGIN"
FEATURES: Final[tuple[FlatSpanFeature, ...]] = ("left", "right", "bilinear")
INITIALIZATION: Final[FlatProjectionInitialization] = "identity"


def build(
    *,
    features: tuple[FlatSpanFeature, ...] = FEATURES,
    projection_initialization: FlatProjectionInitialization = INITIALIZATION,
    input_width: int = INPUT_WIDTH,
    width: int = WIDTH,
    seed: int = SEED,
    dtype: torch.dtype = torch.float32,
) -> FlatSpanEncoder:
    torch.manual_seed(seed)

    return FlatSpanEncoder(
        input_width=input_width,
        width=width,
        component=SPAN,
        features=features,
        projection_initialization=projection_initialization,
    ).to(dtype)


def lengths(items: int, words: int) -> list[int]:
    return [max(1, words - item) for item in range(items)]


def leaves(
    items: int,
    words: int,
    *,
    seed: int = 3,
    dtype: torch.dtype = torch.float32,
    width: int = INPUT_WIDTH,
) -> Encoded:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(items, words, width, generator=generator, dtype=dtype)
    mask = torch.zeros(items, words, dtype=torch.bool)

    for item, length in enumerate(lengths(items, words)):
        mask[item, :length] = True

    return Encoded(values, LEAF_AXES, mask)


def wire(encoder: FlatSpanEncoder, readout: str) -> None:
    if readout not in encoder.publishes:
        msg = f"no pass fills {readout!r}"
        raise ValueError(msg)


def observe(encoder: FlatSpanEncoder) -> Mapping[Site, LogDistribution | Encoded]:
    filled = published(encoder.readouts(leaves(ITEMS, WORDS)))

    return {
        Site(SPAN, Role.CHART_CELL, (name,)): value
        for name, value in filled.items()
        if name == "cells"
    }


def read(encoder: FlatSpanEncoder) -> tuple[Tensor, ...]:
    return tuple(
        value.tensor
        for value in published(encoder.readouts(leaves(ITEMS, WORDS))).values()
    )


def cells_of(encoder: FlatSpanEncoder, words: Encoded) -> Tensor:
    cells = encoder.readouts(words).cells

    assert cells is not None

    return cells.tensor


def projected(encoder: FlatSpanEncoder, word: Tensor) -> Tensor:
    return encoder.get_parameter("_unary.weight") @ word + encoder.get_parameter(
        "_unary.bias"
    )


def bilinear(encoder: FlatSpanEncoder, left: Tensor, right: Tensor) -> Tensor:
    weight = encoder.get_parameter("_pair.weight")

    return torch.stack(
        [left @ weight[row] @ right for row in range(int(weight.shape[0]))]
    ) + encoder.get_parameter("_pair.bias")


def detune(encoder: FlatSpanEncoder, *, seed: int = 17) -> FlatSpanEncoder:
    generator = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for name in ("_unary.weight", "_unary.bias"):
            parameter = encoder.get_parameter(name)
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )

    return encoder


class _Pooled:
    def __init__(self, width: int) -> None:
        self._width = width

    @property
    def output_width(self) -> int:
        return self._width


def pooled(width: int) -> SubwordPooler:
    return cast("SubwordPooler", _Pooled(width))


class TestTheFlatSpanEncoder(ChartScorerSuite):
    @pytest.fixture
    def conformance(self) -> ChartScorerConformance[FlatSpanEncoder]:
        return ChartScorerConformance(
            build=build,
            leaves=leaves,
            axes=AXES,
            split_axis="split",
            ragged_axis="span",
            wire=wire,
            sites=SiteObservation(build=build, observe=observe, read=read),
        )


def test_the_encoder_is_a_chart_scorer() -> None:
    encoder = build()
    scorer: ChartScorer = encoder
    filled = published(scorer.readouts(leaves(ITEMS, WORDS)))

    assert isinstance(encoder, ChartScorer)
    assert encoder.output_width == encoder.width
    assert frozenset(filled) == scorer.publishes
    assert {str(site.key[-1]) for site in scorer.site_map()} <= scorer.publishes
    assert all(
        filled[str(site.key[-1])].axes == spec.axes
        for site, spec in scorer.site_map().items()
    )


def test_the_encoder_publishes_uniform_inside_weights_and_chart_cells() -> None:
    encoder = build()
    readouts = encoder.readouts(leaves(ITEMS, WORDS))
    inside = readouts.inside

    assert encoder.publishes == frozenset({"inside", "cells"})
    assert set(published(readouts)) == {"inside", "cells"}
    assert inside is not None
    assert readouts.cells is not None
    assert inside.mask is not None
    assert bool(inside.tensor[inside.mask].eq(0.0).all())
    assert bool(inside.tensor[~inside.mask].eq(-torch.inf).all())


def test_a_cell_is_the_bilinear_form_of_its_two_projected_endpoints() -> None:
    encoder = detune(build(dtype=torch.float64))
    words = leaves(ITEMS, WORDS, dtype=torch.float64)

    assert not torch.allclose(
        projected(encoder, words.tensor[0, 0]),
        words.tensor[0, 0, :WIDTH],
        atol=TOLERANCE,
    )

    cells = cells_of(encoder, words)
    index = SpanIndex(WORDS)
    checked = 0

    for item, length in enumerate(lengths(ITEMS, WORDS)):
        for start, stop in index.spans_by_length():
            if stop > length:
                continue

            left = projected(encoder, words.tensor[item, start])
            right = projected(encoder, words.tensor[item, stop - 1])
            expected = bilinear(encoder, left, right) + left + right
            checked += 1

            assert torch.allclose(
                cells[item, index.span_id(start, stop)], expected, atol=TOLERANCE
            )

    assert checked == sum(
        length * (length + 1) // 2 for length in lengths(ITEMS, WORDS)
    )


def test_a_word_strictly_inside_a_span_does_not_reach_its_cell() -> None:
    encoder = detune(build(dtype=torch.float64))
    words = leaves(1, WORDS, dtype=torch.float64)
    before = cells_of(encoder, words)
    moved = words.tensor.clone()
    moved[0, 2] = moved[0, 2] + 1.0
    after = cells_of(encoder, Encoded(moved, LEAF_AXES, words.mask))
    index = SpanIndex(WORDS)
    interior = [
        index.span_id(start, stop)
        for start, stop in index.spans_by_length()
        if start < 2 < stop - 1
    ]
    endpoints = [
        index.span_id(start, stop)
        for start, stop in index.spans_by_length()
        if start == 2 or stop - 1 == 2
    ]

    assert interior
    assert endpoints
    assert torch.equal(before[0, interior], after[0, interior])

    for span in endpoints:
        assert not torch.allclose(before[0, span], after[0, span], atol=TOLERANCE)


def test_two_items_sharing_their_endpoints_share_the_cell_over_them() -> None:
    encoder = build(dtype=torch.float64)
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(
        2, WORDS, INPUT_WIDTH, generator=generator, dtype=torch.float64
    )
    values[1, 0] = values[0, 0]
    values[1, WORDS - 1] = values[0, WORDS - 1]
    cells = cells_of(encoder, Encoded(values, LEAF_AXES))
    index = SpanIndex(WORDS)
    whole = index.span_id(0, WORDS)
    shorter = index.span_id(0, 3)

    assert torch.equal(cells[0, whole], cells[1, whole])
    assert not torch.allclose(cells[0, shorter], cells[1, shorter], atol=TOLERANCE)


def test_the_projection_starts_as_the_identity() -> None:
    encoder = build(dtype=torch.float64)

    assert torch.equal(
        encoder.get_parameter("_unary.weight"),
        torch.eye(WIDTH, INPUT_WIDTH, dtype=torch.float64),
    )
    assert torch.equal(
        encoder.get_parameter("_unary.bias"), torch.zeros(WIDTH, dtype=torch.float64)
    )

    with torch.no_grad():
        encoder.get_parameter("_pair.weight").zero_()
        encoder.get_parameter("_pair.bias").zero_()

    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    cells = cells_of(encoder, words)
    index = SpanIndex(WORDS)

    for item, length in enumerate(lengths(ITEMS, WORDS)):
        for start, stop in index.spans_by_length():
            if stop > length:
                continue

            assert torch.allclose(
                cells[item, index.span_id(start, stop)],
                words.tensor[item, start] + words.tensor[item, stop - 1],
                atol=TOLERANCE,
            )


def test_a_projection_onto_fewer_features_keeps_the_leading_ones() -> None:
    wide = INPUT_WIDTH + 2
    encoder = build(input_width=wide, width=WIDTH, dtype=torch.float64)

    with torch.no_grad():
        encoder.get_parameter("_pair.weight").zero_()
        encoder.get_parameter("_pair.bias").zero_()

    words = leaves(1, WORDS, dtype=torch.float64, width=wide)
    cells = cells_of(encoder, words)
    index = SpanIndex(WORDS)

    for word in range(WORDS):
        assert torch.allclose(
            cells[0, index.span_id(word, word + 1)],
            2.0 * words.tensor[0, word, :WIDTH],
            atol=TOLERANCE,
        )


def test_a_projection_onto_more_features_leaves_the_added_ones_empty() -> None:
    tall = INPUT_WIDTH + 2
    encoder = build(input_width=INPUT_WIDTH, width=tall, dtype=torch.float64)

    with torch.no_grad():
        encoder.get_parameter("_pair.weight").zero_()
        encoder.get_parameter("_pair.bias").zero_()

    words = leaves(1, WORDS, dtype=torch.float64)
    cells = cells_of(encoder, words)
    index = SpanIndex(WORDS)

    assert cells.shape == (1, index.num_spans, tall)

    for word in range(WORDS):
        cell = cells[0, index.span_id(word, word + 1)]

        assert torch.allclose(
            cell[:INPUT_WIDTH], 2.0 * words.tensor[0, word], atol=TOLERANCE
        )
        assert torch.allclose(
            cell[INPUT_WIDTH:], torch.zeros(tall - INPUT_WIDTH, dtype=cell.dtype)
        )


def test_the_cell_in_a_column_belongs_to_the_span_that_column_names() -> None:
    encoder = build(input_width=WORDS, width=WORDS, dtype=torch.float64)

    with torch.no_grad():
        encoder.get_parameter("_pair.weight").zero_()
        encoder.get_parameter("_pair.bias").zero_()

    words = Encoded(torch.eye(WORDS, dtype=torch.float64).unsqueeze(0), LEAF_AXES)
    cells = cells_of(encoder, words)
    index = SpanIndex(WORDS)

    assert cells.shape[1] == index.num_spans

    for column in range(index.num_spans):
        start, stop = index.span_of(column)
        expected = torch.zeros(WORDS, dtype=torch.float64)
        expected[start] += 1.0
        expected[stop - 1] += 1.0

        assert torch.equal(cells[0, column], expected), (column, start, stop)


def test_declared_endpoint_features_are_summed_into_the_cell() -> None:
    with_endpoints = detune(build(dtype=torch.float64))
    without = detune(build(features=("bilinear",), dtype=torch.float64))
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    difference = cells_of(with_endpoints, words) - cells_of(without, words)
    index = SpanIndex(WORDS)

    assert without.features == ("bilinear",)
    assert with_endpoints.features == FEATURES

    for item, length in enumerate(lengths(ITEMS, WORDS)):
        for start, stop in index.spans_by_length():
            if stop > length:
                continue

            left = projected(with_endpoints, words.tensor[item, start])
            right = projected(with_endpoints, words.tensor[item, stop - 1])

            assert torch.allclose(
                difference[item, index.span_id(start, stop)],
                left + right,
                atol=TOLERANCE,
            )


def test_endpoint_features_use_the_projection_and_not_the_raw_leaf() -> None:
    encoder = detune(build(dtype=torch.float64))
    words = leaves(1, WORDS, dtype=torch.float64)
    cells = cells_of(encoder, words)
    index = SpanIndex(WORDS)
    span = index.span_id(0, WORDS)
    left = projected(encoder, words.tensor[0, 0])
    right = projected(encoder, words.tensor[0, WORDS - 1])
    raw_left = words.tensor[0, 0, :WIDTH]
    raw_right = words.tensor[0, WORDS - 1, :WIDTH]
    paired = bilinear(encoder, left, right)

    assert not torch.allclose(left, raw_left, atol=TOLERANCE)
    assert torch.allclose(cells[0, span], paired + left + right, atol=TOLERANCE)
    assert not torch.allclose(
        cells[0, span], paired + raw_left + raw_right, atol=TOLERANCE
    )


def test_the_feature_declaration_is_the_only_source_of_cell_contributions() -> None:
    encoder = detune(
        FlatSpanEncoder(
            input_width=INPUT_WIDTH,
            width=WIDTH,
            component=SPAN,
            features=FEATURES,
            projection_initialization=INITIALIZATION,
        ).to(torch.float64)
    )
    words = leaves(1, WORDS, dtype=torch.float64)
    cells = cells_of(encoder, words)
    span = SpanIndex(WORDS).span_id(0, WORDS)
    left = projected(encoder, words.tensor[0, 0])
    right = projected(encoder, words.tensor[0, WORDS - 1])
    paired = bilinear(encoder, left, right)

    assert encoder.features == FEATURES
    assert f"features={FEATURES!r}" in repr(encoder)
    assert torch.allclose(cells[0, span], paired + left + right, atol=TOLERANCE)
    assert not torch.allclose(cells[0, span], paired, atol=TOLERANCE)


def test_every_registered_parameter_receives_a_gradient() -> None:
    encoder = build(dtype=torch.float64)
    named = dict(encoder.named_parameters())

    assert set(dict(encoder.named_modules())) == {"", "_unary", "_pair"}
    assert not list(encoder.buffers())
    assert set(named) == {
        "_unary.weight",
        "_unary.bias",
        "_pair.weight",
        "_pair.bias",
    }

    cells_of(encoder, leaves(ITEMS, WORDS, dtype=torch.float64)).sum().backward()

    for name, parameter in named.items():
        assert parameter.grad is not None, name
        assert float(parameter.grad.abs().sum()) > 0.0, name


def test_a_word_in_the_padding_reaches_neither_a_cell_nor_a_gradient() -> None:
    encoder = build(dtype=torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)

    assert words.mask is not None
    assert bool((~words.mask).any())

    values = words.tensor.clone()
    values[~words.mask] = float("nan")
    cells = cells_of(encoder, Encoded(values, LEAF_AXES, words.mask))

    assert bool(cells.isfinite().all())

    cells.sum().backward()

    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert bool(parameter.grad.isfinite().all()), name


def test_a_span_reaching_past_an_item_is_masked_off_and_left_at_zero() -> None:
    encoder = build(dtype=torch.float64)
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = encoder.readouts(words)
    cells = readouts.cells
    index = SpanIndex(WORDS)

    assert cells is not None
    assert cells.mask is not None
    assert cells.axes == CELL_AXES

    expected = torch.zeros(ITEMS, index.num_spans, dtype=torch.bool)

    for item, length in enumerate(lengths(ITEMS, WORDS)):
        for start, stop in index.spans_by_length():
            expected[item, index.span_id(start, stop)] = stop <= length

    assert torch.equal(cells.mask, expected)
    assert bool(cells.tensor[~expected].eq(0.0).all())
    assert bool(cells.tensor[expected].ne(0.0).any())


def test_a_hole_in_the_middle_of_an_item_kills_every_span_that_covers_it() -> None:
    encoder = detune(build(dtype=torch.float64))
    generator = torch.Generator().manual_seed(23)
    values = torch.randn(
        1, WORDS, INPUT_WIDTH, generator=generator, dtype=torch.float64
    )
    mask = torch.ones(1, WORDS, dtype=torch.bool)
    mask[0, 2] = False
    readouts = encoder.readouts(Encoded(values, LEAF_AXES, mask))
    cells = readouts.cells
    index = SpanIndex(WORDS)

    assert cells is not None
    assert cells.mask is not None

    expected = torch.zeros(1, index.num_spans, dtype=torch.bool)

    for start, stop in index.spans_by_length():
        expected[0, index.span_id(start, stop)] = not start <= 2 < stop

    assert bool((~expected).any())
    assert bool(expected.any())
    assert torch.equal(cells.mask, expected)
    assert bool(cells.tensor[~expected].eq(0.0).all())

    for start, stop in index.spans_by_length():
        if start <= 2 < stop:
            continue

        left = projected(encoder, values[0, start])
        right = projected(encoder, values[0, stop - 1])

        assert torch.allclose(
            cells.tensor[0, index.span_id(start, stop)],
            bilinear(encoder, left, right) + left + right,
            atol=TOLERANCE,
        )


def test_a_hole_leaves_the_spans_on_either_side_of_it_untouched() -> None:
    encoder = detune(build(dtype=torch.float64))
    generator = torch.Generator().manual_seed(29)
    values = torch.randn(
        1, WORDS, INPUT_WIDTH, generator=generator, dtype=torch.float64
    )
    mask = torch.ones(1, WORDS, dtype=torch.bool)
    mask[0, 2] = False
    whole = cells_of(encoder, Encoded(values, LEAF_AXES))
    holed = cells_of(encoder, Encoded(values, LEAF_AXES, mask))
    index = SpanIndex(WORDS)
    clear = [
        index.span_id(start, stop)
        for start, stop in index.spans_by_length()
        if not start <= 2 < stop
    ]
    covering = [
        index.span_id(start, stop)
        for start, stop in index.spans_by_length()
        if start <= 2 < stop
    ]

    assert clear
    assert covering
    assert bool(whole[0, clear].ne(0.0).any())
    assert torch.equal(whole[0, clear], holed[0, clear])
    assert bool(whole[0, covering].ne(0.0).any())
    assert bool(holed[0, covering].eq(0.0).all())


def test_an_unmasked_batch_leaves_every_span_live() -> None:
    encoder = detune(build(dtype=torch.float64))
    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    readouts = encoder.readouts(Encoded(words.tensor, LEAF_AXES))
    cells = readouts.cells
    index = SpanIndex(WORDS)

    assert cells is not None
    assert cells.mask is not None
    assert bool(cells.mask.all())
    assert cells.tensor.shape == (ITEMS, index.num_spans, WIDTH)

    for item in range(ITEMS):
        for start, stop in index.spans_by_length():
            left = projected(encoder, words.tensor[item, start])
            right = projected(encoder, words.tensor[item, stop - 1])

            assert torch.allclose(
                cells.tensor[item, index.span_id(start, stop)],
                bilinear(encoder, left, right) + left + right,
                atol=TOLERANCE,
            )


def test_the_site_map_names_the_trainable_chart_cell() -> None:
    published_sites = build().site_map()
    cell_site = Site(SPAN, Role.CHART_CELL, ("cells",))

    assert published_sites.component == SPAN
    assert published_sites.kind is ComponentKind.PARSER
    assert ensure_publishes(published_sites) is published_sites
    assert set(published_sites) == {cell_site}
    assert published_sites[cell_site].axes == CELL_AXES
    assert published_sites[cell_site].dtype_class is DTypeClass.REAL
    assert published_sites[cell_site].access is Access.OBSERVE
    assert published_sites[cell_site].differentiable


def test_leaves_over_other_axes_are_refused() -> None:
    encoder = build()
    values = torch.zeros(ITEMS, WORDS, INPUT_WIDTH)

    with pytest.raises(ValueError, match="pooled leaves run over"):
        encoder.readouts(Encoded(values, ("item", "slot", "width")))


def test_leaves_at_another_width_are_refused() -> None:
    encoder = build()
    values = torch.zeros(ITEMS, WORDS, INPUT_WIDTH + 1)

    with pytest.raises(ValueError, match="reads leaves"):
        encoder.readouts(Encoded(values, LEAF_AXES))


def test_a_batch_holding_no_word_is_refused() -> None:
    encoder = build()
    values = torch.zeros(ITEMS, 0, INPUT_WIDTH)

    with pytest.raises(ValueError, match="at least one word"):
        encoder.readouts(Encoded(values, LEAF_AXES))


def test_the_representation_names_the_widths_features_and_component() -> None:
    shown = repr(build(features=("left",)))

    assert f"input_width={INPUT_WIDTH}" in shown
    assert f"width={WIDTH}" in shown
    assert "features=('left',)" in shown
    assert "projection_initialization='identity'" in shown
    assert repr(SPAN) in shown


def plugin_built(
    *,
    features: tuple[FlatSpanFeature, ...],
    leaf_width: int,
    projection_initialization: FlatProjectionInitialization = INITIALIZATION,
) -> FlatSpanEncoder:
    """Return the ablation built through its plugin over a pooler of one width."""
    return cast(
        "FlatSpanEncoder",
        build_component(
            FlatParserSpec(
                width=WIDTH,
                features=features,
                projection_initialization=projection_initialization,
            ),
            plugin_context(
                component=SPAN,
                producers={"leaves": pooled(leaf_width)},
                policy=DevicePolicy(torch.device("cpu"), torch.float64),
            ),
        ),
    )


def test_the_plugin_builds_the_ablation_at_the_width_its_pooler_returns() -> None:
    built = plugin_built(features=("bilinear",), leaf_width=INPUT_WIDTH + 3)

    assert type(built) is FlatSpanEncoder
    assert isinstance(built, ChartScorer)
    assert built.input_width == INPUT_WIDTH + 3
    assert built.width == WIDTH
    assert built.features == ("bilinear",)
    assert repr(SPAN) in repr(built)


def test_the_plugin_carries_the_features_the_declaration_states() -> None:
    made: dict[tuple[FlatSpanFeature, ...], FlatSpanEncoder] = {}

    for features in (FEATURES, ("bilinear",)):
        torch.manual_seed(SEED)
        made[features] = plugin_built(features=features, leaf_width=INPUT_WIDTH)

    words = leaves(ITEMS, WORDS, dtype=torch.float64)
    difference = cells_of(made[FEATURES], words) - cells_of(made[("bilinear",)], words)
    span = SpanIndex(WORDS).span_id(0, 2)

    assert made[FEATURES].features == FEATURES
    assert made[("bilinear",)].features == ("bilinear",)
    assert torch.allclose(
        difference[0, span],
        projected(made[FEATURES], words.tensor[0, 0])
        + projected(made[FEATURES], words.tensor[0, 1]),
        atol=TOLERANCE,
    )


def test_the_plugin_carries_the_projection_initialization() -> None:
    identity = plugin_built(features=("left",), leaf_width=INPUT_WIDTH)
    xavier = plugin_built(
        features=("left",),
        leaf_width=INPUT_WIDTH,
        projection_initialization="xavier-uniform",
    )

    assert identity.projection_initialization == "identity"
    assert xavier.projection_initialization == "xavier-uniform"
    assert torch.equal(
        identity.get_parameter("_unary.weight"), torch.eye(WIDTH, INPUT_WIDTH)
    )
    assert not torch.equal(
        xavier.get_parameter("_unary.weight"), torch.eye(WIDTH, INPUT_WIDTH)
    )


def test_the_chart_group_lists_the_ablation_under_its_tag() -> None:
    point = EntryPoint(name=TAG, value=ENTRY_POINT, group=entry_point_group(FAMILY))

    assert entry_point_group(FAMILY) == "lofi.chart"
    assert point.load() is flat.PLUGIN
    assert (
        plugin_for(
            FlatParserSpec(
                width=WIDTH,
                features=FEATURES,
                projection_initialization=INITIALIZATION,
            )
        )
        is flat.PLUGIN
    )


def test_the_flat_variant_is_a_parser_and_not_a_term_encoder() -> None:
    encoder = build()

    assert flat.__all__ == ["FlatSpanEncoder"]
    assert isinstance(encoder, ChartScorer)
    assert not isinstance(encoder, TermEncoder)
    assert not [name for name in dir(flat) if "TermEncoder" in name]
