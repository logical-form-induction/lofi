from typing import TYPE_CHECKING, Final, cast, override

import didactic.api as dx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.configured_component import (
    ConfiguredComponentConformance,
    ConfiguredComponentFixture,
)
from tests.conformance.objective import ObjectiveLawSuite

import lofi.nn.loss._reconstruct as reconstruct_module
from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
)
from lofi.nn.loss._reconstruct import (
    SpanNormalization,
    SpanReconstructionObjective,
    SpanWeighting,
)
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.tokens.embedder import ReservedTokens, TokenVocabulary
from lofi.spec.losses.span_reconstruction import SpanReconstructionLossSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.conftest import ChecksReload
    from torch import Tensor

    from lofi.nn.loss._ports import (
        LossTerm,
        ProofRewardBindings,
        SpanTokenDecoder,
    )
    from lofi.nn.loss._readout import TermsReadout
    from lofi.nn.spaces import Space, SpaceRelationBinding
    from lofi.spec import ConfiguredSpec

type Outputs = ObjectiveOutputs[Words]

RESERVED: Final = ReservedTokens({"<pad>": 0, "<eos>": 1, "<unk>": 2})
INVENTORY: Final = ("a", "b", "c", "d")
VOCABULARY: Final = TokenVocabulary(INVENTORY, reserved=RESERVED, unknown="<unk>")
END: Final = "<eos>"
END_ID: Final = RESERVED.index_of(END)
WIDTH: Final = VOCABULARY.size
SLOPE: Final = 2.0
LENGTH: Final = 3
SPANS: Final = SpanIndex(LENGTH)
ITEMS: Final = (("a", "b", "c"), ("c", "a"))
PADDING: Final = -7.0
SCORED: Final = 9
EXAMPLES: Final = 25
SEED: Final = 11


class Words:
    def __init__(self, items: tuple[tuple[str, ...], ...]) -> None:
        self._items = items

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        return self._items


class TableDecoder:
    def __init__(
        self, vocabulary: TokenVocabulary = VOCABULARY, *, end: str = END
    ) -> None:
        self._vocabulary = vocabulary
        self._end = end
        self.prefixes: list[Encoded] = []
        self.targets: list[Tensor] = []

    @property
    def vocabulary(self) -> TokenVocabulary:
        return self._vocabulary

    @property
    def end(self) -> str:
        return self._end

    def score_tokens(self, prefix: Encoded, /, *, targets: Tensor) -> LogDistribution:
        self.prefixes.append(prefix)
        self.targets.append(targets.detach().clone())

        values = prefix.tensor
        steps = torch.arange(targets.shape[1], dtype=values.dtype, device=values.device)
        scaled = values.unsqueeze(1) * SLOPE * (1.0 + steps.reshape(1, -1, 1) / 10.0)

        return LogDistribution(
            scaled.log_softmax(dim=2),
            ("span", "position", "token"),
            normalized_over="token",
        )


def objective(
    decoder: TableDecoder | None = None,
    weighting: SpanWeighting = SpanWeighting.UNIFORM,
    *,
    max_span_length: int | None = None,
    normalization: SpanNormalization = SpanNormalization.SPAN,
) -> SpanReconstructionObjective[Words]:
    return SpanReconstructionObjective(
        decoder if decoder is not None else TableDecoder(),
        weighting=weighting,
        max_span_length=max_span_length,
        normalization=normalization,
        name="span-reconstruction",
    )


def live_spans(lengths: tuple[int, ...] = (3, 2)) -> Tensor:
    mask = torch.zeros(len(lengths), SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in SPANS.spans_by_length():
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()


def cell_tensor() -> Tensor:
    generator = torch.Generator().manual_seed(SEED)
    values = torch.randn(
        (len(ITEMS), SPANS.num_spans, WIDTH), generator=generator, dtype=torch.float64
    )

    return torch.where(LIVE.unsqueeze(-1), values, PADDING).detach().requires_grad_()


def chart() -> CellChart:
    scores = torch.where(LIVE, -1.0, PADDING).to(torch.float64)

    return CellChart(LogDistribution(scores, SPAN_AXES, LIVE))


def bundle() -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(cell_tensor(), (*SPAN_AXES, "width"), LIVE),
    )


def aligned(sign: float, strength: float = 4.0) -> Outputs:
    values = torch.zeros(len(ITEMS), SPANS.num_spans, WIDTH, dtype=torch.float64)

    for item, words in enumerate(ITEMS):
        for start, end in SPANS.spans_by_length():
            if end > len(words):
                continue

            for token in targets_of(words, start, end):
                values[item, SPANS.span_id(start, end), token] += sign * strength

    filled = torch.where(LIVE.unsqueeze(-1), values, PADDING)

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(filled.detach().requires_grad_(), (*SPAN_AXES, "width"), LIVE),
    )


def empty_bundle() -> Outputs:
    blank = torch.zeros(len(ITEMS), SPANS.num_spans, dtype=torch.bool)
    values = torch.full(
        (len(ITEMS), SPANS.num_spans, WIDTH),
        PADDING,
        dtype=torch.float64,
    ).requires_grad_()

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(values, (*SPAN_AXES, "width"), blank),
    )


def permuted(outputs: Outputs) -> Outputs:
    order = torch.tensor([1, 0])
    published = outputs.cells

    assert published is not None
    assert published.mask is not None

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(tuple(reversed(ITEMS))),
        cells=Encoded(
            published.tensor[order].detach().requires_grad_(),
            published.axes,
            published.mask[order],
        ),
        marginals=None
        if outputs.marginals is None
        else reordered_marginals(outputs.marginals, order),
    )


def refilled(outputs: Outputs) -> Outputs:
    published = outputs.cells

    assert published is not None
    assert published.mask is not None

    values = published.tensor.detach()
    other = torch.where(published.mask.unsqueeze(-1), values, PADDING / 3.0)

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=outputs.batch,
        cells=Encoded(other.requires_grad_(), published.axes, published.mask),
        marginals=outputs.marginals,
    )


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    published = outputs.cells

    assert published is not None

    return (published.tensor,)


def targets_of(words: tuple[str, ...], start: int, end: int) -> tuple[int, ...]:
    return (*(VOCABULARY.index_of(word) for word in words[start:end]), END_ID)


def position_nlls(cell: Tensor, targets: tuple[int, ...]) -> list[float]:
    rows = torch.tensor([targets], dtype=torch.long)
    scored = TableDecoder().score_tokens(
        Encoded(cell.unsqueeze(0), ("span", "width")), targets=rows
    )

    return [
        -float(scored.tensor[0, position, token])
        for position, token in enumerate(targets)
    ]


def row_nll(cell: Tensor, targets: tuple[int, ...]) -> float:
    nlls = position_nlls(cell, targets)

    return sum(nlls) / len(nlls)


def owners_of(outputs: Outputs) -> dict[tuple[float, ...], tuple[int, int, int]]:
    published = outputs.cells

    assert published is not None

    cells = published.tensor.detach()

    return {
        tuple(cells[item, SPANS.span_id(start, end)].tolist()): (item, start, end)
        for item, words in enumerate(outputs.batch.words)
        for start, end in SPANS.spans_by_length()
        if end <= len(words)
    }


def reckoning(outputs: Outputs) -> float:
    published = outputs.cells

    assert published is not None
    assert published.mask is not None

    values = published.tensor.detach()
    weights = weights_of(outputs)
    total = 0.0
    weight = 0.0

    for item, words in enumerate(outputs.batch.words):
        for start, end in SPANS.spans_by_length():
            span = SPANS.span_id(start, end)

            if end > len(words) or not bool(published.mask[item, span]):
                continue

            share = weights[item][span]

            if share is None:
                continue

            total += share * row_nll(values[item, span], targets_of(words, start, end))
            weight += share

    return total / weight if weight else 0.0


def weights_of(outputs: Outputs) -> tuple[tuple[float | None, ...], ...]:
    marginals = outputs.marginals

    if marginals is None:
        return tuple(
            tuple(1.0 for _ in range(SPANS.num_spans)) for _ in outputs.batch.words
        )

    mask = marginals.constituent.mask

    assert mask is not None

    return tuple(
        tuple(
            float(marginals.constituent.tensor[item, span].detach())
            if bool(mask[item, span])
            else None
            for span in range(SPANS.num_spans)
        )
        for item in range(len(outputs.batch.words))
    )


def constituent_mask(lengths: tuple[int, ...] = (3, 2)) -> Tensor:
    mask = torch.zeros(len(lengths), SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(lengths):
        for start, end in SPANS.spans_by_length(min_length=2):
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


BUILDABLE: Final = constituent_mask()


def marginals_over(
    values: tuple[tuple[float, ...], ...],
    weighed: Tensor = BUILDABLE,
) -> SpanMarginals:
    constituent = torch.tensor(values, dtype=torch.float64)
    cutpoints = torch.zeros(
        len(ITEMS), SPANS.num_spans, SPANS.max_splits, dtype=torch.bool
    )

    for item, words in enumerate(ITEMS):
        for start, end in SPANS.spans_by_length(min_length=2):
            if end > len(words):
                continue

            for cut in SPANS.splits(start, end):
                span, column = SPANS.split_id(start, end, cut)
                cutpoints[item, span, column] = True

    rule = torch.where(cutpoints, 0.5, 0.0).to(torch.float64)

    return SpanMarginals(
        SPANS,
        Encoded(constituent.requires_grad_(), SPAN_AXES, weighed),
        Encoded(rule, SPLIT_AXES, cutpoints),
        torch.tensor([3, 2]),
    )


MARGINAL_VALUES: Final = (
    (0.0, 0.0, 0.0, 0.5, 0.75, 1.0),
    (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
)


def reordered_marginals(marginals: SpanMarginals, order: Tensor) -> SpanMarginals:
    constituent = marginals.constituent
    rule = marginals.rule

    assert constituent.mask is not None
    assert rule.mask is not None

    return SpanMarginals(
        SPANS,
        Encoded(
            constituent.tensor[order].detach().requires_grad_(),
            SPAN_AXES,
            constituent.mask[order],
        ),
        Encoded(rule.tensor[order], SPLIT_AXES, rule.mask[order]),
        marginals.lengths[order],
    )


def weighted_bundle() -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(cell_tensor(), (*SPAN_AXES, "width"), LIVE),
        marginals=marginals_over(MARGINAL_VALUES),
    )


def weighted_aligned(sign: float) -> Outputs:
    plain = aligned(sign)
    published = plain.cells

    assert published is not None

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=published,
        marginals=marginals_over(MARGINAL_VALUES),
    )


def weighted_empty() -> Outputs:
    plain = empty_bundle()
    published = plain.cells

    assert published is not None

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=published,
        marginals=marginals_over(MARGINAL_VALUES),
    )


def models_built(call: Callable[[], LossTerm]) -> int:
    counted = 0
    original = dx.Model.__init__

    def counting(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1

        original(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dx.Model, "__init__", counting)
        call()

    return counted


def fixture() -> ObjectiveFixture[Words]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=reckoning,
        monotone=lambda: (aligned(-1.0), aligned(1.0)),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
    )


def weighted_fixture() -> ObjectiveFixture[Words]:
    return ObjectiveFixture(
        objective=objective(weighting=SpanWeighting.MARGINAL),
        outputs=weighted_bundle,
        inputs=inputs_of,
        unweighted=reckoning,
        monotone=lambda: (weighted_aligned(-1.0), weighted_aligned(1.0)),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=weighted_empty,
    )


class TestSpanReconstructionObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Words]:
        """Return the fixture the laws run against.

        Returns
        -------
        ObjectiveFixture
            The objective under the uniform weighting and the bundles it is scored on.
        """
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that cells pointing away from a span's words score above cells at them.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the anti-aligned batch does not score above the aligned one.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestSpanReconstructionUnderMarginalWeightingLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Words]:
        """Return the fixture the laws run against.

        Returns
        -------
        ObjectiveFixture
            The objective under the marginal weighting and the bundles it is scored on.
        """
        return weighted_fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that cells pointing away from a span's words score above cells at them.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the anti-aligned batch does not score above the aligned one.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_term_matches_a_reckoning_made_one_position_at_a_time() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.value.item() == pytest.approx(reckoning(outputs))
    assert term.count == SCORED


def test_the_term_is_a_non_negative_scalar() -> None:
    term = objective().evaluate(bundle())

    assert term.value.ndim == 0
    assert term.value.item() >= 0.0
    assert not bool(torch.isnan(term.value))


def test_the_term_reports_the_unweighted_mean_over_the_positions_it_scored() -> None:
    outputs = bundle()
    published = outputs.cells

    assert published is not None

    values = published.tensor.detach()
    total = 0.0
    positions = 0

    for item, words in enumerate(ITEMS):
        for start, end in SPANS.spans_by_length():
            if end > len(words):
                continue

            targets = targets_of(words, start, end)
            total += row_nll(values[item, SPANS.span_id(start, end)], targets) * len(
                targets
            )
            positions += len(targets)

    term = objective().evaluate(outputs)

    assert set(term.diagnostics) == {"per_token"}
    assert term.diagnostics["per_token"].item() == pytest.approx(total / positions)


def test_a_span_reaching_past_an_item_is_not_scored() -> None:
    decoder = TableDecoder()

    term = objective(decoder).evaluate(bundle())

    assert term.count == SCORED
    assert sum(int(rows.shape[0]) for rows in decoder.targets) == SCORED


def test_every_span_of_one_length_is_scored_in_one_call() -> None:
    decoder = TableDecoder()

    objective(decoder).evaluate(bundle())

    assert [tuple(rows.shape) for rows in decoder.targets] == [(5, 2), (3, 3), (1, 4)]


def test_the_declared_maximum_span_length_bounds_reconstruction() -> None:
    decoder = TableDecoder()

    term = objective(decoder, max_span_length=1).evaluate(bundle())

    assert term.count == 5
    assert [tuple(rows.shape) for rows in decoder.targets] == [(5, 2)]


def test_position_normalization_weights_longer_spans_by_their_positions() -> None:
    outputs = bundle()
    by_span = objective().evaluate(outputs)
    by_position = objective(normalization=SpanNormalization.POSITION).evaluate(outputs)

    assert by_position.value.item() == pytest.approx(
        by_position.diagnostics["per_token"].item()
    )
    assert by_position.value.item() != pytest.approx(by_span.value.item())


def test_a_span_reaches_the_decoder_as_a_prefix_rather_than_a_memory_of_one_row() -> (
    None
):
    decoder = TableDecoder()

    objective(decoder).evaluate(bundle())

    for prefix, rows in zip(decoder.prefixes, decoder.targets, strict=True):
        assert prefix.axes == ("span", "width")
        assert prefix.tensor.ndim == 2
        assert int(prefix.tensor.shape[0]) == int(rows.shape[0])


def test_each_span_is_scored_against_the_cell_of_that_span() -> None:
    decoder = TableDecoder()
    outputs = bundle()
    owners = owners_of(outputs)

    assert len(owners) == SCORED

    objective(decoder).evaluate(outputs)

    for prefix, rows in zip(decoder.prefixes, decoder.targets, strict=True):
        for vector, row in zip(prefix.tensor.detach(), rows, strict=True):
            item, start, end = owners[tuple(vector.tolist())]

            assert tuple(int(value) for value in row) == targets_of(
                ITEMS[item], start, end
            )


def test_the_target_rows_are_the_words_of_the_span_and_then_the_end_token() -> None:
    decoder = TableDecoder()

    objective(decoder).evaluate(bundle())

    wanted = [
        targets_of(words, start, end)
        for words in ITEMS
        for start, end in SPANS.spans_by_length()
        if end <= len(words)
    ]
    found = [
        tuple(int(value) for value in row) for rows in decoder.targets for row in rows
    ]

    assert sorted(found) == sorted(wanted)


def test_every_target_id_is_a_row_of_the_vocabulary() -> None:
    decoder = TableDecoder()

    objective(decoder).evaluate(bundle())

    for rows in decoder.targets:
        assert bool(((rows >= 0) & (rows < VOCABULARY.size)).all())


def test_a_word_outside_the_vocabulary_lands_on_the_declared_unknown_row() -> None:
    decoder = TableDecoder()
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words((("a", "zzz", "c"), ("c", "a"))),
        cells=Encoded(cell_tensor(), (*SPAN_AXES, "width"), LIVE),
    )

    objective(decoder).evaluate(outputs)

    leaves = decoder.targets[0]

    assert int(leaves[1, 0]) == VOCABULARY.unknown_index
    assert int(leaves[0, 0]) == VOCABULARY.index_of("a")


def test_no_target_row_holds_the_padding_row() -> None:
    decoder = TableDecoder()

    objective(decoder).evaluate(bundle())

    for rows in decoder.targets:
        assert not bool((rows == RESERVED.index_of("<pad>")).any())


def test_a_step_with_no_live_cell_is_inapplicable() -> None:
    term = objective().evaluate(empty_bundle())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics == {}


def test_a_chart_over_no_input_is_inapplicable() -> None:
    values = torch.zeros(1, 0, WIDTH, dtype=torch.float64).requires_grad_()
    outputs = ObjectiveOutputs(
        spans=SpanIndex(0),
        chart=CellChart(LogDistribution(torch.zeros(1, 0), SPAN_AXES)),
        batch=Words(((),)),
        cells=Encoded(values, (*SPAN_AXES, "width")),
    )

    term = objective().evaluate(outputs)

    term.value.backward()

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.dtype is values.dtype
    assert values.grad is not None
    assert tuple(values.grad.shape) == tuple(values.shape)


def test_the_gradient_reaches_every_live_cell_and_leaves_the_padding_alone() -> None:
    outputs = bundle()
    published = outputs.cells

    assert published is not None

    term = objective().evaluate(outputs)

    term.value.backward()

    gradient = published.tensor.grad

    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert not bool(torch.isnan(gradient).any())
    assert bool((gradient[~LIVE] == 0.0).all())
    assert bool((gradient[LIVE].abs().sum(dim=-1) > 0.0).all())


def test_an_inapplicable_step_leaves_no_gradient() -> None:
    outputs = empty_bundle()
    published = outputs.cells

    assert published is not None

    objective().evaluate(outputs).value.backward()

    gradient = published.tensor.grad

    assert gradient is not None
    assert float(gradient.abs().max()) == 0.0


def test_a_chart_of_one_word_scores_its_only_span() -> None:
    spans = SpanIndex(1)
    values = torch.full((1, 1, WIDTH), 0.25, dtype=torch.float64).requires_grad_()
    outputs = ObjectiveOutputs(
        spans=spans,
        chart=CellChart(LogDistribution(torch.zeros(1, 1), SPAN_AXES)),
        batch=Words((("a",),)),
        cells=Encoded(values, (*SPAN_AXES, "width")),
    )
    decoder = TableDecoder()

    term = objective(decoder).evaluate(outputs)

    assert term.count == 1
    assert [tuple(rows.shape) for rows in decoder.targets] == [(1, 2)]


class UnnormalizedDecoder(TableDecoder):
    @override
    def score_tokens(self, prefix: Encoded, /, *, targets: Tensor) -> LogDistribution:
        scored = super().score_tokens(prefix, targets=targets)

        return LogDistribution(scored.tensor * 3.0, scored.axes, normalized_over=None)


def test_the_debug_checks_refuse_scores_not_normalized_over_the_tokens(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(reconstruct_module, on=True)
    built = reconstruct_module.SpanReconstructionObjective(
        UnnormalizedDecoder(),
        weighting=SpanWeighting.UNIFORM,
        max_span_length=None,
        normalization=reconstruct_module.SpanNormalization.SPAN,
        name="span-reconstruction",
    )

    with pytest.raises(ValueError, match="normalized over None"):
        built.evaluate(bundle())


def test_the_debug_checks_accept_scores_normalized_over_the_tokens(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(reconstruct_module, on=True)
    outputs = bundle()
    built = reconstruct_module.SpanReconstructionObjective(
        TableDecoder(),
        weighting=SpanWeighting.UNIFORM,
        max_span_length=None,
        normalization=reconstruct_module.SpanNormalization.SPAN,
        name="span-reconstruction",
    )

    term = built.evaluate(outputs)

    assert term.count == SCORED
    assert term.value.item() == pytest.approx(reckoning(outputs))


def test_the_debug_checks_refuse_scores_over_other_axes(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(reconstruct_module, on=True)

    class Transposed(TableDecoder):
        @override
        def score_tokens(
            self, prefix: Encoded, /, *, targets: Tensor
        ) -> LogDistribution:
            scored = super().score_tokens(prefix, targets=targets)

            return LogDistribution(
                scored.tensor.transpose(0, 1),
                ("position", "span", "token"),
                normalized_over="token",
            )

    built = reconstruct_module.SpanReconstructionObjective(
        Transposed(),
        weighting=SpanWeighting.UNIFORM,
        max_span_length=None,
        normalization=reconstruct_module.SpanNormalization.SPAN,
        name="span-reconstruction",
    )

    with pytest.raises(ValueError, match="scored the axes"):
        built.evaluate(bundle())


def test_the_debug_checks_refuse_fewer_rows_than_were_handed_over(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(reconstruct_module, on=True)

    class Short(TableDecoder):
        @override
        def score_tokens(
            self, prefix: Encoded, /, *, targets: Tensor
        ) -> LogDistribution:
            scored = super().score_tokens(prefix, targets=targets)

            return LogDistribution(
                scored.tensor[:, :-1], scored.axes, normalized_over="token"
            )

    built = reconstruct_module.SpanReconstructionObjective(
        Short(),
        weighting=SpanWeighting.UNIFORM,
        max_span_length=None,
        normalization=reconstruct_module.SpanNormalization.SPAN,
        name="span-reconstruction",
    )

    with pytest.raises(ValueError, match="positions and scored"):
        built.evaluate(bundle())


def test_without_the_debug_checks_the_decoder_s_return_is_taken_as_given(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(reconstruct_module, on=False)
    built = reconstruct_module.SpanReconstructionObjective(
        UnnormalizedDecoder(),
        weighting=SpanWeighting.UNIFORM,
        max_span_length=None,
        normalization=reconstruct_module.SpanNormalization.SPAN,
        name="span-reconstruction",
    )
    checked = (
        reconstruct_module.SpanReconstructionObjective(
            TableDecoder(),
            weighting=SpanWeighting.UNIFORM,
            max_span_length=None,
            normalization=reconstruct_module.SpanNormalization.SPAN,
            name="span-reconstruction",
        )
        .evaluate(bundle())
        .value.item()
    )

    term = built.evaluate(bundle())

    assert term.count == SCORED
    assert term.value.item() == pytest.approx(3.0 * checked)


def test_a_step_publishing_no_cell_is_refused_by_name() -> None:
    outputs = ObjectiveOutputs(spans=SPANS, chart=chart(), batch=Words(ITEMS))

    with pytest.raises(MissingOutputError, match="span-reconstruction reads cells"):
        objective().evaluate(outputs)


def test_cells_over_the_wrong_axes_are_refused() -> None:
    values = torch.zeros(len(ITEMS), SPANS.num_spans, WIDTH, dtype=torch.float64)
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(values, ("item", "cell", "width")),
    )

    with pytest.raises(ValueError, match="reads them over"):
        objective().evaluate(outputs)


def test_cells_of_the_wrong_rank_are_refused() -> None:
    values = torch.zeros(len(ITEMS), SPANS.num_spans, dtype=torch.float64)
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(values, SPAN_AXES),
    )

    with pytest.raises(ValueError, match="of rank three"):
        objective().evaluate(outputs)


def test_cells_addressing_other_spans_than_the_index_are_refused() -> None:
    values = torch.zeros(len(ITEMS), SPANS.num_spans + 1, WIDTH, dtype=torch.float64)
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(values, (*SPAN_AXES, "width")),
    )

    with pytest.raises(ValueError, match="positions on the axis 'span'"):
        objective().evaluate(outputs)


def test_an_item_longer_than_the_chart_is_refused() -> None:
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words((("a", "b", "c", "d"), ("c", "a"))),
        cells=Encoded(cell_tensor(), (*SPAN_AXES, "width"), LIVE),
    )

    with pytest.raises(ValueError, match="item 0 holds 4 words"):
        objective().evaluate(outputs)


def test_a_mask_over_the_items_alone_keeps_every_span_of_a_live_item() -> None:
    values = torch.full(
        (len(ITEMS), SPANS.num_spans, WIDTH), 0.1, dtype=torch.float64
    ).requires_grad_()
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(values, (*SPAN_AXES, "width"), torch.tensor([True, False])),
    )

    term = objective().evaluate(outputs)

    assert term.count == SPANS.num_spans


def test_an_unmasked_bundle_scores_every_span_that_fits() -> None:
    values = torch.full(
        (len(ITEMS), SPANS.num_spans, WIDTH), 0.1, dtype=torch.float64
    ).requires_grad_()
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(values, (*SPAN_AXES, "width")),
    )

    term = objective().evaluate(outputs)

    assert term.count == SCORED


def test_the_objective_declares_what_each_weighting_reads() -> None:
    assert objective().requires == frozenset({"spans", "cells", "batch"})
    assert objective(weighting=SpanWeighting.MARGINAL).requires == frozenset(
        {"spans", "cells", "batch", "marginals"}
    )


def test_the_marginal_weighting_scores_only_the_spans_the_marginal_weighs() -> None:
    decoder = TableDecoder()
    weighed = BUILDABLE.clone()
    weighed[0, SPANS.span_id(1, 3)] = False
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(cell_tensor(), (*SPAN_AXES, "width"), LIVE),
        marginals=marginals_over(MARGINAL_VALUES, weighed),
    )
    owners = owners_of(outputs)

    term = objective(decoder, SpanWeighting.MARGINAL).evaluate(outputs)

    scored = {
        owners[tuple(vector.tolist())]
        for prefix in decoder.prefixes
        for vector in prefix.tensor.detach()
    }

    assert scored == {(0, 0, 2), (0, 0, 3), (1, 0, 2)}
    assert term.count == len(scored)
    assert term.value.item() == pytest.approx(reckoning(outputs))


def test_the_marginal_weighting_matches_a_reckoning_over_the_same_weights() -> None:
    outputs = weighted_bundle()

    term = objective(weighting=SpanWeighting.MARGINAL).evaluate(outputs)

    assert term.value.item() == pytest.approx(reckoning(outputs))


def test_the_marginal_weighting_differs_from_the_uniform_one() -> None:
    outputs = weighted_bundle()

    weighted = objective(weighting=SpanWeighting.MARGINAL).evaluate(outputs)
    plain = objective().evaluate(outputs)

    assert weighted.value.item() != pytest.approx(plain.value.item())


def test_the_marginal_weights_take_no_gradient() -> None:
    outputs = weighted_bundle()
    marginals = outputs.marginals

    assert marginals is not None

    objective(weighting=SpanWeighting.MARGINAL).evaluate(outputs).value.backward()

    assert marginals.constituent.tensor.grad is None


def test_a_marginal_that_weighs_nothing_is_inapplicable() -> None:
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=chart(),
        batch=Words(ITEMS),
        cells=Encoded(cell_tensor(), (*SPAN_AXES, "width"), LIVE),
        marginals=marginals_over(((0.0,) * SPANS.num_spans,) * len(ITEMS)),
    )

    term = objective(weighting=SpanWeighting.MARGINAL).evaluate(outputs)

    assert term.count == 0
    assert term.value.item() == 0.0


def test_a_step_publishing_no_marginal_is_refused_by_name() -> None:
    with pytest.raises(MissingOutputError, match="span-reconstruction reads marginals"):
        objective(weighting=SpanWeighting.MARGINAL).evaluate(bundle())


def test_the_end_token_is_the_one_the_decoder_reserves() -> None:
    reserved = ReservedTokens({"<pad>": 0, "<stop>": 1, "<unk>": 2})
    vocabulary = TokenVocabulary(INVENTORY, reserved=reserved, unknown="<unk>")
    decoder = TableDecoder(vocabulary, end="<stop>")
    built = SpanReconstructionObjective(
        decoder,
        weighting=SpanWeighting.UNIFORM,
        max_span_length=None,
        normalization=SpanNormalization.SPAN,
        name="span-reconstruction",
    )

    built.evaluate(bundle())

    assert built.end == "<stop>"
    assert all(
        bool((rows[:, -1] == reserved.index_of("<stop>")).all())
        for rows in decoder.targets
    )


def test_an_end_token_the_decoder_s_vocabulary_does_not_reserve_is_refused() -> None:
    with pytest.raises(KeyError, match="is not reserved here"):
        SpanReconstructionObjective(
            TableDecoder(end="<stop>"),
            weighting=SpanWeighting.UNIFORM,
            max_span_length=None,
            normalization=SpanNormalization.SPAN,
            name="span-reconstruction",
        )


def test_an_empty_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        SpanReconstructionObjective(
            TableDecoder(),
            name="",
            weighting=SpanWeighting.UNIFORM,
            max_span_length=None,
            normalization=SpanNormalization.SPAN,
        )


def test_the_objective_names_its_end_token_its_weighting_and_its_vocabulary() -> None:
    built = objective(weighting=SpanWeighting.MARGINAL)

    assert built.end == END
    assert built.weighting is SpanWeighting.MARGINAL
    assert built.max_span_length is None
    assert built.normalization is SpanNormalization.SPAN
    assert built.vocabulary is VOCABULARY
    assert built.name == "span-reconstruction"
    assert repr(built) == (
        "SpanReconstructionObjective(name='span-reconstruction', "
        "end='<eos>', weighting=MARGINAL, max_span_length=None, "
        "normalization=SPAN)"
    )


@settings(max_examples=EXAMPLES)
@given(
    strength=st.floats(min_value=0.5, max_value=3.0),
    gap=st.floats(min_value=0.25, max_value=2.0),
)
def test_pointing_further_from_a_span_s_words_scores_higher(
    strength: float,
    gap: float,
) -> None:
    built = objective()

    near = built.evaluate(aligned(-1.0, strength)).value.item()
    far = built.evaluate(aligned(-1.0, strength + gap)).value.item()

    assert far > near


class Binding:
    """One span decoder installed under the role the declaration names."""

    def __init__(self) -> None:
        self.decoder = TableDecoder()
        self.requests: list[str] = []

    @property
    def name(self) -> str:
        return "reconstructed-spans"

    def terms_readout(self, role: str, /) -> TermsReadout:
        msg = f"span reconstruction must not resolve term readout {role!r}"
        raise AssertionError(msg)

    def space(self, role: str, /) -> Space:
        msg = f"span reconstruction must not resolve represented space {role!r}"
        raise AssertionError(msg)

    def readout_space(self, role: str, /) -> Space:
        msg = f"span reconstruction must not resolve a readout space for {role!r}"
        raise AssertionError(msg)

    def relation(self, role: str, /) -> SpaceRelationBinding:
        msg = f"span reconstruction must not resolve space relation {role!r}"
        raise AssertionError(msg)

    def span_decoder(self, role: str, /) -> SpanTokenDecoder:
        self.requests.append(role)

        if role != "reconstruction":
            msg = f"no span decoder is bound as {role!r}"
            raise ValueError(msg)

        return self.decoder

    def proof_reward(self) -> ProofRewardBindings:
        msg = "span reconstruction must not resolve proof-reward bindings"
        raise AssertionError(msg)


def declaration(weighting: str = "uniform") -> SpanReconstructionLossSpec:
    return SpanReconstructionLossSpec.model_validate(
        {"decoder": "reconstruction", "weighting": weighting}
    )


def configured(
    weighting: str = "uniform", context: Binding | None = None
) -> SpanReconstructionObjective[Words]:
    """Return the objective the installed plugin builds from a declaration.

    Parameters
    ----------
    weighting : str
        The declared weighting name.
    context : Binding | None
        The binding context, or None for a fresh one.
    """
    return reconstruct_module.PLUGIN.construct(
        declaration(weighting), Binding() if context is None else context
    )


class TestConfiguredSpanReconstructionLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Words]:
        """Return the uniform fixture over the factory-built objective."""
        return fixture()._replace(objective=configured())

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that cells pointing away from a span's words score above cells at them.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the laws run against.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestConfiguredMarginalSpanReconstructionLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Words]:
        """Return the marginal fixture over the factory-built objective."""
        return weighted_fixture()._replace(objective=configured("marginal"))

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that cells pointing away from a span's words score above cells at them.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
            What the laws run against.
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


class TestSpanReconstructionPlugin(ConfiguredComponentConformance):
    @pytest.fixture
    def configured_component_fixture(self) -> ConfiguredComponentFixture:
        context = Binding()
        decoder = context.decoder

        def observe(built: object) -> dict[str, bool]:
            held = cast("SpanReconstructionObjective[Words]", built)

            return {
                "named for its term": held.name == "reconstructed-spans",
                "reads the declared decoder": context.requests[-1] == "reconstruction",
                "scores over the decoder's vocabulary": held.vocabulary is VOCABULARY,
                "holds the declared weighting": held.weighting
                is SpanWeighting.MARGINAL,
            }

        return ConfiguredComponentFixture(
            declaration=cast("ConfiguredSpec[object]", declaration("marginal")),
            context=context,
            expected_type=SpanReconstructionObjective,
            observe=observe,
            unchanged=lambda: context.decoder is decoder and not decoder.prefixes,
        )


@pytest.mark.parametrize(
    ("weighting", "expected"),
    [("uniform", SpanWeighting.UNIFORM), ("marginal", SpanWeighting.MARGINAL)],
)
def test_the_plugin_binds_the_declared_span_decoder_and_weighting(
    weighting: str, expected: SpanWeighting
) -> None:
    context = Binding()
    built = configured(weighting, context)

    assert built.name == "reconstructed-spans"
    assert built.weighting is expected
    assert built.vocabulary is VOCABULARY
    assert context.requests == ["reconstruction"]


def test_the_plugin_preserves_the_declared_span_bound_and_normalization() -> None:
    context = Binding()
    declaration = SpanReconstructionLossSpec.model_validate(
        {
            "decoder": "reconstruction",
            "max_span_length": 2,
            "normalization": "token",
        }
    )
    built = reconstruct_module.PLUGIN.construct(declaration, context)

    assert built.max_span_length == 2
    assert built.normalization is SpanNormalization.POSITION
