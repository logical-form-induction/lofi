import math
import random
from itertools import pairwise
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pytest
import torch
from tests.conformance.component_plugin import plugin_context as component_context

from lofi._types import ReadoutId
from lofi.deduce import (
    Edge,
    ItemKey,
    LogInside,
    PackedForest,
    RootGoal,
    SpanIndex,
    outside_from_edges,
    target_mass,
)
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import DISTRIBUTION_AXES, NeuralCodecView
from lofi.nn.decoders import (
    RecursiveTermDecoder,
    TransformerDecoderSettings,
    TransformerTermDecoder,
    TreeDecoderSettings,
)
from lofi.nn.decoders.hrr import _PathAddressedHolographicTermDecoder
from lofi.nn.encoders import HolographicEncoderSettings, HrrTermEncoder
from lofi.nn.loss._errors import UncomputedCellError
from lofi.nn.loss._linearize import (
    LinearizedTargets,
    linearize_targets,
)
from lofi.nn.loss._mapping import (
    SOFTMAX,
    Entmax,
)
from lofi.nn.loss._parser import (
    SPAN_AXES,
    SPLIT_AXES,
    SpanMarginals,
)
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import (
    CELL_AXES,
    MASS_AXES,
    TERMS_READOUT_AXES,
    TermsReadout,
    _build,
    _log_one_minus,
    log_complement,
    log_mass,
)
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
    RecordingOutputs,
    SpanReadout,
)
from lofi.nn.ports import StandardDeclaration
from lofi.nn.sites import ComponentId
from lofi.nn.strategies import score
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.spec.readouts import DecodedMeaningSettings, ReadoutSource, TermsReadoutSpec
from lofi.spec.readouts.terms import ReadoutMass
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Identity,
    Operator,
    OperatorIndex,
    Signature,
    TargetSet,
    Term,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.deduce import Outside
    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.terms import Matching

type Outputs = ObjectiveOutputs[tuple[str, ...]]

SIGNATURE: Final = Signature(
    "readout-fixture",
    ("Cat",),
    (
        Operator("N", "Cat", family="primitive"),
        Operator("S", "Cat", family="primitive"),
        Operator("/", "Cat", arg_sorts=("Cat", "Cat"), family="constructor"),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"Cat": 0})
LAYOUT: Final = CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=2)
VIEW: Final = NeuralCodecView(TermCodec(INDEX, LAYOUT))
INPUT_WIDTH: Final = 4
SETTINGS: Final = TreeDecoderSettings(INPUT_WIDTH, 5)
SEED: Final = 3
KEY: Final = ReadoutId("terms")

LENGTH: Final = 3
ITEMS: Final = 2
LENGTHS: Final = (3, 2)
SPANS: Final = SpanIndex(LENGTH)
BATCH: Final = ("first", "second")
PADDING: Final = 7.0

UNREACHABLE_SIGNATURE: Final = Signature(
    "unreachable-readout",
    ("Root", "Branch"),
    (
        Operator("root", "Root", family="primitive"),
        Operator(
            "branch",
            "Branch",
            arg_sorts=("Branch", "Branch"),
            family="constructor",
        ),
        Operator("leaf", "Branch", family="primitive"),
    ),
)
UNREACHABLE_INDEX: Final = OperatorIndex(UNREACHABLE_SIGNATURE)
UNREACHABLE_CHOICES: Final = ChoiceIndex(
    UNREACHABLE_SIGNATURE, depth_bounds={"Root": 1, "Branch": 2}
)
UNREACHABLE_VIEW: Final = NeuralCodecView(
    TermCodec(
        UNREACHABLE_INDEX,
        CompleteTree(
            UNREACHABLE_SIGNATURE,
            UNREACHABLE_SIGNATURE.operators,
            order="bf",
            depth=2,
        ),
    )
)


def live_spans() -> Tensor:
    mask = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.bool)

    for item, length in enumerate(LENGTHS):
        for start, end in SPANS.spans_by_length():
            if end <= length:
                mask[item, SPANS.span_id(start, end)] = True

    return mask


LIVE: Final = live_spans()


def decoder() -> RecursiveTermDecoder:
    torch.manual_seed(SEED)
    built = RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("identity"),
        settings=SETTINGS,
        choice_heads="flat",
    )

    return built.to(torch.float64)


class RepresentedSpace:
    """A category-space identity for readout construction tests."""

    @property
    def sort(self) -> str:
        """Return the represented category sort."""
        return "Cat"

    @property
    def codec(self) -> NeuralCodecView:
        """Return the shared category codec."""
        return VIEW


def readout(built: RecursiveTermDecoder | None = None) -> TermsReadout:
    return TermsReadout(
        decoder() if built is None else built,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )


def cell_tensor(fill: float = PADDING) -> Tensor:
    steps = torch.arange(ITEMS * SPANS.num_spans * INPUT_WIDTH, dtype=torch.float64)
    values = torch.sin(steps).reshape(ITEMS, SPANS.num_spans, INPUT_WIDTH)

    return torch.where(LIVE.unsqueeze(-1), values, fill).detach().requires_grad_()


def cells(mask: Tensor | None = LIVE, fill: float = PADDING) -> Encoded:
    return Encoded(cell_tensor(fill), CELL_AXES, mask)


def one_cell(held: Encoded, item: int, span: int) -> Encoded:
    return Encoded(held.tensor[item, span].unsqueeze(0), ("term", "width"))


def test_the_readout_publishes_one_distribution_per_computed_cell() -> None:
    published = readout().publish(cells(), spans=SPANS)[KEY]

    assert published.axes == TERMS_READOUT_AXES
    assert published.spans is SPANS
    assert published.distribution.normalized_over == "symbol"
    assert published.distribution.tensor.shape == (
        ITEMS,
        SPANS.num_spans,
        VIEW.capacity,
        VIEW.n_symbols,
    )
    assert torch.equal(published.computed, LIVE)


def test_the_readout_publishes_payload_aware_choice_marginals() -> None:
    built = decoder()
    held = cells()
    publication = TermsReadout(
        built,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish_with_choices(held, spans=SPANS)
    choices = publication.choices

    assert set(publication.readouts) == {KEY}
    assert choices.axes == (*SPAN_AXES, "slot", "choice")
    assert choices.distribution.normalized_over == "choice"
    assert choices.distribution.size("choice") == CHOICES.width
    assert torch.equal(choices.computed, LIVE)

    for item, length in enumerate(LENGTHS):
        for start, end in SPANS.spans_by_length():
            if end > length:
                continue

            alone = built.marginal_rows(
                one_cell(held, item, SPANS.span_id(start, end)), root_sort="Cat"
            )
            assert torch.allclose(
                choices.at(item, start, end).tensor, alone.tensor[0], atol=1e-6
            )


@pytest.mark.parametrize("kind", ["transformer", "holographic"])
def test_unreachable_decoder_slots_publish_as_one_hot_padding(kind: str) -> None:
    torch.manual_seed(19)

    if kind == "transformer":
        width = 4
        built = TransformerTermDecoder(
            codec=UNREACHABLE_VIEW,
            choices=UNREACHABLE_CHOICES,
            component=ComponentId("unreachable-transformer"),
            settings=TransformerDecoderSettings(
                input_width=width,
                width=8,
                heads=2,
                layers=1,
                feedforward_width=12,
                activation="gelu",
            ),
            structure=StandardDeclaration.PARENT_RECURRENCE.declaration,
            choice_heads="flat",
        ).to(torch.float64)
    else:
        width = 8
        encoder = HrrTermEncoder(
            codec=UNREACHABLE_VIEW,
            component=ComponentId("unreachable-hrr-encoder"),
            settings=HolographicEncoderSettings(width=width, magnitude=1.0, floor=1e-3),
        ).to(torch.float64)
        built = _PathAddressedHolographicTermDecoder(
            codec=UNREACHABLE_VIEW,
            choices=UNREACHABLE_CHOICES,
            component=ComponentId("unreachable-hrr-decoder"),
            codebook=encoder.codebook,
            choice_heads="flat",
        ).to(torch.float64)

    publication = TermsReadout(
        built,
        codec=UNREACHABLE_VIEW,
        index=UNREACHABLE_CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Root",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish_with_choices(
        Encoded(torch.randn(1, 1, width, dtype=torch.float64), CELL_AXES),
        spans=SpanIndex(1),
    )
    choices = publication.choices.at(0, 0, 1)
    symbols = publication.readouts[KEY].at(0, 0, 1)

    assert choices.normalized_over == "choice"
    assert symbols.normalized_over == "symbol"
    assert torch.equal(
        choices.tensor[1:, UNREACHABLE_CHOICES.PAD],
        torch.zeros(choices.tensor.shape[0] - 1, dtype=torch.float64),
    )
    assert bool(choices.tensor[1:, UNREACHABLE_CHOICES.PAD + 1 :].isneginf().all())
    assert torch.equal(
        symbols.tensor[1:, UNREACHABLE_INDEX.PAD],
        torch.zeros(symbols.tensor.shape[0] - 1, dtype=torch.float64),
    )
    assert bool(symbols.tensor[1:, UNREACHABLE_INDEX.PAD + 1 :].isneginf().all())


def test_a_computed_cell_agrees_with_decoding_that_cell_alone() -> None:
    built = decoder()
    held = cells()

    published = TermsReadout(
        built,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(held, spans=SPANS)[KEY]

    for item, length in enumerate(LENGTHS):
        for start, end in SPANS.spans_by_length():
            if end > length:
                continue

            alone = built.marginal_rows(
                one_cell(held, item, SPANS.span_id(start, end)), root_sort="Cat"
            )
            cell = published.at(item, start, end)

            assert cell.axes == DISTRIBUTION_AXES[1:]
            assert torch.allclose(cell.tensor, alone.tensor[0], atol=1e-6)


def test_a_cell_the_mask_excludes_is_never_computed() -> None:
    published = readout().publish(cells(), spans=SPANS)[KEY]

    with pytest.raises(UncomputedCellError):
        published.at(1, 0, 3)

    excluded = published.distribution.tensor[1, SPANS.span_id(0, 3)]

    assert bool(excluded.isneginf().all())


def test_the_padding_never_reaches_a_computed_cell() -> None:
    one = readout().publish(cells(fill=PADDING), spans=SPANS)[KEY]
    other = readout().publish(cells(fill=-PADDING), spans=SPANS)[KEY]

    live = LIVE.reshape(ITEMS, SPANS.num_spans, 1, 1)

    assert torch.allclose(
        one.distribution.tensor[live.expand_as(one.distribution.tensor)],
        other.distribution.tensor[live.expand_as(other.distribution.tensor)],
    )


def test_a_mask_over_the_items_alone_is_widened_to_every_span() -> None:
    mask = torch.tensor([True, False])

    published = readout().publish(cells(mask=mask), spans=SPANS)[KEY]

    assert bool(published.computed[0].all())
    assert not bool(published.computed[1].any())


def test_unmasked_cells_are_all_computed() -> None:
    published = readout().publish(cells(mask=None), spans=SPANS)[KEY]

    assert published.distribution.mask is None
    assert bool(published.computed.all())
    assert published.at(1, 0, 3).axes == DISTRIBUTION_AXES[1:]


def test_the_readout_reports_its_key_and_route() -> None:
    built = readout()

    assert built.key == KEY
    assert built.decoding is None
    assert "terms" in repr(built)


def test_the_readout_refuses_cells_over_other_axes() -> None:
    held = Encoded(cell_tensor(), ("item", "cell", "width"), LIVE)

    with pytest.raises(ValueError, match="cells run over"):
        readout().publish(held, spans=SPANS)


def test_the_readout_refuses_a_cell_count_the_index_does_not_address() -> None:
    with pytest.raises(ValueError, match="addresses"):
        readout().publish(cells(), spans=SpanIndex(LENGTH + 1))


def test_the_readout_refuses_a_mask_reaching_into_a_cell() -> None:
    mask = torch.ones(ITEMS, SPANS.num_spans, INPUT_WIDTH, dtype=torch.bool)

    with pytest.raises(ValueError, match="part of one"):
        readout().publish(cells(mask=mask), spans=SPANS)


def test_an_empty_readout_id_is_refused() -> None:
    with pytest.raises(ValueError, match="readout id"):
        TermsReadout(
            decoder(),
            codec=VIEW,
            index=CHOICES,
            reads=ReadoutSource.INSIDE,
            decoding=None,
            key=ReadoutId(""),
            targets={},
            strategy=ReadoutMass.FACTORIZED,
            mapping=SOFTMAX,
            store=None,
            root_sort="Cat",
            max_nodes=None,
            clamp=None,
            meanings=DecodedMeaningSettings("decoded"),
            proof_terms=None,
            approximation=None,
            run_approximation=None,
        )


class Scorer:
    """A decoder that scores terms and returns no distribution."""

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        return LogDistribution(encoded.tensor, encoded.axes)


class Sideways:
    """A decoder whose distribution comes back over axes it does not declare."""

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        rows = encoded.tensor.new_zeros((encoded.tensor.shape[0], 2, 1))

        return LogDistribution(
            rows.log_softmax(dim=1),
            ("term", "choice", "slot"),
            normalized_over="choice",
        )


def test_a_decoder_returning_no_marginal_rows_is_refused_at_construction() -> None:
    with pytest.raises(TypeError, match="does not provide marginal choice rows"):
        TermsReadout(
            Scorer(),
            codec=VIEW,
            index=CHOICES,
            reads=ReadoutSource.INSIDE,
            decoding=None,
            key=KEY,
            targets={},
            strategy=ReadoutMass.FACTORIZED,
            mapping=SOFTMAX,
            store=None,
            root_sort="Cat",
            max_nodes=None,
            clamp=None,
            meanings=DecodedMeaningSettings("decoded"),
            proof_terms=None,
            approximation=None,
            run_approximation=None,
        )


def test_a_distribution_over_undeclared_axes_is_refused() -> None:
    built = TermsReadout(
        Sideways(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )

    with pytest.raises(ValueError, match="declares"):
        built.publish(cells(), spans=SPANS)


class Withholding:
    """A decoder that marks every other term it is handed as not computed."""

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        terms = encoded.tensor.shape[0]
        kept = torch.arange(terms) % 2 == 0
        rows = encoded.tensor.new_zeros((terms, 1, CHOICES.width)).log_softmax(dim=2)

        return LogDistribution(
            torch.where(kept.reshape(-1, 1, 1), rows, -torch.inf),
            ("term", "slot", "choice"),
            kept,
            normalized_over="choice",
        )


class Partial:
    """A decoder whose mask reaches into a term's slots."""

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        terms = encoded.tensor.shape[0]
        rows = encoded.tensor.new_zeros((terms, 2, CHOICES.width)).log_softmax(dim=2)

        return LogDistribution(
            rows,
            ("term", "slot", "choice"),
            torch.ones(terms, 2, dtype=torch.bool),
            normalized_over="choice",
        )


@pytest.mark.parametrize("mask", [LIVE, None])
def test_a_term_the_decoder_withholds_is_published_as_never_computed(
    mask: Tensor | None,
) -> None:
    built = TermsReadout(
        Withholding(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )
    handed = (
        torch.ones(ITEMS, SPANS.num_spans, dtype=torch.bool) if mask is None else mask
    )
    withheld = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.bool)
    withheld.view(-1)[handed.view(-1).nonzero()[1::2]] = True

    published = built.publish(cells(mask=mask), spans=SPANS)[KEY]

    assert torch.equal(published.computed, handed & ~withheld)
    assert bool(published.distribution.tensor[~published.computed].isneginf().all())

    with pytest.raises(UncomputedCellError):
        published.at(0, 0, 2)

    assert published.at(0, 0, 1).axes == DISTRIBUTION_AXES[1:]


@pytest.mark.parametrize("mask", [LIVE, None])
def test_a_decoder_masking_part_of_a_cell_is_refused(mask: Tensor | None) -> None:
    built = TermsReadout(
        Partial(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )

    with pytest.raises(ValueError, match="part of one"):
        built.publish(cells(mask=mask), spans=SPANS)


class PinnedRoot:
    """An objective pinning every item's root cell to one symbol at its first slot."""

    name = "pinned-root"
    requires = frozenset({"readouts", "spans"})

    def __init__(self, key: ReadoutId, symbol: int) -> None:
        self._key = key
        self._symbol = symbol

    def evaluate(self, outputs: PublishedOutputs[tuple[str, ...]], /) -> LossTerm:
        published = outputs.readouts[self._key]
        total = published.distribution.tensor.new_zeros(())
        count = 0

        for item, length in enumerate(LENGTHS):
            cell = published.at(item, 0, length)
            total = total - cell.tensor[0, self._symbol]
            count += 1

        return LossTerm(total, count)


def test_a_pinning_objective_trains_a_step_with_no_coherence_objective() -> None:
    built = decoder()
    held = cells()
    published = TermsReadout(
        built,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(held, spans=SPANS)
    outputs = ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, -2.0).double(), SPAN_AXES, LIVE)
        ),
        batch=BATCH,
        readouts=published,
    )
    suite = ObjectiveSuite(
        [(1.0, PinnedRoot(KEY, INDEX.index("S")))], published=outputs.published
    )

    result = suite.evaluate(outputs)
    result.total.backward()

    assert math.isfinite(result.total.item())
    assert result.total.item() > 0.0
    assert result.terms["pinned-root"].count == ITEMS
    assert held.tensor.grad is not None
    assert bool(held.tensor.grad[LIVE].abs().sum() > 0.0)
    assert not bool(held.tensor.grad[~LIVE].any())
    gradients = tuple(
        parameter.grad for parameter in built.parameters() if parameter.grad is not None
    )

    assert gradients
    assert all(bool(gradient.isfinite().all()) for gradient in gradients)


CODEC: Final = TermCodec(INDEX, LAYOUT)
STORE: Final = TermStore(SIGNATURE)
ROOTS: Final = ReadoutId("roots")
PERMITTED: Final = ReadoutId("permitted")
MATCHING: Final[Matching[Term]] = Identity()


def term(name: str) -> Term:
    return STORE.op(SIGNATURE.by_name(name))


def slashed(left: str, right: str) -> Term:
    return STORE.op(SIGNATURE.by_name("/"), term(left), term(right))


def targets(*members: Term) -> LinearizedTargets:
    return linearize_targets(TargetSet(members, MATCHING), CODEC)


def test_complete_term_scoring_uses_the_readouts_decoder_and_codec() -> None:
    built = decoder()
    published = readout(built)
    inputs = Encoded(
        torch.randn(2, INPUT_WIDTH, dtype=torch.float64), ("term", "width")
    )
    members = (term("N"), slashed("S", "N"))

    scored = published.score_terms(inputs, members)
    expected = score(built, inputs, CODEC.encode_batch(members), index=CHOICES)

    assert scored.axes == ("term",)
    assert torch.equal(scored.tensor, expected.tensor)


def marginals(interior: float = 0.3) -> SpanMarginals:
    widths = torch.tensor([j - i for i, j in SPANS.spans_by_length()])
    built = LIVE & (widths > 1)
    constituent = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.float64)
    constituent[0, SPANS.span_id(0, 2)] = interior
    constituent[0, SPANS.span_id(1, 3)] = 1.0 - interior
    constituent[0, SPANS.span_id(0, 3)] = 1.0
    constituent[1, SPANS.span_id(0, 2)] = 1.0
    rule = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits, dtype=torch.float64)
    valid = torch.zeros(ITEMS, SPANS.num_spans, SPANS.max_splits, dtype=torch.bool)

    for item, length in enumerate(LENGTHS):
        for i, j in SPANS.spans_by_length(min_length=2):
            if j > length:
                continue

            for k in SPANS.splits(i, j):
                span, column = SPANS.split_id(i, j, k)
                valid[item, span, column] = True

    rule[0, SPANS.span_id(0, 2), 0] = interior
    rule[0, SPANS.span_id(1, 3), 0] = 1.0 - interior
    rule[0, SPANS.span_id(0, 3), 0] = interior
    rule[0, SPANS.span_id(0, 3), 1] = 1.0 - interior
    rule[1, SPANS.span_id(0, 2), 0] = 1.0

    return SpanMarginals(
        SPANS,
        Encoded(constituent.requires_grad_(), SPAN_AXES, built),
        Encoded(rule, SPLIT_AXES, valid),
        torch.tensor(LENGTHS),
    )


def mass_readout(
    built: RecursiveTermDecoder,
    *,
    strategy: ReadoutMass = ReadoutMass.FACTORIZED,
    declared: Mapping[ReadoutId, LinearizedTargets] | None = None,
    mapping: Entmax = SOFTMAX,
) -> TermsReadout:
    return TermsReadout(
        built,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={ROOTS: targets(term("S"))} if declared is None else declared,
        strategy=strategy,
        mapping=mapping,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )


def log_members(
    built: RecursiveTermDecoder,
    held: Encoded,
    item: int,
    span: int,
    members: LinearizedTargets,
    strategy: ReadoutMass,
) -> Tensor:
    inputs = one_cell(held, item, span)
    if strategy is ReadoutMass.COUPLED:
        scored = [
            score(built, inputs, BatchEncoding((row,)), index=CHOICES).tensor[0]
            for row in members.encoded.rows
        ]
    else:
        rows = built.marginal_rows(inputs, root_sort="Cat").tensor[0]
        scored = [
            torch.where(
                torch.tensor(row.occupied),
                rows.gather(
                    1,
                    torch.tensor(CHOICES.codes(row)).reshape(-1, 1),
                ).squeeze(1),
                rows.new_zeros(()),
            ).sum()
            for row in members.encoded.rows
        ]

    return torch.stack(scored).logsumexp(dim=0)


def live_cells() -> list[tuple[int, int, int]]:
    return [
        (item, i, j)
        for item, length in enumerate(LENGTHS)
        for i, j in SPANS.spans_by_length()
        if j <= length
    ]


def test_a_target_sets_mass_is_published_under_its_own_id_beside_the_terms() -> None:
    built = decoder()
    declared = {ROOTS: targets(term("S")), PERMITTED: targets(term("N"), term("S"))}
    readout = mass_readout(built, declared=declared)

    published = readout.publish(cells(), spans=SPANS, marginals=marginals())

    assert set(published) == {KEY, ROOTS, PERMITTED}
    assert readout.targets == (ROOTS, PERMITTED)
    assert readout.strategy is ReadoutMass.FACTORIZED
    assert published[KEY].axes == TERMS_READOUT_AXES

    for key in (ROOTS, PERMITTED):
        pair = published[key]

        assert pair.axes == MASS_AXES
        assert pair.spans is SPANS
        assert pair.distribution.normalized_over == "outcome"
        assert pair.distribution.tensor.shape == (ITEMS, SPANS.num_spans, 2)
        assert torch.equal(pair.computed, LIVE)


@pytest.mark.parametrize(
    "strategy",
    list(ReadoutMass),
)
def test_the_mass_is_the_constituent_factor_times_the_summed_member_probability(
    strategy: ReadoutMass,
) -> None:
    built = decoder()
    held = cells()
    declared = {PERMITTED: targets(term("N"), slashed("S", "N"))}
    published = mass_readout(built, strategy=strategy, declared=declared).publish(
        held, spans=SPANS, marginals=marginals()
    )
    factor = marginals().constituent.tensor

    for item, i, j in live_cells():
        span = SPANS.span_id(i, j)
        summed = log_members(built, held, item, span, declared[PERMITTED], strategy)
        fixed = j - i == 1 or (i, j) == (0, LENGTHS[item])
        expected = summed.item() + (
            0.0 if fixed else math.log(factor[item, span].item())
        )
        cell = published[PERMITTED].at(item, i, j)

        assert cell.axes == ("outcome",)
        assert log_mass(cell).item() == pytest.approx(expected, abs=1e-9)
        assert log_complement(cell).item() == pytest.approx(
            math.log1p(-math.exp(expected)), abs=1e-9
        )


def test_the_constituent_factor_is_one_at_the_root_and_at_width_one_spans() -> None:
    built = decoder()
    held = cells()
    readout = mass_readout(built)

    thin = readout.publish(held, spans=SPANS, marginals=marginals(0.1))[ROOTS]
    thick = readout.publish(held, spans=SPANS, marginals=marginals(0.9))[ROOTS]

    for item, i, j in live_cells():
        span = SPANS.span_id(i, j)
        summed = log_members(
            built, held, item, span, targets(term("S")), ReadoutMass.FACTORIZED
        ).item()

        if j - i == 1 or (i, j) == (0, LENGTHS[item]):
            assert log_mass(thin.at(item, i, j)).item() == pytest.approx(summed)
            assert log_mass(thick.at(item, i, j)).item() == pytest.approx(summed)

        else:
            assert (
                log_mass(thin.at(item, i, j)).item()
                != log_mass(thick.at(item, i, j)).item()
            )


def test_the_two_strategies_agree_on_a_primitive_target() -> None:
    built = decoder()
    held = cells()
    factorized = mass_readout(built, strategy=ReadoutMass.FACTORIZED).publish(
        held, spans=SPANS, marginals=marginals()
    )
    coupled = mass_readout(built, strategy=ReadoutMass.COUPLED).publish(
        held, spans=SPANS, marginals=marginals()
    )

    assert torch.equal(factorized[ROOTS].computed, coupled[ROOTS].computed)
    assert torch.allclose(
        log_mass(factorized[ROOTS].distribution)[LIVE],
        log_mass(coupled[ROOTS].distribution)[LIVE],
    )


def test_the_coupled_strategy_conditions_a_complex_targets_slots() -> None:
    built = decoder()
    held = cells()
    declared = {PERMITTED: targets(slashed("S", "N"))}
    factorized = mass_readout(
        built, strategy=ReadoutMass.FACTORIZED, declared=declared
    ).publish(held, spans=SPANS, marginals=marginals())
    coupled = mass_readout(
        built, strategy=ReadoutMass.COUPLED, declared=declared
    ).publish(held, spans=SPANS, marginals=marginals())

    assert not torch.allclose(
        log_mass(factorized[PERMITTED].distribution)[LIVE],
        log_mass(coupled[PERMITTED].distribution)[LIVE],
    )


class WithholdingScorer:
    """A decoder reading every cell that withholds every third term it scores."""

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        rows = encoded.tensor.new_zeros(
            (encoded.tensor.shape[0], VIEW.capacity, CHOICES.width)
        )

        return LogDistribution(
            rows.log_softmax(dim=2),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        kept = torch.arange(len(terms.rows)) % 3 != 0
        rows = encoded.tensor.new_zeros(
            (len(terms.rows), VIEW.capacity, CHOICES.width)
        ).log_softmax(dim=2)

        return LogDistribution(
            rows.masked_fill(~kept.reshape(-1, 1, 1), -torch.inf),
            ("term", "slot", "choice"),
            kept,
            normalized_over="choice",
        )


@pytest.mark.parametrize("mask", [LIVE, None])
def test_a_cell_with_a_member_the_decoder_withholds_has_no_mass(
    mask: Tensor | None,
) -> None:
    declared = {PERMITTED: targets(term("N"), term("S"))}
    built = TermsReadout(
        WithholdingScorer(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets=declared,
        strategy=ReadoutMass.COUPLED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )
    handed = (
        torch.ones(ITEMS, SPANS.num_spans, dtype=torch.bool) if mask is None else mask
    )
    scored = torch.arange(2 * int(handed.sum())) % 3 != 0
    decided = torch.zeros(ITEMS, SPANS.num_spans, dtype=torch.bool)
    decided.view(-1)[handed.view(-1).nonzero().squeeze(1)] = scored.reshape(-1, 2).all(
        dim=1
    )

    published = built.publish(cells(mask=mask), spans=SPANS, marginals=marginals())

    assert torch.equal(published[KEY].computed, handed)
    assert torch.equal(published[PERMITTED].computed, decided & LIVE)
    assert bool(
        published[PERMITTED]
        .distribution.tensor[~published[PERMITTED].computed]
        .isneginf()
        .all()
    )
    assert bool(decided.any())
    assert bool((handed & ~decided).any())


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_a_superset_of_targets_never_publishes_less_mass(
    strategy: ReadoutMass,
) -> None:
    built = decoder()
    held = cells()
    nested = (
        targets(term("N")),
        targets(term("N"), term("S")),
        targets(term("N"), term("S"), slashed("S", "N")),
        targets(term("N"), term("S"), slashed("S", "N"), slashed("N", "N")),
    )
    declared = {ReadoutId(f"set-{size}"): held for size, held in enumerate(nested)}

    published = mass_readout(built, strategy=strategy, declared=declared).publish(
        held, spans=SPANS, marginals=marginals()
    )
    masses = [log_mass(published[key].distribution)[LIVE] for key in declared]

    for smaller, larger in pairwise(masses):
        assert bool((larger >= smaller).all())
        assert bool((larger > smaller).any())


def test_the_pair_sums_to_one_at_every_computed_cell_and_holds_nothing_elsewhere() -> (
    None
):
    published = mass_readout(decoder()).publish(
        cells(), spans=SPANS, marginals=marginals()
    )[ROOTS]
    pair = published.distribution.tensor

    assert torch.allclose(
        pair[LIVE].logsumexp(dim=-1), torch.zeros(int(LIVE.sum()), dtype=pair.dtype)
    )
    assert bool(pair[~LIVE].isneginf().all())
    assert bool(pair[LIVE].isfinite().all())


def test_a_span_no_derivation_builds_has_no_mass() -> None:
    held = marginals()
    built = held.built.clone()
    built[0, SPANS.span_id(1, 3)] = False
    thinner = SpanMarginals(
        SPANS,
        Encoded(held.constituent.tensor, SPAN_AXES, built),
        held.rule,
        held.lengths,
    )

    published = mass_readout(decoder()).publish(
        cells(), spans=SPANS, marginals=thinner
    )[ROOTS]

    assert torch.equal(published.computed, built | (LIVE & ~held.built))

    with pytest.raises(UncomputedCellError):
        published.at(0, 1, 3)


def test_a_cell_the_terms_never_computed_has_no_mass() -> None:
    mask = torch.tensor([True, False])

    published = mass_readout(decoder()).publish(
        cells(mask=mask), spans=SPANS, marginals=marginals()
    )[ROOTS]

    assert bool(published.computed[0].equal(LIVE[0]))
    assert not bool(published.computed[1].any())


def test_a_leaf_beyond_an_items_length_has_no_mass_even_where_its_type_is_read() -> (
    None
):
    published = mass_readout(decoder()).publish(
        cells(mask=None), spans=SPANS, marginals=marginals()
    )

    assert bool(published[KEY].computed.all())
    assert torch.equal(published[ROOTS].computed, LIVE)

    with pytest.raises(UncomputedCellError):
        published[ROOTS].at(1, 2, 3)

    assert published[ROOTS].at(1, 1, 2).axes == ("outcome",)


def test_duplicate_members_are_counted_once() -> None:
    built = decoder()
    held = cells()
    once = mass_readout(built, declared={ROOTS: targets(term("S"))}).publish(
        held, spans=SPANS, marginals=marginals()
    )
    twice = mass_readout(
        built, declared={ROOTS: targets(term("S"), term("S"))}
    ).publish(held, spans=SPANS, marginals=marginals())

    assert torch.equal(
        once[ROOTS].distribution.tensor, twice[ROOTS].distribution.tensor
    )


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_the_mass_has_finite_gradients_reaching_the_cells_and_the_marginals(
    strategy: ReadoutMass,
) -> None:
    built = decoder()
    held = cells()
    weighted = marginals()
    published = mass_readout(built, strategy=strategy).publish(
        held, spans=SPANS, marginals=weighted
    )[ROOTS]

    total = (
        log_mass(published.distribution)[LIVE].sum()
        + log_complement(published.distribution)[LIVE].sum()
    )
    total.backward()

    constituent = weighted.constituent.tensor
    interior = weighted.built & LIVE
    interior[0, SPANS.span_id(0, 3)] = False
    interior[1, SPANS.span_id(0, 2)] = False

    assert held.tensor.grad is not None
    assert bool(held.tensor.grad.isfinite().all())
    assert bool(held.tensor.grad[LIVE].abs().sum() > 0.0)
    assert not bool(held.tensor.grad[~LIVE].any())
    assert constituent.grad is not None
    assert bool(constituent.grad.isfinite().all())
    assert bool((constituent.grad[interior] != 0.0).all())
    assert not bool(constituent.grad[~interior].any())
    gradients = tuple(
        parameter.grad for parameter in built.parameters() if parameter.grad is not None
    )

    assert gradients
    assert all(bool(gradient.isfinite().all()) for gradient in gradients)


def test_a_mass_of_one_publishes_a_complement_of_zero_without_a_nan_gradient() -> None:
    log_mass_at = torch.tensor([0.0, -1.0, -torch.inf], requires_grad=True)

    rest = _log_one_minus(log_mass_at)
    rest[1].backward()

    assert rest.tolist() == pytest.approx(
        [-torch.inf, math.log1p(-math.exp(-1.0)), 0.0]
    )
    assert log_mass_at.grad is not None
    assert bool(log_mass_at.grad.isfinite().all())


def test_a_target_set_needs_the_marginals() -> None:
    with pytest.raises(ValueError, match="constituent marginal"):
        mass_readout(decoder()).publish(cells(), spans=SPANS)


def test_a_readout_without_targets_publishes_the_terms_alone() -> None:
    published = readout().publish(cells(), spans=SPANS, marginals=marginals())

    assert set(published) == {KEY}


def test_a_readout_without_a_fixed_source_keeps_injected_targets() -> None:
    permitted = TargetSet((term("N"),), MATCHING)
    built = _build(
        TermsReadoutSpec(
            space="categories",
            reads=ReadoutSource.INSIDE,
        ).settings(),
        component_context(
            codec=VIEW,
            component="terms",
            producers={"identity": decoder(), "space": RepresentedSpace()},
            targets={"semantic-clamp": permitted},
        ),
    )

    assert built.targets == (ReadoutId("semantic-clamp"),)
    assert built.target_sets == {ReadoutId("semantic-clamp"): permitted}


def test_marginals_over_another_index_are_refused() -> None:
    shorter = SpanIndex(LENGTH - 1)
    built = torch.tensor([[False, False, True]] * ITEMS)
    other = SpanMarginals(
        shorter,
        Encoded(
            torch.ones(ITEMS, shorter.num_spans, dtype=torch.float64), SPAN_AXES, built
        ),
        Encoded(
            torch.ones(ITEMS, shorter.num_spans, 1, dtype=torch.float64),
            SPLIT_AXES,
            built.unsqueeze(-1),
        ),
        torch.tensor([LENGTH - 1] * ITEMS),
    )

    with pytest.raises(ValueError, match="the readout over"):
        mass_readout(decoder()).publish(cells(), spans=SPANS, marginals=other)


def test_marginals_over_other_items_are_refused() -> None:
    with pytest.raises(ValueError, match="cover 2 items, and the readout 1"):
        mass_readout(decoder()).publish(
            Encoded(cell_tensor()[:1], CELL_AXES, LIVE[:1]),
            spans=SPANS,
            marginals=marginals(),
        )


def test_a_repeated_id_is_refused() -> None:
    with pytest.raises(ValueError, match="each id once"):
        mass_readout(decoder(), declared={KEY: targets(term("S"))})


def test_an_empty_target_set_is_refused() -> None:
    empty = LinearizedTargets(
        BatchEncoding(()), CODEC.index.PAD, MATCHING.name, TargetSet((), MATCHING)
    )

    with pytest.raises(ValueError, match="no member"):
        mass_readout(decoder(), declared={ROOTS: empty})


def test_an_empty_target_id_is_refused() -> None:
    with pytest.raises(ValueError, match="readout id"):
        mass_readout(decoder(), declared={ReadoutId(""): targets(term("S"))})


def test_the_coupled_strategy_needs_a_decoder_that_scores() -> None:
    with pytest.raises(TypeError, match="does not provide teacher-forced rows"):
        TermsReadout(
            Withholding(),
            codec=VIEW,
            index=CHOICES,
            reads=ReadoutSource.INSIDE,
            decoding=None,
            key=KEY,
            targets={ROOTS: targets(term("S"))},
            strategy=ReadoutMass.COUPLED,
            mapping=SOFTMAX,
            store=None,
            root_sort="Cat",
            max_nodes=None,
            clamp=None,
            meanings=DecodedMeaningSettings("decoded"),
            proof_terms=None,
            approximation=None,
            run_approximation=None,
        )


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_a_member_wider_than_the_decoders_slots_is_refused(
    strategy: ReadoutMass,
) -> None:
    deeper = TermCodec(
        INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=3)
    )
    wide = linearize_targets(TargetSet((term("S"),), MATCHING), deeper)

    with pytest.raises(ValueError, match="fills 7 slots"):
        mass_readout(decoder(), strategy=strategy, declared={ROOTS: wide}).publish(
            cells(), spans=SPANS, marginals=marginals()
        )


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_a_member_naming_a_symbol_the_decoder_does_not_read_is_refused(
    strategy: ReadoutMass,
) -> None:
    row = CODEC.encode(term("S"))
    foreign = LinearizedTargets(
        BatchEncoding((row._replace(symbols=(VIEW.n_symbols, *row.symbols[1:])),)),
        CODEC.index.PAD,
        MATCHING.name,
        TargetSet((term("S"),), MATCHING),
    )

    with pytest.raises(ValueError, match=f"names symbol {VIEW.n_symbols}"):
        mass_readout(decoder(), strategy=strategy, declared={ROOTS: foreign}).publish(
            cells(), spans=SPANS, marginals=marginals()
        )


def test_a_value_without_an_outcome_axis_is_not_a_mass_pair() -> None:
    published = readout().publish(cells(), spans=SPANS)[KEY]

    with pytest.raises(ValueError, match="outcome"):
        log_mass(published.distribution)

    with pytest.raises(ValueError, match="outcome"):
        log_complement(published.at(0, 0, 1))


def test_the_readout_reports_its_targets_and_strategy() -> None:
    built = mass_readout(decoder(), strategy=ReadoutMass.COUPLED)

    assert "roots" in repr(built)
    assert built.target_sets == {ROOTS: TargetSet((term("S"),), MATCHING)}

    with pytest.raises(TypeError):
        cast("dict[ReadoutId, TargetSet[Term]]", built.target_sets)[ROOTS] = TargetSet(
            (term("N"),), MATCHING
        )
    assert "COUPLED" in repr(built)


class MassReader:
    """An objective reading one target set's mass at every item's root."""

    requires = frozenset({"readouts"})

    def __init__(self, name: str, key: ReadoutId, *, complement: bool) -> None:
        self.name = name
        self._key = key
        self._complement = complement
        self.seen: list[SpanReadout] = []

    def evaluate(self, outputs: PublishedOutputs[tuple[str, ...]], /) -> LossTerm:
        published = outputs.readouts[self._key]
        self.seen.append(published)
        total = published.distribution.tensor.new_zeros(())

        for item, length in enumerate(LENGTHS):
            cell = published.at(item, 0, length)
            total = total - (
                log_complement(cell) if self._complement else log_mass(cell)
            )

        return LossTerm(total, len(LENGTHS))


def outputs_with(published: Mapping[ReadoutId, SpanReadout]) -> Outputs:
    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(
            LogDistribution(torch.where(LIVE, -1.0, -2.0).double(), SPAN_AXES, LIVE)
        ),
        batch=BATCH,
        readouts=published,
    )


def test_two_consumers_reading_the_mass_id_read_one_tensor_and_touch_no_cell() -> None:
    published = mass_readout(decoder()).publish(
        cells(), spans=SPANS, marginals=marginals()
    )
    outputs = outputs_with(published)
    acceptable = MassReader("acceptable", ROOTS, complement=False)
    unacceptable = MassReader("unacceptable", ROOTS, complement=True)
    terms = []

    for consumer in (acceptable, unacceptable):
        proxy = RecordingOutputs(
            outputs, objective=consumer.name, requires=consumer.requires
        )
        terms.append(consumer.evaluate(proxy))

        assert proxy.touched == {"readouts"}

    assert acceptable.seen[0] is unacceptable.seen[0]
    assert acceptable.seen[0] is published[ROOTS]
    assert (
        acceptable.seen[0].distribution.tensor
        is unacceptable.seen[0].distribution.tensor
    )
    assert terms[0].value.item() > 0.0
    assert terms[1].value.item() > 0.0


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_switching_the_strategy_changes_no_consumer(strategy: ReadoutMass) -> None:
    built = decoder()
    held = cells()
    published = mass_readout(built, strategy=strategy).publish(
        held, spans=SPANS, marginals=marginals()
    )
    consumer = MassReader("acceptable", ROOTS, complement=False)
    suite = ObjectiveSuite(
        [(1.0, consumer)], published=outputs_with(published).published
    )

    result = suite.evaluate(outputs_with(published))
    result.total.backward()

    assert math.isfinite(result.total.item())
    assert result.terms["acceptable"].count == ITEMS
    assert held.tensor.grad is not None
    assert bool(held.tensor.grad[LIVE].abs().sum() > 0.0)


class Counting:
    """A decoder recording how many terms each call reads or scores."""

    def __init__(self) -> None:
        self.read: list[int] = []
        self.scored: list[int] = []

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        terms = encoded.tensor.shape[0]
        self.read.append(terms)
        rows = encoded.tensor.new_zeros((terms, VIEW.capacity, CHOICES.width))

        return LogDistribution(
            rows.log_softmax(dim=2),
            ("term", "slot", "choice"),
            normalized_over="choice",
        )

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        self.scored.append(len(terms.rows))
        rows = encoded.tensor.new_zeros(
            (len(terms.rows), VIEW.capacity, CHOICES.width)
        ).log_softmax(dim=2)

        return LogDistribution(
            rows,
            ("term", "slot", "choice"),
            normalized_over="choice",
        )


def nested_sets(count: int) -> dict[ReadoutId, LinearizedTargets]:
    members = (term("N"), term("S"), slashed("S", "N"), slashed("N", "N"))

    return {
        ReadoutId(f"set-{size}"): targets(*members[: size + 1]) for size in range(count)
    }


@pytest.mark.parametrize("declared", [0, 1, 4])
def test_a_publish_reads_every_cell_once_however_many_target_sets_are_declared(
    declared: int,
) -> None:
    counting = Counting()
    built = TermsReadout(
        counting,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets=nested_sets(declared),
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )

    published = built.publish(cells(), spans=SPANS, marginals=marginals())

    assert len(published) == 1 + declared
    assert counting.read == [int(LIVE.sum())]
    assert counting.scored == []


@pytest.mark.parametrize("declared", [1, 4])
def test_the_coupled_strategy_scores_each_member_at_each_cell_once_per_set(
    declared: int,
) -> None:
    counting = Counting()
    built = TermsReadout(
        counting,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets=nested_sets(declared),
        strategy=ReadoutMass.COUPLED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )

    published = built.publish(cells(), spans=SPANS, marginals=marginals())

    assert len(published) == 1 + declared
    assert counting.read == [int(LIVE.sum())]
    assert counting.scored == [int(LIVE.sum()) * (size + 1) for size in range(declared)]


ORACLE_SIGNATURE: Final = Signature(
    "oracle-fixture",
    ("Cat",),
    tuple(
        Operator(name, "Cat", family="primitive")
        for name in ("S", "NP", "VP", "D", "N", "V")
    ),
)
ORACLE_CODEC: Final = TermCodec(
    OperatorIndex(ORACLE_SIGNATURE),
    CompleteTree(ORACLE_SIGNATURE, ORACLE_SIGNATURE.operators, order="df", depth=1),
)
ORACLE_VIEW: Final = NeuralCodecView(ORACLE_CODEC)
ORACLE_CHOICES: Final = ChoiceIndex(
    ORACLE_SIGNATURE,
    depth_bounds=dict.fromkeys(ORACLE_SIGNATURE.sorts, ORACLE_CODEC.capacity),
)
ORACLE_STORE: Final = TermStore(ORACLE_SIGNATURE)


def category(name: str) -> Term:
    return ORACLE_STORE.op(ORACLE_SIGNATURE.by_name(name))


class Production(NamedTuple):
    """One binary rewrite of the oracle grammar with its weight."""

    parent: str
    left: str
    right: str
    weight: float


PRODUCTIONS: Final = (
    Production("S", "NP", "VP", 0.9),
    Production("NP", "D", "N", 0.8),
    Production("VP", "V", "NP", 0.7),
    Production("NP", "NP", "NP", 0.2),
    Production("VP", "VP", "NP", 0.3),
)
LEXICON: Final[Mapping[str, Mapping[str, float]]] = {
    "the": {"D": 1.0},
    "dogs": {"N": 0.6, "NP": 0.4},
    "cats": {"N": 0.5, "NP": 0.5},
    "see": {"V": 0.7, "VP": 0.3},
    "fish": {"N": 0.3, "NP": 0.3, "V": 0.2, "VP": 0.2},
    "walk": {"V": 0.5, "VP": 0.4, "N": 0.1},
}
SENTENCES: Final = 20
LONGEST: Final = 5


def expand(rng: random.Random, symbol: str) -> list[str]:
    words = [word for word, readings in LEXICON.items() if symbol in readings]
    rewrites = [rule for rule in PRODUCTIONS if rule.parent == symbol]
    lexical = sum(LEXICON[word][symbol] for word in words)
    total = lexical + sum(rule.weight for rule in rewrites)

    if rng.random() * total < lexical:
        return [rng.choice(words)]

    rule = rng.choices(rewrites, weights=[rule.weight for rule in rewrites])[0]

    return expand(rng, rule.left) + expand(rng, rule.right)


def sentences(seed: int) -> tuple[tuple[str, ...], ...]:
    rng = random.Random(seed)
    drawn: list[tuple[str, ...]] = []

    while len(drawn) < SENTENCES:
        words = expand(rng, "S")

        if len(words) <= LONGEST:
            drawn.append(tuple(words))

    return tuple(drawn)


def forest(words: tuple[str, ...]) -> PackedForest[Term, float]:
    index = SpanIndex(len(words))
    built = PackedForest[Term, float](index, LogInside())

    for position, word in enumerate(words):
        for reading, weight in LEXICON[word].items():
            built.add(
                ItemKey((position, position + 1), category(reading), "lex"),
                Edge(word, (), category(reading), math.log(weight)),
            )

    for i, j in index.spans_by_length(min_length=2):
        for k in index.splits(i, j):
            for left in built.items_at((i, k)):
                for right in built.items_at((k, j)):
                    for rule in PRODUCTIONS:
                        if left.reading is category(
                            rule.left
                        ) and right.reading is category(rule.right):
                            built.add(
                                ItemKey((i, j), category(rule.parent), "step"),
                                Edge(
                                    "-".join(rule[:3]),
                                    (left, right),
                                    category(rule.parent),
                                    math.log(rule.weight),
                                    split=k,
                                ),
                            )

    return built


class Parsed(NamedTuple):
    """One sentence of the oracle fixture with its inside-outside result."""

    words: tuple[str, ...]
    chart: PackedForest[Term, float]
    outside: Outside[Term]


ORACLE_ROOTS: Final = TargetSet((category("S"),), MATCHING)


def parsed(seed: int = 20260917) -> tuple[Parsed, ...]:
    charts = tuple(forest(words) for words in sentences(seed))

    return tuple(
        Parsed(words, chart, outside_from_edges(chart, RootGoal(ORACLE_ROOTS)))
        for words, chart in zip(sentences(seed), charts, strict=True)
    )


class Batch(NamedTuple):
    """The oracle fixture as one batch of cells and marginals."""

    spans: SpanIndex
    cells: Encoded
    marginals: SpanMarginals
    table: Tensor


def batched(items: tuple[Parsed, ...]) -> Batch:
    spans = SpanIndex(max(len(item.words) for item in items))
    count = spans.num_spans
    symbols = ORACLE_CODEC.n_symbols
    live = torch.zeros(len(items), count, dtype=torch.bool)
    built = torch.zeros(len(items), count, dtype=torch.bool)
    valid = torch.zeros(len(items), count, spans.max_splits, dtype=torch.bool)
    constituent = torch.zeros(len(items), count, dtype=torch.float64)
    rule = torch.zeros(len(items), count, spans.max_splits, dtype=torch.float64)
    table = torch.full((len(items), count, symbols), -torch.inf, dtype=torch.float64)
    uniform = torch.full((symbols,), -torch.inf, dtype=torch.float64)
    uniform[1:] = -math.log(symbols - 1)

    for item, (words, chart, outside) in enumerate(items):
        for i, j in spans.spans_by_length():
            if j > len(words):
                continue

            span = spans.span_id(i, j)
            live[item, span] = True
            keys = chart.items_at((i, j))
            weights = [outside.constituent_marginal(key) for key in keys]
            constituent[item, span] = sum(weights)

            if sum(weights) > 0.0:
                for key, weight in zip(keys, weights, strict=True):
                    if weight > 0.0:
                        code = ORACLE_CODEC.encode(key.reading).symbols[0]
                        table[item, span, code] = math.log(weight / sum(weights))

            else:
                table[item, span] = uniform

            if j - i > 1:
                built[item, span] = True

                for k in spans.splits(i, j):
                    _, column = spans.split_id(i, j, k)
                    valid[item, span, column] = True

                for key in keys:
                    for edge, weight in outside.rule_marginals(key):
                        if edge.split is not None:
                            _, column = spans.split_id(i, j, edge.split)
                            rule[item, span, column] += weight

    addresses = torch.stack(
        torch.meshgrid(torch.arange(len(items)), torch.arange(count), indexing="ij"),
        dim=-1,
    ).to(torch.float64)

    return Batch(
        spans,
        Encoded(addresses, CELL_AXES, live),
        SpanMarginals(
            spans,
            Encoded(constituent, SPAN_AXES, built),
            Encoded(rule, SPLIT_AXES, valid),
            torch.tensor([len(item.words) for item in items]),
        ),
        table,
    )


class Oracle:
    """A decoder reading a symbolic chart's conditional item distribution per cell."""

    def __init__(self, table: Tensor) -> None:
        self._table = table

    def marginal_rows(self, encoded: Encoded, /, *, root_sort: str) -> LogDistribution:
        del root_sort
        at = encoded.tensor.long()
        rows = self._table[at[:, 0], at[:, 1]].unsqueeze(1)

        return LogDistribution(
            rows, ("term", "slot", "choice"), normalized_over="choice"
        )

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        at = encoded.tensor.long()
        rows = self._table[at[:, 0], at[:, 1]].unsqueeze(1)

        return LogDistribution(
            rows, ("term", "slot", "choice"), normalized_over="choice"
        )


ORACLE_TARGETS: Final = {
    ROOTS: ORACLE_ROOTS,
    PERMITTED: TargetSet((category("NP"), category("N")), MATCHING),
    ReadoutId("phrases"): TargetSet((category("VP"),), MATCHING),
    ReadoutId("everything"): TargetSet(
        tuple(category(op.name) for op in ORACLE_SIGNATURE.operators), MATCHING
    ),
}


def test_the_oracle_fixture_holds_twenty_sentences_with_interior_ambiguity() -> None:
    items = parsed()
    lengths = {len(item.words) for item in items}
    interior = [
        item.outside.constituent_marginal(key)
        for item in items
        for i, j in item.chart.index.spans_by_length(min_length=2)
        if j < len(item.words) or i > 0
        for key in item.chart.items_at((i, j))
    ]

    assert len(items) == SENTENCES
    assert lengths >= {2, 3, 4}
    assert any(0.0 < weight < 1.0 for weight in interior)


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_the_tensor_mass_agrees_with_the_symbolic_item_marginal(
    strategy: ReadoutMass,
) -> None:
    items = parsed()
    batch = batched(items)
    declared = {
        key: linearize_targets(held, ORACLE_CODEC)
        for key, held in ORACLE_TARGETS.items()
    }
    readout = TermsReadout(
        Oracle(batch.table),
        codec=ORACLE_VIEW,
        index=ORACLE_CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets=declared,
        strategy=strategy,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    )

    published = readout.publish(
        batch.cells, spans=batch.spans, marginals=batch.marginals
    )

    for key, held in ORACLE_TARGETS.items():
        for item, (words, chart, outside) in enumerate(items):
            for i, j in batch.spans.spans_by_length():
                if j > len(words):
                    continue

                expected = target_mass(chart, outside, held, (i, j))
                cell = published[key].at(item, i, j)

                assert math.exp(log_mass(cell).item()) == pytest.approx(
                    expected, abs=1e-6
                )
                assert math.exp(log_complement(cell).item()) == pytest.approx(
                    1.0 - expected, abs=1e-6
                )


def test_the_oracle_root_and_leaves_carry_the_whole_conditional_mass() -> None:
    items = parsed()
    batch = batched(items)
    published = TermsReadout(
        Oracle(batch.table),
        codec=ORACLE_VIEW,
        index=ORACLE_CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={
            ReadoutId("everything"): linearize_targets(
                ORACLE_TARGETS[ReadoutId("everything")], ORACLE_CODEC
            )
        },
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(batch.cells, spans=batch.spans, marginals=batch.marginals)

    for item, (words, _, _) in enumerate(items):
        whole = [(0, len(words)), *((k, k + 1) for k in range(len(words)))]

        for i, j in whole:
            cell = published[ReadoutId("everything")].at(item, i, j)

            assert log_mass(cell).item() == pytest.approx(0.0, abs=1e-9)
            assert math.exp(log_complement(cell).item()) == pytest.approx(0.0, abs=1e-9)


def test_a_mass_over_one_by_rounding_is_published_as_one() -> None:
    spans = SpanIndex(1)
    table = torch.full((1, 1, ORACLE_CODEC.n_symbols), -torch.inf, dtype=torch.float64)
    table[0, 0, ORACLE_CODEC.encode(category("S")).symbols[0]] = 1e-9
    published = TermsReadout(
        Oracle(table),
        codec=ORACLE_VIEW,
        index=ORACLE_CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        targets={ROOTS: linearize_targets(ORACLE_ROOTS, ORACLE_CODEC)},
        strategy=ReadoutMass.FACTORIZED,
        mapping=SOFTMAX,
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(
        Encoded(torch.zeros(1, 1, 2, dtype=torch.float64), CELL_AXES),
        spans=spans,
        marginals=SpanMarginals(
            spans,
            Encoded(
                torch.zeros(1, 1, dtype=torch.float64),
                SPAN_AXES,
                torch.zeros(1, 1, dtype=torch.bool),
            ),
            Encoded(
                torch.zeros(1, 1, 0, dtype=torch.float64),
                SPLIT_AXES,
                torch.zeros(1, 1, 0, dtype=torch.bool),
            ),
            torch.tensor([1]),
        ),
    )
    cell = published[ROOTS].at(0, 0, 1)

    assert log_mass(cell).item() == 0.0
    assert log_complement(cell).item() == -torch.inf


def test_the_readout_publishes_through_the_dense_mapping_by_default() -> None:
    built = readout()

    assert built.mapping == SOFTMAX
    assert "Entmax(1.0, bisections=0)" in repr(built)


def test_a_sparse_readout_leaves_most_symbols_at_exactly_zero_mass() -> None:
    shared = decoder()
    dense = readout(shared).publish(cells(), spans=SPANS)[KEY]
    sparse = TermsReadout(
        shared,
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        mapping=Entmax(2.0, bisections=50),
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(cells(), spans=SPANS)[KEY]
    computed = dense.computed
    legal = dense.distribution.tensor[computed].exp() > 0.0
    kept = sparse.distribution.tensor[computed].exp() > 0.0

    assert bool((kept <= legal).all())
    assert int(kept.sum()) < int(legal.sum())


def test_a_sparse_readout_still_publishes_a_normalized_distribution() -> None:
    published = TermsReadout(
        decoder(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        mapping=Entmax(1.5, bisections=50),
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(cells(), spans=SPANS)[KEY]
    kept = published.distribution.tensor[published.computed].exp().sum(dim=-1)

    assert torch.allclose(kept, torch.ones_like(kept))


def test_a_sparse_readout_leaves_the_cells_it_never_computed_at_no_mass() -> None:
    published = TermsReadout(
        decoder(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        mapping=Entmax(1.5, bisections=50),
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(cells(), spans=SPANS)[KEY]

    assert bool(published.distribution.tensor[~published.computed].isneginf().all())


def test_a_sparse_readout_keeps_a_gradient_to_the_cells_it_read() -> None:
    held = cells()
    published = TermsReadout(
        decoder(),
        codec=VIEW,
        index=CHOICES,
        reads=ReadoutSource.INSIDE,
        decoding=None,
        key=KEY,
        mapping=Entmax(1.5, bisections=50),
        store=None,
        root_sort="Cat",
        max_nodes=None,
        clamp=None,
        targets={},
        strategy=ReadoutMass.FACTORIZED,
        meanings=DecodedMeaningSettings("decoded"),
        proof_terms=None,
        approximation=None,
        run_approximation=None,
    ).publish(held, spans=SPANS)[KEY]
    kept = published.distribution.tensor[published.computed]
    torch.where(kept.isneginf(), kept.new_zeros(()), kept).sum().backward()

    assert held.tensor.grad is not None
    assert bool(held.tensor.grad[LIVE].abs().sum() > 0.0)


def zeroed(published: Mapping[ReadoutId, SpanReadout], member: Term) -> Tensor:
    """Return where the published terms leave a member's symbols at zero mass."""
    row = CODEC.encode(member)
    rows = published[KEY].distribution.tensor
    picked = rows[:, :, torch.arange(len(row.symbols)), torch.tensor(row.symbols)]

    return (picked.isneginf() & torch.tensor(row.occupied)).any(dim=-1)


def published_mass(
    built: RecursiveTermDecoder,
    held: Encoded,
    strategy: ReadoutMass,
    members: tuple[Term, ...],
    mapping: Entmax = SOFTMAX,
) -> Mapping[ReadoutId, SpanReadout]:
    """Return one target set's mass, published at one strategy and one mapping."""
    return mass_readout(
        built,
        strategy=strategy,
        declared={PERMITTED: targets(*members)},
        mapping=mapping,
    ).publish(held, spans=SPANS, marginals=marginals())


def sparse_and_dense(
    strategy: ReadoutMass, member: Term
) -> tuple[Mapping[ReadoutId, SpanReadout], Mapping[ReadoutId, SpanReadout]]:
    """Return one target set published through a sparse and through a dense mapping."""
    built = decoder()
    held = cells()

    return (
        published_mass(built, held, strategy, (member,), Entmax(2.0, bisections=50)),
        published_mass(built, held, strategy, (member,)),
    )


def dropped_cells(
    sparse: Mapping[ReadoutId, SpanReadout],
    dense: Mapping[ReadoutId, SpanReadout],
    gone: Term,
    held_on: Term,
) -> Tensor:
    """Return the published cells where the sparse rows zero one member, not another."""
    return zeroed(sparse, gone) & ~zeroed(sparse, held_on) & dense[PERMITTED].computed


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_a_sparse_readout_gives_no_mass_to_a_target_it_zeroes(
    strategy: ReadoutMass,
) -> None:
    sparse, dense = sparse_and_dense(strategy, term("N"))
    off = zeroed(sparse, term("N")) & LIVE

    assert bool(off.any())
    assert bool(log_mass(sparse[PERMITTED].distribution)[off].isneginf().all())
    assert bool(log_mass(dense[PERMITTED].distribution)[off].isfinite().all())


@pytest.mark.parametrize("strategy", list(ReadoutMass))
def test_the_mapping_restricts_a_coupled_mass_and_reweights_a_factorized_one(
    strategy: ReadoutMass,
) -> None:
    built = decoder()
    held = cells()
    both = (term("N"), term("S"))
    sparse = published_mass(built, held, strategy, both, Entmax(2.0, bisections=50))
    dense = published_mass(built, held, strategy, both)
    surviving = published_mass(built, held, strategy, both[1:])
    dropped = dropped_cells(sparse, dense, term("N"), term("S"))

    assert bool(dropped.any())

    restricted = log_mass(sparse[PERMITTED].distribution)[dropped]
    whole = log_mass(dense[PERMITTED].distribution)[dropped]
    remaining = log_mass(surviving[PERMITTED].distribution)[dropped]

    if strategy is ReadoutMass.COUPLED:
        assert bool((restricted < whole - 1e-9).all())
        assert torch.allclose(restricted, remaining)

    else:
        assert not torch.allclose(restricted, whole)
        assert not torch.allclose(restricted, remaining)


def test_a_sparse_coupled_mass_drops_a_zeroed_member_with_no_cell_mask() -> None:
    built = decoder()
    held = cells(mask=None)
    both = (term("N"), term("S"))
    strategy = ReadoutMass.COUPLED
    sparse = published_mass(built, held, strategy, both, Entmax(2.0, bisections=50))
    dense = published_mass(built, held, strategy, both)
    surviving = published_mass(built, held, strategy, both[1:])
    dropped = dropped_cells(sparse, dense, term("N"), term("S"))

    assert sparse[KEY].distribution.mask is None
    assert bool(dropped.any())

    restricted = log_mass(sparse[PERMITTED].distribution)[dropped]

    assert bool(
        (restricted < log_mass(dense[PERMITTED].distribution)[dropped] - 1e-9).all()
    )
    assert torch.allclose(
        restricted, log_mass(surviving[PERMITTED].distribution)[dropped]
    )


def test_the_coupled_mass_follows_the_mapping_the_readout_publishes_through() -> None:
    built = decoder()
    held = cells()
    declared = {PERMITTED: targets(term("N"), term("S"), slashed("S", "N"))}
    sparse = mass_readout(
        built,
        strategy=ReadoutMass.COUPLED,
        declared=declared,
        mapping=Entmax(2.0, bisections=50),
    ).publish(held, spans=SPANS, marginals=marginals())
    dense = mass_readout(
        built, strategy=ReadoutMass.COUPLED, declared=declared
    ).publish(held, spans=SPANS, marginals=marginals())
    thinned = log_mass(sparse[PERMITTED].distribution)[LIVE]
    whole = log_mass(dense[PERMITTED].distribution)[LIVE]

    assert bool((thinned <= whole + 1e-9).all())
    assert bool((thinned < whole - 1e-9).any())


def test_the_readout_hands_out_its_decoder_s_codec() -> None:
    built = decoder()

    assert readout(built).codec is built.codec


def test_each_target_set_is_handed_out_as_its_distinct_members_under_its_id() -> None:
    nominal = targets(term("N"), term("N"), slashed("N", "N"))
    clausal = targets(term("S"))
    published = mass_readout(decoder(), declared={ROOTS: clausal, PERMITTED: nominal})

    encodings = published.target_encodings

    assert tuple(encodings) == (ROOTS, PERMITTED)
    assert encodings[ROOTS] == clausal.encoded
    assert encodings[PERMITTED].rows == tuple(dict.fromkeys(nominal.encoded.rows))
    assert len(encodings[PERMITTED].rows) == 2
    assert readout().target_encodings == {}
