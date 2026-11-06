from typing import TYPE_CHECKING, Final

import pytest
import torch
from torch import nn

from lofi.data import Alignment
from lofi.judgment import (
    PARTICIPANT,
    ApproximateMarginalLikelihood,
    Cumulative,
    DesignCodec,
    FixedDiscrimination,
    GroupedJudgmentSet,
    JudgmentRow,
    OffsetTerm,
    OrdinalHead,
    PooledEncoderObjective,
    PriorBlock,
    RandomIntercept,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    TruncatedMarginalLikelihood,
)
from lofi.nn._values import Encoded, TokenEncoding
from lofi.nn.sites import ComponentId, SiteMap
from lofi.train import (
    LossAssembly,
    OptimizationResult,
    WeightedObjective,
)
from lofi.train._optimizing import over_no_span

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from lofi.judgment.grouped import _GroupedJudgmentBatch
    from lofi.nn.loss._values import ObjectiveOutputs

pytestmark = pytest.mark.integration

SCORE: Final = "score"


class Encoder(nn.Module):
    """One scalar row per word."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        trainable = False
        self.scale.requires_grad_(trainable)

    @property
    def output_width(self) -> int:
        """Return the scalar row width."""
        return 1

    def encode_words(self, items: Sequence[Sequence[str]], /) -> TokenEncoding:
        """Return one live scalar row per word.

        Parameters
        ----------
        items : Sequence[Sequence[str]]
            Words of each neural item.
        """
        longest = max(map(len, items))
        values = self.scale.new_zeros((len(items), longest, 1))
        mask = torch.zeros((len(items), longest), dtype=torch.bool)
        alignments: list[Alignment] = []

        for item, words in enumerate(items):
            length = len(words)
            values[item, :length, 0] = torch.arange(
                1,
                length + 1,
                dtype=values.dtype,
            )
            mask[item, :length] = True
            alignments.append(
                Alignment(tuple((position, position + 1) for position in range(length)))
            )

        return TokenEncoding(
            Encoded(values, ("item", "row", "width"), mask),
            tuple(alignments),
            words=items,
        )

    def site_map(self) -> SiteMap:
        """Return an empty published site map."""
        return SiteMap.custom(ComponentId("grouped-integration"), {})


class BatchForward:
    """Publish a grouped judgment batch without a parser chart."""

    @property
    def publishes(self) -> frozenset[str]:
        """Return the sole output this forward pass publishes."""
        return frozenset({"batch"})

    def publish(
        self, batch: _GroupedJudgmentBatch, /
    ) -> ObjectiveOutputs[_GroupedJudgmentBatch]:
        """Publish the batch over an empty chart.

        Parameters
        ----------
        batch : _GroupedJudgmentBatch
            Complete participant-group view to publish.
        """
        return over_no_span(batch, len(batch.words))


class Descent:
    """One finite stochastic-gradient step."""

    def __init__(self, parameters: Sequence[nn.Parameter]) -> None:
        self._optimizer = torch.optim.SGD(parameters, lr=0.01)
        self.steps = 0

    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        """Take one optimizer step over a freshly evaluated loss.

        Parameters
        ----------
        forward : Callable[[], Tensor]
            Computation returning the scalar loss.
        """
        self._optimizer.zero_grad()
        loss = forward()
        loss.backward()
        self._optimizer.step()
        self.steps += 1

        return OptimizationResult(
            loss_finite=bool(torch.isfinite(loss)),
            stepped=True,
            scale=None,
        )


def grouped_set(*, reverse_rows: bool = False) -> GroupedJudgmentSet:
    """Return two complete participant groups over two neural items."""
    item_ids = ("item-a", "item-b")
    rows = tuple(
        JudgmentRow(
            participant,
            item_id,
            response,
            RowColumns({}, {SCORE: 0.0}),
        )
        for participant, responses in (("p-a", (0, 2)), ("p-b", (1, 2)))
        for item_id, response in zip(item_ids, responses, strict=True)
    )

    if reverse_rows:
        rows = tuple(reversed(rows))

    return GroupedJudgmentSet(
        DesignCodec(3, readouts=(SCORE,)).encode(rows),
        item_ids,
        (("short",), ("two", "words")),
    )


def training_components() -> tuple[
    PooledEncoderObjective,
    LossAssembly[_GroupedJudgmentBatch],
    Descent,
]:
    """Return one real grouped objective, assembly and optimization policy."""
    objective = PooledEncoderObjective(Encoder(), OrdinalHead(3)).double()
    assembly = LossAssembly((WeightedObjective(objective.name, 1.0, objective),))
    policy = Descent(tuple(objective.parameters()))

    return objective, assembly, policy


def test_row_and_requested_group_order_preserve_layout_mapping_and_objective() -> None:
    forward = grouped_set().select((0, 1))
    reordered = grouped_set(reverse_rows=True).select((1, 0))
    objective, _, _ = training_components()
    forward_term = objective.evaluate(BatchForward().publish(forward))
    reordered_term = objective.evaluate(BatchForward().publish(reordered))

    assert reordered.active_groups == forward.active_groups == (0, 1)
    assert reordered.design.participant_levels == forward.design.participant_levels
    assert reordered.design.item_levels == forward.design.item_levels
    assert reordered.item_ids == forward.item_ids
    assert reordered.words == forward.words
    assert reordered.response_items.tolist() == list(
        reversed(forward.response_items.tolist())
    )
    assert torch.allclose(reordered_term.value, forward_term.value)


def test_the_same_batch_mapping_binds_neural_readouts_for_the_marginal() -> None:
    batch = grouped_set().select((1,))
    declaration = ResponseSpec(
        family=Cumulative(),
        levels=3,
        objective=TruncatedMarginalLikelihood(
            group=PARTICIPANT,
            radius=2.5,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8,
            quadrature_points=8,
            quadrature_seed=37,
        ),
        predictor=(RandomIntercept(group=PARTICIPANT), OffsetTerm(readout=SCORE)),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    program = ResponseCompiler({}).emit(declaration, batch.design)
    neural = torch.tensor([0.25, -0.4], dtype=torch.float64, requires_grad=True)
    response_readout = batch.response_rows(neural)
    marginal = ApproximateMarginalLikelihood(
        program.with_readouts({SCORE: response_readout}),
        active_groups=batch.active_groups,
    )
    population = torch.full(
        (marginal.population_size,),
        0.2,
        dtype=torch.float64,
        requires_grad=True,
    )

    value = marginal(population)
    value.backward()

    assert response_readout.tolist() == [0.25, -0.4]
    assert marginal.active_groups == (1,)
    assert bool(torch.isfinite(value))
    assert neural.grad is not None
    assert bool(torch.isfinite(neural.grad).all())
    assert population.grad is not None
    assert bool(torch.isfinite(population.grad).all())
