"""Benchmarks the cumulative-link head with crossed random effects."""

from typing import TYPE_CHECKING, Final, Literal

import pytest
import torch
from tests.conftest import counted_constructions

from lofi.judgment import (
    ConditionalLikelihood,
    Cumulative,
    CumulativeLinkHead,
    CumulativeParameters,
    DesignCodec,
    FixedDiscrimination,
    JudgmentRow,
    Predictor,
    PriorBlock,
    RandomEffectStore,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget
    from torch import Tensor

    from lofi.judgment import JudgmentDesign

BATCH: Final = 512
LEVELS: Final = 7
PARTICIPANTS: Final = 727
ITEMS: Final = 2_000
READOUT: Final = "p_gram"
DTYPE: Final = torch.float32


class Readout(Predictor, extra="forbid"):
    kind: Literal["head_benchmark_readout"] = "head_benchmark_readout"
    name: str


class ReadoutPredictor:
    def __init__(self, name: str) -> None:
        self._terms = (Readout(name=name),)
        self._name = name

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        return design.readouts[self._name]


class Fitted:
    """The head, its parameters, the two stores and one gathered batch."""

    def __init__(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        participants = tuple(f"participant-{index}" for index in range(PARTICIPANTS))
        items = tuple(f"item-{index}" for index in range(ITEMS))
        drawn_participants = torch.randint(
            0, PARTICIPANTS, (BATCH,), generator=generator
        )
        drawn_items = torch.randint(0, ITEMS, (BATCH,), generator=generator)
        responses = torch.randint(0, LEVELS, (BATCH,), generator=generator)
        scores = torch.randn(BATCH, generator=generator)
        rows = tuple(
            JudgmentRow(
                participants[int(participant)],
                items[int(item)],
                int(response),
                RowColumns(factors={}, readouts={READOUT: float(score)}),
            )
            for participant, item, response, score in zip(
                drawn_participants, drawn_items, responses, scores, strict=True
            )
        )
        declared = ResponseSpec(
            family=Cumulative(),
            levels=LEVELS,
            objective=ConditionalLikelihood(),
            predictor=(Readout(name=READOUT),),
            identification=FixedDiscrimination(value=1.0),
            priors=PriorBlock(thresholds=ThresholdPriors()),
        )
        self.head = CumulativeLinkHead(declared, ReadoutPredictor(READOUT))
        self.design = DesignCodec(LEVELS, readouts=(READOUT,)).encode(rows)
        self.parameters = CumulativeParameters(
            torch.tensor(-1.0, dtype=DTYPE, requires_grad=True),
            torch.zeros(LEVELS - 1, dtype=DTYPE, requires_grad=True),
        )
        self.by_participant = RandomEffectStore(
            "participant", participants, cutpoints=LEVELS - 1, dtype=DTYPE
        )
        self.by_item = RandomEffectStore("item", items, cutpoints=0, dtype=DTYPE)
        self.participant_index = self.by_participant.align(
            self.design.participant_levels
        )[self.design.participants]
        self.item_index = self.by_item.align(self.design.item_levels)[self.design.items]

    def step(self) -> Tensor:
        """Run one forward and backward pass over the batch."""
        self.by_participant.zero_grad(set_to_none=True)
        self.by_item.zero_grad(set_to_none=True)
        effects = self.by_participant.effects(
            self.participant_index
        ) + self.by_item.effects(self.item_index)
        loss = -self.head.log_likelihood(
            self.design, self.parameters, effects=effects
        ).mean()
        loss.backward()

        return loss.detach()


def test_the_benchmark_batch_gathers_every_row_through_both_stores() -> None:
    fitted = Fitted(seed=3)

    assert fitted.design.n_rows == BATCH
    assert int(fitted.participant_index.max()) < fitted.by_participant.fallback
    assert int(fitted.item_index.max()) < fitted.by_item.fallback
    assert torch.isfinite(fitted.step())


def test_a_hundred_head_steps_construct_no_didactic_model() -> None:
    fitted = Fitted(seed=3)

    with counted_constructions() as constructions:
        for _ in range(100):
            fitted.step()

    assert constructions.count == 0


def test_the_head_runs_a_forward_and_backward_over_a_batch_of_512(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    fitted = Fitted(seed=3)
    loss = benchmark.pedantic(
        fitted.step,
        rounds=20,
        warmup_rounds=2,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the head ran but pytest-benchmark timed nothing")

    benchmark.extra_info["batch"] = BATCH
    benchmark.extra_info["levels"] = LEVELS
    benchmark.extra_info["participants"] = PARTICIPANTS
    benchmark.extra_info["items"] = ITEMS
    benchmark.extra_info["dtype"] = str(DTYPE)
    benchmark.extra_info["device"] = loss.device.type
    benchmark.extra_info["budget"] = "judgment-head"
    benchmark.extra_info["median_ms_per_step"] = measured.stats.median * 1e3
    mean_ms = measured.stats.mean * 1e3
    benchmark.extra_info["mean_ms_per_step"] = mean_ms

    if wall_clock_budget.enforced:
        assert mean_ms <= 8.0
