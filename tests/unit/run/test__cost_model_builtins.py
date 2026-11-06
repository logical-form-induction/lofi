from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl, UniformLengthSpec
from lofi.run import (
    GridCell,
    GridTiming,
    LengthBatchPricing,
    StackBootstrapPricing,
)
from lofi.run._cost_model import cost_model
from lofi.run._cost_model_builtins import reference_stacks
from lofi.spec.trainers.parser import ParserTrainerSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.run import PointSeed
    from lofi.spec import RunSpec


class Grid:
    def __init__(self) -> None:
        self.timed: list[tuple[tuple[tuple[int, int], ...], str, int, int]] = []

    def time_grid(
        self,
        stacks: Mapping[tuple[int, int], RunSpec],
        point: PointSeed,
        /,
        *,
        device: str,
        warmup: int,
        repeats: int,
    ) -> GridTiming:
        del point
        self.timed.append((tuple(stacks), device, warmup, repeats))
        return GridTiming(
            [
                GridCell(length, batch, tuple(0.1 for _ in range(repeats)))
                for length, batch in stacks
            ]
        )


def test_a_stack_bootstrap_times_no_reference_grid() -> None:
    timer = Grid()
    model = cost_model(StackBootstrapPricing())

    assert model.reference(timer, device="cpu", warmup=0, repeats=2) is None
    assert timer.timed == []
    assert model.resamples == 10_000


def test_a_length_batch_model_times_its_declared_grid() -> None:
    timer = Grid()
    model = cost_model(LengthBatchPricing(lengths=(5, 9), batches=(2, 4), resamples=7))

    reference = model.reference(timer, device="cpu", warmup=1, repeats=2)

    assert reference is not None
    assert timer.timed == [(((5, 2), (5, 4), (9, 2), (9, 4)), "cpu", 1, 2)]
    assert model.resamples == 7


def test_the_reference_document_is_composed_under_each_cells_overrides() -> None:
    stacks = reference_stacks(LengthBatchPricing(lengths=(5, 9), batches=(2, 4)))

    for (length, batch), spec in stacks.items():
        dataset = spec.data.datasets["train"].dataset
        trainer = spec.trainer

        assert isinstance(dataset, SyntheticCorpusDecl)
        assert dataset.length == UniformLengthSpec(min_length=length, max_length=length)
        assert isinstance(trainer, ParserTrainerSpec)
        assert trainer.batch_size == batch


def test_a_cell_override_naming_another_placeholder_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="substitutes only"):
        LengthBatchPricing(cell_overrides=("trainer.batch_size={size}",))
