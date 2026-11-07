"""Benchmark pair-valued featurization against a parser step."""

import statistics
import time
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from tests.conftest import counted_constructions

from lofi import _checks
from lofi.data import (
    Example,
    Inventory,
    Item,
    ItemHasher,
    ItemId,
    Relational,
    Supervision,
    Token,
    _batch,
)
from lofi.data._join import ItemLookup
from lofi.data.featurizers import _core
from lofi.data.featurizers import pair as pair_featurizer
from lofi.run import ExperimentSeed, LengthBatchPricing
from lofi.run._cost_model_builtins import reference_stacks
from lofi.run._timing import ReferenceStep, prepare_reference

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.conftest import ChecksReload
    from torch import Tensor

    from lofi.data.featurizers.pair import PairFeaturizer

EXAMPLES: Final = 8
"""The number of pair examples in one featurizer call."""

ROWS_PER_EXAMPLE: Final = 2
"""The distinct items a relational example adds to the lexical batch."""

FEATURIZED_ROWS: Final = EXAMPLES * ROWS_PER_EXAMPLE
"""The item rows consumed by the paired parser step."""

SENTENCE_LENGTH: Final = 10
"""The number of tokens in every priced item."""

STEP_ROUNDS: Final = 5
"""The number of parser steps summarized by the denominator median."""

PAD: Final = "<pad>"
"""The reserved padding symbol used by the featurizer."""

WORDS: Final = (
    "the",
    "big",
    "dog",
    "quietly",
    "sees",
    "a",
    "small",
    "cat",
    "near",
    "fish",
)
"""One fixture sentence drawn from the packaged grammar's vocabulary."""


class TorchCodeTensorizer:
    """A tensorizer producing padded integer rows and presence masks."""

    def __call__(
        self,
        rows: Sequence[tuple[int, ...]],
        /,
        *,
        padding: int,
    ) -> tuple[Tensor, Tensor]:
        """Return padded rows and their boolean presence mask.

        Parameters
        ----------
        rows : Sequence[tuple[int, ...]]
            Encoded lexical rows.
        padding : int
            Reserved value filling absent positions.

        Returns
        -------
        tuple[Tensor, Tensor]
            Padded rows and their boolean presence mask.
        """
        width = max((len(row) for row in rows), default=0)
        vectors = torch.full((len(rows), width), padding, dtype=torch.long)
        present = torch.zeros((len(rows), width), dtype=torch.bool)

        for position, row in enumerate(rows):
            if row:
                vectors[position, : len(row)] = torch.tensor(row, dtype=torch.long)
                present[position, : len(row)] = True

        return vectors, present


class Featurization(NamedTuple):
    """A pair featurizer, its examples, and their resolved items."""

    featurizer: PairFeaturizer
    examples: tuple[Example[Supervision[str]], ...]
    items: tuple[Item, ...]


def _item(handle: str) -> Item:
    """Return one benchmark item.

    Parameters
    ----------
    handle : str
        Item identifier.

    Returns
    -------
    Item
        Fixed-width lexical item.
    """
    return Item(
        ItemId(handle),
        tuple(Token(position, word) for position, word in enumerate(WORDS)),
    )


def _pair_featurization() -> Featurization:
    """Return the prejoining featurizer over one fixed batch shape."""
    anchors = tuple(_item(f"anchor-{position}") for position in range(EXAMPLES))
    others = tuple(_item(f"other-{position}") for position in range(EXAMPLES))
    items = tuple(item for pair in zip(anchors, others, strict=True) for item in pair)
    examples: tuple[Example[Supervision[str]], ...] = tuple(
        Example(anchor, Relational(other.item_id, "contrast"))
        for anchor, other in zip(anchors, others, strict=True)
    )
    hasher = ItemHasher(
        Inventory("words", tuple(dict.fromkeys(WORDS)), (PAD,)),
        Inventory("categories", ()),
    )
    return Featurization(
        pair_featurizer.PairFeaturizer(
            hasher,
            PAD,
            TorchCodeTensorizer(),
            ItemLookup(items),
            Relational,
        ),
        examples,
        items,
    )


@pytest.fixture(scope="module")
def featurization() -> Featurization:
    """Return the pair-valued boundary fixture."""
    return _pair_featurization()


@pytest.fixture(scope="module")
def step() -> ReferenceStep:
    """Return the pricing reference step at the featurized batch shape."""
    pricing = LengthBatchPricing(
        lengths=(SENTENCE_LENGTH, SENTENCE_LENGTH + 1),
        batches=(FEATURIZED_ROWS, FEATURIZED_ROWS + 1),
    )
    spec = reference_stacks(pricing)[SENTENCE_LENGTH, FEATURIZED_ROWS]

    return prepare_reference(
        spec, ExperimentSeed(0).point("pricing-reference", 0), device="cpu"
    )


@pytest.fixture
def unchecked_benchmark() -> None:
    """Require performance measurements to compile invariant checks out."""
    if _checks.CHECKS:
        pytest.fail("the featurization benchmark requires LOFI_CHECK=0")


def _step_seconds(step: ReferenceStep) -> float:
    """Return median wall seconds over warm parser steps.

    Parameters
    ----------
    step : ReferenceStep
        The prepared pricing reference step.

    Returns
    -------
    float
        Median seconds over the declared repetitions.
    """
    step.run()
    measured: list[float] = []

    for _ in range(STEP_ROUNDS):
        started = time.perf_counter()
        step.run()
        measured.append(time.perf_counter() - started)

    return statistics.median(measured)


def test_the_priced_featurizer_builds_the_declared_pair_batch(
    featurization: Featurization,
) -> None:
    built = featurization.featurizer(featurization.examples)
    expected_ids = tuple(item.item_id for item in featurization.items)

    assert len(built) == EXAMPLES
    assert built.batch.vectors.shape == (FEATURIZED_ROWS, SENTENCE_LENGTH)
    assert built.batch.lengths == (SENTENCE_LENGTH,) * FEATURIZED_ROWS
    assert built.batch.item_ids == expected_ids
    assert built.batch.mask.values.shape == (FEATURIZED_ROWS, SENTENCE_LENGTH)
    assert bool(built.batch.mask.values.all())
    assert all(
        alignment.rows == tuple((token, token + 1) for token in range(SENTENCE_LENGTH))
        for alignment in built.batch.alignments
    )
    assert tuple(hashed.item_id for hashed in built.hashed) == expected_ids[::2]
    assert built.supervision == tuple(
        example.supervision for example in featurization.examples
    )
    assert built.member_rows == tuple(
        (position, position + 1) for position in range(0, FEATURIZED_ROWS, 2)
    )


def test_one_unchecked_pair_batch_constructs_no_didactic_models(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, _core, pair_featurizer, on=False)
    featurization = _pair_featurization()

    with counted_constructions() as constructions:
        featurization.featurizer(featurization.examples)

    assert _checks.CHECKS is False
    assert constructions.count == 0


@pytest.mark.usefixtures("unchecked_benchmark")
def test_featurization_stays_under_two_percent_of_step_wall(
    benchmark: BenchmarkFixture,
    featurization: Featurization,
    step: ReferenceStep,
) -> None:
    step_seconds = _step_seconds(step)
    benchmark(featurization.featurizer, featurization.examples)
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the featurizer ran but pytest-benchmark timed nothing")

    ratio = measured.stats.median / step_seconds
    benchmark.extra_info["examples"] = EXAMPLES
    benchmark.extra_info["item_rows"] = FEATURIZED_ROWS
    benchmark.extra_info["tokens_per_item"] = SENTENCE_LENGTH
    benchmark.extra_info["median_ms_per_batch"] = measured.stats.median * 1e3
    benchmark.extra_info["mean_ms_per_batch"] = measured.stats.mean * 1e3
    benchmark.extra_info["median_step_ms"] = step_seconds * 1e3
    benchmark.extra_info["fraction_of_step"] = ratio
    benchmark.extra_info["checks"] = _checks.CHECKS
    benchmark.extra_info["budget"] = "featurization"

    assert ratio <= 0.02
