from typing import TYPE_CHECKING, Final

import pytest
import torch

from lofi.nn._values import Encoded
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.sites import ComponentId
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
)

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

TERMS: Final = 32
INPUT_WIDTH: Final = 64
WIDTH: Final = 64
DEPTH: Final = 4
PASS_BUDGET_MS: Final = 40.0

PRIMITIVES: Final = ("N", "NP", "S", "PP", "ADJ", "ADV", "CONJ", "DET")
CONSTRUCTORS: Final = ("/", "\\", "^")

SIGNATURE: Final = Signature(
    "decoder-benchmark",
    ("Cat",),
    (
        *(Operator(name, "Cat", family="primitive") for name in PRIMITIVES),
        *(
            Operator(name, "Cat", arg_sorts=("Cat", "Cat"), family="constructor")
            for name in CONSTRUCTORS
        ),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"Cat": DEPTH})
VIEW: Final = NeuralCodecView(
    TermCodec(
        INDEX,
        CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=DEPTH),
    )
)


def _step(decoder: RecursiveTermDecoder, encoded: Encoded) -> torch.Tensor:
    decoder.zero_grad(set_to_none=True)
    weights = decoder.marginal_rows(encoded, root_sort="Cat")
    loss = weights.tensor.masked_select(weights.tensor.isfinite()).mean()
    loss.backward()

    return loss.detach()


@pytest.mark.benchmark
def test_the_marginalizing_pass_meets_its_budget(
    benchmark: BenchmarkFixture,
    wall_clock_budget: WallClockBudget,
) -> None:
    torch.manual_seed(31)
    decoder = RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("tree-rnn-benchmark"),
        settings=TreeDecoderSettings(INPUT_WIDTH, WIDTH),
        choice_heads="flat",
    )
    encoded = Encoded(torch.randn(TERMS, INPUT_WIDTH), ("term", "width"))
    benchmark.pedantic(
        _step,
        args=(decoder, encoded),
        rounds=20,
        warmup_rounds=2,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the marginalizing pass ran but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    conditionals = VIEW.capacity * CHOICES.width * CHOICES.width
    benchmark.extra_info["budget_ms"] = PASS_BUDGET_MS
    benchmark.extra_info["device"] = encoded.tensor.device.type
    benchmark.extra_info["terms"] = TERMS
    benchmark.extra_info["slots"] = VIEW.capacity
    benchmark.extra_info["choices"] = CHOICES.width
    benchmark.extra_info["choice_entries_per_term"] = conditionals
    benchmark.extra_info["median_ms"] = median_ms

    if wall_clock_budget.enforced:
        assert median_ms <= PASS_BUDGET_MS


def test_the_marginalizing_pass_returns_a_finite_loss() -> None:
    torch.manual_seed(31)
    decoder = RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("tree-rnn-benchmark"),
        settings=TreeDecoderSettings(INPUT_WIDTH, WIDTH),
        choice_heads="flat",
    )
    encoded = Encoded(torch.randn(TERMS, INPUT_WIDTH), ("term", "width"))

    assert torch.isfinite(_step(decoder, encoded))
