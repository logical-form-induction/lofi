import pytest

from lofi.nn.loss._decoding import (
    BEAM,
    EXHAUSTIVE,
    GREEDY,
    SAMPLE,
    DecodingStrategy,
    decoding_strategy,
)
from lofi.registry import RegistryError
from lofi.spec.decoding import (
    BeamDecodingSpec,
    ExhaustiveDecodingSpec,
    GreedyDecodingSpec,
    SampleDecodingSpec,
)


@pytest.mark.parametrize(
    ("name", "strategy"),
    [
        ("greedy", GREEDY),
        ("beam", BEAM),
        ("exhaustive", EXHAUSTIVE),
        ("sample", SAMPLE),
    ],
)
def test_each_declared_strategy_is_installed_under_its_name(
    name: str, strategy: DecodingStrategy
) -> None:
    assert decoding_strategy(name) is strategy


def test_an_uninstalled_strategy_is_refused_by_the_family() -> None:
    with pytest.raises(RegistryError, match="decoding_strategy"):
        decoding_strategy("viterbi")


def test_each_strategy_states_what_it_needs_and_bounds_its_frontier() -> None:
    greedy = GreedyDecodingSpec().settings()
    beam = BeamDecodingSpec(width=3).settings()
    exhaustive = ExhaustiveDecodingSpec(cap=7, residual_mass=0.1).settings()
    sample = SampleDecodingSpec(draws=5, rescore_tolerance=0.0).settings()

    assert (GREEDY.frontier(greedy), BEAM.frontier(beam)) == (1, 3)
    assert (EXHAUSTIVE.frontier(exhaustive), SAMPLE.frontier(sample)) == (7, 5)
    assert [s.enumerates for s in (GREEDY, BEAM, EXHAUSTIVE, SAMPLE)] == [
        False,
        False,
        True,
        False,
    ]
    assert [s.rescores for s in (GREEDY, BEAM, EXHAUSTIVE, SAMPLE)] == [
        False,
        False,
        False,
        True,
    ]
