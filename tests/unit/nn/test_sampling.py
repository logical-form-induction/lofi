import math
from typing import Final, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.nn._values import LogDistribution
from lofi.nn.sampling import (
    FULL,
    KEEP_OFFERED,
    NUCLEUS,
    TOP_K,
    SamplingStrategy,
    restrict,
    sample_from_logits,
    sampling_strategy,
)
from lofi.registry import RegistryError
from lofi.spec.decoding import SamplingSettings

AXES: Final = ("row", "symbol")
GOLDEN: Final = (0.0, 1.0, 2.0, 3.0)
DRAWS: Final = 200


def full(temperature: float = 1.0) -> SamplingStrategy:
    return sampling_strategy(SamplingSettings("full", temperature, None, None))


def top_k(count: int, temperature: float = 1.0) -> SamplingStrategy:
    return sampling_strategy(SamplingSettings("top-k", temperature, count, None))


def nucleus(mass: float, temperature: float = 1.0) -> SamplingStrategy:
    return sampling_strategy(SamplingSettings("nucleus", temperature, None, mass))


def softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    weights = [math.exp(score - top) for score in scores]
    total = sum(weights)

    return [weight / total for weight in weights]


def logits(
    *rows: tuple[float, ...], mask: torch.Tensor | None = None
) -> LogDistribution:
    return LogDistribution(torch.tensor(rows), AXES, mask)


def probabilities(distribution: LogDistribution) -> list[list[float]]:
    return distribution.tensor.exp().tolist()


def nucleus_by_hand(scores: list[float], mass: float) -> set[int]:
    weights = softmax(scores)
    order = sorted(range(len(scores)), key=lambda at: (-scores[at], at))
    kept: set[int] = set()
    ahead = 0.0

    for at in order:
        if ahead >= mass:
            break

        kept.add(at)
        ahead += weights[at]

    return kept


def logit_rows(width: int = 6) -> st.SearchStrategy[tuple[float, ...]]:
    return st.lists(
        st.floats(min_value=-8.0, max_value=8.0, width=32),
        min_size=width,
        max_size=width,
        unique=True,
    ).map(tuple)


def test_temperature_scaling_is_the_softmax_of_the_divided_logits() -> None:
    warm = restrict(
        logits(GOLDEN),
        strategy=full(2.0),
        over="symbol",
    )

    by_hand = softmax([score / 2.0 for score in GOLDEN])

    assert warm.normalized_over == "symbol"
    assert warm.axes == AXES
    assert probabilities(warm)[0] == pytest.approx(by_hand, abs=1e-6)


def test_a_low_temperature_concentrates_on_the_argmax_within_its_own_bound() -> None:
    temperature = 0.05
    ordered = sorted(GOLDEN, reverse=True)
    gap = ordered[0] - ordered[1]
    floor = 1.0 / (1.0 + (len(GOLDEN) - 1) * math.exp(-gap / temperature))

    cold = restrict(
        logits(GOLDEN),
        strategy=full(temperature),
        over="symbol",
    )
    at_top = probabilities(cold)[0][GOLDEN.index(max(GOLDEN))]

    assert floor < 1.0
    assert at_top >= floor
    assert at_top == pytest.approx(1.0, abs=1e-6)


def test_top_k_leaves_no_mass_outside_the_k_highest_symbols() -> None:
    kept = restrict(
        logits(GOLDEN),
        strategy=top_k(2),
        over="symbol",
    )
    by_hand = softmax(list(GOLDEN[2:]))

    assert kept.mask is not None
    assert kept.mask.tolist() == [[False, False, True, True]]
    assert probabilities(kept)[0][:2] == [0.0, 0.0]
    assert probabilities(kept)[0][2:] == pytest.approx(by_hand, abs=1e-6)


def test_top_k_wider_than_the_alphabet_keeps_what_the_alphabet_holds() -> None:
    kept = restrict(
        logits(GOLDEN),
        strategy=top_k(99),
        over="symbol",
    )

    assert kept.mask is not None
    assert kept.mask.all()
    assert probabilities(kept)[0] == pytest.approx(softmax(list(GOLDEN)), abs=1e-6)


def test_nucleus_keeps_the_top_position_however_little_mass_is_asked_for() -> None:
    narrow = restrict(
        logits(GOLDEN),
        strategy=nucleus(1e-6),
        over="symbol",
    )

    assert narrow.mask is not None
    assert narrow.mask.tolist() == [[False, False, False, True]]
    assert probabilities(narrow)[0] == [0.0, 0.0, 0.0, 1.0]


def test_nucleus_keeps_the_shortest_prefix_reaching_the_mass_it_was_asked_for() -> None:
    for mass in (0.3, 0.6, 0.9, 1.0):
        kept = restrict(
            logits(GOLDEN),
            strategy=nucleus(mass),
            over="symbol",
        )

        assert kept.mask is not None
        assert {
            at for at, held in enumerate(kept.mask[0].tolist()) if held
        } == nucleus_by_hand(list(GOLDEN), mass)
        assert sum(probabilities(kept)[0]) == pytest.approx(1.0, abs=1e-6)


@settings(max_examples=50)
@given(row=logit_rows(), count=st.integers(min_value=1, max_value=6))
def test_top_k_holds_of_any_generated_row(row: tuple[float, ...], count: int) -> None:
    kept = restrict(
        logits(row),
        strategy=top_k(count),
        over="symbol",
    )
    weights = probabilities(kept)[0]
    wanted = sorted(range(len(row)), key=lambda at: (-row[at], at))[:count]

    assert kept.mask is not None
    assert int(kept.mask.sum()) == count
    assert sum(1 for weight in weights if weight > 0.0) <= count
    assert {at for at, held in enumerate(kept.mask[0].tolist()) if held} == set(wanted)
    assert sum(weights) == pytest.approx(1.0, abs=1e-5)


@settings(max_examples=50)
@given(row=logit_rows(), mass=st.floats(min_value=0.05, max_value=1.0))
def test_nucleus_holds_of_any_generated_row(
    row: tuple[float, ...], mass: float
) -> None:
    kept = restrict(
        logits(row),
        strategy=nucleus(mass),
        over="symbol",
    )

    assert kept.mask is not None

    weights = softmax(list(row))
    order = sorted(range(len(row)), key=lambda at: (-row[at], at))
    held = [at for at in order if kept.mask[0, at]]
    ahead = sum(weights[at] for at in held[:-1])

    assert held == order[: len(held)]
    assert sum(probabilities(kept)[0]) == pytest.approx(1.0, abs=1e-5)
    assert ahead < mass + 1e-4
    assert len(held) == len(row) or ahead + weights[held[-1]] >= mass - 1e-4


@settings(max_examples=50)
@given(row=logit_rows(), temperature=st.floats(min_value=0.2, max_value=5.0))
def test_a_full_draw_agrees_with_a_softmax_computed_apart_from_torch(
    row: tuple[float, ...], temperature: float
) -> None:
    drawn = restrict(
        logits(row),
        strategy=full(temperature),
        over="symbol",
    )

    assert probabilities(drawn)[0] == pytest.approx(
        softmax([score / temperature for score in row]), abs=1e-5
    )


def test_a_symbol_left_out_is_left_out_by_the_mask_and_never_drawn() -> None:
    mask = torch.tensor([[True, False, True, False]])
    offered = logits((0.0, 5.0, 0.0, 5.0), mask=mask)

    kept = restrict(offered, strategy=full(), over="symbol")
    generator = torch.Generator().manual_seed(11)
    drawn = {
        int(
            sample_from_logits(
                offered,
                strategy=full(),
                over="symbol",
                generator=generator,
            )
        )
        for _ in range(DRAWS)
    }

    assert probabilities(kept)[0] == pytest.approx([0.5, 0.0, 0.5, 0.0], abs=1e-6)
    assert bool(kept.tensor[0, 1].isneginf())
    assert bool(kept.tensor[0, 3].isneginf())
    assert drawn == {0, 2}


def test_a_top_k_wider_than_what_a_row_offers_keeps_only_what_it_offers() -> None:
    mask = torch.tensor([[True, False, False, False]])
    offered = logits((0.0, 9.0, 9.0, 9.0), mask=mask)

    kept = restrict(
        offered,
        strategy=top_k(3),
        over="symbol",
    )

    assert kept.mask is not None
    assert kept.mask.tolist() == [[True, False, False, False]]
    assert probabilities(kept)[0] == [1.0, 0.0, 0.0, 0.0]


def test_a_mask_over_the_outer_axes_alone_offers_every_symbol_of_a_live_row() -> None:
    mask = torch.tensor([True, False])
    offered = logits((0.0, 1.0, 2.0, 3.0), (4.0, 4.0, 4.0, 4.0), mask=mask)

    kept = restrict(offered, strategy=full(), over="symbol")

    assert kept.mask is not None
    assert kept.mask.tolist() == [[True] * 4, [False] * 4]
    assert probabilities(kept)[0] == pytest.approx(softmax(list(GOLDEN)), abs=1e-6)
    assert probabilities(kept)[1] == [0.0, 0.0, 0.0, 0.0]


def test_a_row_offering_nothing_cannot_be_drawn_for() -> None:
    mask = torch.tensor([True, False])
    offered = logits((0.0, 1.0, 2.0, 3.0), (4.0, 4.0, 4.0, 4.0), mask=mask)

    with pytest.raises(ValueError, match="1 of 2 rows offer no symbol"):
        sample_from_logits(offered, strategy=full(), over="symbol")


def test_a_fill_among_the_offered_scores_is_refused_rather_than_read() -> None:
    filled = logits((0.0, 1.0, -math.inf, 3.0))

    with pytest.raises(ValueError, match="1 of the 4 positions on offer"):
        restrict(filled, strategy=full(), over="symbol")

    with pytest.raises(ValueError, match="positions on offer"):
        sample_from_logits(
            logits((0.0, math.nan)),
            strategy=full(),
            over="symbol",
        )


def test_a_draw_runs_along_the_axis_it_was_told_to() -> None:
    scores = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    cube = LogDistribution(scores, ("item", "slot", "symbol"))

    over_symbols = sample_from_logits(
        cube,
        strategy=top_k(1),
        over="symbol",
    )
    over_slots = sample_from_logits(
        cube,
        strategy=top_k(1),
        over="slot",
    )

    assert over_symbols.shape == (2, 3)
    assert over_symbols.tolist() == [[3, 3, 3], [3, 3, 3]]
    assert over_slots.shape == (2, 4)
    assert over_slots.tolist() == [[2, 2, 2, 2], [2, 2, 2, 2]]


def test_two_draws_from_one_seed_agree_and_two_seeds_disagree() -> None:
    scores = logits(tuple(float(step % 5) for step in range(64)))
    strategy = full(3.0)

    def drawn(seed: int) -> list[int]:
        generator = torch.Generator().manual_seed(seed)

        return [
            int(
                sample_from_logits(
                    scores, strategy=strategy, over="symbol", generator=generator
                )
            )
            for _ in range(DRAWS)
        ]

    assert drawn(7) == drawn(7)
    assert drawn(7) != drawn(8)


def test_each_kind_resolves_to_its_installed_restriction() -> None:
    assert full().restriction is KEEP_OFFERED
    assert FULL.build(SamplingSettings("full", 1.0, None, None)) is KEEP_OFFERED
    assert TOP_K.build(SamplingSettings("top-k", 1.0, 3, None)).identity == (
        "top-k",
        3.0,
    )
    assert NUCLEUS.build(SamplingSettings("nucleus", 1.0, None, 0.5)).identity == (
        "nucleus",
        0.5,
    )

    with pytest.raises(RegistryError, match="greedy"):
        sampling_strategy(SamplingSettings._make(("greedy", 1.0, None, None)))

    with pytest.raises(TypeError, match="restricted by a SamplingStrategy"):
        restrict(
            logits(GOLDEN),
            strategy=cast("SamplingStrategy", KEEP_OFFERED),
            over="symbol",
        )


def test_a_restriction_without_its_setting_is_refused() -> None:
    with pytest.raises(ValueError, match="without a count"):
        TOP_K.build(SamplingSettings("top-k", 1.0, None, None))

    with pytest.raises(ValueError, match="without a mass"):
        NUCLEUS.build(SamplingSettings("nucleus", 1.0, None, None))


def test_a_temperature_outside_its_range_is_refused_where_it_is_named() -> None:
    with pytest.raises(ValueError, match="is above zero"):
        full(0.0)

    with pytest.raises(ValueError, match="not finite"):
        full(math.inf)


def test_two_strategies_describing_the_same_draw_are_one_value() -> None:
    narrow = top_k(3, 0.5)
    same = top_k(3, 0.5)

    assert narrow == same
    assert hash(narrow) == hash(same)
    assert narrow != top_k(4, 0.5)
    assert narrow != full(0.5)
    assert narrow.__eq__(KEEP_OFFERED) is NotImplemented
    assert repr(narrow) == "SamplingStrategy('top-k', setting=3.0, temperature=0.5)"
    assert repr(full()) == "SamplingStrategy('full', temperature=1.0)"

    for field in ("restriction", "temperature"):
        with pytest.raises(AttributeError):
            setattr(narrow, field, None)


def test_a_draw_along_an_axis_that_was_not_declared_is_refused() -> None:
    with pytest.raises(ValueError, match="there is no axis named 'piece'"):
        restrict(logits(GOLDEN), strategy=full(), over="piece")
