"""Strategies for pairing measured premise and hypothesis forms."""

from typing import TYPE_CHECKING, Protocol, final

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random

    from lofi.data.synthetic._complexity import FormComplexity

__all__ = [
    "INDEPENDENT_PAIRING",
    "IndependentPairing",
    "OverlapMatchedPairing",
    "PairingStrategy",
]


class PairingStrategy(Protocol):
    """Select one premise index and one hypothesis index from measured pools."""

    def choose(
        self,
        random: Random,
        premises: Sequence[FormComplexity],
        hypotheses: Sequence[FormComplexity],
        /,
    ) -> tuple[int, int]:
        """Return selected indices into the two nonempty candidate pools."""
        ...


def _nonempty(
    premises: Sequence[FormComplexity], hypotheses: Sequence[FormComplexity]
) -> None:
    """Refuse a pairing request over an empty candidate pool."""
    if not premises or not hypotheses:
        msg = "pairing needs a nonempty premise and hypothesis candidate pool"
        raise ValueError(msg)


@final
class IndependentPairing(FrozenValue):
    """Draw one premise and one hypothesis independently with replacement."""

    _noun = "independent pairing strategy"
    __slots__ = ()

    def choose(
        self,
        random: Random,
        premises: Sequence[FormComplexity],
        hypotheses: Sequence[FormComplexity],
        /,
    ) -> tuple[int, int]:
        """Draw one index uniformly from each candidate pool."""
        _nonempty(premises, hypotheses)
        return random.randrange(len(premises)), random.randrange(len(hypotheses))


@final
class OverlapMatchedPairing(FrozenValue):
    """Draw uniformly among pairs with greatest predicate-vocabulary overlap."""

    _noun = "overlap-matched pairing strategy"
    __slots__ = ()

    def choose(
        self,
        random: Random,
        premises: Sequence[FormComplexity],
        hypotheses: Sequence[FormComplexity],
        /,
    ) -> tuple[int, int]:
        """Draw one of the candidate pairs attaining maximum overlap."""
        _nonempty(premises, hypotheses)
        scored = tuple(
            (premise.overlap(hypothesis), left, right)
            for left, premise in enumerate(premises)
            for right, hypothesis in enumerate(hypotheses)
        )
        maximum = max(score for score, _, _ in scored)
        choices = tuple(
            (left, right) for score, left, right in scored if score == maximum
        )
        return random.choice(choices)


INDEPENDENT_PAIRING = IndependentPairing()
"""The default independent-with-replacement pairing strategy."""
