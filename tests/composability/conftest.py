"""Composability test fixtures."""

from __future__ import annotations

import enum
from itertools import product
from typing import TYPE_CHECKING, NamedTuple, Protocol

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.registry import Registry

type TagPair = tuple[str, str]
type Incompatibility = tuple[type[Exception], str]


class CombinationOutcome(enum.StrEnum):
    """Identifies how a component pair was accounted for."""

    CONSTRUCTED = "constructed"
    DECLARED_INCOMPATIBLE = "declared_incompatible"


class Combination(NamedTuple):
    """Records one checked component pair."""

    pair: TagPair
    outcome: CombinationOutcome


class ComposabilityHarness[Left, Right, Result]:
    """Checks every pair supplied by two registries."""

    __slots__ = ("_construct", "_incompatibilities", "_left", "_right")

    def __init__(
        self,
        left: Registry[Left],
        right: Registry[Right],
        /,
        *,
        construct: Callable[[Left, Right], Result],
        incompatibilities: Mapping[TagPair, Incompatibility],
    ) -> None:
        self._left = left
        self._right = right
        self._construct = construct
        self._incompatibilities = dict(incompatibilities)

        pairs = frozenset(self.pairs())

        for pair, (_, reason) in self._incompatibilities.items():
            if pair not in pairs:
                msg = f"incompatibility {pair!r} is outside the registry product"

                raise ValueError(msg)

            if not reason.strip():
                msg = f"incompatibility {pair!r} needs a nonempty reason"

                raise ValueError(msg)

    def pairs(self) -> tuple[TagPair, ...]:
        """Return the registry cross product."""
        return tuple(product(self._left.available(), self._right.available()))

    def check(self, pair: TagPair, /) -> CombinationOutcome:
        """Check one component pair."""
        left_tag, right_tag = pair
        left = self._left.get(left_tag)
        right = self._right.get(right_tag)
        expected = self._incompatibilities.get(pair)

        if expected is None:
            self._construct(left, right)

            return CombinationOutcome.CONSTRUCTED

        error, reason = expected

        try:
            self._construct(left, right)

        except error as refusal:
            if reason not in str(refusal):
                msg = (
                    f"{pair!r} raised {refusal!r}; its message does not contain "
                    f"{reason!r}"
                )

                raise AssertionError(msg) from refusal

            return CombinationOutcome.DECLARED_INCOMPATIBLE

        msg = f"{pair!r} constructed despite declared incompatibility: {reason}"

        raise AssertionError(msg)

    def evaluate(self) -> tuple[Combination, ...]:
        """Check and classify every component pair."""
        return tuple(Combination(pair, self.check(pair)) for pair in self.pairs())


class HarnessFactory(Protocol):
    """Builds a composability harness."""

    def __call__[Left, Right, Result](
        self,
        left: Registry[Left],
        right: Registry[Right],
        /,
        *,
        construct: Callable[[Left, Right], Result],
        incompatibilities: Mapping[TagPair, Incompatibility],
    ) -> ComposabilityHarness[Left, Right, Result]: ...


@pytest.fixture
def composability_harness() -> HarnessFactory:
    return ComposabilityHarness
