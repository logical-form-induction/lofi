"""Bind the pair-sampler laws to the synthetic implementation."""

from typing import TYPE_CHECKING, Final, cast, override

import pytest
from tests.conformance.pair_sampler import PairSamplerConformance, PairSamplerFixture

from lofi.data import PairDistribution, PairSampler
from lofi.data.synthetic import RecursiveTypeSampler, RulePairSampler
from lofi.judge import Action, ActionSet
from lofi.terms import Operator, Signature, Term, TermStore

if TYPE_CHECKING:
    from collections.abc import Iterator


SIGNATURE: Final = Signature(
    "pair-sampler-conformance",
    ("Cat",),
    (Operator("n", "Cat"), Operator("s", "Cat")),
)


class CopyRule:
    """A rule that copies the left daughter as its conclusion."""

    name = "copy"
    arity = 2
    directions = frozenset({"forward"})
    forward_is_complete = True

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Yield the left daughter."""
        yield inputs[0]

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Accept exactly a copied left daughter."""
        return len(inputs) == 2 and output is inputs[0]


class UnaryCopyRule(CopyRule):
    """A unary rule that preserves its premise."""

    name = "unary-copy"
    arity = 1

    @override
    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Accept exactly a copied sole premise."""
        return len(inputs) == 1 and output is inputs[0]


class CopyInventory:
    """An inventory containing one binary rule."""

    def __init__(self) -> None:
        self.rule = CopyRule()
        self.unary = UnaryCopyRule()

    def of_arity(self, k: int) -> tuple[CopyRule, ...]:
        """Select copy rules at their declared arities."""
        if k == 1:
            return (self.unary,)
        if k == 2:
            return (self.rule,)
        return ()

    def __iter__(self) -> Iterator[CopyRule]:
        """Return the unary and binary copy rules in declaration order."""
        return iter((self.unary, self.rule))


class TestRulePairSampler(PairSamplerConformance):
    """Conformance tests for exact labels over a concrete pair sampler."""

    @pytest.fixture
    def pair_sampler_fixture(self) -> PairSamplerFixture:
        """Build a copy-rule fixture."""
        store = TermStore(SIGNATURE)
        rules = CopyInventory()
        identity = Action("identity")
        actions = ActionSet(
            (
                identity,
                Action("unary-copy", "unary-copy"),
                Action("copy", "copy"),
            ),
            identity,
        )
        sampler = RulePairSampler(
            RecursiveTypeSampler(store, max_size=7, sort="Cat"),
            rules,
            actions,
        )

        return PairSamplerFixture(
            cast("PairSampler[Term, Action]", sampler),
            PairDistribution(max_depth=1),
            (rules.unary, rules.rule),
        )
