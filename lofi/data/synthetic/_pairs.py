"""Synthetic sampler for labeled rule instances of arbitrary arity."""

from itertools import product
from random import Random
from typing import TYPE_CHECKING, Final, Protocol

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data._sampling import (
    PairDistribution,
    PairSampleReport,
    SampledPair,
    term_depth,
)
from lofi.registry import ComponentPlugin
from lofi.spec.pair_samplers.rule import RulePairSamplerSettings, RulePairSamplerSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lofi.data._ports import InventoriedTypeSampler
    from lofi.terms import Term


class _Rule[T](Protocol):
    """Rule operations used by rule-instance sampling."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        ...

    @property
    def arity(self) -> int:
        """Return the premise count."""
        ...

    @property
    def directions(self) -> frozenset[str]:
        """Return the supported computation directions."""
        ...

    @property
    def forward_is_complete(self) -> bool:
        """Return whether forward enumeration is complete."""
        ...

    def apply(self, inputs: tuple[T, ...], /) -> Iterator[T]:
        """Return forward conclusions for the inputs."""
        ...

    def admits(self, inputs: tuple[T, ...], output: T, /) -> bool:
        """Return whether the rule admits the step."""
        ...


class _RuleInventory[T](Protocol):
    """Rules in declaration order, queryable by premise count."""

    def of_arity(self, k: int) -> tuple[_Rule[T], ...]:
        """Return rules with the requested premise count."""
        ...

    def __iter__(self) -> Iterator[_Rule[T]]:
        """Return rules in declaration order."""
        ...


class _Action(Protocol):
    """An action name and its optional rule name."""

    @property
    def name(self) -> str:
        """Return the action name."""
        ...

    @property
    def rule(self) -> str | None:
        """Return the associated rule name, if any."""
        ...


class _ActionInventory(Protocol):
    """Actions in controller-column order."""

    @property
    def actions(self) -> tuple[_Action, ...]:
        """Return the ordered actions."""
        ...


__all__ = ["RulePairSampler", "RulePairSamplerContext"]


class RulePairSamplerContext(FrozenValue):
    """The type inventory, rules and action labels a sampler reads."""

    _noun = "rule-pair sampler context"
    __slots__ = ("_actions", "_rules", "_types")

    def __init__(
        self,
        types: InventoriedTypeSampler,
        rules: _RuleInventory[Term],
        actions: _ActionInventory,
    ) -> None:
        object.__setattr__(self, "_types", types)
        object.__setattr__(self, "_rules", rules)
        object.__setattr__(self, "_actions", actions)

    @property
    def types(self) -> InventoriedTypeSampler:
        """Return the finite premise inventory."""
        return self._types

    @property
    def rules(self) -> _RuleInventory[Term]:
        """Return the declared rule inventory."""
        return self._rules

    @property
    def actions(self) -> _ActionInventory:
        """Return the ordered action labels."""
        return self._actions


class RulePairSampler:
    """Sample bounded labeled instances from every generative rule arity."""

    __slots__ = ("_actions", "_arities", "_pools", "_rules", "_types")

    def __init__(
        self,
        type_sampler: InventoriedTypeSampler,
        rules: _RuleInventory[Term],
        actions: _ActionInventory,
    ) -> None:
        held_rules = tuple(rules)
        arities = tuple(
            ensure_non_negative(f"the arity of rule {rule.name!r}", rule.arity)
            for rule in held_rules
        )

        action_names = {
            action.rule for action in actions.actions if action.rule is not None
        }
        missing = {rule.name for rule in held_rules} - action_names

        if missing:
            msg = f"rules lack controller actions: {sorted(missing)!r}"
            raise ValueError(msg)

        self._types = type_sampler
        self._rules = rules
        self._actions = actions
        self._arities = tuple(sorted(set(arities)))
        self._pools: dict[int, tuple[SampledPair[Term, _Action], ...]] = {}

    def sample(
        self,
        count: int,
        distribution: PairDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[SampledPair[Term, _Action], ...], PairSampleReport]:
        """Return a seeded sample and its observed rule-arity counts."""
        held_count = ensure_non_negative("count", count)

        if held_count == 0:
            return (), PairSampleReport(0, {})

        pool = self._pool(distribution.max_depth)

        if not pool:
            msg = (
                f"no admissible rule instance exists at depth {distribution.max_depth}"
            )
            raise ValueError(msg)

        random = Random(seed)
        samples = tuple(random.choice(pool) for _ in range(held_count))
        arities: dict[int, int] = {}

        for sample in samples:
            arity = len(sample.inputs)
            arities[arity] = arities.get(arity, 0) + 1

        return samples, PairSampleReport(len(samples), arities)

    def _pool(self, max_depth: int) -> tuple[SampledPair[Term, _Action], ...]:
        """Return every bounded labeled instance at every declared arity."""
        cached = self._pools.get(max_depth)

        if cached is not None:
            return cached

        terms = self._types.inventory.through_depth(max_depth)
        pairs: list[SampledPair[Term, _Action]] = []

        for arity in self._arities:
            for inputs in product(terms, repeat=arity):
                for output, labels in self._candidates(inputs, max_depth):
                    pair = SampledPair(inputs, output, labels)

                    if pair not in pairs:
                        pairs.append(pair)

        result = tuple(pairs)
        self._pools[max_depth] = result

        return result

    def _candidates(
        self,
        inputs: tuple[Term, ...],
        max_depth: int,
    ) -> tuple[tuple[Term, tuple[_Action, ...]], ...]:
        """Return bounded conclusions and their exact action labels."""
        rules = self._rules.of_arity(len(inputs))
        outputs: list[Term] = []

        for rule in rules:
            if "forward" not in rule.directions or not rule.forward_is_complete:
                continue

            for output in rule.apply(inputs):
                if term_depth(output) <= max_depth and output not in outputs:
                    outputs.append(output)

        candidates = []

        for output in outputs:
            admitted = {rule.name for rule in rules if rule.admits(inputs, output)}
            labels = tuple(
                action
                for action in self._actions.actions
                if action.rule is not None and action.rule in admitted
            )

            if labels:
                candidates.append((output, labels))

        return tuple(candidates)


def _build_rule_pairs(
    settings: RulePairSamplerSettings, context: RulePairSamplerContext, /
) -> RulePairSampler:
    """Build the rule-instance sampler selected by a run declaration."""
    del settings
    return RulePairSampler(context.types, context.rules, context.actions)


PLUGIN: Final = ComponentPlugin(spec=RulePairSamplerSpec, build=_build_rule_pairs)
"""The rule-instance sampler's plugin."""
