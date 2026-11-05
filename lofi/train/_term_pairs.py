"""Generated combinator pairs for represented term spaces."""

from itertools import product
from random import Random
from typing import TYPE_CHECKING, Protocol

from lofi._checks import ensure_non_negative
from lofi.data import PairSampleReport, SampledPair, term_depth
from lofi.interpret import CategoryWitnessRequiredError, run_action

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data import InventoriedTypeSampler, PairDistribution, PairSampler
    from lofi.interpret import MeaningActions
    from lofi.judge import Action, ActionSet
    from lofi.terms import Normalization, Term

__all__ = ["MeaningPairSampler", "MorphismPairSampler"]


class _Rule(Protocol):
    """The rule fields semantic-action sampling reads."""

    @property
    def name(self) -> str:
        """Return the rule name."""
        ...

    @property
    def arity(self) -> int:
        """Return the rule's premise count."""
        ...


class MorphismPairSampler:
    """Map generated combinator instances into another term space."""

    __slots__ = ("_morphism", "_source")

    def __init__(
        self,
        source: PairSampler[Term, Action],
        morphism: Callable[[Term], Term],
    ) -> None:
        self._source = source
        self._morphism = morphism

    def sample(
        self,
        count: int,
        distribution: PairDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[SampledPair[Term, Action], ...], PairSampleReport]:
        """Return source pairs after applying the declared term morphism."""
        pairs, report = self._source.sample(count, distribution, seed=seed)
        mapped = tuple(
            SampledPair(
                tuple(map(self._morphism, pair.inputs)),
                self._morphism(pair.output),
                pair.actions,
            )
            for pair in pairs
        )

        return mapped, report


class MeaningPairSampler:
    """Generate beta-normal form pairs from declared semantic actions."""

    __slots__ = ("_actions", "_meanings", "_normalization", "_pools", "_types")

    def __init__(
        self,
        types: InventoriedTypeSampler,
        actions: ActionSet,
        meanings: MeaningActions,
        rules: tuple[_Rule, ...],
        *,
        normalization: Normalization,
    ) -> None:
        composing = actions.composing
        by_name = {rule.name: rule for rule in rules}
        by_action: dict[Action, tuple[tuple[Term, ...], int]] = {}
        missing: list[str] = []
        conditioned: set[str] = set()

        for action in composing:
            rule_name = action.rule
            rule = None if rule_name is None else by_name.get(rule_name)

            if rule is None:
                missing.append(action.name)
                continue

            try:
                terms = meanings.resolve(rule.name, None)
            except CategoryWitnessRequiredError:
                conditioned.add(rule.name)
                continue

            if terms:
                by_action[action] = (terms, rule.arity)
            else:
                missing.append(action.name)

        if conditioned:
            msg = (
                "semantic pair supervision requires category-invariant actions; "
                f"category-conditioned rules: {sorted(conditioned)!r}"
            )
            raise ValueError(msg)

        if missing:
            msg = (
                "semantic actions and rule arities do not cover composing actions: "
                f"{sorted(missing)}"
            )
            raise ValueError(msg)

        self._types = types
        self._actions = by_action
        self._meanings = meanings
        self._normalization = normalization
        self._pools: dict[int, tuple[SampledPair[Term, Action], ...]] = {}

    def sample(
        self,
        count: int,
        distribution: PairDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[SampledPair[Term, Action], ...], PairSampleReport]:
        """Return seeded inputs, actions and beta-normal semantic outputs."""
        held_count = ensure_non_negative("count", count)

        pool = self._pool(distribution.max_depth)

        if held_count and not pool:
            msg = (
                "no admissible semantic-action pair exists at depth "
                f"{distribution.max_depth}"
            )
            raise ValueError(msg)

        random = Random(seed)
        samples = tuple(random.choice(pool) for _ in range(held_count))
        arities: dict[int, int] = {}

        for sample in samples:
            arity = len(sample.inputs)
            arities[arity] = arities.get(arity, 0) + 1

        return (
            samples,
            PairSampleReport(len(samples), arities),
        )

    def _pool(self, max_depth: int, /) -> tuple[SampledPair[Term, Action], ...]:
        """Return the cached admissible semantic-action pool at one bound."""
        cached = self._pools.get(max_depth)

        if cached is not None:
            return cached

        terms = self._types.inventory.through_depth(max_depth)
        pool: list[SampledPair[Term, Action]] = []

        outputs: dict[tuple[tuple[Term, ...], Term], list[Action]] = {}

        for action, (meanings, arity) in self._actions.items():
            self._add_outputs(outputs, action, meanings, arity, terms, max_depth)

        for (inputs, output), labels in outputs.items():
            pair = SampledPair(inputs, output, tuple(labels))

            if pair not in pool:
                pool.append(pair)

        result = tuple(pool)
        self._pools[max_depth] = result

        return result

    def _add_outputs(
        self,
        outputs: dict[tuple[tuple[Term, ...], Term], list[Action]],
        action: Action,
        meanings: tuple[Term, ...],
        arity: int,
        terms: tuple[Term, ...],
        max_depth: int,
        /,
    ) -> None:
        """Add bounded logical-form pairs for selected action terms."""
        for meaning in meanings:
            for inputs in product(terms, repeat=arity):
                output = run_action(
                    meaning,
                    inputs,
                    calculus=self._meanings.calculus,
                    normalization=self._normalization,
                )

                if term_depth(output) <= max_depth:
                    labels = outputs.setdefault((inputs, output), [])

                    if action not in labels:
                        labels.append(action)
