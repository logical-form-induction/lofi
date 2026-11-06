"""Push lexical-form and chart distributions through semantic proof actions."""

from __future__ import annotations

from math import log
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, override, runtime_checkable

import torch

from lofi._checks import ensure_positive
from lofi._frozen import FrozenValue
from lofi.interpret import (
    CategoryWitnessRequiredError,
    MeaningActions,
    run_action,
)
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.typecheck import MeaningWeighting, meaning_shares

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.judge import Action, ActionSet
    from lofi.nn._values import LogDistribution
    from lofi.terms import Normalization, Term

__all__ = ["ProofTermMeaning", "ProofTermPushforward"]


class SplitWeights(Protocol):
    """The per-span split posterior consumed by proof term induction."""

    @property
    def spans(self) -> SpanIndex:
        """Return the addressing of the posterior."""

    @property
    def items(self) -> int:
        """Return how many items the posterior covers."""

    def at(self, item: int, i: int, j: int, k: int) -> Tensor:
        """Return one gradient-connected log split probability."""


@runtime_checkable
class DerivedResidualBound(Protocol):
    """The residual mass check applied to each derived phrase cell."""

    def check_terms(
        self,
        weighted: WeightedTerms,
        /,
        *,
        component: str,
    ) -> WeightedTerms:
        """Return a term frontier after checking its truncation residual."""
        ...


@runtime_checkable
class ProofTermMeaning(Protocol):
    """The semantic resources required to construct proof terms."""

    @property
    def actions(self) -> MeaningActions:
        """Return semantic actions keyed by rule name."""

    @property
    def normalization(self) -> Normalization:
        """Return the declared proof term normal form and reduction bound."""


def _estimate(children: tuple[WeightedTerms, ...]) -> Estimate:
    """Return the least exact estimate among child form distributions."""
    estimates = {child.estimate for child in children}

    if Estimate.SAMPLED in estimates:
        return Estimate.SAMPLED

    if Estimate.TRUNCATED in estimates:
        return Estimate.TRUNCATED

    return Estimate.EXACT


def _log_residual(weights: Tensor) -> Tensor:
    """Return the stable complement of a finite subdistribution."""
    if weights.numel() == 0:
        return weights.new_zeros(())

    mass = torch.logsumexp(weights, dim=0).clamp(max=0.0)
    finite = mass < 0.0
    safe = torch.where(finite, mass, mass.new_full((), -1.0))
    complement = torch.log1p(-safe.exp())

    return torch.where(finite, complement, mass.new_full((), -torch.inf))


class ProofTermPushforward(FrozenValue):
    """Derive each span's forms from lexical forms and parser decisions.

    The chart posterior supplies a split distribution and conditional action
    weights. Only binary actions with a semantic proof action construct an internal
    form. Every other derivation, as well as child residual mass, remains represented
    by the output residual rather than being assigned a fabricated meaning.

    Each derived phrase cell is checked against the pushforward's own residual
    bound. Lexical leaves pass through unchanged, since the word decoder that
    produced them has already checked them against the readout's bound.
    """

    __slots__ = (
        "_action_set",
        "_binary",
        "_bound",
        "_cap",
        "_meaning_actions",
        "_normalization",
        "_weighting",
    )
    _noun = "proof-term pushforward"

    def __init__(
        self,
        action_set: ActionSet,
        meaning: ProofTermMeaning,
        *,
        cap: int,
        rule_arities: Mapping[str, int],
        weighting: MeaningWeighting,
        bound: DerivedResidualBound,
    ) -> None:
        if not isinstance(meaning, ProofTermMeaning):
            msg = (
                "a proof-term readout requires semantic actions and normalization "
                "settings"
            )
            raise TypeError(msg)

        if not isinstance(weighting, MeaningWeighting):
            msg = "a proof-term readout requires a meaning-weighting policy"
            raise TypeError(msg)

        if not isinstance(bound, DerivedResidualBound):
            msg = "a proof-term readout requires a derived-cell residual bound"
            raise TypeError(msg)

        meaning_actions = meaning.actions
        arities = dict(rule_arities)
        binary: dict[Action, tuple[Term, ...]] = {}
        category_conditioned: set[str] = set()

        for action in action_set.composing:
            rule = action.rule

            if rule is None:
                continue

            try:
                terms = meaning_actions.resolve(rule, None)
            except CategoryWitnessRequiredError:
                category_conditioned.add(rule)
                continue

            if arities.get(rule) == 2 and terms:
                binary[action] = terms

        if category_conditioned:
            msg = (
                "proof-term pushforward requires category-invariant semantic "
                "actions; category-conditioned rules: "
                f"{sorted(category_conditioned)!r}"
            )
            raise ValueError(msg)
        object.__setattr__(self, "_action_set", action_set)
        object.__setattr__(self, "_bound", bound)
        object.__setattr__(self, "_cap", ensure_positive("a proof-term frontier", cap))
        object.__setattr__(
            self,
            "_binary",
            MappingProxyType(
                {action: terms for action, terms in binary.items() if terms}
            ),
        )
        object.__setattr__(self, "_meaning_actions", meaning_actions)
        object.__setattr__(self, "_normalization", meaning.normalization)
        object.__setattr__(self, "_weighting", weighting)

    def derive(
        self,
        lexical: Mapping[tuple[int, int], WeightedTerms],
        /,
        *,
        spans: SpanIndex,
        splits: SplitWeights,
        actions: Mapping[Action, LogDistribution],
    ) -> Mapping[tuple[int, int], WeightedTerms]:
        """Return lexical leaves and their proof term pushforward at every span."""
        if splits.spans is not spans:
            msg = "proof-term splits and lexical forms use different span addressing"
            raise ValueError(msg)

        expected = frozenset(self._action_set.actions)

        if frozenset(actions) != expected:
            missing = sorted(action.name for action in expected - frozenset(actions))
            extra = sorted(action.name for action in frozenset(actions) - expected)
            msg = (
                "proof-term action weights differ from the declared inventory; "
                f"missing={missing}, extra={extra}"
            )
            raise ValueError(msg)

        cells = dict(lexical)

        for item in range(splits.items):
            for i, j in spans.spans_by_length(min_length=2):
                contributions: dict[Term, list[Tensor]] = {}
                estimates: list[Estimate] = []
                basis: Tensor | None = None

                for k in spans.splits(i, j):
                    left = cells.get((item, spans.span_id(i, k)))
                    right = cells.get((item, spans.span_id(k, j)))

                    if left is None or right is None:
                        continue

                    children = (left, right)
                    estimates.append(_estimate(children))
                    split_weight = splits.at(item, i, j, k)
                    basis = split_weight
                    span_id, column = spans.split_id(i, j, k)
                    non_null = tuple(
                        distribution.tensor[item, span_id, column]
                        for action, distribution in actions.items()
                        if action is not self._action_set.null
                    )
                    denominator = torch.logsumexp(torch.stack(non_null), dim=0)

                    for action, semantics in self._binary.items():
                        action_weight = actions[action].tensor[item, span_id, column]
                        route_weight = split_weight + action_weight - denominator

                        for left_term, left_weight in zip(
                            left.terms, left.log_weights, strict=True
                        ):
                            for right_term, right_weight in zip(
                                right.terms, right.log_weights, strict=True
                            ):
                                formed = tuple(
                                    run_action(
                                        semantic,
                                        (left_term, right_term),
                                        calculus=self._meaning_actions.calculus,
                                        normalization=self._normalization,
                                    )
                                    for semantic in semantics
                                )
                                for term, share in meaning_shares(
                                    self._weighting, formed
                                ).items():
                                    contributions.setdefault(term, []).append(
                                        route_weight
                                        + left_weight
                                        + right_weight
                                        + log(share)
                                    )

                if basis is None:
                    continue

                component = f"proof-term span ({i}, {j})"

                if not contributions:
                    cells[item, spans.span_id(i, j)] = self._bound.check_terms(
                        WeightedTerms(
                            (),
                            basis.new_empty((0,)),
                            basis.new_zeros(()),
                            (
                                Estimate.SAMPLED
                                if Estimate.SAMPLED in estimates
                                else (
                                    Estimate.TRUNCATED
                                    if Estimate.TRUNCATED in estimates
                                    else Estimate.EXACT
                                )
                            ),
                        ),
                        component=component,
                    )
                    continue

                terms = tuple(contributions)
                log_weights = torch.stack(
                    tuple(
                        torch.logsumexp(torch.stack(contributions[term]), dim=0)
                        for term in terms
                    )
                )
                dropped = len(terms) > self._cap

                if dropped:
                    order = sorted(
                        range(len(terms)),
                        key=lambda position: (
                            -float(log_weights[position].detach()),
                            position,
                        ),
                    )[: self._cap]
                    terms = tuple(terms[position] for position in order)
                    log_weights = log_weights[
                        log_weights.new_tensor(order, dtype=torch.long)
                    ]

                estimate = (
                    Estimate.EXACT
                    if not estimates
                    else (
                        Estimate.SAMPLED
                        if Estimate.SAMPLED in estimates
                        else (
                            Estimate.TRUNCATED
                            if Estimate.TRUNCATED in estimates or dropped
                            else Estimate.EXACT
                        )
                    )
                )
                cells[item, spans.span_id(i, j)] = self._bound.check_terms(
                    WeightedTerms(
                        terms,
                        log_weights,
                        _log_residual(log_weights),
                        estimate,
                    ),
                    component=component,
                )

        return MappingProxyType(dict(sorted(cells.items())))

    @override
    def __repr__(self) -> str:
        """Return the semantic-action coverage of the pushforward."""
        return (
            f"ProofTermPushforward(binary={len(self._binary)}, "
            f"actions={len(self._meaning_actions)}, cap={self._cap})"
        )
