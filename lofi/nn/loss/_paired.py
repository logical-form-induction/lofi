"""Divergences between decoded complete term distributions."""

from typing import TYPE_CHECKING

import torch

from lofi._frozen import FrozenValue
from lofi.nn._values import LogDistribution
from lofi.nn.forms import Estimate
from lofi.nn.loss._kernels import node_divergence
from lofi.nn.loss._ports import LossTerm

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._divergence import Divergence
    from lofi.terms import Term

__all__ = ["PairedTermDivergence"]

_AXES = ("term",)


def _identity(term: Term) -> Term:
    """Return a term unchanged."""
    return term


def _aggregate(
    terms: tuple[Term, ...],
    log_weights: Tensor,
    transform: Callable[[Term], Term],
) -> dict[Term, Tensor]:
    """Aggregate log weights after one term morphism."""
    grouped: dict[Term, list[Tensor]] = {}

    for term, weight in zip(terms, log_weights, strict=True):
        grouped.setdefault(transform(term), []).append(weight)

    return {
        term: weights[0] if len(weights) == 1 else torch.stack(weights).logsumexp(0)
        for term, weights in grouped.items()
    }


def _distribution(
    weights: dict[Term, Tensor], support: tuple[Term, ...], /
) -> LogDistribution:
    """Return one normalized distribution over a shared finite support."""
    like = next(iter(weights.values()))
    absent = like.new_full((), -torch.inf)
    scores = torch.stack([weights.get(term, absent) for term in support])
    normalized = scores - scores.logsumexp(0)

    return LogDistribution(normalized, _AXES, normalized_over=_AXES[0])


def _mass(weights: dict[Term, Tensor], support: tuple[Term, ...], /) -> Tensor:
    """Return the probability one side assigns to a set of terms."""
    like = next(iter(weights.values()))

    if not support:
        return like.new_zeros(())

    return torch.stack([weights[term] for term in support]).logsumexp(0).exp()


class PairedTermDivergence(FrozenValue):
    """Compare two finite decoded distributions over complete terms.

    Each side may be mapped into a common language before equal images are
    aggregated. Exact inputs compare their whole supports. When either input is
    truncated, a term only one side lists has unknown mass on the other side, so
    the comparison is conditioned on the terms both sides list and publishes the
    probability that at least one decoder lies outside them. Listings that share
    no term compare nothing: the divergence is zero and all mass is omitted.
    """

    _noun = "paired term divergence"

    __slots__ = ("_candidate", "_divergence", "_reference")

    def __init__(
        self,
        divergence: Divergence,
        /,
        *,
        reference: Callable[[Term], Term] | None = None,
        candidate: Callable[[Term], Term] | None = None,
    ) -> None:
        object.__setattr__(self, "_divergence", divergence)
        object.__setattr__(self, "_reference", reference or _identity)
        object.__setattr__(self, "_candidate", candidate or _identity)

    def __call__(
        self, reference: WeightedTerms, candidate: WeightedTerms, /
    ) -> LossTerm:
        """Return the declared divergence and the union of omitted mass."""
        if (
            reference.estimate is Estimate.SAMPLED
            or candidate.estimate is Estimate.SAMPLED
        ):
            msg = "a paired term divergence is undefined for sampled weights"
            raise ValueError(msg)

        reference_by_term = _aggregate(
            reference.terms, reference.log_weights, self._reference
        )
        candidate_by_term = _aggregate(
            candidate.terms, candidate.log_weights, self._candidate
        )

        if not reference_by_term or not candidate_by_term:
            msg = "a paired term divergence needs both decoders to reach a term"
            raise ValueError(msg)

        if (
            reference.estimate is Estimate.EXACT
            and candidate.estimate is Estimate.EXACT
        ):
            support = tuple(
                dict.fromkeys((*reference_by_term.keys(), *candidate_by_term.keys()))
            )
            value = node_divergence(
                self._divergence,
                _distribution(reference_by_term, support),
                _distribution(candidate_by_term, support),
            )

            return LossTerm(value, 1, residual=reference.log_residual.new_zeros(()))

        support = tuple(term for term in reference_by_term if term in candidate_by_term)
        residual = 1.0 - _mass(reference_by_term, support) * _mass(
            candidate_by_term, support
        )

        if not support:
            return LossTerm(reference.log_weights.new_zeros(()), 1, residual=residual)

        value = node_divergence(
            self._divergence,
            _distribution(reference_by_term, support),
            _distribution(candidate_by_term, support),
        )

        return LossTerm(value, 1, residual=residual)
