"""Composing pair verdicts into one probability of entailment."""

from typing import TYPE_CHECKING, NamedTuple, Never, final, override

import torch

from lofi._checks import CHECKS, ensure_finite
from lofi.nn.grounding._values import GradientPath

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from torch import Tensor

    from lofi.nn.forms import WeightedTerms
    from lofi.nn.grounding._values import Composition


__all__ = ["Composed", "PairTable", "compose", "heaviest", "pair_log_weights"]


def heaviest(log_weights: Tensor, width: int, /) -> frozenset[int]:
    """Return the indices of the heaviest entries up to a width.

    Parameters
    ----------
    log_weights : Tensor
        The weights to rank, of rank 1.
    width : int
        The maximum number of entries to keep.
    """
    order = log_weights.detach().argsort(descending=True).tolist()

    return frozenset(order[:width])


def pair_log_weights(premise: WeightedTerms, hypothesis: WeightedTerms, /) -> Tensor:
    """Return the log joint weight of every pair of a premise and a hypothesis form.

    Parameters
    ----------
    premise : WeightedTerms
        The premise's forms.
    hypothesis : WeightedTerms
        The hypothesis's forms.

    Returns
    -------
    Tensor
        Of rank 1 over the pairs in the order ``itertools.product`` visits the
        premise forms and then the hypothesis forms, each the sum of the two
        log weights and with the same gradient connections.
    """
    return (premise.log_weights[:, None] + hypothesis.log_weights[None, :]).reshape(-1)


@final
class PairTable:
    """The pairs one alignment was scored under, with what each was scored.

    Parameters
    ----------
    log_joint : Tensor
        The log joint weight of each pair, of rank 1 at floating precision,
        attached to the beam weights.
    truth : Tensor
        How far each pair entails, of the same shape: one for a proof, zero for
        a countermodel, and a value between them for a relaxation, which may be
        attached to the graph. Ignored where the pair is undecided.
    decided : Tensor
        Whether each pair was decided, of the same shape at boolean precision.
        A pair not decided, whether never consulted or abstained on, moves its
        mass to the residual.

    Raises
    ------
    TypeError
        If the joint weights or the truth values are not floating point, or the
        decided mask is not boolean.
    ValueError
        If the three do not share one shape of rank 1, or, when the debug checks
        are on, if a decided truth value lies outside the unit interval.
    """

    __slots__ = ("_decided", "_log_joint", "_truth")

    def __init__(self, log_joint: Tensor, truth: Tensor, decided: Tensor) -> None:
        if not log_joint.is_floating_point() or not truth.is_floating_point():
            msg = (
                "the joint weights and the truth values are floating point, not "
                f"{log_joint.dtype} and {truth.dtype}"
            )

            raise TypeError(msg)

        if decided.dtype is not torch.bool:
            msg = f"the decided mask is boolean, not {decided.dtype}"

            raise TypeError(msg)

        if log_joint.ndim != 1 or truth.shape != log_joint.shape:
            msg = (
                "a pair table holds one truth value per pair, so its truth values are "
                f"of shape {tuple(log_joint.shape)}, not {tuple(truth.shape)}"
            )

            raise ValueError(msg)

        if decided.shape != log_joint.shape:
            msg = (
                "a pair table holds one decision per pair, so its decided mask is of "
                f"shape {tuple(log_joint.shape)}, not {tuple(decided.shape)}"
            )

            raise ValueError(msg)

        if CHECKS:
            held = truth.detach()[decided]

            if held.numel() and (bool(held.min() < 0.0) or bool(held.max() > 1.0)):
                msg = "a decided truth value lies in the unit interval"

                raise ValueError(msg)

        object.__setattr__(self, "_log_joint", log_joint)
        object.__setattr__(self, "_truth", truth)
        object.__setattr__(self, "_decided", decided)

    @classmethod
    def from_values(
        cls, log_joint: Tensor, values: Iterable[float | None], /
    ) -> PairTable:
        """Return the table of pairs a prover decided by one, zero or nothing.

        Parameters
        ----------
        log_joint : Tensor
            The log joint weight of each pair, of rank 1.
        values : Iterable[float | None]
            What each pair was paid, in the order of the weights: one where the
            hypothesis was proved from the premise, zero where it was refuted,
            and None where the prover abstained.

        Returns
        -------
        PairTable
            The table, with the abstained pairs undecided.

        Raises
        ------
        ValueError
            If there is not one value per pair.
        """
        held = tuple(values)

        if len(held) != log_joint.shape[0]:
            msg = (
                f"a pair table holds one value per pair, so over "
                f"{int(log_joint.shape[0])} pairs there are that many values, not "
                f"{len(held)}"
            )

            raise ValueError(msg)

        truth = log_joint.new_tensor(
            [0.0 if value is None else value for value in held]
        )
        decided = torch.tensor(
            [value is not None for value in held],
            dtype=torch.bool,
            device=log_joint.device,
        )

        return cls(log_joint, truth, decided)

    @property
    def log_joint(self) -> Tensor:
        """Return the log joint weight of each pair."""
        return self._log_joint

    @property
    def truth(self) -> Tensor:
        """Return how far each pair entails."""
        return self._truth

    @property
    def decided(self) -> Tensor:
        """Return whether each pair was decided."""
        return self._decided

    def __len__(self) -> int:
        """Return how many pairs the table holds."""
        return int(self._log_joint.shape[0])

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a pair table is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a pair table is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return how many pairs the table holds and how many were decided."""
        return f"PairTable(pairs={len(self)}, decided={int(self._decided.sum())})"


class Composed(NamedTuple):
    """What one prediction composes to, before the prover's ledger is attached.

    Attributes
    ----------
    log_probability : Tensor | None
        A scalar whose forward value is the estimated log probability of
        entailment and whose gradient follows the declared path, or None where
        no pair of any alignment was decided.
    residual : Tensor
        The mass left unscored, a detached scalar probability.
    """

    log_probability: Tensor | None
    residual: Tensor


def _at_most_zero(log_probability: Tensor) -> Tensor:
    """Return a log probability with any rounding above zero subtracted.

    Parameters
    ----------
    log_probability : Tensor
        A scalar that is zero up to rounding where every share entails.

    Returns
    -------
    Tensor
        The scalar, at most zero, with its gradient unchanged.
    """
    return log_probability - log_probability.detach().clamp_min(0.0)


def _log_entailing(log_shares: Tensor, truth: Tensor) -> Tensor:
    """Return the log probability of entailment from shares and truth values.

    Parameters
    ----------
    log_shares : Tensor
        The log share of each decided pair, some of which may be negative
        infinity.
    truth : Tensor
        How far each decided pair entails.

    Returns
    -------
    Tensor
        A scalar: the log of the sum of each share times its truth value, which
        is negative infinity, attached to nothing, where that sum is zero.
    """
    live = (truth.detach() > 0.0) & (log_shares.detach() > -torch.inf)

    if not bool(live.any()):
        return log_shares.new_full((), -torch.inf)

    return (log_shares[live] + truth[live].log()).logsumexp(dim=0)


def _pathwise(log_joint: Tensor, truth: Tensor, composition: Composition) -> Tensor:
    """Return the log probability of entailment differentiated as computed.

    Parameters
    ----------
    log_joint : Tensor
        The log joint weight of each decided pair.
    truth : Tensor
        How far each decided pair entails.
    composition : Composition
        The rule the pairs are weighed by.
    """
    return _log_entailing(composition.log_shares(log_joint), truth)


def _score_function(
    log_joint: Tensor, truth: Tensor, composition: Composition, baseline: float
) -> Tensor:
    """Return the log probability of entailment with the score-function gradient.

    Parameters
    ----------
    log_joint : Tensor
        The log joint weight of each decided pair.
    truth : Tensor
        How far each decided pair entails, taken as constant.
    composition : Composition
        The rule the pairs are weighed by, at detached weights.
    baseline : float
        What is subtracted from each truth value in the surrogate. Where the
        joint weights are the log probabilities of drawn pairs and each pair
        takes an equal share, this moves the estimator's variance and not its
        expectation. Over an enumerated table it moves the gradient, which
        equals the pathwise one exactly when the baseline is the estimate.

    Returns
    -------
    Tensor
        A scalar whose forward value is the estimate under detached shares and
        whose gradient is that of the surrogate: each pair's log joint weight
        times its share times its centered truth value, divided by the
        estimate. Negative infinity, attached to nothing, where the estimate
        is zero.
    """
    centered = truth.detach() - baseline
    log_shares = composition.log_shares(log_joint.detach())
    estimate = _log_entailing(log_shares, truth.detach())

    if bool(estimate == -torch.inf):
        return estimate

    weight = centered.sign() * (log_shares - estimate + centered.abs().log()).exp()
    surrogate = (weight * log_joint).sum()

    return estimate + surrogate - surrogate.detach()


def _mixed(
    log_joint: Tensor, truth: Tensor, composition: Composition, baseline: float
) -> Tensor:
    """Return the sampled estimate with scorer and score-function gradients."""
    log_shares = composition.log_shares(log_joint.detach())
    estimate = _log_entailing(log_shares, truth)

    if bool(estimate == -torch.inf):
        return estimate

    centered = truth.detach() - baseline
    weight = (
        centered.sign() * (log_shares - estimate.detach() + centered.abs().log()).exp()
    )
    surrogate = (weight * log_joint).sum()
    return estimate + surrogate - surrogate.detach()


def _one(
    table: PairTable,
    *,
    composition: Composition,
    gradient_path: GradientPath,
    baseline: float,
) -> Composed:
    """Return what one alignment's table composes to.

    Parameters
    ----------
    table : PairTable
        The pairs scored under the alignment.
    composition : Composition
        The rule the pairs are weighed by.
    gradient_path : GradientPath
        How the gradient reaches the joint weights.
    baseline : float
        The score-function baseline, unused on the pathwise path.
    """
    decided = table.decided
    log_joint = table.log_joint[decided]
    residual = (1.0 - log_joint.detach().exp().sum()).clamp(0.0, 1.0)

    if log_joint.numel() == 0:
        return Composed(None, residual)

    truth = table.truth[decided]

    if gradient_path is GradientPath.PATHWISE:
        return Composed(_pathwise(log_joint, truth, composition), residual)

    if gradient_path is GradientPath.MIXED:
        return Composed(_mixed(log_joint, truth, composition, baseline), residual)

    return Composed(_score_function(log_joint, truth, composition, baseline), residual)


def compose(
    tables: Sequence[PairTable],
    /,
    *,
    composition: Composition,
    gradient_path: GradientPath,
    baseline: float = 0.0,
) -> Composed:
    """Return the probability of entailment composed over pairs and alignments.

    The composition rule combines decided pairs within an alignment, then combines
    the alignment estimates. Abstained mass contributes to the residual, not to a
    negative verdict.

    Parameters
    ----------
    tables : Sequence[PairTable]
        The pairs scored under each alignment, one table per alignment, at
        least one.
    composition : Composition
        The rule the pairs and the alignments are composed by.
    gradient_path : GradientPath
        How the gradient reaches the joint weights: through the forward value,
        or through the score-function surrogate with the truth values held
        constant.
    baseline : float
        What the score-function surrogate subtracts from each truth value,
        which leaves the estimate alone and moves its gradient. Unused on the
        pathwise path.

    Returns
    -------
    Composed
        The estimate, or None where no alignment decided a pair, and the
        residual.

    Raises
    ------
    ValueError
        If there is no table, or the baseline is not finite.
    TypeError
        If the baseline is not a number.
    """
    if not tables:
        msg = "a prediction composes over at least one alignment"

        raise ValueError(msg)

    centered = ensure_finite("a score-function baseline", baseline)
    composed = [
        _one(
            table,
            composition=composition,
            gradient_path=gradient_path,
            baseline=centered,
        )
        for table in tables
    ]
    residual = composition.over_residuals(
        torch.stack([one.residual for one in composed])
    )
    estimates = [
        one.log_probability for one in composed if one.log_probability is not None
    ]

    if not estimates:
        return Composed(None, residual)

    if len(estimates) == 1:
        return Composed(_at_most_zero(estimates[0]), residual)

    return Composed(
        _at_most_zero(composition.over_alignments(torch.stack(estimates))), residual
    )
