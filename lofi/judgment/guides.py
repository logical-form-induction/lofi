"""Variational guides over crossed random-effects models.

Two families are fitted by coordinate ascent. The mean-field family writes
the coefficients and every random effect as independent Gaussians. The
partially factorized family of Goplerud, Papaspiliopoulos and Zanella
(arXiv:2312.13148) keeps each effect Gaussian given the coefficients rather
than independent of them, so an effect's conditional mean is an affine
function of the coefficient vector and its marginal variance picks up the
coefficients' own variance through that function.

Both sweeps cost one pass over the rows per grouping factor plus one solve of
a system the size of the coefficient vector, so a sweep is linear in the rows
and in the levels and the two families differ only in whether the affine part
is carried. Setting it to zero recovers mean field exactly, which is what
makes the second family a relaxation of the first rather than a different
algorithm.
"""

import math
from typing import TYPE_CHECKING, Final, final, override

import torch

from lofi.judgment._spec import InferenceMethod
from lofi.judgment.crossed import CrossedPosterior
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment._ports import VariationalGuide
    from lofi.judgment.crossed import CrossedModel

__all__ = [
    "CrossedState",
    "MeanFieldGuide",
    "PartiallyFactorizedGuide",
    "fit_guide",
    "resolve_guide",
]


@final
class CrossedState:
    """One member of a variational family over a crossed model.

    The coefficients are Gaussian with a full covariance. Each effect is
    Gaussian given the coefficients, with a variance that does not depend on
    them and a mean that is its recorded mean plus its loading against the
    coefficients' departure from their own mean. A family with no loadings is
    the mean-field one, whose effects are independent of the coefficients.

    Parameters
    ----------
    fixed_mean : Tensor
        Mean of the coefficient vector.
    fixed_covariance : Tensor
        Covariance of the coefficient vector.
    effect_means : Tensor
        Mean of each effect, in effect order.
    effect_variances : Tensor
        Variance of each effect given the coefficients, in effect order.
    effect_loadings : Tensor | None
        Rate at which each effect's conditional mean follows the
        coefficients, over effects and then coefficients, or None where the
        effects do not follow them.

    Raises
    ------
    ValueError
        If the covariance is not square against the coefficient mean, the
        effect means and variances are not one dimension of the same length,
        a variance is not positive, or the loadings are not one row per
        effect and one column per coefficient.
    """

    __slots__ = (
        "_effect_loadings",
        "_effect_means",
        "_effect_variances",
        "_fixed_covariance",
        "_fixed_mean",
    )

    def __init__(
        self,
        fixed_mean: Tensor,
        fixed_covariance: Tensor,
        effect_means: Tensor,
        effect_variances: Tensor,
        effect_loadings: Tensor | None,
    ) -> None:
        coefficients = int(fixed_mean.numel())

        if fixed_mean.ndim != 1 or fixed_covariance.shape != (
            coefficients,
            coefficients,
        ):
            msg = (
                f"a coefficient mean of shape {tuple(fixed_mean.shape)} takes a "
                f"square covariance, and shape "
                f"{tuple(fixed_covariance.shape)} was given"
            )

            raise ValueError(msg)

        if effect_means.ndim != 1 or effect_means.shape != effect_variances.shape:
            msg = (
                f"a state takes one mean and one variance per effect, and "
                f"shapes {tuple(effect_means.shape)} and "
                f"{tuple(effect_variances.shape)} were given"
            )

            raise ValueError(msg)

        if bool((effect_variances <= 0.0).any()):
            worst = int(torch.argmin(effect_variances))
            msg = (
                f"effect {worst} has variance {float(effect_variances[worst])}, "
                f"which is not positive"
            )

            raise ValueError(msg)

        if effect_loadings is not None and effect_loadings.shape != (
            int(effect_means.numel()),
            coefficients,
        ):
            msg = (
                f"loadings over {int(effect_means.numel())} effects and "
                f"{coefficients} coefficients have shape "
                f"{tuple(effect_loadings.shape)}"
            )

            raise ValueError(msg)

        self._fixed_mean = fixed_mean
        self._fixed_covariance = fixed_covariance
        self._effect_means = effect_means
        self._effect_variances = effect_variances
        self._effect_loadings = effect_loadings

    @property
    def fixed_mean(self) -> Tensor:
        """Return the mean of the coefficient vector."""
        return self._fixed_mean

    @property
    def fixed_covariance(self) -> Tensor:
        """Return the covariance of the coefficient vector."""
        return self._fixed_covariance

    @property
    def effect_means(self) -> Tensor:
        """Return the mean of each effect, in effect order."""
        return self._effect_means

    @property
    def effect_variances(self) -> Tensor:
        """Return each effect's variance given the coefficients."""
        return self._effect_variances

    @property
    def effect_loadings(self) -> Tensor | None:
        """Return how each effect's mean follows the coefficients, or None."""
        return self._effect_loadings

    @override
    def __repr__(self) -> str:
        """Return the coefficient and effect counts and whether loadings are held."""
        return (
            f"CrossedState(fixed={int(self._fixed_mean.numel())}, "
            f"effects={int(self._effect_means.numel())}, "
            f"loadings={self._effect_loadings is not None})"
        )


def _conditional_precisions(model: CrossedModel, /) -> Tensor:
    """Return the precision of each effect given everything else.

    Parameters
    ----------
    model : CrossedModel
        Model the effects belong to.

    Returns
    -------
    Tensor
        One precision per effect, in effect order.
    """
    return (
        model.effect_counts / model.scales.residual**2 + model.effect_prior_precisions
    )


def _initial(model: CrossedModel, /, *, loadings: bool) -> CrossedState:
    """Return the state a fit starts from.

    Parameters
    ----------
    model : CrossedModel
        Model to fit.
    loadings : bool
        Whether the effects follow the coefficients.

    Returns
    -------
    CrossedState
        Coefficients at their prior, effects at zero with their conditional
        variances.
    """
    dtype = model.fixed.dtype
    device = model.fixed.device
    coefficients = model.n_fixed

    return CrossedState(
        torch.zeros((coefficients,), dtype=dtype, device=device),
        model.scales.fixed**2 * torch.eye(coefficients, dtype=dtype, device=device),
        torch.zeros((model.n_effects,), dtype=dtype, device=device),
        1.0 / _conditional_precisions(model),
        torch.zeros((model.n_effects, coefficients), dtype=dtype, device=device)
        if loadings
        else None,
    )


def _sweep(
    model: CrossedModel, state: CrossedState, /, *, loadings: bool
) -> CrossedState:
    """Return the state after one coordinate ascent sweep.

    Each grouping factor's effects are set to their optimum given the other
    factors and the coefficients, in factor order, and the coefficients are
    then set to their optimum given the effects. Where the effects follow the
    coefficients, the same recursion that sets an effect's mean sets its
    loading, since the two are the value and the slope of one affine function
    of the coefficient vector.

    Parameters
    ----------
    model : CrossedModel
        Model to fit.
    state : CrossedState
        State to improve on.
    loadings : bool
        Whether the effects follow the coefficients.

    Returns
    -------
    CrossedState
        The state after the sweep.
    """
    dtype = model.fixed.dtype
    device = model.fixed.device
    noise = model.scales.residual**2
    prior = model.effect_prior_precisions
    gain = 1.0 / (noise * _conditional_precisions(model))
    means = state.effect_means.clone()
    held = state.effect_loadings
    follows = (
        (
            torch.zeros(
                (model.n_effects, model.n_fixed),
                dtype=dtype,
                device=device,
            )
            if held is None
            else held.clone()
        )
        if loadings
        else None
    )
    offset = model.responses - model.fixed @ state.fixed_mean
    pooled = means[model.stacked_indices].sum(dim=1)
    carried = (
        model.fixed
        if follows is None
        else model.fixed + follows[model.stacked_indices].sum(dim=1)
    )

    for index, start, size in zip(
        model.group_indices, model.group_offsets, model.group_sizes, strict=True
    ):
        block = slice(start, start + size)
        without = pooled - means[index]
        residual = torch.zeros_like(means)
        residual.index_add_(0, index, offset - without)
        means[block] = residual[block] * gain[block]
        pooled = without + means[index]

        if follows is None:
            continue

        rest = carried - follows[index]
        gathered = torch.zeros_like(follows)
        gathered.index_add_(0, index, rest)
        follows[block] = -gathered[block] * gain[block].unsqueeze(1)
        carried = rest + follows[index]

    precision = (
        carried.T @ carried / noise
        + torch.eye(model.n_fixed, dtype=dtype, device=device) / model.scales.fixed**2
    )
    gradient = (
        carried.T @ (offset - pooled) / noise - state.fixed_mean / model.scales.fixed**2
    )

    if follows is not None:
        weighted = follows * prior.unsqueeze(1)
        precision = precision + follows.T @ weighted
        gradient = gradient - weighted.T @ means

    factor = torch.linalg.cholesky(precision)
    covariance = torch.cholesky_inverse(factor)
    step = torch.cholesky_solve(gradient.unsqueeze(1), factor).squeeze(1)

    if follows is not None:
        means = means + follows @ step

    return CrossedState(
        state.fixed_mean + step,
        covariance,
        means,
        1.0 / _conditional_precisions(model),
        follows,
    )


def _evidence_bound(model: CrossedModel, state: CrossedState, /) -> Tensor:
    """Return the evidence lower bound at a state.

    Parameters
    ----------
    model : CrossedModel
        Model the state approximates the posterior of.
    state : CrossedState
        State to score.

    Returns
    -------
    Tensor
        The bound, as a scalar.

    Raises
    ------
    ValueError
        If the state's coefficient or effect counts differ from the model's.
    """
    if (
        int(state.fixed_mean.numel()) != model.n_fixed
        or int(state.effect_means.numel()) != model.n_effects
    ):
        msg = (
            f"a model over {model.n_fixed} coefficients and "
            f"{model.n_effects} effects was scored against a state over "
            f"{int(state.fixed_mean.numel())} and "
            f"{int(state.effect_means.numel())}"
        )

        raise ValueError(msg)

    noise = model.scales.residual**2
    spread = model.scales.fixed**2
    covariance = state.fixed_covariance
    means = state.effect_means
    variances = state.effect_variances
    follows = state.effect_loadings
    carried = (
        model.fixed
        if follows is None
        else model.fixed + follows[model.stacked_indices].sum(dim=1)
    )
    residual = (
        model.responses
        - model.fixed @ state.fixed_mean
        - means[model.stacked_indices].sum(dim=1)
    )
    squared = (
        (residual**2).sum()
        + ((carried @ covariance) * carried).sum()
        + (variances * model.effect_counts).sum()
    )
    spread_of_effects = (
        torch.zeros_like(means)
        if follows is None
        else ((follows @ covariance) * follows).sum(dim=1)
    )
    second = means**2 + variances + spread_of_effects
    density = (
        -0.5 * model.n_rows * math.log(2.0 * math.pi * noise)
        - squared / (2.0 * noise)
        - 0.5 * model.n_fixed * math.log(2.0 * math.pi * spread)
        - ((state.fixed_mean**2).sum() + torch.diagonal(covariance).sum())
        / (2.0 * spread)
        - 0.5 * torch.log(2.0 * math.pi / model.effect_prior_precisions).sum()
        - 0.5 * (model.effect_prior_precisions * second).sum()
    )
    entropy = (
        0.5 * (model.n_fixed + model.n_effects) * (1.0 + math.log(2.0 * math.pi))
        + 0.5 * torch.logdet(covariance)
        + 0.5 * torch.log(variances).sum()
    )

    return density + entropy


def _posterior(state: CrossedState, /) -> CrossedPosterior:
    """Return the marginal means and variances a state implies.

    An effect's marginal variance is its variance given the coefficients plus
    the variance its loading carries over from them, which is what a family
    with no loadings leaves out.

    Parameters
    ----------
    state : CrossedState
        State to read.

    Returns
    -------
    CrossedPosterior
        One mean and one variance per parameter, in layout order.
    """
    follows = state.effect_loadings
    covariance = state.fixed_covariance
    carried = (
        torch.zeros_like(state.effect_variances)
        if follows is None
        else ((follows @ covariance) * follows).sum(dim=1)
    )

    return CrossedPosterior(
        torch.cat([state.fixed_mean, state.effect_means]),
        torch.cat([torch.diagonal(covariance), state.effect_variances + carried]),
    )


@final
class MeanFieldGuide:
    """The Gaussian guide whose effects are independent of the coefficients.

    Each effect's variance is the reciprocal of its precision given every
    other parameter, so nothing of the coefficients' own variance reaches it.
    """

    __slots__ = ()

    @property
    def method(self) -> InferenceMethod:
        """Return the inference a fit under this guide is recorded as."""
        return InferenceMethod.MEAN_FIELD

    def initialize(self, model: CrossedModel, /) -> CrossedState:
        """Return the state a fit starts from.

        Parameters
        ----------
        model : CrossedModel
            Model to fit.

        Returns
        -------
        CrossedState
            Coefficients at their prior and effects at zero.
        """
        return _initial(model, loadings=False)

    def update(self, model: CrossedModel, state: CrossedState, /) -> CrossedState:
        """Return the state after one coordinate ascent sweep.

        Parameters
        ----------
        model : CrossedModel
            Model to fit.
        state : CrossedState
            State to improve on.

        Returns
        -------
        CrossedState
            The state after the sweep.
        """
        return _sweep(model, state, loadings=False)

    def evidence_bound(self, model: CrossedModel, state: CrossedState, /) -> Tensor:
        """Return the evidence lower bound at a state.

        Parameters
        ----------
        model : CrossedModel
            Model the state approximates the posterior of.
        state : CrossedState
            State to score.

        Returns
        -------
        Tensor
            The bound, as a scalar.
        """
        return _evidence_bound(model, state)

    def posterior(self, state: CrossedState, /) -> CrossedPosterior:
        """Return the marginal means and variances a state implies.

        Parameters
        ----------
        state : CrossedState
            State to read.

        Returns
        -------
        CrossedPosterior
            One mean and one variance per parameter.
        """
        return _posterior(state)


@final
class PartiallyFactorizedGuide:
    """The Gaussian guide whose effects follow the coefficients.

    Each effect is Gaussian given the coefficients, with a mean that is an
    affine function of them and a variance that is not. Its marginal variance
    is therefore its conditional variance plus what its loading carries over
    from the coefficients' covariance, which is the term mean field drops.
    """

    __slots__ = ()

    @property
    def method(self) -> InferenceMethod:
        """Return the inference a fit under this guide is recorded as."""
        return InferenceMethod.STRUCTURED

    def initialize(self, model: CrossedModel, /) -> CrossedState:
        """Return the state a fit starts from.

        Parameters
        ----------
        model : CrossedModel
            Model to fit.

        Returns
        -------
        CrossedState
            Coefficients at their prior, effects at zero and loadings at zero.
        """
        return _initial(model, loadings=True)

    def update(self, model: CrossedModel, state: CrossedState, /) -> CrossedState:
        """Return the state after one coordinate ascent sweep.

        Parameters
        ----------
        model : CrossedModel
            Model to fit.
        state : CrossedState
            State to improve on.

        Returns
        -------
        CrossedState
            The state after the sweep.
        """
        return _sweep(model, state, loadings=True)

    def evidence_bound(self, model: CrossedModel, state: CrossedState, /) -> Tensor:
        """Return the evidence lower bound at a state.

        Parameters
        ----------
        model : CrossedModel
            Model the state approximates the posterior of.
        state : CrossedState
            State to score.

        Returns
        -------
        Tensor
            The bound, as a scalar.
        """
        return _evidence_bound(model, state)

    def posterior(self, state: CrossedState, /) -> CrossedPosterior:
        """Return the marginal means and variances a state implies.

        Parameters
        ----------
        state : CrossedState
            State to read.

        Returns
        -------
        CrossedPosterior
            One mean and one variance per parameter.
        """
        return _posterior(state)


MEAN_FIELD_GUIDE: Final = MeanFieldGuide()
"""The mean-field guide, listed as ``mean_field`` in ``lofi.guide``."""

PARTIALLY_FACTORIZED_GUIDE: Final = PartiallyFactorizedGuide()
"""The partially factorized guide, listed as ``structured`` in ``lofi.guide``."""


def resolve_guide(method: InferenceMethod, /) -> VariationalGuide[CrossedState]:
    """Return the installed guide a fit under one inference method runs.

    Guides are listed in the ``lofi.guide`` entry point group, each under the
    inference method a fit under it is recorded as.

    Parameters
    ----------
    method : InferenceMethod
        The inference the fit is recorded as.

    Returns
    -------
    VariationalGuide[CrossedState]
        The guide listed under the method.

    Raises
    ------
    UnknownTagError
        If no installed guide is listed under the method.
    RegistryError
        If the guide listed under the method records another method.
    """
    guides: Registry[VariationalGuide[CrossedState]] = Registry("guide")
    guide = guides.get(str(method))

    if guide.method is not method:
        msg = (
            f"the guide listed as {str(method)!r} records the inference {guide.method}"
        )
        raise RegistryError(msg, family="guide")

    return guide


def fit_guide(
    guide: VariationalGuide[CrossedState],
    model: CrossedModel,
    /,
    *,
    iterations: int,
) -> CrossedPosterior:
    """Return the posterior a guide reaches after a number of sweeps.

    Parameters
    ----------
    guide : VariationalGuide[CrossedState]
        Guide to fit with.
    model : CrossedModel
        Model to fit.
    iterations : int
        Number of sweeps.

    Returns
    -------
    CrossedPosterior
        One mean and one variance per parameter.

    Raises
    ------
    ValueError
        If fewer than one sweep is asked for.
    """
    if iterations < 1:
        msg = f"a fit runs at least one sweep, and {iterations} was asked for"

        raise ValueError(msg)

    state = guide.initialize(model)

    for _ in range(iterations):
        state = guide.update(model, state)

    return guide.posterior(state)
