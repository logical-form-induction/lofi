"""Gaussian crossed random-effects models, their exact posterior and a sampler.

A crossed model holds a fixed-effect design, one grouping factor index per
row for each crossed factor, and a continuous response. Its parameters are
laid out with the fixed-effect coefficients first and then each factor's
levels in factor order, and every posterior over it reports one mean and one
variance per parameter in that layout.

With the scales known the posterior is Gaussian, so the reference draws come
from one Cholesky factorization of the posterior precision and are
independent. That reference costs a cubic in the parameter count and is meant
for checking an approximation against, not for fitting a large model.
"""

import math
from typing import TYPE_CHECKING, NamedTuple, final, override

import torch

from lofi._checks import ensure_finite

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign

__all__ = [
    "CredibleIntervals",
    "CrossedModel",
    "CrossedPosterior",
    "CrossedScales",
    "PosteriorDraws",
    "SimulatedCrossed",
    "coverage_of",
    "crossed_model_of",
    "exact_posterior",
    "sample_posterior",
    "simulate_crossed",
]


def _positive(field: str, value: float) -> float:
    """Return a finite positive number.

    Parameters
    ----------
    field : str
        Name of the number, used in the error message.
    value : float
        Number to check.

    Returns
    -------
    float
        The number.

    Raises
    ------
    ValueError
        If the number is not finite or is not positive.
    """
    checked = ensure_finite(field, value)

    if checked <= 0.0:
        msg = f"{field} is not positive: {value!r}"

        raise ValueError(msg)

    return checked


def _proportion(field: str, value: float) -> float:
    """Return a number strictly between zero and one.

    Parameters
    ----------
    field : str
        Name of the number, used in the error message.
    value : float
        Number to check.

    Returns
    -------
    float
        The number.

    Raises
    ------
    ValueError
        If the number is not finite or does not lie strictly inside the unit
        interval.
    """
    checked = ensure_finite(field, value)

    if not 0.0 < checked < 1.0:
        msg = f"{field} is not a proportion: {value!r}"

        raise ValueError(msg)

    return checked


@final
class CrossedScales:
    """The standard deviations a crossed random-effects model is written with.

    Parameters
    ----------
    residual : float
        Spread of a response around its linear predictor.
    fixed : float
        Spread of the prior on each fixed-effect coefficient.
    groups : Sequence[float]
        Spread of each grouping factor's effects, in factor order.

    Raises
    ------
    ValueError
        If a scale is not finite and positive, or no grouping factor is named.
    """

    __slots__ = ("_fixed", "_groups", "_residual")

    def __init__(
        self,
        residual: float,
        fixed: float,
        groups: Sequence[float],
    ) -> None:
        self._residual = _positive("a residual scale", residual)
        self._fixed = _positive("a fixed-effect prior scale", fixed)
        self._groups = tuple(
            _positive(f"the scale of grouping factor {position}", scale)
            for position, scale in enumerate(groups)
        )

        if not self._groups:
            msg = "a crossed model has at least one grouping factor"

            raise ValueError(msg)

    @property
    def residual(self) -> float:
        """Return the spread of a response around its linear predictor."""
        return self._residual

    @property
    def fixed(self) -> float:
        """Return the spread of the prior on each fixed-effect coefficient."""
        return self._fixed

    @property
    def groups(self) -> tuple[float, ...]:
        """Return the spread of each grouping factor's effects, in factor order."""
        return self._groups

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same three scales."""
        if type(other) is not CrossedScales:
            return NotImplemented

        return (self._residual, self._fixed, self._groups) == (
            other._residual,
            other._fixed,
            other._groups,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the three scales."""
        return hash((CrossedScales, self._residual, self._fixed, self._groups))

    @override
    def __repr__(self) -> str:
        """Return the three scales."""
        return (
            f"CrossedScales(residual={self._residual!r}, fixed={self._fixed!r}, "
            f"groups={self._groups!r})"
        )


@final
class CrossedModel:
    r"""A Gaussian crossed random-effects model over recorded rows.

    A row's linear predictor is its fixed-effect row against the coefficients
    plus one effect of every grouping factor. The parameter layout puts the
    fixed-effect coefficients first and then each factor's levels in factor
    order, so a parameter vector has $n_{\mathrm{fixed}}+n_{\mathrm{effects}}$ entries,
    where the two counts are `n_fixed` and `n_effects`.

    Parameters
    ----------
    fixed : Tensor
        Fixed-effect design, one row per response and one column per
        coefficient.
    groups : Sequence[Tensor]
        Zero-based level of each row under each grouping factor, in factor
        order.
    group_sizes : Sequence[int]
        Number of levels of each grouping factor, in factor order.
    responses : Tensor
        Recorded response of each row.
    scales : CrossedScales
        Scales the model is written with.

    Raises
    ------
    ValueError
        If the fixed-effect design is not a floating matrix, the responses are
        not one finite float per row, a factor index is not one integer per
        row, a factor size is not positive, a level lies outside its factor,
        or the number of factors differs from the number of group scales.
    """

    __slots__ = (
        "_counts",
        "_fixed",
        "_indices",
        "_offsets",
        "_prior_precisions",
        "_responses",
        "_scales",
        "_sizes",
        "_stacked",
    )

    def __init__(
        self,
        fixed: Tensor,
        groups: Sequence[Tensor],
        group_sizes: Sequence[int],
        responses: Tensor,
        scales: CrossedScales,
    ) -> None:
        if fixed.ndim != 2 or not fixed.is_floating_point():
            msg = (
                f"a fixed-effect design is a floating matrix, and "
                f"{fixed.dtype} values of shape {tuple(fixed.shape)} were given"
            )

            raise ValueError(msg)

        rows = fixed.shape[0]

        if responses.shape != (rows,) or not responses.is_floating_point():
            msg = (
                f"a model over {rows} rows takes one floating response per row, "
                f"and {responses.dtype} values of shape "
                f"{tuple(responses.shape)} were given"
            )

            raise ValueError(msg)

        if not bool(torch.isfinite(responses).all()) or not bool(
            torch.isfinite(fixed).all()
        ):
            msg = "a crossed model's design and responses are finite"

            raise ValueError(msg)

        sizes = tuple(int(size) for size in group_sizes)
        indices = tuple(groups)

        if len(indices) != len(sizes) or len(indices) != len(scales.groups):
            msg = (
                f"a crossed model takes one index column and one scale per "
                f"grouping factor, and {len(indices)} columns, {len(sizes)} "
                f"sizes and {len(scales.groups)} scales were given"
            )

            raise ValueError(msg)

        for position, (index, size) in enumerate(zip(indices, sizes, strict=True)):
            if index.shape != (rows,) or index.is_floating_point():
                msg = (
                    f"grouping factor {position} takes one integer level per "
                    f"row of {rows}, and {index.dtype} values of shape "
                    f"{tuple(index.shape)} were given"
                )

                raise ValueError(msg)

            if size <= 0:
                msg = f"grouping factor {position} has {size} levels"

                raise ValueError(msg)

            if rows and (int(index.min()) < 0 or int(index.max()) >= size):
                msg = (
                    f"grouping factor {position} holds levels "
                    f"{int(index.min())} to {int(index.max())} against its "
                    f"{size} levels"
                )

                raise ValueError(msg)

        offsets: list[int] = []
        running = 0

        for size in sizes:
            offsets.append(running)
            running += size

        self._fixed = fixed
        self._responses = responses
        self._scales = scales
        self._sizes = sizes
        self._offsets = tuple(offsets)
        self._indices = tuple(
            index.to(dtype=torch.int64) + offset
            for index, offset in zip(indices, self._offsets, strict=True)
        )
        self._stacked = (
            torch.stack(self._indices, dim=1)
            if self._indices
            else torch.zeros((rows, 0), dtype=torch.int64, device=fixed.device)
        )
        counts = torch.zeros(running, dtype=fixed.dtype, device=fixed.device)

        for index in self._indices:
            counts.index_add_(0, index, torch.ones_like(index, dtype=fixed.dtype))

        self._counts = counts
        self._prior_precisions = torch.cat(
            [
                torch.full(
                    (size,), 1.0 / scale**2, dtype=fixed.dtype, device=fixed.device
                )
                for size, scale in zip(sizes, scales.groups, strict=True)
            ]
        )

    @property
    def fixed(self) -> Tensor:
        """Return the fixed-effect design, one row per response."""
        return self._fixed

    @property
    def responses(self) -> Tensor:
        """Return the recorded response of each row."""
        return self._responses

    @property
    def scales(self) -> CrossedScales:
        """Return the scales the model is written with."""
        return self._scales

    @property
    def group_indices(self) -> tuple[Tensor, ...]:
        """Return each row's effect position under each factor, in factor order.

        A position indexes the stacked effect vector rather than the factor's
        own levels, so it already carries the factor's offset.
        """
        return self._indices

    @property
    def stacked_indices(self) -> Tensor:
        """Return the effect positions of every row, over rows and then factors."""
        return self._stacked

    @property
    def group_sizes(self) -> tuple[int, ...]:
        """Return the number of levels of each grouping factor, in factor order."""
        return self._sizes

    @property
    def group_offsets(self) -> tuple[int, ...]:
        """Return where each factor's levels start in the stacked effect vector."""
        return self._offsets

    @property
    def effect_counts(self) -> Tensor:
        """Return how many rows each effect appears in, in effect order."""
        return self._counts

    @property
    def effect_prior_precisions(self) -> Tensor:
        """Return the prior precision of each effect, in effect order."""
        return self._prior_precisions

    @property
    def n_rows(self) -> int:
        """Return the number of recorded responses."""
        return int(self._fixed.shape[0])

    @property
    def n_fixed(self) -> int:
        """Return the number of fixed-effect coefficients."""
        return int(self._fixed.shape[1])

    @property
    def n_effects(self) -> int:
        """Return the number of random effects over every grouping factor."""
        return sum(self._sizes)

    @property
    def n_parameters(self) -> int:
        """Return the number of coefficients and effects together."""
        return self.n_fixed + self.n_effects

    @override
    def __repr__(self) -> str:
        """Return the row, coefficient and level counts with the scales."""
        return (
            f"CrossedModel(rows={self.n_rows}, fixed={self.n_fixed}, "
            f"groups={self._sizes!r}, scales={self._scales!r})"
        )


@final
class CredibleIntervals:
    """One interval per parameter at a stated coverage.

    Parameters
    ----------
    lower : Tensor
        Lower bound of each parameter's interval, in layout order.
    upper : Tensor
        Upper bound of each parameter's interval, in layout order.
    level : float
        Nominal coverage the bounds were computed at.

    Raises
    ------
    ValueError
        If the bounds are not one dimension of the same length, the level is
        not a proportion, or a lower bound exceeds its upper bound.
    """

    __slots__ = ("_level", "_lower", "_upper")

    def __init__(self, lower: Tensor, upper: Tensor, level: float) -> None:
        if lower.ndim != 1 or lower.shape != upper.shape:
            msg = (
                f"an interval takes one lower and one upper bound per "
                f"parameter, and shapes {tuple(lower.shape)} and "
                f"{tuple(upper.shape)} were given"
            )

            raise ValueError(msg)

        if bool((lower > upper).any()):
            worst = int(torch.argmax(lower - upper))
            msg = (
                f"parameter {worst} has lower bound {float(lower[worst])} above "
                f"its upper bound {float(upper[worst])}"
            )

            raise ValueError(msg)

        self._lower = lower
        self._upper = upper
        self._level = _proportion("an interval's coverage level", level)

    @property
    def lower(self) -> Tensor:
        """Return the lower bound of each parameter's interval."""
        return self._lower

    @property
    def upper(self) -> Tensor:
        """Return the upper bound of each parameter's interval."""
        return self._upper

    @property
    def level(self) -> float:
        """Return the nominal coverage the bounds were computed at."""
        return self._level

    def covers(self, values: Tensor, /) -> Tensor:
        """Return whether each parameter's interval holds its value.

        Parameters
        ----------
        values : Tensor
            One value per parameter, in layout order.

        Returns
        -------
        Tensor
            One boolean per parameter.

        Raises
        ------
        ValueError
            If the values are not one per parameter.
        """
        if values.shape != self._lower.shape:
            msg = (
                f"intervals over {int(self._lower.numel())} parameters take one "
                f"value each, and shape {tuple(values.shape)} was given"
            )

            raise ValueError(msg)

        return (self._lower <= values) & (values <= self._upper)

    @override
    def __repr__(self) -> str:
        """Return the parameter count and the coverage level."""
        return (
            f"CredibleIntervals(parameters={int(self._lower.numel())}, "
            f"level={self._level!r})"
        )


@final
class CrossedPosterior:
    """A Gaussian posterior over a crossed model's parameters.

    Parameters
    ----------
    means : Tensor
        Posterior mean of each parameter, in layout order.
    variances : Tensor
        Posterior variance of each parameter, in layout order.

    Raises
    ------
    ValueError
        If the means and variances are not one dimension of the same length,
        a value is not finite, or a variance is not positive.
    """

    __slots__ = ("_means", "_variances")

    def __init__(self, means: Tensor, variances: Tensor) -> None:
        if means.ndim != 1 or means.shape != variances.shape:
            msg = (
                f"a posterior takes one mean and one variance per parameter, "
                f"and shapes {tuple(means.shape)} and "
                f"{tuple(variances.shape)} were given"
            )

            raise ValueError(msg)

        if not bool(torch.isfinite(means).all()) or not bool(
            torch.isfinite(variances).all()
        ):
            msg = "a posterior's means and variances are finite"

            raise ValueError(msg)

        if bool((variances <= 0.0).any()):
            worst = int(torch.argmin(variances))
            msg = (
                f"parameter {worst} has variance {float(variances[worst])}, "
                f"which is not positive"
            )

            raise ValueError(msg)

        self._means = means
        self._variances = variances

    @property
    def means(self) -> Tensor:
        """Return the posterior mean of each parameter."""
        return self._means

    @property
    def variances(self) -> Tensor:
        """Return the posterior variance of each parameter."""
        return self._variances

    @property
    def n_parameters(self) -> int:
        """Return the number of parameters the posterior is over."""
        return int(self._means.numel())

    def intervals(self, level: float, /) -> CredibleIntervals:
        """Return each parameter's equal-tailed Gaussian interval.

        Parameters
        ----------
        level : float
            Nominal coverage, strictly between zero and one.

        Returns
        -------
        CredibleIntervals
            One interval per parameter.
        """
        checked = _proportion("a coverage level", level)
        tail = (1.0 + checked) / 2.0
        quantile = math.sqrt(2.0) * torch.erfinv(
            torch.tensor(2.0 * tail - 1.0, dtype=self._means.dtype)
        )
        half = quantile.to(device=self._means.device) * torch.sqrt(self._variances)

        return CredibleIntervals(self._means - half, self._means + half, checked)

    @override
    def __repr__(self) -> str:
        """Return the parameter count."""
        return f"CrossedPosterior(parameters={self.n_parameters})"


@final
class PosteriorDraws:
    """Draws from a posterior over a crossed model's parameters.

    Parameters
    ----------
    values : Tensor
        One row per draw and one column per parameter, in layout order.

    Raises
    ------
    ValueError
        If the draws are not a matrix, hold no draw, or are not finite.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Tensor) -> None:
        if values.ndim != 2 or values.shape[0] == 0:
            msg = (
                f"draws are a matrix of at least one row, and shape "
                f"{tuple(values.shape)} was given"
            )

            raise ValueError(msg)

        if not bool(torch.isfinite(values).all()):
            msg = "posterior draws are finite"

            raise ValueError(msg)

        self._values = values

    @property
    def values(self) -> Tensor:
        """Return the draws, over draws and then parameters."""
        return self._values

    @property
    def n_draws(self) -> int:
        """Return the number of draws."""
        return int(self._values.shape[0])

    @property
    def n_parameters(self) -> int:
        """Return the number of parameters each draw covers."""
        return int(self._values.shape[1])

    def intervals(self, level: float, /) -> CredibleIntervals:
        """Return each parameter's equal-tailed interval over the draws.

        Parameters
        ----------
        level : float
            Nominal coverage, strictly between zero and one.

        Returns
        -------
        CredibleIntervals
            One interval per parameter, at the empirical quantiles.
        """
        checked = _proportion("a coverage level", level)
        tail = (1.0 - checked) / 2.0
        bounds = torch.quantile(
            self._values,
            torch.tensor([tail, 1.0 - tail], dtype=self._values.dtype),
            dim=0,
        )

        return CredibleIntervals(bounds[0], bounds[1], checked)

    @override
    def __repr__(self) -> str:
        """Return the draw and parameter counts."""
        return f"PosteriorDraws(draws={self.n_draws}, parameters={self.n_parameters})"


class SimulatedCrossed(NamedTuple):
    """A crossed model drawn from its own prior, with the values drawn.

    Attributes
    ----------
    model : CrossedModel
        The model, holding the responses the drawn parameters generated.
    truth : Tensor
        The drawn parameters, in the model's layout order.
    """

    model: CrossedModel
    truth: Tensor


def _dense_design(model: CrossedModel, /) -> Tensor:
    """Return the effect design, one row per response and one column per effect.

    Parameters
    ----------
    model : CrossedModel
        Model whose grouping factors are spread into columns.

    Returns
    -------
    Tensor
        A matrix holding one in each row's effect columns and zero elsewhere.
    """
    dense = torch.zeros(
        (model.n_rows, model.n_effects),
        dtype=model.fixed.dtype,
        device=model.fixed.device,
    )
    rows = torch.arange(model.n_rows, device=model.fixed.device)

    for index in model.group_indices:
        dense[rows, index] += 1.0

    return dense


def _precision_and_mean(model: CrossedModel, /) -> tuple[Tensor, Tensor]:
    """Return the exact posterior precision and mean of a crossed model.

    Parameters
    ----------
    model : CrossedModel
        Model to solve.

    Returns
    -------
    tuple[Tensor, Tensor]
        The posterior precision and the posterior mean, each over the parameter
        layout.
    """
    dense = torch.cat([model.fixed, _dense_design(model)], dim=1)
    noise = model.scales.residual**2
    prior = torch.cat(
        [
            torch.full(
                (model.n_fixed,),
                1.0 / model.scales.fixed**2,
                dtype=model.fixed.dtype,
                device=model.fixed.device,
            ),
            model.effect_prior_precisions,
        ]
    )
    precision = dense.T @ dense / noise + torch.diag(prior)
    weighted = dense.T @ model.responses / noise
    factor = torch.linalg.cholesky(precision)
    mean = torch.cholesky_solve(weighted.unsqueeze(1), factor).squeeze(1)

    return precision, mean


def exact_posterior(model: CrossedModel, /) -> CrossedPosterior:
    """Return the posterior a crossed model has when its scales are known.

    The model is conjugate, so the posterior is the Gaussian whose precision
    is the design's weighted cross-product plus the prior precision. The cost
    is cubic in the parameter count.

    Parameters
    ----------
    model : CrossedModel
        Model to solve.

    Returns
    -------
    CrossedPosterior
        Marginal mean and variance of every parameter.
    """
    precision, mean = _precision_and_mean(model)
    covariance = torch.cholesky_inverse(torch.linalg.cholesky(precision))

    return CrossedPosterior(mean, torch.diagonal(covariance).clone())


def sample_posterior(
    model: CrossedModel,
    /,
    *,
    draws: int,
    generator: torch.Generator,
) -> PosteriorDraws:
    """Return independent draws from a crossed model's exact posterior.

    The posterior is Gaussian, so one Cholesky factorization of its precision
    turns standard normal noise into independent draws. Nothing here mixes,
    so the draws carry no autocorrelation and need no burn-in.

    Parameters
    ----------
    model : CrossedModel
        Model to draw from.
    draws : int
        Number of draws.
    generator : torch.Generator
        Source of the draws, so that a seed fixes the sample.

    Returns
    -------
    PosteriorDraws
        The draws, over draws and then parameters.

    Raises
    ------
    ValueError
        If fewer than one draw is asked for.
    """
    if draws < 1:
        msg = f"a sample holds at least one draw, and {draws} was asked for"

        raise ValueError(msg)

    precision, mean = _precision_and_mean(model)
    factor = torch.linalg.cholesky(precision)
    noise = torch.randn(
        (model.n_parameters, draws),
        generator=generator,
        dtype=model.fixed.dtype,
        device=model.fixed.device,
    )
    offsets = torch.linalg.solve_triangular(factor.mT, noise, upper=True)

    return PosteriorDraws(mean.unsqueeze(0) + offsets.T)


def coverage_of(intervals: CredibleIntervals, values: Tensor, /) -> float:
    """Return the fraction of parameters whose interval holds its value.

    Parameters
    ----------
    intervals : CredibleIntervals
        One interval per parameter.
    values : Tensor
        One value per parameter, in layout order.

    Returns
    -------
    float
        The covered fraction.
    """
    return float(intervals.covers(values).to(dtype=torch.float64).mean())


def simulate_crossed(
    fixed: Tensor,
    groups: Sequence[Tensor],
    group_sizes: Sequence[int],
    scales: CrossedScales,
    /,
    *,
    generator: torch.Generator,
) -> SimulatedCrossed:
    """Return a crossed model whose responses were drawn from its own prior.

    The coefficients, the effects and the residuals are all drawn at the
    scales given, so the drawn parameters are one draw from the prior and the
    responses one draw from the model at them. Posterior intervals over data
    generated this way are calibrated, which is what makes coverage over
    repeated draws comparable against its nominal level.

    Parameters
    ----------
    fixed : Tensor
        Fixed-effect design, one row per response.
    groups : Sequence[Tensor]
        Zero-based level of each row under each grouping factor.
    group_sizes : Sequence[int]
        Number of levels of each grouping factor.
    scales : CrossedScales
        Scales to draw at.
    generator : torch.Generator
        Source of the draws, so that a seed fixes the sample.

    Returns
    -------
    SimulatedCrossed
        The model and the parameters its responses were drawn at.
    """
    sizes = tuple(int(size) for size in group_sizes)
    coefficients = scales.fixed * torch.randn(
        (fixed.shape[1],),
        generator=generator,
        dtype=fixed.dtype,
        device=fixed.device,
    )
    effects = torch.cat(
        [
            scale
            * torch.randn(
                (size,), generator=generator, dtype=fixed.dtype, device=fixed.device
            )
            for size, scale in zip(sizes, scales.groups, strict=True)
        ]
    )
    predictor = fixed @ coefficients
    offset = 0

    for index, size in zip(groups, sizes, strict=True):
        predictor = predictor + effects[offset : offset + size][index.to(torch.int64)]
        offset += size

    responses = predictor + scales.residual * torch.randn(
        (fixed.shape[0],),
        generator=generator,
        dtype=fixed.dtype,
        device=fixed.device,
    )

    return SimulatedCrossed(
        CrossedModel(fixed, groups, sizes, responses, scales),
        torch.cat([coefficients, effects]),
    )


def crossed_model_of(
    design: JudgmentDesign,
    scales: CrossedScales,
    /,
    *,
    response: str,
    covariates: Sequence[str] = (),
) -> CrossedModel:
    """Return the crossed model over a design's participants and items.

    The two grouping factors are the design's participants and its items, in
    that order. The fixed-effect design is an intercept column followed by one
    column per named covariate, read from the design's readouts, and the
    response is read from a readout as well, since a crossed model is over a
    continuous response rather than an ordered level.

    Parameters
    ----------
    design : JudgmentDesign
        Design the rows are read from.
    scales : CrossedScales
        Scales the model is written with.
    response : str
        Readout holding each row's response.
    covariates : Sequence[str]
        Readouts to enter as fixed effects, in column order.

    Returns
    -------
    CrossedModel
        The model over the design's rows.

    Raises
    ------
    ValueError
        If a named readout is absent from the design.
    """
    named = (response, *covariates)
    missing = [name for name in named if name not in design.readouts]

    if missing:
        msg = (
            f"the design holds the readouts {sorted(design.readouts)!r} and "
            f"{missing!r} were asked for"
        )

        raise ValueError(msg)

    responses = design.readouts[response].to(dtype=torch.float64)
    intercept = torch.ones(
        (design.n_rows, 1), dtype=responses.dtype, device=responses.device
    )
    columns = [intercept] + [
        design.readouts[name].to(dtype=responses.dtype).unsqueeze(1)
        for name in covariates
    ]

    return CrossedModel(
        torch.cat(columns, dim=1),
        (design.participants, design.items),
        (len(design.participant_levels), len(design.item_levels)),
        responses,
        scales,
    )
