"""Predictive distributions over held-out rows from fitted response models.

The posterior-draw treatment pairs each draw with either fresh effects for an
unseen participant or their population mean. The marginal-point treatment
instead reuses the fitted integration kernel: a seen participant's candidate
is added to that participant's retained rows, while a new participant's
candidate is integrated alone under the local prior.
"""

import math
from typing import TYPE_CHECKING, NamedTuple, Never, Protocol, final, override

import torch

from lofi.judgment._design import JudgmentRow, RowColumns
from lofi.judgment._spec import (
    Measurement,
    PosteriorVarianceRow,
    PredictiveDensityRow,
    PseudoRSquaredRow,
)
from lofi.judgment.backends import latent
from lofi.judgment.compiler import ITEM, PARTICIPANT
from lofi.judgment.marginal import ApproximateMarginalLikelihood
from lofi.judgment.ordinal import log_category_probabilities
from lofi.spec import HeldOutParticipantPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judgment._design import DesignCodec, JudgmentDesign
    from lofi.judgment._program import Prior
    from lofi.judgment.compiler import BoundProgram, ResponseProgram
    from lofi.judgment.splits import Fold

__all__ = [
    "FoldPredictions",
    "HeldOutPredictor",
    "MarginalPointPredictor",
    "PointEstimate",
    "fold_predictions",
    "posterior_variance_row",
    "predictive_density_row",
    "predictive_log_density",
    "pseudo_r_squared",
]


class PointEstimate(Protocol):
    """A fitted point in an objective owner's parameter layout."""

    @property
    def parameters(self) -> Tensor:
        """Return the fitted one-dimensional parameter vector."""
        ...


def _levels_of(design: JudgmentDesign, factor: str) -> tuple[str, ...]:
    """Return the level table of a grouping factor, in the design's index order.

    Parameters
    ----------
    design : JudgmentDesign
        Design holding the factor.
    factor : str
        The participant column, the item column, or a declared factor.
    """
    if factor == PARTICIPANT:
        return design.participant_levels

    if factor == ITEM:
        return design.item_levels

    return design.factor_levels[factor]


def _population_mean(prior: Prior, site: str) -> float:
    """Return the mean of a prior, for the value plugged in at an unseen level.

    Parameters
    ----------
    prior : Prior
        Prior the site is drawn from.
    site : str
        Site drawn, named in the error message.

    Raises
    ------
    ValueError
        If the prior's family has no mean this module computes.
    """
    match prior.family, prior.arguments:
        case "Normal", (location, _):
            return location
        case "HalfNormal", (scale,):
            return scale * math.sqrt(2.0 / math.pi)
        case "LogNormal", (location, scale):
            return math.exp(location + scale * scale / 2.0)
        case "Uniform", (low, high):
            return (low + high) / 2.0
        case "Exponential", (rate,):
            return 1.0 / rate

    msg = f"the site {site!r} is drawn from {prior}, which has no population mean here"

    raise ValueError(msg)


def _prior_sample(
    prior: Prior,
    site: str,
    shape: tuple[int, ...],
    *,
    like: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Return independent draws from a prior at a shape.

    Parameters
    ----------
    prior : Prior
        Prior drawn from.
    site : str
        Site drawn, named in the error message.
    shape : tuple[int, ...]
        Shape of the draws.
    like : Tensor
        Tensor whose floating-point kind and device the draws are written at.
    generator : torch.Generator
        Source of the draws.

    Raises
    ------
    ValueError
        If the prior's family has no sampler this module implements.
    """

    def standard() -> Tensor:
        return torch.randn(
            shape, dtype=like.dtype, device=like.device, generator=generator
        )

    def uniform() -> Tensor:
        return torch.rand(
            shape, dtype=like.dtype, device=like.device, generator=generator
        )

    match prior.family, prior.arguments:
        case "Normal", (location, scale):
            return location + scale * standard()
        case "HalfNormal", (scale,):
            return scale * standard().abs()
        case "LogNormal", (location, scale):
            return torch.exp(location + scale * standard())
        case "Uniform", (low, high):
            return low + (high - low) * uniform()
        case "Exponential", (rate,):
            return -torch.log1p(-uniform()) / rate

    msg = f"the site {site!r} is drawn from {prior}, which has no sampler here"

    raise ValueError(msg)


class _Axis(NamedTuple):
    """One axis of a drawn site's grid: its factor, if any, and its fitted size."""

    factor: str | None
    size: int


class _Alignment(NamedTuple):
    """Where each target level of a factor sits among the fitted levels."""

    index: tuple[int, ...]
    unseen: int


def _align(fitted: tuple[str, ...], target: tuple[str, ...]) -> _Alignment:
    """Return the fitted index of each target level, unseen levels numbered after.

    Parameters
    ----------
    fitted : tuple[str, ...]
        Levels the fit saw, in its index order.
    target : tuple[str, ...]
        Levels of the design predicted, in its index order.
    """
    positions = {level: position for position, level in enumerate(fitted)}
    index: list[int] = []
    unseen = 0

    for level in target:
        position = positions.get(level)

        if position is None:
            position = len(fitted) + unseen
            unseen += 1

        index.append(position)

    return _Alignment(tuple(index), unseen)


def predictive_log_density(probabilities: Tensor, responses: Tensor) -> Tensor:
    """Return the log probability of each row's recorded level.

    Parameters
    ----------
    probabilities : Tensor
        Predicted probability of each level, one row per response row.
    responses : Tensor
        Zero-based recorded level of each row.
    """
    return torch.log(probabilities.gather(1, responses.unsqueeze(1)).squeeze(1))


def predictive_density_row(fit: str, log_densities: Tensor, /) -> PredictiveDensityRow:
    """Reduce row log densities to one named fit report row.

    Parameters
    ----------
    fit : str
        Name of the fitted specification being reported.
    log_densities : Tensor
        Finite log predictive density of each held-out response row.

    Raises
    ------
    TypeError
        If the densities are not floating point.
    ValueError
        If the tensor is not a nonempty finite vector.
    """
    if log_densities.ndim != 1 or log_densities.numel() == 0:
        msg = (
            "predictive densities must be a nonempty vector, got shape "
            f"{tuple(log_densities.shape)}"
        )

        raise ValueError(msg)

    if not log_densities.dtype.is_floating_point:
        msg = f"predictive densities must be floating point, not {log_densities.dtype}"

        raise TypeError(msg)

    if not bool(torch.isfinite(log_densities).all()):
        msg = "predictive densities must be finite"

        raise ValueError(msg)

    return PredictiveDensityRow(
        fit=fit,
        predictive_density=Measurement(
            value=float(log_densities.mean()),
            n=log_densities.numel(),
            unit="response",
        ),
    )


def posterior_variance_row(
    fit: str,
    group: str,
    variances: Tensor,
    /,
    *,
    level: float = 0.95,
) -> PosteriorVarianceRow:
    """Reduce posterior variance-component draws to one named report row.

    Parameters
    ----------
    fit : str
        Name of the fitted specification being reported.
    group : str
        Random-effect group whose variance component was drawn.
    variances : Tensor
        Nonnegative posterior draws of the variance component.
    level : float
        Equal-tail interval coverage.

    Raises
    ------
    TypeError
        If the draws are not floating point.
    ValueError
        If the draws are not a nonempty finite nonnegative vector, or the
        requested coverage is not a proportion.
    """
    if variances.ndim != 1 or variances.numel() == 0:
        msg = (
            "posterior variances must be a nonempty vector, got shape "
            f"{tuple(variances.shape)}"
        )

        raise ValueError(msg)

    if not variances.dtype.is_floating_point:
        msg = f"posterior variances must be floating point, not {variances.dtype}"

        raise TypeError(msg)

    if not bool(torch.isfinite(variances).all()):
        msg = "posterior variances must be finite"

        raise ValueError(msg)

    if bool((variances < 0.0).any()):
        msg = "posterior variances must be nonnegative"

        raise ValueError(msg)

    if not math.isfinite(level) or not 0.0 < level < 1.0:
        msg = f"posterior interval coverage is not a proportion: {level!r}"

        raise ValueError(msg)

    tail = (1.0 - level) / 2.0
    bounds = torch.quantile(
        variances,
        torch.tensor(
            (tail, 1.0 - tail), dtype=variances.dtype, device=variances.device
        ),
    )

    return PosteriorVarianceRow(
        fit=fit,
        group=group,
        posterior_variance=Measurement(
            value=float(variances.mean()),
            n=variances.numel(),
            unit="posterior_draw",
            lower=float(bounds[0]),
            upper=float(bounds[1]),
            level=level,
        ),
    )


def pseudo_r_squared(
    fit: str,
    log_likelihood: Measurement,
    /,
    *,
    reference: str,
    references: Mapping[str, Measurement],
) -> PseudoRSquaredRow:
    """Return a pseudo-R-squared report row against a named reference.

    The fit and reference log-likelihoods must summarize the same number and
    kind of units. Their values may be totals or means because the common
    count makes their ratio identical.

    Parameters
    ----------
    fit : str
        Name of the fitted specification being reported.
    log_likelihood : Measurement
        Fit log likelihood.
    reference : str
        Name of the specification whose likelihood supplies the denominator.
    references : Mapping[str, Measurement]
        Reference log-likelihoods over the same units, by specification name.

    Raises
    ------
    ValueError
        If the named reference is unavailable, either likelihood is undefined,
        the reference is zero, or the two measurements cover different counts
        or units.
    """
    try:
        reference_log_likelihood = references[reference]
    except KeyError:
        available = ", ".join(sorted(references))
        msg = (
            f"pseudo-R-squared reference {reference!r} is missing from the "
            f"available references: {available}"
        )
        raise ValueError(msg) from None

    if (
        log_likelihood.n != reference_log_likelihood.n
        or log_likelihood.unit != reference_log_likelihood.unit
    ):
        msg = (
            "pseudo-R-squared compares log-likelihoods over the same units, and "
            f"the fit covers {log_likelihood.n} {log_likelihood.unit!r} units "
            f"while the reference covers {reference_log_likelihood.n} "
            f"{reference_log_likelihood.unit!r} units"
        )
        raise ValueError(msg)

    fitted = log_likelihood.value
    baseline = reference_log_likelihood.value

    if fitted is None or baseline is None:
        msg = "pseudo-R-squared is undefined when either log-likelihood is undefined"
        raise ValueError(msg)

    if baseline == 0.0:
        msg = "pseudo-R-squared has no ratio against a zero reference log-likelihood"
        raise ValueError(msg)

    measured = Measurement(
        value=1.0 - fitted / baseline,
        n=log_likelihood.n,
        unit=log_likelihood.unit,
    )

    return PseudoRSquaredRow(
        fit=fit,
        reference=reference,
        pseudo_r_squared=measured,
    )


@final
class FoldPredictions:
    """The predicted level probabilities of every row under each convention.

    Parameters
    ----------
    marginal : Tensor
        Probability of each level per row, integrating over the population
        of participants for a participant the fit never saw.
    plug_in : Tensor
        Probability of each level per row, at the population mean effects
        for a participant the fit never saw.

    Raises
    ------
    ValueError
        If the two tables are not one row of probabilities per response
        row at the same shape.
    """

    __slots__ = ("_marginal", "_plug_in")

    def __init__(self, marginal: Tensor, plug_in: Tensor) -> None:
        if marginal.ndim != 2 or marginal.shape != plug_in.shape:
            msg = (
                f"predictions are one row of probabilities per response row under "
                f"each convention, and were given shapes {tuple(marginal.shape)} "
                f"and {tuple(plug_in.shape)}"
            )

            raise ValueError(msg)

        object.__setattr__(self, "_marginal", marginal)
        object.__setattr__(self, "_plug_in", plug_in)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "fold predictions are immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "fold predictions are immutable"

        raise AttributeError(msg)

    @property
    def marginal(self) -> Tensor:
        """Return the probabilities under the marginal convention."""
        return self._marginal

    @property
    def plug_in(self) -> Tensor:
        """Return the probabilities under the plug-in convention."""
        return self._plug_in

    @property
    def rows(self) -> int:
        """Return the number of response rows predicted."""
        return self._marginal.shape[0]

    @property
    def levels(self) -> int:
        """Return the number of response levels."""
        return self._marginal.shape[1]

    def under(self, policy: HeldOutParticipantPolicy, /) -> Tensor:
        """Return the probabilities under one convention.

        Parameters
        ----------
        policy : HeldOutParticipantPolicy
            Convention read.
        """
        match policy:
            case HeldOutParticipantPolicy.MARGINAL:
                return self._marginal
            case HeldOutParticipantPolicy.PLUG_IN:
                return self._plug_in

    @override
    def __repr__(self) -> str:
        """Return the row and level counts."""
        return f"FoldPredictions(rows={self.rows}, levels={self.levels})"


def _with_response(row: JudgmentRow, response: int) -> JudgmentRow:
    """Return one boundary row with another candidate response level."""
    return JudgmentRow(
        participant=row.participant,
        item=row.item,
        response=response,
        columns=RowColumns(
            factors=dict(row.factors),
            readouts=dict(row.readouts),
            alternative_readouts=dict(row.alternative_readouts),
        ),
        alternatives=row.alternatives,
    )


@final
class MarginalPointPredictor:
    """Predict held-out responses from a fitted marginal population point.

    Each candidate response is scored by the same complete-local-block
    integration kernel used to fit the population point. For a participant in
    the retained design, the integral contains that participant's retained
    rows and the candidate row. For a new participant, the integral contains
    only the candidate row. Candidate log scores are normalized over the
    response levels, so the finite integration region does not contribute a
    response-independent normalization constant.

    Parameters
    ----------
    program : ResponseProgram
        Compiled truncated marginal response program.
    codec : DesignCodec
        Codec that produced the retained and target designs.
    retained : JudgmentDesign
        Response rows used to fit the population point.
    fit : PointEstimate
        Fitted point in the marginal kernel's population layout.

    Raises
    ------
    TypeError
        If the point does not hold real floating-point parameters.
    ValueError
        If the point is not a finite vector in the kernel's population layout.
    MarginalLikelihoodError
        If the program or retained design cannot form a marginal kernel.
    """

    __slots__ = ("_codec", "_layout", "_parameters", "_program", "_retained")

    def __init__(
        self,
        program: ResponseProgram,
        codec: DesignCodec,
        retained: JudgmentDesign,
        fit: PointEstimate,
    ) -> None:
        retained_kernel = ApproximateMarginalLikelihood(
            program.over(retained),
            active_groups=tuple(range(len(retained.participant_levels))),
        )
        parameters = fit.parameters

        if (
            not isinstance(parameters, torch.Tensor)
            or not parameters.is_floating_point()
        ):
            kind = type(parameters).__name__

            if isinstance(parameters, torch.Tensor):
                kind = str(parameters.dtype)

            msg = f"a marginal point holds {kind}, not real floating-point parameters"

            raise TypeError(msg)

        if parameters.is_complex():
            msg = "a marginal point holds complex, not real floating-point parameters"

            raise TypeError(msg)

        if (
            parameters.ndim != 1
            or parameters.numel() != retained_kernel.population_size
        ):
            msg = (
                "a marginal point follows the kernel's population layout of "
                f"{retained_kernel.population_size} values, not shape "
                f"{tuple(parameters.shape)!r}"
            )

            raise ValueError(msg)

        if not bool(torch.isfinite(parameters).all()):
            msg = "a marginal point contains a nonfinite parameter"

            raise ValueError(msg)

        self._program = program
        self._codec = codec
        self._retained = codec.decode(retained)
        self._layout = retained_kernel.population_layout
        self._parameters = parameters.detach().clone()

    @property
    def parameters(self) -> Tensor:
        """Return an independent snapshot of the fitted population point."""
        return self._parameters.clone()

    def _log_score(self, row: JudgmentRow, response: int) -> Tensor:
        """Return one candidate's integrated log score for its participant."""
        candidate = _with_response(row, response)
        design = self._codec.encode((*self._retained, candidate))
        target_group = design.participant_levels.index(row.participant)
        kernel = ApproximateMarginalLikelihood(
            self._program.over(design),
            active_groups=tuple(range(len(design.participant_levels))),
        )

        if kernel.population_layout != self._layout:
            msg = (
                "a held-out row changes the marginal population layout from "
                f"{self._layout!r} to {kernel.population_layout!r}"
            )

            raise ValueError(msg)

        return kernel.group_log_likelihood(self._parameters, target_group)

    def probabilities(self, design: JudgmentDesign, /) -> Tensor:
        """Return normalized marginal probabilities for the design's rows.

        Parameters
        ----------
        design : JudgmentDesign
            Held-out rows. Their recorded responses are placeholders; every
            response level is scored as a candidate.

        Returns
        -------
        Tensor
            One row of level probabilities per held-out response row.
        """
        rows = self._codec.decode(design)

        if not rows:
            return self._parameters.new_empty((0, design.response_levels))

        probabilities = tuple(
            torch.softmax(
                torch.stack(
                    tuple(
                        self._log_score(row, response)
                        for response in range(design.response_levels)
                    )
                ),
                dim=0,
            )
            for row in rows
        )

        return torch.stack(probabilities)

    @override
    def __repr__(self) -> str:
        """Return the retained-row and population-parameter counts."""
        return (
            f"MarginalPointPredictor(rows={len(self._retained)}, "
            f"parameters={self._parameters.numel()})"
        )


@final
class HeldOutPredictor:
    """Posterior draws over one fitted design, applied to other designs.

    Every drawn site indexed by a grouping factor is realigned from the
    fitted design's levels to the predicted design's levels. A participant
    the fit never saw takes the convention's value at each site indexed by
    the participant: the population mean of the site's prior, or fresh
    draws from it. An unseen level of any other factor is refused.

    Parameters
    ----------
    program : ResponseProgram
        Compiled response program the draws are over.
    fitted : JudgmentDesign
        Design the program was bound to when the draws were made.
    draws : Tensor
        One parameter vector per row, laid out as the program bound to the
        fitted design lays its sites out.
    samples : int
        Number of draws of an unseen participant's effects paired with each
        posterior draw under the marginal convention.

    Raises
    ------
    TypeError
        If the draws hold values of a nonfloating kind, or the sample count
        is not an integer.
    ValueError
        If the draws are not a matrix with at least one row of the bound
        program's length, or the sample count is below one.
    """

    __slots__ = ("_draws", "_fit", "_fitted", "_program", "_samples")

    def __init__(
        self,
        program: ResponseProgram,
        fitted: JudgmentDesign,
        draws: Tensor,
        *,
        samples: int,
    ) -> None:
        fit = program.over(fitted)

        if not draws.is_floating_point():
            msg = (
                f"posterior draws hold {draws.dtype}, which is not a floating-point "
                f"kind"
            )

            raise TypeError(msg)

        if draws.ndim != 2 or draws.shape[0] == 0 or draws.shape[1] != fit.size:
            msg = (
                f"posterior draws are at least one row of {fit.size} parameters, "
                f"and were given at shape {tuple(draws.shape)}"
            )

            raise ValueError(msg)

        if isinstance(samples, bool) or not isinstance(samples, int):
            msg = f"a sample count is not an integer: {samples!r}"

            raise TypeError(msg)

        if samples < 1:
            msg = f"each posterior draw is paired with at least one sample: {samples}"

            raise ValueError(msg)

        self._program = program
        self._fitted = fitted
        self._fit = fit
        self._draws = draws
        self._samples = samples

    @property
    def program(self) -> ResponseProgram:
        """Return the compiled program the draws are over."""
        return self._program

    @property
    def fit(self) -> BoundProgram:
        """Return the program bound to the fitted design."""
        return self._fit

    @property
    def draws(self) -> Tensor:
        """Return the posterior draws, one parameter vector per row."""
        return self._draws

    @property
    def samples(self) -> int:
        """Return the effect draws paired with each posterior draw when marginal."""
        return self._samples

    def _axes(self, index: tuple[str, ...]) -> tuple[_Axis, ...]:
        """Return the grid axes of a site indexed by a product of finite sets.

        A set that is the product of factors contributes one axis per
        factor, sized by the fitted design's level table. Any other set has
        the same size under every design and contributes one axis of it.

        Parameters
        ----------
        index : tuple[str, ...]
            Finite sets the site is indexed by.
        """
        axes: list[_Axis] = []

        for name in index:
            factors = self._program.objects.get(name)

            if factors is None:
                axes.append(_Axis(None, self._fit.sizes[name]))
            else:
                axes.extend(
                    _Axis(factor, len(_levels_of(self._fitted, factor)))
                    for factor in factors
                )

        return tuple(axes)

    def _realigned(
        self,
        grid: Tensor,
        position: int,
        axis: _Axis,
        design: JudgmentDesign,
        *,
        site: str,
        prior: Prior,
        policy: HeldOutParticipantPolicy,
        generator: torch.Generator,
    ) -> Tensor:
        """Return a grid with one factor's axis reordered to a design's levels.

        Parameters
        ----------
        grid : Tensor
            Site values with a leading draw axis, then one axis per grid axis.
        position : int
            Axis of the grid the factor indexes.
        axis : _Axis
            The factor realigned and its fitted size.
        design : JudgmentDesign
            Design whose level order the axis takes.
        site : str
            Site the grid holds, named in an error message.
        prior : Prior
            Prior the site is drawn from.
        policy : HeldOutParticipantPolicy
            Convention for a participant the fit never saw.
        generator : torch.Generator
            Source of the draws under the marginal convention.

        Raises
        ------
        ValueError
            If the design has a level of a factor other than the participant
            the fit never saw, or the prior has no mean or sampler.
        """
        factor = axis.factor

        if factor is None:
            return grid

        alignment = _align(_levels_of(self._fitted, factor), _levels_of(design, factor))

        if alignment.unseen:
            if factor != PARTICIPANT:
                msg = (
                    f"the design has {alignment.unseen} levels of {factor!r} the "
                    f"fit never saw, and only an unseen participant is predicted"
                )

                raise ValueError(msg)

            shape = (
                *grid.shape[:position],
                alignment.unseen,
                *grid.shape[position + 1 :],
            )
            fill = self._fill(prior, site, shape, policy, generator)
            grid = torch.cat((grid, fill), dim=position)

        index = torch.tensor(alignment.index, dtype=torch.int64, device=grid.device)

        return grid.index_select(position, index)

    def parameters(
        self,
        design: JudgmentDesign,
        /,
        *,
        policy: HeldOutParticipantPolicy,
        generator: torch.Generator,
    ) -> Tensor:
        """Return parameter vectors over a design, one per row of the result.

        Under the plug-in convention there is one vector per posterior
        draw. Under the marginal convention each posterior draw is repeated
        once per sample, and each repetition pairs it with fresh draws of
        every unseen participant's effects.

        Parameters
        ----------
        design : JudgmentDesign
            Design the vectors are laid out over.
        policy : HeldOutParticipantPolicy
            Convention for a participant the fit never saw.
        generator : torch.Generator
            Source of the effect draws under the marginal convention.

        Raises
        ------
        ValueError
            If the design cannot bind the program, has a level of a factor
            other than the participant the fit never saw, or a participant
            site's prior has no mean or sampler.
        """
        self._program.over(design)

        return self._vectors(design, policy, generator)

    def _vectors(
        self,
        design: JudgmentDesign,
        policy: HeldOutParticipantPolicy,
        generator: torch.Generator,
    ) -> Tensor:
        """Return the draws realigned to a design the program binds.

        Parameters
        ----------
        design : JudgmentDesign
            Design the vectors are laid out over.
        policy : HeldOutParticipantPolicy
            Convention for a participant the fit never saw.
        generator : torch.Generator
            Source of the effect draws under the marginal convention.
        """
        draws = self._draws

        if policy is HeldOutParticipantPolicy.MARGINAL:
            draws = draws.repeat_interleave(self._samples, dim=0)

        count = draws.shape[0]
        pieces: list[Tensor] = []
        offset = 0
        priors = {draw.site: draw.prior for draw in self._program.draws}
        indices = {draw.site: draw.index for draw in self._program.draws}

        for slot in self._fit.layout:
            axes = self._axes(indices[slot.site])
            grid = draws[:, offset : offset + slot.size].reshape(
                count, *(axis.size for axis in axes)
            )
            offset += slot.size

            for position, axis in enumerate(axes, start=1):
                grid = self._realigned(
                    grid,
                    position,
                    axis,
                    design,
                    site=slot.site,
                    prior=priors[slot.site],
                    policy=policy,
                    generator=generator,
                )

            pieces.append(grid.reshape(count, -1))

        return torch.cat(pieces, dim=1)

    def _fill(
        self,
        prior: Prior,
        site: str,
        shape: tuple[int, ...],
        policy: HeldOutParticipantPolicy,
        generator: torch.Generator,
    ) -> Tensor:
        """Return an unseen participant's values at a site under a convention.

        Parameters
        ----------
        prior : Prior
            Prior the site is drawn from.
        site : str
            Site filled, named in an error message.
        shape : tuple[int, ...]
            Shape of the values.
        policy : HeldOutParticipantPolicy
            Convention for a participant the fit never saw.
        generator : torch.Generator
            Source of the draws under the marginal convention.
        """
        like = self._draws

        match policy:
            case HeldOutParticipantPolicy.MARGINAL:
                return _prior_sample(prior, site, shape, like=like, generator=generator)
            case HeldOutParticipantPolicy.PLUG_IN:
                return like.new_full(shape, _population_mean(prior, site))

    def probabilities(
        self,
        design: JudgmentDesign,
        /,
        *,
        policy: HeldOutParticipantPolicy,
        generator: torch.Generator,
    ) -> Tensor:
        """Return the predicted probability of each level for each row of a design.

        The prediction is the mean over parameter vectors of the level
        probabilities each vector gives the row.

        Parameters
        ----------
        design : JudgmentDesign
            Design whose rows are predicted.
        policy : HeldOutParticipantPolicy
            Convention for a participant the fit never saw.
        generator : torch.Generator
            Source of the effect draws under the marginal convention.

        Raises
        ------
        ValueError
            If the design cannot bind the program, has a level of a factor
            other than the participant the fit never saw, or a participant
            site's prior has no mean or sampler.
        """
        target = self._program.over(design)
        vectors = self._vectors(design, policy, generator)
        total = vectors.new_zeros((design.n_rows, design.response_levels))

        for vector in vectors:
            location, thresholds = latent(target, vector)
            total = total + torch.exp(log_category_probabilities(location, thresholds))

        return total / vectors.shape[0]

    def both(
        self, design: JudgmentDesign, /, *, generator: torch.Generator
    ) -> FoldPredictions:
        """Return the predictions for a design under both conventions.

        Parameters
        ----------
        design : JudgmentDesign
            Design whose rows are predicted.
        generator : torch.Generator
            Source of the effect draws under the marginal convention.
        """
        return FoldPredictions(
            self.probabilities(
                design, policy=HeldOutParticipantPolicy.MARGINAL, generator=generator
            ),
            self.probabilities(
                design, policy=HeldOutParticipantPolicy.PLUG_IN, generator=generator
            ),
        )

    @override
    def __repr__(self) -> str:
        """Return the draw count, the parameter count and the sample count."""
        return (
            f"HeldOutPredictor(draws={self._draws.shape[0]}, "
            f"parameters={self._fit.size}, samples={self._samples})"
        )


def fold_predictions(
    program: ResponseProgram,
    codec: DesignCodec,
    fold: Fold,
    draws: Tensor,
    *,
    samples: int,
    generator: torch.Generator,
) -> FoldPredictions:
    """Return the predictions for a fold's held-out rows under both conventions.

    Parameters
    ----------
    program : ResponseProgram
        Compiled response program the draws are over.
    codec : DesignCodec
        Codec encoding the fold's rows as designs.
    fold : Fold
        Fold whose retained rows the draws were fitted on and whose held-out
        rows are predicted.
    draws : Tensor
        One parameter vector per row, laid out as the program bound to the
        retained rows' design lays its sites out.
    samples : int
        Number of draws of an unseen participant's effects paired with each
        posterior draw under the marginal convention.
    generator : torch.Generator
        Source of those draws.
    """
    predictor = HeldOutPredictor(
        program, codec.encode(fold.retained), draws, samples=samples
    )

    return predictor.both(codec.encode(fold.held_out), generator=generator)
